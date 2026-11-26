"""Option portfolio analytics for the Portfolio tab. numpy/pandas only (no scipy).

Pieces:
  * Black-Scholes price / greeks / implied vol (continuous r and q).
  * Multi-leg payoff curves: today, and at the first expiry (calendars / diagonals: the short leg has settled, the
    long leg is still alive and valued with its IV held fixed). Breakevens are the zero crossings of those curves.
    There is no curve at a later expiry: the long leg's outcome after the first expiry depends on what you do next.
  * A rules-based advisor: BOOK PROFIT / HOLD / CUT LOSS / ADJUST per trade, with the reasons spelled out.
  * Portfolio summary (cash, equity, unrealised / realised P&L, return) and closed-trade statistics.

Conventions: qty is signed contracts (+ long, - short), 100 shares per option contract, prices are per share.
cost = sum(qty * mult * entry_price), so a debit is positive and a credit negative.  Everything is a model: marks
you type in win over model prices, IVs are backed out of marks, and nothing here knows about margin rules.
"""
import datetime as dt
import math

import numpy as np
import pandas as pd

MULT = 100
POSITION_COLS = ["trade", "underlying", "type", "strike", "expiry", "qty", "entry_price", "iv_pct", "mark"]
CLOSED_COLS = ["trade", "underlying", "close_date", "realized_pnl"]

DEFAULT_RULES = {
    "debit_target": 50.0,      # book a long-premium trade at +X% of the amount at risk
    "calendar_target": 25.0,   # book a calendar / multi-expiry trade at +X% of its debit
    "credit_target": 50.0,     # book a credit trade after capturing X% of the credit
    "stop_loss": 50.0,         # cut at -X% of the amount at risk (any trade type)
    "calendar_exit_dte": 2,    # close / roll a calendar when the front leg has <= N days left
    "credit_exit_dte": 7,      # same for credit trades
    "debit_exit_dte": 5,       # same for long-premium trades
    "tested_em": 0.35,         # "tested" = spot within this many expected moves of a breakeven
    "short_delta_tested": 0.30,  # credit trade tested when a short leg's |delta| reaches this
    "lock_in_frac": 0.6,       # long-premium trade: suggest locking in gains at this fraction of the target
}

_erfc = np.vectorize(math.erfc, otypes=[float])
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------------------------------- pricing
def norm_cdf(x):
    return 0.5 * _erfc(-np.asarray(x, float) / _SQRT2)


def norm_pdf(x):
    x = np.asarray(x, float)
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _date(x):
    if isinstance(x, dt.datetime):
        return x.date()
    return x


def years(expiry, on_date):
    """Calendar-day year fraction from on_date to expiry, floored at zero."""
    return max((_date(expiry) - _date(on_date)).days, 0) / 365.0


def bs_price(kind, spot, strike, t, iv, r=0.0, q=0.0):
    """Black-Scholes value per share. kind 'C', 'P' or 'S' (stock). spot may be an array. t <= 0 gives intrinsic."""
    s = np.asarray(spot, float)
    if kind == "S":
        return s
    if t <= 0:
        return np.maximum(s - strike, 0.0) if kind == "C" else np.maximum(strike - s, 0.0)
    iv = max(float(iv), 1e-4)
    sd = iv * math.sqrt(t)
    d1 = (np.log(s / strike) + (r - q + 0.5 * iv * iv) * t) / sd
    d2 = d1 - sd
    df_r, df_q = math.exp(-r * t), math.exp(-q * t)
    if kind == "C":
        return s * df_q * norm_cdf(d1) - strike * df_r * norm_cdf(d2)
    return strike * df_r * norm_cdf(-d2) - s * df_q * norm_cdf(-d1)


def option_value(kind, spot, strike, t, iv, r=0.0, q=0.0):
    """Black-Scholes floored at intrinsic: a cheap stand-in for early exercise (QQQ / SPY / stock options are American).
    Without the floor a deep in-the-money long put is worth less than intrinsic at positive rates, which would make a
    calendar's modelled loss exceed its debit."""
    s = np.asarray(spot, float)
    intrinsic = np.maximum(s - strike, 0.0) if kind == "C" else np.maximum(strike - s, 0.0)
    return np.maximum(bs_price(kind, s, strike, t, iv, r, q), intrinsic)


def bs_greeks(kind, spot, strike, t, iv, r=0.0, q=0.0):
    """Per-share greeks at a scalar spot: delta, gamma (per $), theta ($ per calendar day), vega ($ per vol point)."""
    if kind == "S":
        return {"delta": 1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    if t <= 0:
        itm = spot > strike if kind == "C" else spot < strike
        return {"delta": (1.0 if kind == "C" else -1.0) if itm else 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    iv = max(float(iv), 1e-4)
    sd = iv * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t) / sd
    d2 = d1 - sd
    pdf, nd1, nd2 = float(norm_pdf(d1)), float(norm_cdf(d1)), float(norm_cdf(d2))
    df_r, df_q = math.exp(-r * t), math.exp(-q * t)
    decay = -spot * df_q * pdf * iv / (2.0 * math.sqrt(t))
    if kind == "C":
        delta = df_q * nd1
        theta = decay - r * strike * df_r * nd2 + q * spot * df_q * nd1
    else:
        delta = -df_q * (1.0 - nd1)
        theta = decay + r * strike * df_r * (1.0 - nd2) - q * spot * df_q * (1.0 - nd1)
    return {"delta": delta, "gamma": df_q * pdf / (spot * sd), "theta": theta / 365.0,
            "vega": spot * df_q * pdf * math.sqrt(t) / 100.0}


def implied_vol(kind, price, spot, strike, t, r=0.0, q=0.0, lo=0.01, hi=5.0):
    """Bisection implied vol (decimal). None when the price is outside what the model can produce."""
    if kind == "S" or t <= 0 or price is None or not np.isfinite(price) or price <= 0:
        return None

    def f(v):
        return float(bs_price(kind, spot, strike, t, v, r, q)) - price

    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def round_strike(x, step=1.0):
    return round(round(float(x) / step) * step, 4)


def default_step(spot):
    return 0.5 if spot < 25 else (1.0 if spot < 1000 else 5.0)


# ------------------------------------------------------------------------------------------- positions
_TYPE_MAP = {"c": "C", "call": "C", "calls": "C", "p": "P", "put": "P", "puts": "P",
             "s": "S", "stock": "S", "share": "S", "shares": "S", "equity": "S"}


def empty_positions():
    return pd.DataFrame({"trade": pd.Series(dtype=object), "underlying": pd.Series(dtype=object),
                         "type": pd.Series(dtype=object), "strike": pd.Series(dtype=float),
                         "expiry": pd.Series(dtype=object), "qty": pd.Series(dtype=float),
                         "entry_price": pd.Series(dtype=float), "iv_pct": pd.Series(dtype=float),
                         "mark": pd.Series(dtype=float)})


def empty_closed():
    return pd.DataFrame({"trade": pd.Series(dtype=object), "underlying": pd.Series(dtype=object),
                         "close_date": pd.Series(dtype=object), "realized_pnl": pd.Series(dtype=float)})


def normalize_positions(df, require_price=True):
    """Clean a user-edited or uploaded positions table. Returns (clean frame, list of issue strings).

    Columns are matched case-insensitively. Type accepts C/P/call/put/stock. A blank trade name makes each row its
    own trade. Rows that cannot be used (missing underlying, type, qty, entry price, or strike/expiry on an option)
    are dropped and reported. require_price=False keeps rows with a blank entry price (the trade evaluator fills
    those from the model)."""
    issues = []
    if df is None or len(df) == 0:
        return empty_positions(), issues
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    for c in POSITION_COLS:
        if c not in d:
            d[c] = np.nan
    d = d[POSITION_COLS].copy()
    d = d.dropna(how="all")
    if d.empty:
        return empty_positions(), issues
    for c in ("strike", "qty", "entry_price", "iv_pct", "mark"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["expiry"] = pd.to_datetime(d["expiry"], errors="coerce").dt.date
    d["type"] = d["type"].astype(str).str.strip().str.lower().map(_TYPE_MAP)
    und = d["underlying"].astype(str).str.strip().str.upper()
    d["underlying"] = und.where(~und.isin(["", "NAN", "NONE", "<NA>"]), np.nan)
    is_opt = d["type"].isin(["C", "P"])
    bad = (d["underlying"].isna() | d["type"].isna() | d["qty"].isna() | (d["qty"] == 0)
           | (d["entry_price"].isna() if require_price else False) | (is_opt & (d["strike"].isna() | d["expiry"].isna())))
    for pos, (idx, is_bad) in enumerate(bad.items(), start=1):
        if is_bad and len(issues) < 6:
            issues.append(f"Row {pos} skipped: needs underlying, type (C/P/stock), non-zero qty, entry price, "
                          f"and strike + expiry for options.")
    d = d[~bad].copy()
    names = []
    for row in d.itertuples():
        t = row.trade
        if not isinstance(t, str) or not t.strip() or t.strip().lower() == "nan":
            if row.type == "S":
                t = f"{row.underlying} shares"
            else:
                t = f"{row.underlying} {row.type}{row.strike:g} {row.expiry:%d%b%y}"
        names.append(t.strip())
    d["trade"] = names
    d.loc[d["type"] == "S", ["strike", "expiry"]] = np.nan
    return d.reset_index(drop=True), issues


def normalize_closed(df):
    if df is None or len(df) == 0:
        return empty_closed()
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    for c in CLOSED_COLS:
        if c not in d:
            d[c] = np.nan
    d = d[CLOSED_COLS].copy()
    d["realized_pnl"] = pd.to_numeric(d["realized_pnl"], errors="coerce")
    d["close_date"] = pd.to_datetime(d["close_date"], errors="coerce").dt.date
    d = d[d["realized_pnl"].notna()]
    return d.reset_index(drop=True)


def with_mult(d):
    d = d.copy()
    d["mult"] = np.where(d["type"] == "S", 1, MULT)
    return d


def resolve_legs(legs, spots, default_iv, today, r=0.0, q=0.0):
    """Add `iv` (decimal) and `mark_eff` to each leg.

    iv: the iv_pct you typed, else implied from the mark, else the underlying's default IV.
    mark_eff: the mark you typed, else the Black-Scholes value at today's spot with that iv.
    spots and default_iv are {underlying: value}; legs whose underlying has no spot get NaN marks."""
    d = with_mult(legs)
    ivs, marks = [], []
    for row in d.itertuples():
        spot = spots.get(row.underlying)
        have_mark = pd.notna(row.mark)
        if row.type == "S":
            ivs.append(0.0)
            marks.append(float(row.mark) if have_mark else (float(spot) if spot else np.nan))
            continue
        dflt = float(default_iv.get(row.underlying, 0.25))
        t = years(row.expiry, today)
        if pd.notna(row.iv_pct) and row.iv_pct > 0:
            iv = row.iv_pct / 100.0
        elif have_mark and row.mark > 0 and spot:
            iv = implied_vol(row.type, float(row.mark), float(spot), row.strike, t, r, q) or dflt
        else:
            iv = dflt
        ivs.append(iv)
        if have_mark:
            marks.append(float(row.mark))
        elif spot:
            marks.append(float(option_value(row.type, float(spot), row.strike, t, iv, r, q)))
        else:
            marks.append(np.nan)
    d["iv"] = ivs
    d["mark_eff"] = marks
    return d


# ----------------------------------------------------------------------------------------- trade analysis
def trade_value(legs, grid, on_date, r=0.0, q=0.0, iv_shift=0.0):
    """Position value ($) on each spot in `grid` at `on_date`; legs past expiry are worth intrinsic."""
    grid = np.asarray(grid, float)
    total = np.zeros(len(grid))
    for row in legs.itertuples():
        if row.type == "S":
            total += row.qty * row.mult * grid
            continue
        t = years(row.expiry, on_date)
        total += row.qty * row.mult * np.asarray(option_value(row.type, grid, row.strike, t, max(row.iv + iv_shift, 0.01), r, q))
    return total


def breakevens(grid, pnl):
    """Spot levels where the P&L curve crosses zero (linear interpolation)."""
    grid, pnl = np.asarray(grid, float), np.asarray(pnl, float)
    out = []
    for i in range(len(grid) - 1):
        a, b = pnl[i], pnl[i + 1]
        if a == 0.0:
            x = grid[i]
        elif a * b < 0:
            x = grid[i] - a * (grid[i + 1] - grid[i]) / (b - a)
        else:
            continue
        if not out or abs(x - out[-1]) > (grid[1] - grid[0]) * 1.5:
            out.append(float(x))
    return [round(x, 2) for x in out]


def prob_stats(grid, pnl, spot, t, iv, r=0.0, q=0.0):
    """(probability of profit, expected P&L) at a horizon of t years under a lognormal at `iv`. Model numbers."""
    grid, pnl = np.asarray(grid, float), np.asarray(pnl, float)
    here = float(np.interp(spot, grid, pnl))
    if t <= 0 or iv <= 0:
        return (1.0 if here > 0 else 0.0), here
    sd = iv * math.sqrt(t)
    z = (np.log(grid / spot) - (r - q - 0.5 * iv * iv) * t) / sd
    w = norm_pdf(z) / (grid * sd)
    w = w / w.sum()
    return float(w[pnl > 0].sum()), float((w * pnl).sum())


def classify_trade(legs, cost):
    opts = legs[legs["type"].isin(["C", "P"])]
    if opts.empty:
        return "stock"
    if opts["expiry"].nunique() > 1:
        return "calendar"
    return "credit" if cost < 0 else "debit"


def _curve_stats(grid, pnl):
    return {"max_profit": float(np.max(pnl)), "max_loss": float(np.min(pnl)), "breakevens": breakevens(grid, pnl)}


def analyze_trade(legs, spot, today, r=0.0, q=0.0, iv_shift=0.0, ref_iv=None, span=None, n=361):
    """Analyse one trade (all legs on one underlying; legs from resolve_legs). Returns a dict of metrics and curves."""
    legs = legs.reset_index(drop=True)
    opts = legs[legs["type"].isin(["C", "P"])]
    cost = float((legs["qty"] * legs["mult"] * legs["entry_price"]).sum())
    value_now = float((legs["qty"] * legs["mult"] * legs["mark_eff"]).sum())
    kind = classify_trade(legs, cost)
    first_exp = min(opts["expiry"]) if not opts.empty else None
    last_exp = max(opts["expiry"]) if not opts.empty else None
    ref_iv = float(ref_iv) if ref_iv else (float(opts["iv"].mean()) if not opts.empty else 0.25)
    t_first = years(first_exp, today) if first_exp else 0.0
    if span is None:
        span = float(np.clip(4.5 * ref_iv * math.sqrt(max(t_first, 1 / 365.0)), 0.20, 0.80))
    grid = np.linspace(spot * (1 - span), spot * (1 + span), n)

    pnl_today = trade_value(legs, grid, today, r, q, iv_shift) - cost
    curves = {"Today": pnl_today}
    pnl_first = None
    if first_exp is not None:
        # The horizon is the FIRST expiry: legs expiring then settle at intrinsic, later legs are still alive and valued
        # with their IV held fixed. There is deliberately no curve at the last expiry: a short leg's payoff depends on the
        # spot at ITS expiry, so one "spot at the last expiry" axis would misprice calendars and diagonals.
        pnl_first = trade_value(legs, grid, first_exp, r, q, iv_shift) - cost
        curves[f"At {first_exp:%d %b} expiry"] = pnl_first

    # Unlimited upside / downside at the horizon: net call + share exposure as spot -> infinity (a call still alive at the
    # horizon has slope e^(-q t)). A calendar nets to ~0 and is treated as bounded; a naked short call does not.
    gross = net = 0.0
    for row in legs.itertuples():
        if row.type == "S" or row.type == "C":
            alive = row.type == "C" and first_exp is not None and row.expiry > first_exp
            f = math.exp(-q * years(row.expiry, first_exp)) if alive else 1.0
            gross += abs(row.qty * row.mult)
            net += row.qty * row.mult * f
    unlimited_profit = gross > 0 and net > 0.01 * gross
    unlimited_loss = gross > 0 and net < -0.01 * gross
    end_curve = pnl_first if pnl_first is not None else pnl_today
    stats_first = _curve_stats(grid, end_curve)
    if unlimited_profit:
        stats_first["max_profit"] = math.inf
    if unlimited_loss:
        stats_first["max_loss"] = -math.inf

    if unlimited_loss:
        risk = 2.0 * (-cost) if cost < 0 else max(abs(cost), 1.0)
        risk_note = "undefined risk: 2x the credit is used as the risk base"
    else:
        floor = min(end_curve.min(), 0.0)
        if first_exp is not None:  # the grid stops short of spot -> 0, so check the zero-price tail too (naked short puts)
            floor = min(floor, float(trade_value(legs, [0.01], first_exp, r, q, iv_shift)[0]) - cost)
        risk = max(-floor, 1.0)
        risk_note = "worst case at the first expiry"
    # shares only: no defined loss, use cost as the base
    if kind == "stock":
        risk = max(abs(cost), 1.0)
        risk_note = "position cost"

    pop_first = ev_first = None
    if pnl_first is not None:
        pop_first, ev_first = prob_stats(grid, pnl_first, spot, t_first, ref_iv, r, q)

    greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    short_deltas = []
    for row in legs.itertuples():
        t = years(row.expiry, today) if row.type != "S" else 0.0
        g = bs_greeks(row.type, spot, row.strike, t, row.iv, r, q)
        for k in greeks:
            greeks[k] += row.qty * row.mult * g[k]
        if row.type != "S" and row.qty < 0 and t > 0:
            short_deltas.append(abs(g["delta"]))
    shorts = opts[opts["qty"] < 0]

    return {
        "kind": kind, "spot": float(spot), "cost": cost, "value_now": value_now, "pnl_now": value_now - cost,
        "risk": risk, "risk_note": risk_note, "ref_iv": ref_iv,
        "first_exp": first_exp, "last_exp": last_exp,
        "dte_first": (first_exp - _date(today)).days if first_exp else None,
        "dte_last": (last_exp - _date(today)).days if last_exp else None,
        "em_first": spot * ref_iv * math.sqrt(t_first) if first_exp else 0.0,
        "grid": grid, "curves": curves, "pnl_today": pnl_today, "pnl_first": pnl_first,
        "stats_first": stats_first,
        "breakevens_today": breakevens(grid, pnl_today),
        "unlimited_profit": unlimited_profit, "unlimited_loss": unlimited_loss,
        "pop_first": pop_first, "ev_first": ev_first,
        "greeks": greeks, "dollar_delta": greeks["delta"] * spot,
        "max_short_delta": max(short_deltas) if short_deltas else 0.0,
        "short_strikes": sorted(set(float(k) for k in shorts["strike"])) if not shorts.empty else [],
        "short_types": sorted(set(shorts["type"])) if not shorts.empty else [],
    }


# ------------------------------------------------------------------------------------------------ advisor
def _fmt_money(x):
    if round(x) == 0:
        return "$0"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def advise(m, rules=None, step=None):
    """Rules-based management call for one analysed trade. Returns action, reasons, how, and the numbers used.

    Order: expired -> stop-loss -> profit target -> time exit -> tested / adjust -> hold. It is a checklist for
    discipline, not a forecast: the thresholds are yours (sidebar) and the model assumes IV is held fixed."""
    R = {**DEFAULT_RULES, **(rules or {})}
    kind, pnl, risk, cost = m["kind"], m["pnl_now"], m["risk"], m["cost"]
    spot, dte = m["spot"], m["dte_first"]
    step = step or default_step(spot)
    pnl_pct = 100.0 * pnl / risk
    stop_amt = R["stop_loss"] / 100.0 * risk
    if kind == "calendar":
        base = cost if cost > 0 else risk
        target_amt, target_pct = R["calendar_target"] / 100.0 * base, R["calendar_target"]
    elif kind == "credit":
        target_amt, target_pct = R["credit_target"] / 100.0 * (-cost), R["credit_target"]
    else:
        target_amt, target_pct = R["debit_target"] / 100.0 * risk, R["debit_target"]

    out = {"pnl_pct": pnl_pct, "target_amt": target_amt, "stop_amt": stop_amt, "margin_em": None}
    reasons = []
    be = m["stats_first"]["breakevens"] if m["stats_first"] else m["breakevens_today"]
    em = m["em_first"] or 0.0
    pnl_first = m["pnl_first"]
    in_profit = (float(np.interp(spot, m["grid"], pnl_first)) > 0) if pnl_first is not None else None
    if be and em > 0:
        out["margin_em"] = min(abs(spot - b) for b in be) / em

    def done(action, how):
        out.update(action=action, reasons=reasons, how=how)
        return out

    if dte is not None and dte < 0:
        reasons.append(f"The front leg expired {-dte} day(s) ago, so the table is stale or the legs have settled.")
        return done("CLOSE (EXPIRED)", "Remove or update the expired legs; expired options settle at intrinsic value.")
    if dte == 0:
        reasons.append("The front leg expires today: pin and assignment risk are at their peak.")
        return done("BOOK PROFIT" if pnl > 0 else "CUT LOSS",
                    "Close every leg as one order before the close instead of carrying expiry risk.")

    if pnl <= -stop_amt:
        reasons.append(f"P&L {_fmt_money(pnl)} is {pnl_pct:.0f}% of the amount at risk, past the -{R['stop_loss']:.0f}% stop.")
        return done("CUT LOSS", "Close all legs together as a single multi-leg order at the mid and work it toward the natural; "
                                "only leg out if a single leg is the problem.")
    if pnl >= target_amt > 0:
        what = {"calendar": "of the debit", "credit": "of the credit", "debit": "of the amount at risk", "stock": "of cost"}[kind]
        reasons.append(f"P&L {_fmt_money(pnl)} has reached the {target_pct:.0f}% target {what} ({_fmt_money(target_amt)}).")
        how = "Close all legs together."
        if kind == "calendar":
            how += " Or close the short leg only and keep the long leg if you still want the directional / vol exposure."
        return done("BOOK PROFIT", how)

    if dte is not None:
        limit = {"calendar": R["calendar_exit_dte"], "credit": R["credit_exit_dte"], "debit": R["debit_exit_dte"]}.get(kind)
        if limit is not None and dte <= limit:
            if kind == "calendar":
                reasons.append(f"Front leg has {dte} day(s) left (rule: close at {limit} or fewer): the gamma / pin risk now outweighs the theta left.")
                return done("BOOK PROFIT" if pnl > 0 else "CUT LOSS",
                            "Close the calendar, or roll the short leg to the next expiry if you want to keep the long leg working.")
            if kind == "credit":
                if pnl > 0:
                    reasons.append(f"{dte} day(s) left (rule: manage at {limit} or fewer) with a gain of {_fmt_money(pnl)}: the remaining credit is not worth the gamma risk.")
                    return done("BOOK PROFIT", "Buy the spread back and redeploy.")
                tested = (out["margin_em"] is not None and out["margin_em"] < R["tested_em"]) or in_profit is False \
                    or m["max_short_delta"] >= R["short_delta_tested"]
                if tested:
                    reasons.append(f"{dte} day(s) left and the spread is tested (short delta {m['max_short_delta']:.2f}) with no gain to protect.")
                    return done("CUT LOSS", "Close it; rolling this close to expiry rarely buys enough time for the credit it costs.")
            if kind == "debit":
                reasons.append(f"{dte} day(s) left (rule: exit at {limit} or fewer): theta decay is accelerating on a long-premium trade.")
                return done("BOOK PROFIT" if pnl > 0 else "CUT LOSS", "Close it or roll to a later expiry if the thesis is intact.")

    # tested / adjust
    if kind in ("calendar", "credit") and pnl_first is not None:
        if in_profit is False:
            reasons.append(f"Spot {spot:,.2f} is outside the profit range at the {m['first_exp']:%d %b} expiry "
                           f"(breakevens {', '.join(f'{b:,.2f}' for b in be) or 'none'}).")
        elif out["margin_em"] is not None and out["margin_em"] < R["tested_em"]:
            reasons.append(f"Spot is {out['margin_em']:.2f} expected move(s) from a breakeven (rule: adjust inside {R['tested_em']:.2f}).")
        elif kind == "credit" and m["max_short_delta"] >= R["short_delta_tested"]:
            reasons.append(f"A short leg's delta has reached {m['max_short_delta']:.2f} (rule: {R['short_delta_tested']:.2f}).")
        if reasons:
            if kind == "calendar":
                center = round_strike(spot, step)
                old = ", ".join(f"{k:g}" for k in m["short_strikes"]) or "its strike"
                how = (f"Re-centre: close the tested calendar and re-open at the strike nearest spot (about {center:g}, was {old}), "
                       f"or add a calendar on the other side to make it a double calendar. Skip the adjustment if it costs "
                       f"more than the time premium it buys, and respect the {R['stop_loss']:.0f}% stop.")
            else:
                how = ("Roll the tested short strike to a later expiry and further out of the money for a net credit; if no credit is "
                       "available, close it rather than adding risk.")
            return done("ADJUST", how)
    if kind == "debit" and target_amt > 0 and pnl >= R["lock_in_frac"] * target_amt:
        reasons.append(f"P&L {_fmt_money(pnl)} is past {R['lock_in_frac'] * 100:.0f}% of the way to the {target_pct:.0f}% target: protect it.")
        return done("ADJUST", "Lock in gains: sell a further out-of-the-money option against it to make a spread, or move your stop to breakeven.")

    reasons.append(f"P&L {_fmt_money(pnl)} ({pnl_pct:.0f}% of risk) sits between the -{R['stop_loss']:.0f}% stop and the {target_pct:.0f}% target.")
    if out["margin_em"] is not None and out["margin_em"] > 0:
        reasons.append(f"Spot is {out['margin_em']:.2f} expected move(s) from the nearest breakeven.")
    if dte is not None:
        reasons.append(f"{dte} day(s) to the first expiry.")
    return done("HOLD", f"No action. Re-check after a move of about a third of the expected move (~${em / 3:,.2f}) or as the first expiry nears." if em else "No action.")


def evaluate_trade(legs, spot, today, rules=None, r=0.0, q=0.0, iv_shift=0.0, ref_iv=None, step=None):
    """analyze_trade + advise in one dict (advice keys are prefixed with `advice_`)."""
    m = analyze_trade(legs, spot, today, r=r, q=q, iv_shift=iv_shift, ref_iv=ref_iv)
    a = advise(m, rules, step)
    m.update({f"advice_{k}": v for k, v in a.items()})
    m["advice"] = a
    return m


# -------------------------------------------------------------------------------------------- new trades
def next_weekday(start, weekday=4, min_days=7):
    """First date on or after start + min_days that falls on `weekday` (Mon=0 ... Fri=4)."""
    d = _date(start) + dt.timedelta(days=min_days)
    return d + dt.timedelta(days=(weekday - d.weekday()) % 7)


TEMPLATES = ["Long call", "Long put", "Call debit spread", "Put debit spread", "Call credit spread", "Put credit spread",
             "Long straddle", "Long strangle", "Iron condor", "Call calendar", "Put calendar", "Double calendar",
             "Call diagonal", "Put diagonal", "Custom"]
NEEDS_BACK_EXPIRY = {"Call calendar", "Put calendar", "Double calendar", "Call diagonal", "Put diagonal"}
NEEDS_WIDTH = {"Call debit spread", "Put debit spread", "Call credit spread", "Put credit spread", "Iron condor"}
NEEDS_OFFSET = {"Long strangle", "Iron condor", "Double calendar", "Call diagonal", "Put diagonal"}


def build_template(name, underlying, center, width, offset, exp1, exp2, qty=1):
    """Legs (POSITION_COLS, prices blank) for a named structure. Short leg(s) use exp1, long back leg(s) exp2."""
    def leg(t, k, e, n):
        return {"trade": "", "underlying": underlying, "type": t, "strike": float(k), "expiry": e, "qty": float(n),
                "entry_price": np.nan, "iv_pct": np.nan, "mark": np.nan}
    c, w, o, n = center, width, offset, qty
    legs = {
        "Long call": [leg("C", c, exp1, n)],
        "Long put": [leg("P", c, exp1, n)],
        "Call debit spread": [leg("C", c, exp1, n), leg("C", c + w, exp1, -n)],
        "Put debit spread": [leg("P", c, exp1, n), leg("P", c - w, exp1, -n)],
        "Call credit spread": [leg("C", c, exp1, -n), leg("C", c + w, exp1, n)],
        "Put credit spread": [leg("P", c, exp1, -n), leg("P", c - w, exp1, n)],
        "Long straddle": [leg("C", c, exp1, n), leg("P", c, exp1, n)],
        "Long strangle": [leg("C", c + o, exp1, n), leg("P", c - o, exp1, n)],
        "Iron condor": [leg("P", c - o - w, exp1, n), leg("P", c - o, exp1, -n), leg("C", c + o, exp1, -n), leg("C", c + o + w, exp1, n)],
        "Call calendar": [leg("C", c, exp1, -n), leg("C", c, exp2, n)],
        "Put calendar": [leg("P", c, exp1, -n), leg("P", c, exp2, n)],
        "Double calendar": [leg("P", c - o, exp1, -n), leg("P", c - o, exp2, n), leg("C", c + o, exp1, -n), leg("C", c + o, exp2, n)],
        "Call diagonal": [leg("C", c + o, exp1, -n), leg("C", c, exp2, n)],
        "Put diagonal": [leg("P", c - o, exp1, -n), leg("P", c, exp2, n)],
        "Custom": [leg("C", c, exp1, n)],
    }[name]
    return pd.DataFrame(legs, columns=POSITION_COLS)


def fill_model_prices(legs, spot, default_iv, today, r=0.0, q=0.0):
    """Fill blank entry_price with the model price at the leg's iv (iv_pct, else default_iv). Returns a copy."""
    d = legs.copy()
    for i, row in d.iterrows():
        if pd.isna(row["entry_price"]):
            iv = row["iv_pct"] / 100.0 if pd.notna(row["iv_pct"]) and row["iv_pct"] > 0 else default_iv
            d.at[i, "entry_price"] = round(float(option_value(row["type"], spot, row["strike"], years(row["expiry"], today), iv, r, q)), 2)
    return d


def size_trade(analysis, equity, cash, risk_pct):
    """Contracts allowed by a per-trade risk budget (risk_pct of equity) and by cash. Returns (contracts, why)."""
    loss = analysis["risk"]
    if not np.isfinite(loss) or loss <= 0:
        return 0, "risk is undefined"
    budget = equity * risk_pct / 100.0
    by_risk = int(budget // loss)
    by_cash = int(max(cash, 0.0) // loss)
    n = max(min(by_risk, by_cash), 0)
    if by_risk <= by_cash:
        why = f"limited by the {risk_pct:g}% risk budget ({_fmt_money(budget)})"
    else:
        why = f"limited by cash ({_fmt_money(cash)})"
    return n, why


# --------------------------------------------------------------------------------------------- portfolio
def closed_stats(closed):
    p = pd.to_numeric(closed["realized_pnl"], errors="coerce").dropna() if closed is not None and len(closed) else pd.Series(dtype=float)
    if p.empty:
        return {"n": 0, "total": 0.0, "win_rate": None, "avg_win": None, "avg_loss": None, "profit_factor": None, "expectancy": None}
    wins, losses = p[p > 0], p[p < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else (math.inf if len(wins) else None)
    return {"n": int(len(p)), "total": float(p.sum()), "win_rate": float(len(wins) / len(p)),
            "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
            "profit_factor": pf, "expectancy": float(p.mean())}


def portfolio_summary(rows, realized, capital, deposits=0.0, cash_override=None):
    """Account-level numbers from per-trade dicts (each with cost, value_now, pnl_now, risk, kind, greeks).

    cash = starting capital + net deposits + realised P&L - cost of open trades (credits add cash). Return is simple
    (not time-weighted): (equity - capital - deposits) / (capital + deposits)."""
    cost = sum(r["cost"] for r in rows)
    mv = sum(r["value_now"] for r in rows)
    unreal = sum(r["pnl_now"] for r in rows)
    base = capital + deposits
    cash_calc = base + realized - cost
    cash = cash_calc if cash_override is None else float(cash_override)
    equity = cash + mv
    at_risk = sum(r["risk"] for r in rows)
    collateral = sum(r["risk"] for r in rows if r["kind"] == "credit")
    g = {k: sum(r["greeks"][k] for r in rows) for k in ("delta", "gamma", "theta", "vega")}
    return {
        "cost": cost, "market_value": mv, "unrealized": unreal, "realized": realized, "total_pnl": unreal + realized,
        "cash": cash, "cash_calc": cash_calc, "equity": equity, "base": base,
        "total_return_pct": 100.0 * (equity - base) / base if base else None,
        "return_on_risk_pct": 100.0 * unreal / at_risk if at_risk else None,
        "at_risk": at_risk, "at_risk_pct": 100.0 * at_risk / equity if equity else None,
        "cash_pct": 100.0 * cash / equity if equity else None,
        "collateral": collateral, "free_cash": cash - collateral, "greeks": g,
        "dollar_delta": sum(r["dollar_delta"] for r in rows), "n_trades": len(rows),
    }


# ------------------------------------------------------------------------------------------ live quotes
def _yf_loader(underlying, expiries):
    import yfinance as yf
    tk = yf.Ticker(underlying)
    hist = tk.history(period="5d", auto_adjust=False)
    if hist is None or hist.empty:
        raise ValueError("no price history")
    spot = float(hist["Close"].iloc[-1])
    listed = set(tk.options or [])
    chains = {}
    for e in expiries:
        iso = e.isoformat()
        if iso not in listed:
            continue
        oc = tk.option_chain(iso)
        frames = []
        for cp, df in (("C", oc.calls), ("P", oc.puts)):
            if df is not None and not df.empty:
                frames.append(pd.DataFrame({"strike": df["strike"].astype(float), "cp": cp,
                                            "bid": pd.to_numeric(df["bid"], errors="coerce"),
                                            "ask": pd.to_numeric(df["ask"], errors="coerce"),
                                            "last": pd.to_numeric(df["lastPrice"], errors="coerce")}))
        if frames:
            chains[e] = pd.concat(frames, ignore_index=True)
    return spot, chains


def fetch_spot(underlying):
    """Last close for one ticker from Yahoo (raises on no data)."""
    return _yf_loader(underlying, [])[0]


def fetch_quotes(legs, loader=None):
    """Spot per underlying and a mark per leg from Yahoo (bid/ask mid, else last). Returns (spots, marks, errors).

    marks is a Series aligned to `legs`' index (NaN where nothing was found); errors maps underlying / leg -> text.
    loader(underlying, [expiry dates]) -> (spot, {expiry: DataFrame[strike, cp, bid, ask, last]}) can be injected."""
    loader = loader or _yf_loader
    spots, errors = {}, {}
    marks = pd.Series(np.nan, index=legs.index, dtype=float)
    for und, grp in legs.groupby("underlying"):
        expiries = sorted({e for e in grp["expiry"] if pd.notna(e)})
        try:
            spot, chains = loader(und, expiries)
        except Exception as e:  # network, bad ticker, rate limit
            errors[und] = str(e)[:120]
            continue
        spots[und] = float(spot)
        for idx, row in grp.iterrows():
            if row["type"] == "S":
                marks[idx] = float(spot)
                continue
            ch = chains.get(row["expiry"])
            if ch is None:
                errors[f"{und} {row['expiry']}"] = "expiry not listed"
                continue
            hit = ch[(ch["cp"] == row["type"]) & np.isclose(ch["strike"], row["strike"])]
            if hit.empty:
                errors[f"{und} {row['type']}{row['strike']:g} {row['expiry']}"] = "strike not found"
                continue
            h = hit.iloc[0]
            if pd.notna(h["bid"]) and pd.notna(h["ask"]) and h["bid"] > 0 and h["ask"] > 0:
                marks[idx] = round(float((h["bid"] + h["ask"]) / 2.0), 4)
            elif pd.notna(h["last"]) and h["last"] > 0:
                marks[idx] = float(h["last"])
    return spots, marks, errors
