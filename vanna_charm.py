"""Dealer vanna / charm positioning and the short-squeeze (relief-rally) alert. Pure functions, no Streamlit.

core.analyse() calls positioning() and squeeze_alert(); app.py only displays the results.

What this measures, and what it cannot
--------------------------------------
Vanna = change in an option's delta per 1 vol point of implied volatility.  Charm = change in delta per day of
time passing (both are read from the chain file; the file's charm sign was checked against a Black-Scholes
one-day finite difference: OTM calls negative, OTM puts positive).

Dealer hedge flow, in shares (positive = dealers BUY the underlying):
    IV falls 1 vol point :  + sum(side * OI * 100 * vanna)
    one day passes       :  - sum(side * OI * 100 * charm)
where side = +1 for options dealers are long and -1 for options dealers are short.

Open interest does not say who is long or short.  The one assumption that is reasonably defensible is that
customers own OTM puts (hedgers), so dealers are SHORT them.  OTM calls are ambiguous (overwriting makes dealers
long, speculation makes them short), so the call side is reported separately as a +/- range and is excluded from
the headline unless you pick a convention that includes it.  ITM options are ignored: they are mostly hedges and
financing trades, not customer directional positions.

The squeeze alert is a checklist, not a forecast.  It needs (1) a washout, (2) dealer positioning that would
support a bounce, and (3) a trigger: implied volatility actually rolling over and price stabilising.  Without the
vol turn, vanna and charm are only potential.  The thresholds are judgment calls, not fitted values; the history
file is used to show how often the same trigger fired and what followed, and that sample is small.
"""
import numpy as np
import pandas as pd

VC_VERSION = "2026-11-21.1"

CONVENTIONS = {
    "puts": "OTM puts only (dealers short puts; calls ignored)",
    "long_calls": "OTM puts short + OTM calls long (dealers overwrite calls)",
    "short_calls": "OTM puts short + OTM calls short (customers long calls)",
}

DEFAULTS = {
    "convention": "puts",
    "min_dte": 1,
    "max_dte": 120,           # back-month IV moves less than front-month IV, so very long dated OI is left out by default
    "near_band": 0.10,        # "near spot" = within this fraction of spot
    "shock_pts": (1, 3, 5),   # parallel IV drops shown in the scenario table
    # --- squeeze alert thresholds (judgment calls)
    "sq_ret5": 0.04,          # price stretched: 5-session drop at least this big ...
    "sq_ret20": 0.08,         # ... or 20-session drop at least this big
    "sq_dex_pct": 0.10,       # dealer delta washout: DEX in the bottom X of its last 120 sessions
    "sq_ivrank": 70.0,        # vol elevated: IV rank at least this
    "sq_recent": 5,           # the washout must have been seen within this many sessions
    "sq_min_vanna": 0.5e6,    # support: put-side shares bought per 1 vol point of IV drop
    "sq_min_charm": 0.25e6,   # support: put-side shares bought per day of decay
    "sq_iv_range": 5.0,       # trigger: 30d IV had a range of at least this many vol points over the last 10 sessions
}


def _empty(reason: str) -> dict:
    return {"ok": False, "reason": reason}


def positioning(chain: pd.DataFrame, spot: float, asof: pd.Timestamp, params: dict | None = None) -> dict:
    """Net dealer vanna / charm hedge flow implied by the chain.  `chain` is core.parse_chain() output."""
    p = {**DEFAULTS, **(params or {})}
    if "vanna" not in chain.columns or "charm" not in chain.columns:
        return _empty("The option chain file has no vanna / charm columns.")
    c = chain.copy()
    c["dte"] = (c["exp"] - pd.Timestamp(asof).normalize()).dt.days
    c = c[(c["dte"] >= p["min_dte"]) & (c["dte"] <= p["max_dte"]) & (c["open_interest"] > 0)]
    c = c.dropna(subset=["vanna", "charm", "strike"])
    if c.empty:
        return _empty("No open interest with vanna / charm inside the DTE window.")
    c["pct"] = c["strike"] / spot - 1.0
    otm = c[((c["cp"] == "P") & (c["strike"] < spot)) | ((c["cp"] == "C") & (c["strike"] > spot))].copy()
    otm["oi100"] = otm["open_interest"] * 100.0
    otm["v"] = otm["vanna"] * otm["oi100"]         # delta shares per vol point, before the dealer side
    otm["ch"] = otm["charm"] * otm["oi100"]        # delta shares per day, before the dealer side
    puts, calls = otm[otm["cp"] == "P"].copy(), otm[otm["cp"] == "C"].copy()
    # dealers short puts: IV down -> buy, time passing -> buy when the put's delta decays toward zero
    puts["vflow"] = -puts["v"]
    puts["cflow"] = puts["ch"]
    # call side as if dealers were LONG the calls; flip the sign for the short-calls convention
    calls["vflow"] = calls["v"]
    calls["cflow"] = -calls["ch"]

    put_v, put_c = float(puts["vflow"].sum()), float(puts["cflow"].sum())
    call_v, call_c = float(calls["vflow"].sum()), float(calls["cflow"].sum())
    cs = {"puts": 0.0, "long_calls": 1.0, "short_calls": -1.0}[p["convention"]]
    net_v, net_c = put_v + cs * call_v, put_c + cs * call_c

    near = puts[puts["pct"] >= -p["near_band"]]
    zones = pd.cut(puts["pct"], [-1.0, -0.20, -0.10, -0.05, 0.0],
                   labels=["20%+ below spot", "10-20% below", "5-10% below", "0-5% below"])
    by_zone = (puts.groupby(zones, observed=False)[["vflow", "cflow"]].sum()
               .rename(columns={"vflow": "vanna_sh_per_pt", "cflow": "charm_sh_per_day"}).reset_index()
               .rename(columns={"pct": "put strikes"}))
    buckets = pd.cut(puts["dte"], [0, 3, 8, 15, 30, 60, 120, 400, 5000],
                     labels=["1-3d", "4-8d", "9-15d", "16-30d", "31-60d", "61-120d", "121-400d", "400d+"])
    by_dte = (puts.groupby(buckets, observed=False)[["vflow", "cflow"]].sum()
              .rename(columns={"vflow": "vanna_sh_per_pt", "cflow": "charm_sh_per_day"}).reset_index()
              .rename(columns={"dte": "expiry bucket"}))
    by_dte = by_dte[(by_dte["vanna_sh_per_pt"] != 0) | (by_dte["charm_sh_per_day"] != 0)].reset_index(drop=True)
    top = (puts.groupby(["exp", "strike"])[["vflow", "cflow", "open_interest"]].sum().reset_index()
           .sort_values("vflow", ascending=False).head(8))

    scen = pd.DataFrame({"IV change (vol pts)": [-float(s) for s in p["shock_pts"]]})
    scen["put-side shares (M)"] = -scen["IV change (vol pts)"] * put_v / 1e6
    scen["put-side $ (B)"] = scen["put-side shares (M)"] * 1e6 * spot / 1e9
    scen["net, selected convention, shares (M)"] = -scen["IV change (vol pts)"] * net_v / 1e6
    return {
        "ok": True, "convention": p["convention"], "convention_label": CONVENTIONS[p["convention"]],
        "spot": float(spot), "window": (int(p["min_dte"]), int(p["max_dte"])), "near_band": float(p["near_band"]),
        "put_vanna": put_v, "put_charm": put_c, "call_vanna": call_v, "call_charm": call_c,
        "net_vanna": net_v, "net_charm": net_c,
        "put_vanna_near": float(near["vflow"].sum()), "put_charm_near": float(near["cflow"].sum()),
        "put_oi": float(puts["open_interest"].sum()), "call_oi": float(calls["open_interest"].sum()),
        "by_zone": by_zone, "by_dte": by_dte, "top": top, "scenarios": scen,
    }


# --------------------------------------------------------------------------- squeeze alert
def _pct_rank(s: pd.Series, n: int = 120) -> pd.Series:
    """Share of the trailing n values that are at or below today's value (low = most extreme negative)."""
    return s.rolling(n, min_periods=60).apply(lambda x: float((x <= x[-1]).mean()), raw=True)


def washout_vol_series(hc: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Daily history-file conditions.  hc is core.history_components() output (has a Date column)."""
    d = hc.set_index("Date") if "Date" in hc.columns else hc.copy()
    px = d["Close Price"].astype(float)
    ivr = d["IV Rank"].astype(float)
    if "ivr_own" in d.columns:
        ivr = ivr.fillna(d["ivr_own"] * 100.0)
    iv30 = d["IV 30d"].astype(float)
    out = pd.DataFrame(index=d.index)
    out["px"] = px
    out["ret5"] = px.pct_change(5)
    out["ret20"] = px.pct_change(20)
    out["s_price"] = (out["ret5"] <= -p["sq_ret5"]) | (out["ret20"] <= -p["sq_ret20"])
    out["dex_pct"] = _pct_rank(d["DEX Net OI"].astype(float))
    out["s_dex"] = out["dex_pct"] <= p["sq_dex_pct"]
    out["ivrank"] = ivr
    out["s_iv"] = ivr >= p["sq_ivrank"]
    out["setup_day"] = out[["s_price", "s_dex", "s_iv"]].sum(axis=1) >= 2
    out["setup_recent"] = out["setup_day"].astype(float).rolling(int(p["sq_recent"]), min_periods=1).max() > 0
    dv = iv30.diff()
    out["iv30"] = iv30
    out["iv_range10"] = iv30.rolling(10).max() - iv30.rolling(10).min()
    out["t_vol"] = (dv < 0) & (dv.shift(1) < 0) & (out["iv_range10"] >= p["sq_iv_range"])
    out["t_px"] = px.diff() >= 0
    out["trigger"] = out["t_vol"] & out["t_px"]
    for n in (5, 10, 20):
        out[f"f{n}"] = px.shift(-n) / px - 1.0
    return out


def _events(mask: pd.Series, gap: int = 10) -> list:
    """First day of each cluster of True days (clusters are at least `gap` sessions apart)."""
    keep, last = [], -10 ** 9
    for i, (idx, v) in enumerate(mask.items()):
        if bool(v) and i - last >= gap:
            keep.append(idx)
            last = i
    return keep


def trigger_history(series: pd.DataFrame) -> dict:
    """How often the trigger fired in the history file, alone and after a washout, and what followed."""
    out = {"baseline": series[["f5", "f10", "f20"]].mean(), "n_days": int(len(series)),
           "baseline_hit": {n: float((series[n].dropna() > 0).mean()) for n in ("f5", "f10", "f20")}}
    for key, mask in (("trigger", series["trigger"]), ("washout_then_trigger", series["trigger"] & series["setup_recent"])):
        ev = _events(mask)
        t = series.loc[ev, ["px", "iv30", "ivrank", "f5", "f10", "f20"]].copy() if ev else pd.DataFrame(
            columns=["px", "iv30", "ivrank", "f5", "f10", "f20"])
        stats = {}
        for n in ("f5", "f10", "f20"):
            v = t[n].dropna()
            stats[n] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else float("nan"),
                        "hit": float((v > 0).mean()) if len(v) else float("nan")}
        out[key] = {"events": t, "stats": stats, "n_events": len(ev)}
    return out


def squeeze_alert(hc: pd.DataFrame | None, vc: dict | None, gamma_effective: str, spot: float,
                  agg_flip: float | None, call_wall: float | None, params: dict | None = None) -> dict:
    """Checklist-style alert.  level is one of: none, watch, armed, alert."""
    p = {**DEFAULTS, **(params or {})}
    if hc is None or len(hc) < 30:
        return {"ok": False, "reason": "Not enough option-history rows to evaluate the alert."}
    s = washout_vol_series(hc, p)
    last, prev = s.iloc[-1], s.iloc[-2]
    put_v = vc["put_vanna"] if vc and vc.get("ok") else float("nan")
    put_c = vc["put_charm"] if vc and vc.get("ok") else float("nan")

    def pct(x):
        return "n/a" if pd.isna(x) else f"{x:+.1%}"

    rows = [
        ("Setup", "Price stretched down", bool(last["s_price"]),
         f"5d {pct(last['ret5'])}, 20d {pct(last['ret20'])} (needs 5d <= -{p['sq_ret5']:.0%} or 20d <= -{p['sq_ret20']:.0%})"),
        ("Setup", "Dealer delta washed out", bool(last["s_dex"]),
         f"DEX at the {last['dex_pct']:.0%} percentile of 120 sessions (needs <= {p['sq_dex_pct']:.0%})"
         if pd.notna(last["dex_pct"]) else "n/a"),
        ("Setup", "Implied vol elevated", bool(last["s_iv"]),
         f"IV rank {last['ivrank']:.0f} (needs >= {p['sq_ivrank']:.0f})" if pd.notna(last["ivrank"]) else "n/a"),
        ("Context", "Negative gamma (amplifies a bounce too)", gamma_effective == "Negative",
         f"effective regime: {gamma_effective}"),
        ("Support", "Put-side vanna: dealers buy as IV falls", bool(put_v >= p["sq_min_vanna"]) if pd.notna(put_v) else False,
         "n/a" if pd.isna(put_v) else f"{put_v / 1e6:+.2f}M shares per 1 vol pt (needs >= {p['sq_min_vanna'] / 1e6:.2f}M)"),
        ("Support", "Put-side charm: decay adds buying", bool(put_c >= p["sq_min_charm"]) if pd.notna(put_c) else False,
         "n/a" if pd.isna(put_c) else f"{put_c / 1e6:+.2f}M shares per day (needs >= {p['sq_min_charm'] / 1e6:.2f}M)"),
        ("Trigger", "30d IV falling two sessions in a row", bool(last["t_vol"]),
         f"IV30 {last['iv30']:.1f} vs {prev['iv30']:.1f} the session before; 10-session range {last['iv_range10']:.1f} pts "
         f"(needs a fall on both of the last two sessions and a range >= {p['sq_iv_range']:.0f})"),
        ("Trigger", "Price stabilising (close >= prior close)", bool(last["t_px"]),
         f"close {last['px']:.2f} vs {prev['px']:.2f}"),
    ]
    chk = pd.DataFrame(rows, columns=["group", "condition", "met", "reading"])
    setup_n = int(chk[chk["group"] == "Setup"]["met"].sum())
    setup_ok = bool(last["setup_recent"])
    support_ok = bool(chk[(chk["group"] == "Support")]["met"].any())
    vol_turn, px_ok = bool(last["t_vol"]), bool(last["t_px"])
    if not setup_ok:
        level, headline = "none", "No squeeze setup: the washout conditions are not in place."
    elif not vol_turn:
        level, headline = "watch", ("WATCH: washout in place, but implied vol has not rolled over yet. "
                                    "Vanna and charm only turn into buying once IV falls.")
    elif not (px_ok and support_ok):
        level, headline = "armed", ("ARMED: vol is rolling over after a washout, but price has not stabilised "
                                    "or dealer put-side support is weak.")
    else:
        level, headline = "alert", ("SQUEEZE ALERT: washout, then IV falling and price stabilising, with put-side "
                                    "vanna/charm tailwind. Expect a relief bounce; it is not a guaranteed squeeze.")
    levels = []
    if agg_flip is not None and not pd.isna(agg_flip):
        levels.append(("Aggregate gamma flip", float(agg_flip)))
    if call_wall is not None and not pd.isna(call_wall):
        levels.append(("Call wall", float(call_wall)))
    levels.append(("Spot", float(spot)))
    return {"ok": True, "level": level, "headline": headline, "checks": chk, "setup_n_today": setup_n,
            "setup_recent": setup_ok, "recent_window": int(p["sq_recent"]), "levels": levels,
            "history": trigger_history(s), "series": s, "params": p}
