"""Gamma (GEX) screener using free yfinance option chains. numpy/pandas only (no scipy).

Per ticker: net dealer GEX in $ per 1% move, GEX as % of 20-day average dollar volume (so names are comparable),
zero-gamma flip level, distance from spot to the flip, and the largest call / put gamma walls.

Convention (the usual, and a weak assumption on single stocks): dealers are long calls and short puts, so call GEX
is positive and put GEX negative.  GEX per contract = gamma * OI * 100 * spot^2 * 0.01.
yfinance open interest is end-of-day and its IVs can be stale or zero, so strikes are filtered (see load_chain).
"""
import datetime as dt
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Liquid optionable names: a sensible default so a scan takes ~1 minute instead of pulling 500 chains.
LIQUID = ("SPY QQQ IWM DIA TLT GLD SLV XLF XLE XLK XLV XLY XLP XLI XLU SMH ARKK "
          "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX ORCL CRM ADBE INTC MU QCOM CSCO IBM "
          "JPM BAC WFC GS MS C V MA PYPL COF AXP "
          "XOM CVX COP OXY SLB LLY UNH JNJ PFE MRK ABBV AMGN "
          "WMT COST HD LOW TGT NKE MCD SBUX KO PEP DIS "
          "BA CAT GE F GM UBER SHOP PLTR COIN SOFI RIVN ABNB SNOW CRWD PANW NOW").split()

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def bs_gamma(spot, strike, t_years, iv):
    """Black-Scholes gamma (same for calls and puts), vectorised, zero rates/dividends."""
    spot = np.asarray(spot, float)
    strike = np.asarray(strike, float)
    t = np.maximum(np.asarray(t_years, float), 0.5 / 365.0)
    iv = np.maximum(np.asarray(iv, float), 1e-4)
    sd = iv * np.sqrt(t)
    d1 = (np.log(spot / strike) + 0.5 * sd * sd) / sd
    return _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1) / (spot * sd)


def clean_chain(chain, spot, min_oi=1, band=0.30, iv_lo=0.03, iv_hi=3.0):
    """Keep strikes within +/-band of spot, real OI, and sane IV (yfinance returns ~0 IV for stale quotes)."""
    c = chain.copy()
    c = c[(c["oi"] >= min_oi) & c["iv"].between(iv_lo, iv_hi) & c["strike"].between(spot * (1 - band), spot * (1 + band))]
    return c.reset_index(drop=True)


def gex_at(chain, s):
    """Net dealer GEX ($ per 1% move) if spot were `s`, holding IV and OI fixed."""
    if chain.empty:
        return 0.0
    g = bs_gamma(s, chain["strike"].to_numpy(), chain["dte"].to_numpy() / 365.0, chain["iv"].to_numpy())
    sign = np.where(chain["cp"].to_numpy() == "C", 1.0, -1.0)
    return float(np.sum(g * chain["oi"].to_numpy() * 100.0 * s * s * 0.01 * sign))


def strike_table(chain, spot):
    """Per-strike call / put / net GEX at the current spot."""
    g = bs_gamma(spot, chain["strike"].to_numpy(), chain["dte"].to_numpy() / 365.0, chain["iv"].to_numpy())
    val = g * chain["oi"].to_numpy() * 100.0 * spot * spot * 0.01
    x = chain.assign(gex=np.where(chain["cp"] == "C", val, -val))
    piv = x.pivot_table(index="strike", columns="cp", values="gex", aggfunc="sum", fill_value=0.0)
    for k in ("C", "P"):
        if k not in piv:
            piv[k] = 0.0
    out = pd.DataFrame({"call": piv["C"], "put": piv["P"]})
    out["net"] = out["call"] + out["put"]
    return out.reset_index()


def zero_gamma(chain, spot, span=0.15, n=61):
    """Spot level where net GEX changes sign, nearest to spot, searched within +/-span. None if no crossing."""
    grid = np.linspace(spot * (1 - span), spot * (1 + span), n)
    vals = np.array([gex_at(chain, s) for s in grid])
    cross = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if len(cross) == 0:
        return None
    best = None
    for i in cross:
        x0, x1, y0, y1 = grid[i], grid[i + 1], vals[i], vals[i + 1]
        x = x0 - y0 * (x1 - x0) / (y1 - y0)
        if best is None or abs(x - spot) < abs(best - spot):
            best = x
    return float(best)


def regime_label(net, dist_flip_pct, near=1.5):
    if net < 0:
        return "Negative"
    if dist_flip_pct is not None and abs(dist_flip_pct) <= near:
        return "Near flip"
    return "Positive"


def scan_chain(ticker, spot, chain, adv_dollar=None, min_oi=1):
    """One result row (and the strike table) from a raw chain frame: columns strike, cp, oi, iv, dte, volume."""
    c = clean_chain(chain, spot, min_oi=min_oi)
    if c.empty:
        raise ValueError("no usable option rows (OI/IV filters removed everything)")
    st_ = strike_table(c, spot)
    net = float(st_["net"].sum())
    flip = zero_gamma(c, spot)
    dist = None if flip is None else 100.0 * (spot - flip) / spot
    calls, puts = st_.loc[st_["call"].idxmax()], st_.loc[st_["put"].idxmin()]
    row = {
        "ticker": ticker, "spot": round(spot, 2), "regime": regime_label(net, dist),
        "net_gex_m": round(net / 1e6, 2),
        "gex_pct_adv": None if not adv_dollar else round(100.0 * net / adv_dollar, 3),
        "flip": None if flip is None else round(flip, 2),
        "spot_vs_flip_pct": None if dist is None else round(dist, 2),
        "call_wall": float(calls["strike"]), "put_wall": float(puts["strike"]),
        "total_oi": int(c["oi"].sum()), "put_call_oi": round(float(c.loc[c.cp == "P", "oi"].sum()) / max(float(c.loc[c.cp == "C", "oi"].sum()), 1.0), 2),
        "n_expiries": int(c["dte"].nunique()),
    }
    return row, st_


def yf_chain(ticker, max_dte=45, today=None):
    """(spot, avg dollar volume, raw chain) from yfinance. Raises on no data."""
    import yfinance as yf
    today = today or dt.date.today()
    tk = yf.Ticker(ticker)
    hist = tk.history(period="2mo", auto_adjust=False)
    if hist is None or hist.empty:
        raise ValueError("no price history")
    spot = float(hist["Close"].iloc[-1])
    adv = float((hist["Close"] * hist["Volume"]).tail(20).mean())
    frames = []
    for e in tk.options or []:
        dte = (dt.date.fromisoformat(e) - today).days
        if dte < 0 or dte > max_dte:
            continue
        oc = tk.option_chain(e)
        for cp, df in (("C", oc.calls), ("P", oc.puts)):
            if df is None or df.empty:
                continue
            frames.append(pd.DataFrame({
                "strike": df["strike"].astype(float), "cp": cp,
                "oi": pd.to_numeric(df["openInterest"], errors="coerce").fillna(0.0),
                "iv": pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0.0),
                "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0.0), "dte": dte}))
    if not frames:
        raise ValueError("no options in the DTE window")
    return spot, adv, pd.concat(frames, ignore_index=True)


def run(tickers, max_dte=45, min_oi=1, loader=None, progress=None, workers=6):
    """Scan tickers. Returns (results DataFrame sorted by gex_pct_adv ascending, {ticker: strike table}, {ticker: error})."""
    loader = loader or yf_chain
    rows, strikes, failed = [], {}, {}

    def one(t):
        spot, adv, chain = loader(t, max_dte)
        return scan_chain(t, spot, chain, adv, min_oi)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {t: ex.submit(one, t) for t in tickers}
        for t, f in futs.items():
            try:
                row, st_ = f.result()
                rows.append(row)
                strikes[t] = st_
            except Exception as e:
                failed[t] = str(e)[:120]
            done += 1
            if progress:
                progress(done, len(tickers))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("gex_pct_adv", na_position="last").reset_index(drop=True)
    return df, strikes, failed
