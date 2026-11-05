"""Core logic for the layered market-regime scorer. No Streamlit imports, so it can be tested alone.

Layers
  1. Direction score  = weighted price structure + flow + DEX (GEX is NOT part of direction)
  2. Gates            = IV rank + IV term structure (decide whether calendars/credit trades qualify)
  3. Regime           = sign of net GEX over the trade's expiry window (picks strategy family)
  4. Placement        = expected-move bands and GEX walls (pick strikes)
"""
from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BANDS = ["Bearish", "Mild Bearish", "Neutral", "Mild Bullish", "Bullish"]

# Strategy matrix: (band, gamma regime) -> ordered strategy names
MATRIX = {
    ("Bullish", "Positive"): ["ATM call calendar", "OTM call calendar", "Bull put credit spread"],
    ("Bullish", "Negative"): ["Call diagonal", "Bull call debit spread", "Call BWB"],
    ("Mild Bullish", "Positive"): ["OTM call calendar", "ATM call calendar", "Bull put credit spread"],
    ("Mild Bullish", "Negative"): ["Call diagonal", "OTM call calendar", "Bull call debit spread"],
    ("Neutral", "Positive"): ["ATM calendar", "Double calendar", "BWB"],
    ("Neutral", "Negative"): ["Wide double calendar", "OTM calendars", "Iron condor"],
    ("Mild Bearish", "Positive"): ["Put calendar", "Put diagonal"],
    ("Mild Bearish", "Negative"): ["OTM put calendar", "Put diagonal", "Bear put debit spread", "Put BWB"],
    ("Bearish", "Positive"): ["Put calendar", "Bear call credit spread", "Bear put debit spread"],
    ("Bearish", "Negative"): ["Bear put debit spread", "Put diagonal", "Put BWB"],
}

DEFAULT_WEIGHTS = {"price": 60.0, "flow": 25.0, "dex": 15.0}


# --------------------------------------------------------------------------- loading
def detect_kind(columns) -> str | None:
    c = {str(x).strip() for x in columns}
    if {"Bid x Size", "Premium", "Side"} <= c:
        return "flow"
    if {"Vol/OI", "Moneyness"} <= c:
        return "unusual"
    if {"implied_volatility", "expiration_date"} <= c:
        return "chain"
    if {"Close Price", "IV Rank"} <= c:
        return "history"
    if "net_dex" in c:
        return "dex_exp"
    if "gamma_flip" in c:
        return "gex_exp"
    if "expected_move_amt" in c:
        return "exp_move"
    if {"strike", "call_gex", "put_gex"} <= c:
        return "gex_strike"
    return None


def read_source(src) -> tuple[str | None, pd.DataFrame]:
    """src: path or bytes. Returns (kind, raw dataframe)."""
    buf = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    head = pd.read_csv(buf, nrows=0)
    kind = detect_kind(head.columns)
    if isinstance(buf, io.BytesIO):
        buf.seek(0)
    return kind, pd.read_csv(buf)


def load_folder(path: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for fn in sorted(os.listdir(path)):
        if fn.lower().endswith(".csv"):
            try:
                kind, df = read_source(os.path.join(path, fn))
            except Exception:
                continue
            if kind and kind not in out:
                out[kind] = df
    return out


def load_uploads(files) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for f in files or []:
        try:
            kind, df = read_source(f.getvalue())
        except Exception:
            continue
        if kind and kind not in out:
            out[kind] = df
    return out


def _num(s: pd.Series) -> pd.Series:
    """Parse '1.2B', '875.91M', '754.93K', '21.23%', '6,780', '—' into floats (percent kept in % units)."""
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    mult = s.str[-1].map({"K": 1e3, "M": 1e6, "B": 1e9}).fillna(1.0)
    base = pd.to_numeric(s.str.replace(r"[KMB%]$", "", regex=True), errors="coerce")
    return base * mult


def parse_history(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)
    for col in ["GEX Net OI", "DEX Net OI", "IV 7d", "IV 30d", "IV 90d", "IV Rank"]:
        if col in d:
            d[col] = _num(d[col])
    return d


def parse_chain(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["exp"] = pd.to_datetime(d["expiration_date"].astype(str).str[:10])
    d["cp"] = d["type"].astype(str).str[0].str.upper()
    for c in ["strike", "bid", "ask", "mid", "implied_volatility", "delta", "volume", "open_interest"]:
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    d = d.dropna(subset=["strike"])
    d = d.drop_duplicates(["exp", "cp", "strike"], keep="first").reset_index(drop=True)
    return d


def parse_gex_exp(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["expiration"], format="%b %d, %Y")
    return d


def parse_dex_exp(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["expiration"], format="%b %d, %Y")
    return d


def parse_em(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d


# --------------------------------------------------------------------------- price / spot helpers
def clip(x, lo=-1.0, hi=1.0):
    return np.clip(x, lo, hi)


def _z(x: pd.Series, n: int = 120) -> pd.Series:
    return (x - x.rolling(n, min_periods=60).mean()) / x.rolling(n, min_periods=60).std()


def history_components(h: pd.DataFrame, flow_contrarian: bool = False) -> pd.DataFrame:
    """Daily component scores in [-1, 1] (+ = bullish), computed from the history file only."""
    d = h.copy()
    c = d["Close Price"]
    ema200 = c.ewm(span=200, adjust=False).mean()
    hi20, lo20 = c.rolling(20).max(), c.rolling(20).min()
    hi55, lo55 = c.rolling(55).max(), c.rolling(55).min()
    d["ema200"], d["hi20"], d["lo20"], d["hi55"], d["lo55"] = ema200, hi20, lo20, hi55, lo55
    pos20 = (c - lo20) / (hi20 - lo20)
    pos55 = (c - lo55) / (hi55 - lo55)
    dist = (c / ema200 - 1) / 0.04
    slope = (ema200 - ema200.shift(20)) / ema200.shift(20) / 0.02
    d["price"] = clip(0.35 * clip(dist) + 0.25 * (2 * pos20 - 1) + 0.25 * (2 * pos55 - 1) + 0.15 * clip(slope))
    d["dex"] = clip(_z(d["DEX Net OI"]) / 2)
    d["gex"] = clip(_z(d["GEX Net OI"]) / 2)
    sign = 1.0 if flow_contrarian else -1.0
    d["flow_pcr"] = clip(sign * _z(d["Option Volume Put-Call Ratio"]) / 2)
    ivmin = d["IV 30d"].rolling(252, min_periods=120).min()
    ivmax = d["IV 30d"].rolling(252, min_periods=120).max()
    d["ivr_own"] = (d["IV 30d"] - ivmin) / (ivmax - ivmin)
    term = d["IV 7d"] - d["IV 90d"]  # percentage points
    d["vol"] = clip(-(0.5 * (2 * d["ivr_own"] - 1) + 0.5 * clip(term / 5)))
    return d


def estimate_spot(chain: pd.DataFrame, fallback: float | None = None) -> float:
    """Spot from put-call parity at the nearest few expiries."""
    ests = []
    for e in sorted(chain["exp"].unique())[:5]:
        s = chain[(chain["exp"] == e) & (chain["bid"] > 0)]
        c = s[s["cp"] == "C"].set_index("strike")["mid"]
        p = s[s["cp"] == "P"].set_index("strike")["mid"]
        j = pd.concat([c, p], axis=1, keys=["c", "p"]).dropna()
        if j.empty:
            continue
        j["diff"] = (j["c"] - j["p"]).abs()
        k = j["diff"].idxmin()
        ests.append(k + j.loc[k, "c"] - j.loc[k, "p"])
    if ests:
        return float(np.median(ests))
    if fallback is None:
        raise ValueError("cannot estimate spot")
    return float(fallback)


def atm_iv_table(chain: pd.DataFrame, spot: float, asof: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for e, g in chain.groupby("exp"):
        g = g[g["bid"] > 0]
        c = g[g["cp"] == "C"].set_index("strike")["implied_volatility"]
        p = g[g["cp"] == "P"].set_index("strike")["implied_volatility"]
        j = pd.concat([c, p], axis=1, keys=["c", "p"]).dropna()
        if j.empty:
            continue
        k = (j.index.to_series() - spot).abs().idxmin()
        rows.append({"exp": e, "dte": (e - asof).days, "atm_strike": k, "atm_iv": float(j.loc[k].mean())})
    return pd.DataFrame(rows).sort_values("exp").reset_index(drop=True)


def term_edge(ivtab: pd.DataFrame, front_dte: int = 10, back_dte: int = 45) -> dict:
    t = ivtab[ivtab["dte"] >= 1]
    f = t.iloc[(t["dte"] - front_dte).abs().argsort()[:1]].iloc[0]
    b = t.iloc[(t["dte"] - back_dte).abs().argsort()[:1]].iloc[0]
    return {
        "front_exp": f["exp"], "front_dte": int(f["dte"]), "front_iv": float(f["atm_iv"]),
        "back_exp": b["exp"], "back_dte": int(b["dte"]), "back_iv": float(b["atm_iv"]),
        "edge_pts": float(f["atm_iv"] - b["atm_iv"]) * 100,
    }


# --------------------------------------------------------------------------- flow / dex / regime
def flow_live(flow: pd.DataFrame | None, unusual: pd.DataFrame | None, min_dte: int = 1,
              contrarian: bool = False) -> dict:
    out = {"score": np.nan, "aggr": np.nan, "unusual": np.nan, "mid_share": np.nan,
           "bull_prem": np.nan, "bear_prem": np.nan, "mid_prem": np.nan}
    parts = []
    if flow is not None:
        f = flow[flow["Symbol"] == "QQQ"].copy()
        f["Premium"] = pd.to_numeric(f["Premium"], errors="coerce")
        f["DTE"] = pd.to_numeric(f["DTE"], errors="coerce")
        f = f[(f["Premium"] > 0) & (f["DTE"] >= min_dte)]
        side, typ = f["Side"].astype(str).str.lower(), f["Type"].astype(str).str.lower()
        bull = f.loc[((typ == "call") & (side == "ask")) | ((typ == "put") & (side == "bid")), "Premium"].sum()
        bear = f.loc[((typ == "put") & (side == "ask")) | ((typ == "call") & (side == "bid")), "Premium"].sum()
        mid = f.loc[side == "mid", "Premium"].sum()
        out.update(bull_prem=bull, bear_prem=bear, mid_prem=mid,
                   mid_share=mid / f["Premium"].sum() if f["Premium"].sum() else np.nan)
        if bull + bear > 0:
            out["aggr"] = (bull - bear) / (bull + bear)
            parts.append((0.6, float(clip(2 * out["aggr"]))))
    if unusual is not None:
        u = unusual[unusual["Symbol"] == "QQQ"].copy()
        for col in ["Volume", "Latest", "DTE"]:
            u[col] = pd.to_numeric(u[col].astype(str).str.replace(",", ""), errors="coerce")
        u = u[u["DTE"] >= max(min_dte, 2)]
        u["prem"] = u["Volume"] * u["Latest"] * 100
        cp = u.loc[u["Type"].astype(str).str.lower() == "call", "prem"].sum()
        pp = u.loc[u["Type"].astype(str).str.lower() == "put", "prem"].sum()
        if cp + pp > 0:
            out["unusual"] = (cp - pp) / (cp + pp)
            parts.append((0.4, float(clip(2 * out["unusual"]))))
    if parts:
        w = sum(p[0] for p in parts)
        s = sum(p[0] * p[1] for p in parts) / w
        out["score"] = float(-s if contrarian else s)
    return out


def dex_exp_score(dex_exp: pd.DataFrame | None, asof: pd.Timestamp, max_dte: int = 60) -> float:
    if dex_exp is None:
        return np.nan
    d = dex_exp.copy()
    d = d[((d["date"] - asof).dt.days >= 0) & ((d["date"] - asof).dt.days <= max_dte)]
    tot = (d["call_dex"].abs() + d["put_dex"].abs()).sum()
    if tot == 0:
        return np.nan
    return float(clip((d["net_dex"].sum() / tot) / 0.5))


def gamma_regime(gex_exp: pd.DataFrame | None, hist_last_gex: float, spot: float, asof: pd.Timestamp,
                 min_dte: int = 1, max_dte: int = 45, flip_buffer: float = 0.01) -> dict:
    if gex_exp is None:
        state = "Negative" if hist_last_gex < 0 else "Positive"
        return {"state": state, "effective": state, "near_flip": False, "net": hist_last_gex, "flip": np.nan,
                "table": pd.DataFrame(), "source": "history file (net GEX OI sign)"}
    g = gex_exp.copy()
    g["dte"] = (g["date"] - asof).dt.days
    w = g[(g["dte"] >= min_dte) & (g["dte"] <= max_dte)].copy()
    net = float(w["net_gex"].sum())
    size = w["call_gex"].abs() + w["put_gex"].abs()
    ok = w["gamma_flip"].notna() & (size > 0)
    flip = float(np.average(w.loc[ok, "gamma_flip"], weights=size[ok])) if ok.any() else np.nan
    state = "Negative" if net < 0 else "Positive"
    near = bool(not np.isnan(flip) and abs(spot - flip) / spot < flip_buffer)
    return {"state": state, "effective": "Negative" if near else state, "near_flip": near, "net": net,
            "flip": flip, "table": w, "source": f"net GEX over expiries {min_dte}-{max_dte} DTE"}


def band_from_score(score: float, t_mild: float = 0.10, t_strong: float = 0.30) -> str:
    if score <= -t_strong:
        return "Bearish"
    if score <= -t_mild:
        return "Mild Bearish"
    if score < t_mild:
        return "Neutral"
    if score < t_strong:
        return "Mild Bullish"
    return "Bullish"


def direction_score(components: dict, weights: dict, t_mild=0.10, t_strong=0.30) -> dict:
    """components: name -> score or NaN. Weights renormalised over available components."""
    avail = {k: v for k, v in components.items() if v is not None and not np.isnan(v) and weights.get(k, 0) > 0}
    if not avail:
        return {"score": np.nan, "band": "Neutral", "used": {}, "agreement": np.nan}
    wsum = sum(weights[k] for k in avail)
    score = sum(weights[k] * avail[k] for k in avail) / wsum
    sgn = np.sign(score)
    agree = np.mean([np.sign(v) == sgn for v in avail.values()]) if sgn != 0 else np.nan
    return {"score": float(score), "band": band_from_score(score, t_mild, t_strong),
            "used": {k: weights[k] / wsum for k in avail}, "agreement": float(agree)}


# --------------------------------------------------------------------------- backtest tables
def backtest_tables(hc: pd.DataFrame, weights: dict, t_mild=0.10, t_strong=0.30, warmup: int = 200) -> dict:
    d = hc.copy()
    c = d["Close Price"]
    comp = {"price": d["price"], "flow": d["flow_pcr"], "dex": d["dex"]}
    wsum = sum(weights[k] for k in comp)
    d["score"] = sum(weights[k] * comp[k] for k in comp) / wsum
    d["band"] = pd.cut(d["score"], [-9, -t_strong, -t_mild, t_mild, t_strong, 9], labels=BANDS)
    d["gamma"] = np.where(d["GEX Net OI"] < 0, "Negative", "Positive")
    for h in (1, 5, 10):
        d[f"f{h}"] = c.shift(-h) / c - 1
    r = np.log(c).diff()
    d["rv5"] = r[::-1].rolling(5).std()[::-1].shift(-1) * np.sqrt(252) * 100
    t = d.iloc[warmup:].dropna(subset=["score"]).copy()
    ic_rows = []
    for k in ["price", "gex", "flow_pcr", "vol", "dex", "score"]:
        row = {"signal": k}
        for h in (1, 5, 10):
            s = t[[k, f"f{h}"]].dropna()
            row[f"IC_{h}d"] = round(s[k].corr(s[f"f{h}"], method="spearman"), 3)
        ic_rows.append(row)
    band_rows = []
    for b in BANDS:
        s = t[t["band"] == b]
        band_rows.append({"band": b, "n": len(s),
                          "5d_mean_%": round(s["f5"].mean() * 100, 2) if len(s) else np.nan,
                          "5d_up_%": round((s["f5"] > 0).mean() * 100) if len(s) else np.nan,
                          "10d_mean_%": round(s["f10"].mean() * 100, 2) if len(s) else np.nan})
    reg = t.groupby("gamma").agg(n=("f5", "size"), realised_vol_5d=("rv5", "mean"),
                                 abs_move_5d=("f5", lambda x: x.abs().mean() * 100)).round(2)
    half = len(t) // 2
    stab = {}
    for name, s in (("first half", t.iloc[:half]), ("second half", t.iloc[half:])):
        s = s[["score", "f5"]].dropna()
        stab[name] = round(s["score"].corr(s["f5"], method="spearman"), 3)
    return {"ic": pd.DataFrame(ic_rows), "bands": pd.DataFrame(band_rows), "regime": reg,
            "stability": stab, "n": len(t), "start": t["Date"].iloc[0], "end": t["Date"].iloc[-1],
            "rows": t}


# --------------------------------------------------------------------------- Black-Scholes + payoff engine
def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(cp: str, S: float, K: float, T: float, iv: float, r: float = 0.04, q: float = 0.006) -> float:
    if T <= 1e-9 or iv <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if cp == "C":
        return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


@dataclass
class Leg:
    cp: str
    strike: float
    exp: pd.Timestamp
    qty: int
    bid: float
    ask: float
    iv: float


@dataclass
class Strategy:
    name: str
    legs: list
    eval_date: pd.Timestamp
    debit_nat: float
    debit_mid: float
    max_profit: float
    max_loss: float
    breakevens: list
    pop: float
    grid: pd.DataFrame
    eligible: bool = True
    reason: str = ""
    warnings: list = field(default_factory=list)
    source: str = "matrix cell"
    model_based: bool = False


def evaluate_legs(legs: list, spot: float, asof: pd.Timestamp, eval_iv: float, r=0.04, q=0.006):
    ev = min(l.exp for l in legs)
    grid = np.linspace(spot * 0.65, spot * 1.35, 561)
    debit_nat = sum(l.qty * (l.ask if l.qty > 0 else l.bid) for l in legs)
    debit_mid = sum(l.qty * (l.bid + l.ask) / 2 for l in legs)
    val = np.zeros_like(grid)
    model = False
    for l in legs:
        T = (l.exp - ev).days / 365.0
        if T > 0:
            model = True
        for i, s in enumerate(grid):
            val[i] += l.qty * bs(l.cp, s, l.strike, T, l.iv, r, q)
    pnl = (val - debit_nat) * 100
    # probability of profit under lognormal(eval_iv) to eval date
    T0 = max((ev - asof).days, 0.5) / 365.0
    sig = max(eval_iv, 0.05) * math.sqrt(T0)
    lg = np.log(grid / spot)
    w = np.exp(-0.5 * ((lg + 0.5 * sig * sig) / sig) ** 2) / grid
    pop = float((w * (pnl > 0)).sum() / w.sum())
    # breakevens
    be = []
    sign = np.sign(pnl)
    for i in range(1, len(grid)):
        if sign[i] != sign[i - 1] and sign[i - 1] != 0:
            x0, x1, y0, y1 = grid[i - 1], grid[i], pnl[i - 1], pnl[i]
            be.append(round(x0 - y0 * (x1 - x0) / (y1 - y0), 2))
    window = (grid >= spot * 0.85) & (grid <= spot * 1.15)
    return ev, debit_nat, debit_mid, float(pnl[window].max()), float(pnl[window].min()), be, pop, \
        pd.DataFrame({"spot": grid, "pnl": pnl}), model


# --------------------------------------------------------------------------- strategy construction
class Builder:
    def __init__(self, chain, spot, asof, em, gex_strike, ivtab, params, iv_rank, edge_threshold_pts):
        self.ch = chain[(chain["bid"] >= 0) & (chain["ask"] > 0)].copy()
        self.spot, self.asof, self.em, self.gs, self.ivtab = spot, asof, em, gex_strike, ivtab
        self.p = params
        self.iv_rank = iv_rank
        self.edge_thr = edge_threshold_pts
        self.exps = sorted(e for e in self.ch["exp"].unique() if (e - asof).days >= 1)

    # --- helpers
    def pick_exp(self, target_dte: int, after: pd.Timestamp | None = None):
        cand = [e for e in self.exps if after is None or e > after]
        return min(cand, key=lambda e: (abs((e - self.asof).days - target_dte), -((e - self.asof).days)))

    def em_for(self, exp):
        if self.em is None or self.em.empty:
            iv = self.iv_at(exp)
            T = max((exp - self.asof).days, 1) / 365.0
            m = self.spot * iv * math.sqrt(T)
            return self.spot - m, self.spot + m
        d = self.em.iloc[(self.em["date"] - exp).abs().argsort()[:1]].iloc[0]
        return float(d["lower_price"]), float(d["upper_price"])

    def iv_at(self, exp):
        r = self.ivtab[self.ivtab["exp"] == exp]
        return float(r["atm_iv"].iloc[0]) if len(r) else 0.25

    def snap(self, exp, cp, target, step: int = 5):
        t = round(target / step) * step if step else target
        g = self.ch[(self.ch["exp"] == exp) & (self.ch["cp"] == cp) & (self.ch["bid"] > 0)]
        if g.empty:
            return None
        return float(g.iloc[(g["strike"] - t).abs().argsort()[:1]]["strike"].iloc[0])

    def leg(self, exp, cp, strike, qty):
        g = self.ch[(self.ch["exp"] == exp) & (self.ch["cp"] == cp) & (self.ch["strike"] == strike)]
        if g.empty:
            return None
        r = g.iloc[0]
        return Leg(cp, float(strike), exp, qty, float(r["bid"]), float(r["ask"]), float(r["implied_volatility"]))

    def put_wall(self, lo, hi):
        if self.gs is None:
            return None
        g = self.gs[(self.gs["strike"] >= lo) & (self.gs["strike"] <= hi)]
        return float(g.loc[g["put_gex"].idxmin(), "strike"]) if len(g) else None

    def call_wall(self, lo, hi):
        if self.gs is None:
            return None
        g = self.gs[(self.gs["strike"] >= lo) & (self.gs["strike"] <= hi)]
        return float(g.loc[g["call_gex"].idxmax(), "strike"]) if len(g) else None

    def finish(self, name, legs, notes=None, eligible=True, reason=""):
        if any(l is None for l in legs):
            return None
        ev = min(l.exp for l in legs)
        res = evaluate_legs(legs, self.spot, self.asof, self.iv_at(ev))
        ev, dn, dm, mp, ml, be, pop, grid, model = res
        s = Strategy(name, legs, ev, dn, dm, mp, ml, be, pop, grid, eligible, reason,
                     list(notes or []), model_based=model)
        return s

    def width_ok(self, a, b):
        return a is not None and b is not None and a != b

    # --- verticals
    def bear_put_debit(self):
        e = self.pick_exp(self.p["vert_dte"])
        lo, _ = self.em_for(e)
        f = self.p["em_fraction"]
        long = self.snap(e, "P", self.spot)
        short = self.snap(e, "P", self.spot - f * (self.spot - lo))
        if not self.width_ok(long, short) or short >= long:
            return None
        s = self.finish("Bear put debit spread", [self.leg(e, "P", long, 1), self.leg(e, "P", short, -1)])
        if s and self.iv_rank is not None and self.iv_rank >= 80:
            s.warnings.append("IV rank >= 80: long premium is expensive")
        return s

    def bull_call_debit(self):
        e = self.pick_exp(self.p["vert_dte"])
        _, hi = self.em_for(e)
        f = self.p["em_fraction"]
        long = self.snap(e, "C", self.spot)
        short = self.snap(e, "C", self.spot + f * (hi - self.spot))
        if not self.width_ok(long, short) or short <= long:
            return None
        s = self.finish("Bull call debit spread", [self.leg(e, "C", long, 1), self.leg(e, "C", short, -1)])
        if s and self.iv_rank is not None and self.iv_rank >= 80:
            s.warnings.append("IV rank >= 80: long premium is expensive")
        return s

    def bull_put_credit(self):
        e = self.pick_exp(self.p["vert_dte"])
        lo, _ = self.em_for(e)
        short = self.snap(e, "P", lo)
        long = self.snap(e, "P", (short or lo) - self.p["width"], step=0)
        if not self.width_ok(long, short) or long >= short:
            return None
        s = self.finish("Bull put credit spread", [self.leg(e, "P", short, -1), self.leg(e, "P", long, 1)])
        if s and self.iv_rank is not None and self.iv_rank < 40:
            s.warnings.append("IV rank < 40: premium is thin for a credit trade")
        return s

    def bear_call_credit(self):
        e = self.pick_exp(self.p["vert_dte"])
        _, hi = self.em_for(e)
        short = self.snap(e, "C", hi)
        long = self.snap(e, "C", (short or hi) + self.p["width"], step=0)
        if not self.width_ok(long, short) or long <= short:
            return None
        s = self.finish("Bear call credit spread", [self.leg(e, "C", short, -1), self.leg(e, "C", long, 1)])
        if s and self.iv_rank is not None and self.iv_rank < 40:
            s.warnings.append("IV rank < 40: premium is thin for a credit trade")
        return s

    def iron_condor(self):
        e = self.pick_exp(self.p["vert_dte"])
        lo, hi = self.em_for(e)
        sp, sc = self.snap(e, "P", lo), self.snap(e, "C", hi)
        if sp is None or sc is None:
            return None
        lp = self.snap(e, "P", sp - self.p["width"], step=0)
        lc = self.snap(e, "C", sc + self.p["width"], step=0)
        legs = [self.leg(e, "P", sp, -1), self.leg(e, "P", lp, 1), self.leg(e, "C", sc, -1), self.leg(e, "C", lc, 1)]
        ok = self.iv_rank is not None and self.iv_rank >= 50
        s = self.finish("Iron condor", legs, eligible=ok,
                        reason="" if ok else f"IV rank {self.iv_rank:.0f} < 50: condor needs rich premium")
        if s:
            s.warnings.append("Selective: negative-gamma regimes can run through short strikes")
        return s

    # --- broken-wing butterflies
    def put_bwb(self, name="Put BWB"):
        e = self.pick_exp(self.p["vert_dte"])
        lo, _ = self.em_for(e)
        upper = self.snap(e, "P", self.spot)
        body = self.put_wall(lo, self.spot * 0.99)
        body = self.snap(e, "P", body if body else self.spot - 0.5 * (self.spot - lo))
        if not self.width_ok(upper, body) or body >= upper:
            return None
        lower = self.snap(e, "P", body - round(1.33 * (upper - body) / 5) * 5, step=0)
        if lower is None or lower >= body:
            return None
        return self.finish(name, [self.leg(e, "P", upper, 1), self.leg(e, "P", body, -2), self.leg(e, "P", lower, 1)])

    def call_bwb(self, name="Call BWB"):
        e = self.pick_exp(self.p["vert_dte"])
        _, hi = self.em_for(e)
        lower = self.snap(e, "C", self.spot)
        body = self.call_wall(self.spot * 1.01, hi)
        body = self.snap(e, "C", body if body else self.spot + 0.5 * (hi - self.spot))
        if not self.width_ok(lower, body) or body <= lower:
            return None
        upper = self.snap(e, "C", body + round(1.33 * (body - lower) / 5) * 5, step=0)
        if upper is None or upper <= body:
            return None
        return self.finish(name, [self.leg(e, "C", lower, 1), self.leg(e, "C", body, -2), self.leg(e, "C", upper, 1)])

    def neutral_bwb(self):
        e = self.pick_exp(self.p["vert_dte"])
        lo, hi = self.em_for(e)
        pw, cw = self.put_wall(lo, self.spot), self.call_wall(self.spot, hi)
        dp = (self.spot - pw) if pw else 1e9
        dc = (cw - self.spot) if cw else 1e9
        s = self.put_bwb("BWB (put side)") if dp <= dc else self.call_bwb("BWB (call side)")
        return s

    # --- calendars
    def _calendar(self, cp, strike_mode, name):
        fe = self.pick_exp(self.p["cal_front_dte"])
        be = self.pick_exp((fe - self.asof).days + self.p["cal_gap"], after=fe)
        lo, hi = self.em_for(fe)
        if strike_mode == "atm":
            k = self.snap(fe, cp, self.spot, step=0)
        elif cp == "P":
            w = self.put_wall(lo, self.spot)
            k = self.snap(fe, "P", w if w else self.spot - 0.5 * (self.spot - lo), step=0)
        else:
            w = self.call_wall(self.spot, hi)
            k = self.snap(fe, "C", w if w else self.spot + 0.5 * (hi - self.spot), step=0)
        legs = [self.leg(fe, cp, k, -1), self.leg(be, cp, k, 1)] if k else [None]
        if any(l is None for l in legs):
            return None
        edge = (legs[0].iv - legs[1].iv) * 100
        ok = edge >= self.edge_thr
        s = self.finish(name, legs, eligible=ok,
                        reason="" if ok else f"No IV edge: front-back {edge:.1f} pts < {self.edge_thr:.1f}")
        if s:
            s.warnings.append(f"Front-back IV edge {edge:.1f} pts (threshold {self.edge_thr:.1f})")
        return s

    def _double_calendar(self, f, name):
        fe = self.pick_exp(self.p["cal_front_dte"])
        be = self.pick_exp((fe - self.asof).days + self.p["cal_gap"], after=fe)
        lo, hi = self.em_for(fe)
        kp = self.snap(fe, "P", self.spot - f * (self.spot - lo))
        kc = self.snap(fe, "C", self.spot + f * (hi - self.spot))
        legs = [self.leg(fe, "P", kp, -1), self.leg(be, "P", kp, 1), self.leg(fe, "C", kc, -1), self.leg(be, "C", kc, 1)]
        if any(l is None for l in legs):
            return None
        edges = [(legs[0].iv - legs[1].iv) * 100, (legs[2].iv - legs[3].iv) * 100]
        ok = min(edges) >= self.edge_thr
        s = self.finish(name, legs, eligible=ok,
                        reason="" if ok else f"No IV edge: min front-back {min(edges):.1f} pts < {self.edge_thr:.1f}")
        if s:
            s.warnings.append(f"Front-back IV edge put {edges[0]:.1f} / call {edges[1]:.1f} pts")
        return s

    # --- diagonals
    def _diagonal(self, cp, name):
        se = self.pick_exp(self.p["diag_front_dte"])
        le = self.pick_exp((se - self.asof).days + self.p["diag_gap"], after=se)
        lo, hi = self.em_for(se)
        f = self.p["em_fraction"]
        if cp == "P":
            short = self.snap(se, "P", self.spot - f * (self.spot - lo))
            long = self.snap(le, "P", self.spot * 0.995)
            bad = short is None or long is None or short >= long
        else:
            short = self.snap(se, "C", self.spot + f * (hi - self.spot))
            long = self.snap(le, "C", self.spot * 1.005)
            bad = short is None or long is None or short <= long
        if bad:
            return None
        s = self.finish(name, [self.leg(se, cp, short, -1), self.leg(le, cp, long, 1)])
        if s:
            s.warnings.append("After the short leg expires you hold only the long option: plan the exit")
        return s

    # --- dispatch
    def build(self, name):
        table = {
            "Bear put debit spread": self.bear_put_debit,
            "Bull call debit spread": self.bull_call_debit,
            "Bull put credit spread": self.bull_put_credit,
            "Bear call credit spread": self.bear_call_credit,
            "Iron condor": self.iron_condor,
            "Put BWB": self.put_bwb,
            "Call BWB": self.call_bwb,
            "BWB": self.neutral_bwb,
            "Put calendar": lambda: self._calendar("P", "wall", "Put calendar"),
            "OTM put calendar": lambda: self._calendar("P", "wall", "OTM put calendar"),
            "ATM calendar": lambda: self._calendar("P", "atm", "ATM calendar"),
            "ATM call calendar": lambda: self._calendar("C", "atm", "ATM call calendar"),
            "OTM call calendar": lambda: self._calendar("C", "wall", "OTM call calendar"),
            "Double calendar": lambda: self._double_calendar(0.5, "Double calendar"),
            "OTM calendars": lambda: self._double_calendar(0.5, "OTM calendars (double)"),
            "Wide double calendar": lambda: self._double_calendar(0.9, "Wide double calendar"),
            "Put diagonal": lambda: self._diagonal("P", "Put diagonal"),
            "Call diagonal": lambda: self._diagonal("C", "Call diagonal"),
        }
        fn = table.get(name)
        return fn() if fn else None


def adjacent_cells(band: str, regime: str):
    i = BANDS.index(band)
    order = []
    for j in (i - 1, i + 1, i - 2, i + 2):
        if 0 <= j < len(BANDS):
            order.append((BANDS[j], regime))
    return order


def pick_strategies(builder: Builder, band: str, regime: str, top_n: int = 5) -> list[Strategy]:
    out: list[Strategy] = []
    seen = set()

    def add(names, source):
        for n in names:
            if len(out) >= top_n:
                return
            s = builder.build(n)
            if s is None:
                continue
            key = (s.name, tuple((l.cp, l.strike, l.exp, l.qty) for l in s.legs))
            if key in seen:
                continue
            seen.add(key)
            s.source = source
            out.append(s)

    add(MATRIX[(band, regime)], f"{band} / {regime} gamma")
    for b, r in adjacent_cells(band, regime):
        if len(out) >= top_n:
            break
        add(MATRIX[(b, r)], f"adjacent: {b} / {r} gamma")
    out.sort(key=lambda s: (not s.eligible,))  # stable: eligible first, matrix order kept
    return out


# --------------------------------------------------------------------------- one-call analysis
def analyse(data: dict, weights: dict, opts: dict) -> dict:
    """data: kind -> raw dataframe. Returns everything the UI needs."""
    if "history" not in data or "chain" not in data:
        raise ValueError("Need at least the option history and option chain files.")
    h = parse_history(data["history"])
    chain = parse_chain(data["chain"])
    asof = pd.Timestamp(opts.get("asof") or h["Date"].iloc[-1]).normalize()
    spot = estimate_spot(chain, fallback=float(h["Close Price"].iloc[-1]))
    hc = history_components(h, flow_contrarian=opts.get("flow_contrarian", False))
    last = hc.iloc[-1]
    gex_exp = parse_gex_exp(data["gex_exp"]) if "gex_exp" in data else None
    dex_exp = parse_dex_exp(data["dex_exp"]) if "dex_exp" in data else None
    em = parse_em(data["exp_move"]) if "exp_move" in data else None
    gs = data["gex_strike"].copy() if "gex_strike" in data else None
    ivtab = atm_iv_table(chain, spot, asof)
    edge = term_edge(ivtab)
    fl = flow_live(data.get("flow"), data.get("unusual"), opts.get("flow_min_dte", 1),
                   opts.get("flow_contrarian", False))
    dexs = dex_exp_score(dex_exp, asof)
    dex_parts = [v for v in (last["dex"], dexs) if not np.isnan(v)]
    dex_score = float(np.mean(dex_parts)) if dex_parts else np.nan
    flow_score = fl["score"] if not np.isnan(fl["score"]) else float(last["flow_pcr"])
    flow_source = "live flow files" if not np.isnan(fl["score"]) else "history put/call ratio (fallback)"
    comps = {"price": float(last["price"]), "flow": flow_score, "dex": dex_score}
    ds = direction_score(comps, weights, opts.get("t_mild", 0.10), opts.get("t_strong", 0.30))
    reg = gamma_regime(gex_exp, float(last["GEX Net OI"]), spot, asof, opts.get("reg_min_dte", 1),
                       opts.get("reg_max_dte", 45), opts.get("flip_buffer", 0.01))
    ivr = last["IV Rank"] if not np.isnan(last["IV Rank"]) else last["ivr_own"] * 100
    ivr = None if np.isnan(ivr) else float(ivr)
    walls = {}
    if gs is not None:
        lo, hi = spot * 0.93, spot * 1.07
        near = gs[(gs["strike"] >= lo) & (gs["strike"] <= hi)]
        walls = {
            "put_wall": float(near.loc[near["put_gex"].idxmin(), "strike"]),
            "call_wall": float(near.loc[near["call_gex"].idxmax(), "strike"]),
            "table": near.assign(net=near["call_gex"] + near["put_gex"]),
        }
    builder = Builder(chain, spot, asof, em, gs, ivtab, opts["structure"], ivr, opts.get("edge_threshold", 1.5))
    strategies = pick_strategies(builder, ds["band"], reg["effective"], opts.get("top_n", 5))
    return {"hist": hc, "asof": asof, "spot": spot, "components": comps, "direction": ds, "regime": reg,
            "flow": fl, "flow_source": flow_source, "dex_hist": float(last["dex"]), "dex_exp": dexs,
            "ivrank": ivr, "edge": edge, "ivtab": ivtab, "walls": walls, "em": em, "gex_exp": gex_exp,
            "dex_exp_df": dex_exp, "strategies": strategies, "builder": builder,
            "stay_out": not any(s.eligible for s in strategies)}
