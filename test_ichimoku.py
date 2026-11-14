"""Tests for ichimoku_screener.py (no network, no scipy)."""
import contextlib, io, os, sys, tempfile
sys.modules["scipy"] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import ichimoku_screener as ich
import donchian_screener as dsc


def mk(closes, spread=0.4, vol=1e6, volumes=None):
    c = np.asarray(closes, float); n = len(c)
    idx = pd.bdate_range("2023-01-02", periods=n)
    o = np.r_[c[0], c[:-1]]
    v = np.full(n, vol) if volumes is None else np.asarray(volumes, float)
    return pd.DataFrame({"Open": o, "High": np.maximum(c, o) + spread, "Low": np.minimum(c, o) - spread, "Close": c, "Volume": v}, index=idx)


def ind(df): return ich.add_indicators(df)


def at(d, n, setup, side=None, lookback=0):
    """Signals of one setup using only the first n bars (so each case sits on an exact bar)."""
    return [r for r in ich.scan_symbol(d.iloc[:n], "T", lookback=lookback)
            if r["setup"] == setup and (side is None or r["side"] == side)]


UP = [100 + 0.3 * i + 0.4 * (i % 3) for i in range(300)]
DN = [200 - 0.3 * i for i in range(200)]
UPDIP = UP + [UP[-1] - 1.5, UP[-1] - 3, UP[-1] - 4.5, UP[-1] - 3, UP[-1]]
TURN_UP = DN + [DN[-1] + 0.5 * i for i in range(1, 120)]           # long downtrend, then a steady reversal
TURN_DN = UP + [UP[-1] - 0.5 * i for i in range(1, 120)]           # long uptrend, then a steady reversal

# ---- 1. indicators vs a slow row-by-row reference, and no look-ahead
rng = np.random.default_rng(5)
cc = 100 + np.cumsum(rng.normal(0.05, 1.0, 400))
df = mk(cc, spread=0.6)
d = ind(df)
h, l = df.High.values, df.Low.values
N = len(df)
ten = np.full(N, np.nan); kij = np.full(N, np.nan); sbr = np.full(N, np.nan)
for i in range(N):
    if i >= 8: ten[i] = (h[i - 8:i + 1].max() + l[i - 8:i + 1].min()) / 2
    if i >= 25: kij[i] = (h[i - 25:i + 1].max() + l[i - 25:i + 1].min()) / 2
    if i >= 51: sbr[i] = (h[i - 51:i + 1].max() + l[i - 51:i + 1].min()) / 2
sar = (ten + kij) / 2
assert np.allclose(d["tenkan"], ten, equal_nan=True) and np.allclose(d["kijun"], kij, equal_nan=True)
assert np.allclose(d["sa_raw"], sar, equal_nan=True) and np.allclose(d["sb_raw"], sbr, equal_nan=True)
assert np.allclose(d["sa"].iloc[100:], sar[100 - 26:N - 26]) and np.allclose(d["sb"].iloc[100:], sbr[100 - 26:N - 26])
assert d["sa"].iloc[:50].isna().all() and d["sb"].iloc[:77].isna().all() and d["cloud_top"].iloc[77:].notna().all()
assert (d["cloud_top"].iloc[77:] >= d["cloud_bot"].iloc[77:]).all()
assert np.allclose(d["past"].iloc[30:], df.Close.shift(26).iloc[30:])
# truncating the history must not change any value at the last kept bar (nothing reads the future)
for cut in (150, 233, 300):
    dc = ind(df.iloc[:cut])
    for col in ("tenkan", "kijun", "sa", "sb", "cloud_top", "cloud_bot", "past", "adx", "atr", "fut_bull"):
        a, b = dc[col].iloc[-1], d[col].iloc[cut - 1]
        assert (pd.isna(a) and pd.isna(b)) or np.isclose(a, b), (cut, col, a, b)
print("1 Tenkan/Kijun/Senkou/chikou match reference; truncation leaves the last bar unchanged")

# ---- 2. Tenkan / Kijun cross (both sides, grading by cloud position)
dm = ind(mk(TURN_UP))
r = at(dm, 210, "tk_cross", "long")
assert len(r) == 1 and r[0]["bars_ago"] == 0 and r[0]["cloud"] == "below" and r[0]["grade"] == "C" and "price vs cloud" in r[0]["note"], r
assert r[0]["stop"] < r[0]["close"] and r[0]["level"] == round(dm["kijun"].iloc[209], 2)
assert at(dm, 211, "tk_cross") == [], "a cross is reported on its own bar, not on the day after (lookback 0)"
r3 = at(dm, 213, "tk_cross", "long", lookback=3)
assert len(r3) == 1 and r3[0]["bars_ago"] == 3, r3
ds = ind(mk(TURN_DN))
r = at(ds, 310, "tk_cross", "short")
assert len(r) == 1 and r[0]["grade"] == "C" and r[0]["stop"] > r[0]["close"] > 0, r
print("2 TK cross: long/short, freshness, stop side, grade C against the cloud")

# ---- 3. Cloud break (fresh close out of the cloud)
r = at(dm, 220, "cloud_break", "long")
assert len(r) == 1 and r[0]["grade"] == "A" and r[0]["bars_ago"] == 0 and "all filters pass" in r[0]["note"], r
assert r[0]["stop"] < r[0]["level"] < r[0]["close"] and r[0]["cloud"] == "above" and "ATR thick" in r[0]["note"]
assert at(dm, 221, "cloud_break") == [] and at(dm, 219, "cloud_break") == [], "only the bar of the break"
assert at(dm, 222, "cloud_break", lookback=3)[0]["bars_ago"] == 2
r = at(ds, 320, "cloud_break", "short")
assert len(r) == 1 and r[0]["grade"] == "A" and r[0]["stop"] > r[0]["level"] > r[0]["close"], r
# no volume, flat ADX: lower grade and the failures are named
low_vol = mk(TURN_UP, volumes=np.r_[np.full(215, 1e6), np.full(len(TURN_UP) - 215, 1e5)])
rl = at(ind(low_vol), 220, "cloud_break", "long")
assert len(rl) == 1 and "volume" in rl[0]["note"] and rl[0]["grade"] in ("A", "B"), rl
print("3 cloud break: long/short, freshness, stop beyond far side of cloud, filters named")

# ---- 4. Kumo twist
r = at(dm, 218, "kumo_twist", "long")
assert len(r) == 1 and "turned bullish" in r[0]["note"] and r[0]["stop"] < r[0]["close"], r
assert at(dm, 219, "kumo_twist") == []
r = at(ds, 318, "kumo_twist", "short")
assert len(r) == 1 and "turned bearish" in r[0]["note"] and r[0]["stop"] > r[0]["close"], r
print("4 kumo twist: both colours")

# ---- 5. Kijun bounce (latest bar only)
du = ind(mk(UPDIP))
r = at(du, 304, "kijun_bounce", "long")
assert len(r) == 1 and r[0]["grade"] == "A" and r[0]["bars_ago"] == 0 and r[0]["cloud"] == "above", r
assert r[0]["stop"] < r[0]["close"] and r[0]["level"] == round(du["kijun"].iloc[303], 2) and "vs Kijun" in r[0]["note"]
assert at(du, 300, "kijun_bounce") == [], "no pullback yet in a clean uptrend"
assert at(du, 305, "kijun_bounce", lookback=3, side="long")[0]["bars_ago"] == 0
rs = at(dm, 205, "kijun_bounce", "short")
assert len(rs) == 1 and rs[0]["stop"] > rs[0]["close"], rs
# closing through Kijun is a break, not a bounce
broke = UP + [UP[-1] - 12]
assert at(ind(mk(broke)), len(broke), "kijun_bounce") == []
print("5 Kijun bounce: long/short, needs trend side of cloud, not a break through Kijun")

# ---- 6. stops sit on the correct side and one row per (setup, side) over many random walks
rng = np.random.default_rng(21)
n_sig = 0
for s in range(60):
    cl = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, 420)))
    dd = ind(mk(cl, spread=0.8))
    rows = ich.scan_symbol(dd, "R%d" % s, lookback=5)
    keys = [(x["setup"], x["side"]) for x in rows]
    assert len(keys) == len(set(keys))
    for x in rows:
        n_sig += 1
        assert x["grade"] in ("A", "B", "C") and x["cloud"] in ("above", "inside", "below") and 0 <= x["bars_ago"] <= 5
        assert (x["stop"] < x["close"]) if x["side"] == "long" else (x["stop"] > x["close"]), x
assert 0 < n_sig < 60 * 8, n_sig
print("6 random walks: stops on the right side, one row per setup/side, signals:", n_sig)

# ---- 7. end-to-end with a fake loader, short history, missing ticker, CSV
rng = np.random.default_rng(7)
def fake(tickers, start):
    parts = {}
    for t in tickers:
        if t == "NONE": continue
        n = 100 if t == "SHORT" else 420
        parts[t] = mk(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))), spread=0.6)
    return pd.concat(parts, axis=1)
with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()) as buf:
    outp = os.path.join(td, "o.csv")
    res = ich.run(["A", "B", "C", "D", "NONE", "SHORT"] + ["T%d" % i for i in range(40)], out=outp, loader=fake, chunk=15, lookback=5)
    txt = buf.getvalue()
    assert "no data/errors 1" in txt and "too little history 1" in txt, txt[:300]
    assert res.attrs["failed"] == ["NONE"] and res.attrs["short"] == ["SHORT"] and res.attrs["n_tickers"] == 46
    assert res.attrs["min_rows"] == ich.MIN_ROWS
    assert len(res) == 0 or os.path.exists(outp)
    need = {"ticker", "setup", "side", "grade", "bars_ago", "date", "close", "level", "dist_atr", "stop", "adx", "atr", "atr_pct", "cloud", "note"}
    assert len(res) == 0 or need <= set(res.columns)
    assert set(res["setup"]) <= {"tk_cross", "cloud_break", "kijun_bounce", "kumo_twist"}
    assert len(res) < 100, len(res)
    prog = []
    ich.run(["A", "B"], out=None, loader=fake, verbose=False, progress=lambda a, b: prog.append((a, b)))
    assert prog[-1] == (2, 2)
print("7 end to end ok; signals on 44 random walks:", len(res), dict(res["setup"].value_counts()) if len(res) else {})

# ---- 8. robustness
assert ich.scan_symbol(ind(mk([100, 101, 102])), "T") == []
nanv = mk(UP); nanv["Volume"] = np.nan
assert isinstance(ich.scan_symbol(ind(nanv), "T"), list)
assert isinstance(ich.scan_symbol(ind(mk([100.0] * 200)), "T"), list)    # perfectly flat data must not raise
try:
    def boom(tickers, start): raise ImportError("No module named 'yfinance'")
    ich.run(["A"], out=None, loader=boom, verbose=False)
    raise SystemExit("ImportError should propagate")
except ImportError:
    pass
print("8 robustness ok")
print("ALL ICHIMOKU CHECKS PASSED")
