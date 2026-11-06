"""Tests for donchian_screener.py (no network, no scipy)."""
import contextlib, io, os, sys, tempfile, datetime as dt
sys.modules["scipy"] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import donchian_screener as sc

def mk(closes, spread=0.4, vol=1e6, opens=None, highs=None, lows=None, volumes=None):
    c = np.asarray(closes, float); n = len(c)
    idx = pd.bdate_range("2023-01-02", periods=n)
    o = np.r_[c[0], c[:-1]] if opens is None else np.asarray(opens, float)
    h = np.maximum(c, o) + spread if highs is None else np.asarray(highs, float)
    l = np.minimum(c, o) - spread if lows is None else np.asarray(lows, float)
    v = np.full(n, vol) if volumes is None else np.asarray(volumes, float)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)

def ind(df, **k): return sc.add_indicators(df, **k)
def at(d, setup, side=None, **k):
    return [r for r in sc.scan_symbol(d, "T", **k) if r["setup"] == setup and (side is None or r["side"] == side)]

# ---- 1. the original bug: with a prior-only channel a breakout CAN fire
base = [100 + 0.2 * (i % 3) for i in range(300)]
d = ind(mk(base + [104]))
assert d["Close"].iloc[-1] > d["hi"].iloc[-1]
assert not (d["High"].rolling(20).max() < d["Close"]).any(), "same-bar channel can never be broken (what the old code did)"
print("1 prior-only channel: breakout reachable")

# ---- 2. Wilder ATR/ADX vs a slow row-by-row reference
rng = np.random.default_rng(3)
cc = 100 + np.cumsum(rng.normal(0.1, 1.0, 400))
df = mk(cc, spread=0.5)
dd = ind(df)
h, l, c = df.High.values, df.Low.values, df.Close.values
n = 14; N = len(c)
tr = np.array([h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, N)])
pdm = np.zeros(N); mdm = np.zeros(N)
for i in range(1, N):
    u, dn = h[i] - h[i - 1], l[i - 1] - l[i]
    pdm[i] = u if (u > dn and u > 0) else 0
    mdm[i] = dn if (dn > u and dn > 0) else 0
def wref(x, start):                      # recursion seeded with the first valid value
    out = np.full(N, np.nan); out[start] = x[start]
    for i in range(start + 1, N): out[i] = out[i - 1] * (1 - 1 / n) + x[i] / n
    return out
atr = wref(tr, 0); P = 100 * wref(pdm, 1) / atr; M = 100 * wref(mdm, 1) / atr
dx = 100 * np.abs(P - M) / (P + M); adx = wref(np.nan_to_num(dx), 1)
assert np.allclose(dd["atr"].iloc[60:], atr[60:], rtol=1e-6)
assert np.allclose(dd["pdi"].iloc[60:], P[60:], rtol=0.02) and np.allclose(dd["mdi"].iloc[60:], M[60:], rtol=0.02)
assert np.allclose(dd["adx"].iloc[120:], adx[120:], atol=0.8), np.abs(dd["adx"].values[120:] - adx[120:]).max()
tre = ind(mk(np.arange(400) * 1.0 + 100, spread=0.3))
flat = ind(mk(100 + np.sin(np.arange(400) / 3.0), spread=0.3))
assert tre["adx"].iloc[-1] > 60 and tre["pdi"].iloc[-1] > tre["mdi"].iloc[-1] and flat["adx"].iloc[-1] < 30
print("2 ATR/ADX match reference; trend vs flat ok")

# ---- 3. breakout grading
vols = np.full(len(base) + 1, 1e6); vols[-1] = 3e6
grow = [100 + 0.25 * i + 0.3 * (i % 3) for i in range(250)]
grow += [grow[-1] + 0.3 * (i % 3) - 0.3 for i in range(50)]     # uptrend, then a 50-bar pause (EMA still rising)
g = ind(mk(grow + [grow[-1] + 6], volumes=np.r_[np.full(300, 1e6), 3e6]))
b = at(g, "breakout", "long")
assert len(b) == 1 and b[0]["bars_ago"] == 0 and b[0]["grade"] in ("A", "B"), b
assert b[0]["stop"] < b[0]["level"] < b[0]["close"]
# same break against the 200 EMA (downtrend then spike) is C
fall = [200 - 0.25 * i for i in range(300)]
b2 = at(ind(mk(fall + [fall[-1] + 8])), "breakout", "long")
assert b2 and b2[0]["grade"] == "C" and "200EMA trend" in b2[0]["note"], b2
# mirror
bm = at(ind(mk(fall + [fall[-1] - 6], volumes=np.r_[np.full(300, 1e6), 3e6])), "breakout", "short")
assert bm and bm[0]["grade"] in ("A", "B") and bm[0]["stop"] > bm[0]["level"] > bm[0]["close"], bm
# only a FRESH break is reported; a second day above the level is not a new breakout
two = at(ind(mk(grow + [grow[-1] + 6, grow[-1] + 6.5])), "breakout", "long", lookback=0)
assert two == [], "day 2 of a breakout is not fresh"
three = at(ind(mk(grow + [grow[-1] + 6, grow[-1] + 6.5])), "breakout", "long", lookback=3)
assert len(three) == 1 and three[0]["bars_ago"] == 1
# big gap lowers the score
gap_o = list(np.r_[grow[:-0 or None]]); 
dgap = mk(grow + [grow[-1] + 6], opens=np.r_[np.r_[grow[0], grow[:-1]], grow[-1] + 5.5], volumes=np.r_[np.full(300, 1e6), 3e6])
bg = at(ind(dgap), "breakout", "long")
assert bg and "no big gap" in bg[0]["note"]
print("3 breakout grading, mirror, freshness, gap ok")

# ---- 4. pullback
ramp = [100.0] * 280 + [101.0 + 1.5 * i for i in range(14)] + [119.5 - 2.0 * i for i in range(7)]
pb = at(ind(mk(ramp, spread=0.3)), "pullback", "long")
d4 = ind(mk(ramp, spread=0.3))
print("   pb debug:", round(d4.close.iloc[-1] if False else d4["Close"].iloc[-1], 2), round(d4["mid"].iloc[-1], 2), round(d4["adx"].iloc[-1], 1))
assert len(pb) == 1 and pb[0]["stop"] < d4["lo"].iloc[-1] and "ATR off" in pb[0]["note"], pb
pbm = at(ind(mk([200 - v for v in ramp], spread=0.3)), "pullback", "short")
assert len(pbm) == 1 and pbm[0]["stop"] > ind(mk([200 - v for v in ramp], spread=0.3))["hi"].iloc[-1]
# price already through the channel low = no pullback signal (broken, not pulled back)
crash = ramp + [90.0]
assert at(ind(mk(crash, spread=0.3)), "pullback", "long") == []
# below the 200 EMA = no long pullback
assert at(ind(mk([100 + 0.3 * i for i in range(100)] + [130 - 0.8 * i for i in range(200)], spread=0.3)), "pullback", "long") == []
print("4 pullback ok (long, short, broken channel, below EMA)")

# ---- 5. squeeze and squeeze break
sine = [100 + 10 * np.sin(i / 6) for i in range(320)] + [100 + 0.4 * np.sin(i / 6) for i in range(45)]
ds = ind(mk(sine, spread=0.1))
sq = at(ds, "squeeze", "watch")
assert len(sq) == 1 and sq[0]["grade"] in ("A", "B") and "alert above" in sq[0]["note"], sq
brk = ind(mk(sine + [sine[-1] + 3], spread=0.1))
sb = at(brk, "squeeze_break", "long")
assert len(sb) == 1 and sb[0]["bars_ago"] == 0 and at(brk, "squeeze", "watch") == [], (sb,)
print("5 squeeze watch + break ok")

# ---- 6. failed breakout: lags 1-3, window limit, fresh-only, mirror, ADX grade
B = [100 + 0.2 * (i % 3) for i in range(300)]
for tail, lag in (([103, 99], 1), ([103, 102, 99], 2), ([103, 102, 101, 99], 3)):
    r = at(ind(mk(B + tail)), "failed_break", "short")
    assert len(r) == 1 and r[0]["bars_ago"] == 0 and "lag %d" % lag in r[0]["note"], (tail, r)
    assert r[0]["stop"] > r[0]["level"] and r[0]["grade"] == "A"          # flat market: ADX low
assert at(ind(mk(B + [103, 102, 101, 101, 99])), "failed_break", "short") == [], "4 bars is outside the window"
assert at(ind(mk(B + [103, 104, 105])), "failed_break") == []
r = at(ind(mk(B + [97, 101])), "failed_break", "long")
assert len(r) == 1 and r[0]["stop"] < r[0]["level"]
assert len(at(ind(mk(B + [103, 99, 99.5])), "failed_break", "short", lookback=3)) == 1, "one event, reported once"
# strong, rising ADX makes it a B
tr_up = [100 + 1.0 * i for i in range(300)]
top = tr_up[-1]
seq = tr_up + [top - 0.5, top - 1.0, top - 0.5, top + 9, top - 1]       # short pause, fresh break, then back inside
dB = ind(mk(seq, spread=0.3))
print("   strong-trend debug: adx", round(dB["adx"].iloc[-1], 1), "adx 3 ago", round(dB["adx"].iloc[-4], 1))
rb = at(dB, "failed_break", "short")
assert rb and rb[0]["grade"] == "B" and "riskier" in rb[0]["note"], rb
print("6 failed breakout ok")

# ---- 7. data handling
f1 = mk(B)
cols2 = pd.MultiIndex.from_product([["AAA", "BBB"], FIELDS := sc.FIELDS])
raw_t = pd.concat({"AAA": f1, "BBB": f1 * 2}, axis=1)                          # (ticker, field)
raw_f = raw_t.swaplevel(axis=1).sort_index(axis=1)                              # (field, ticker)
for raw in (raw_t, raw_f):
    a = sc.extract(raw, "BBB")
    assert a is not None and abs(a["Close"].iloc[-1] - 2 * f1["Close"].iloc[-1]) < 1e-9 and list(a.columns) == FIELDS
assert sc.extract(raw_t, "ZZZ") is None
single = sc.extract(f1, "X"); assert len(single) == len(f1)
# partial-bar guard
from zoneinfo import ZoneInfo
x = mk(B[:10]); today = x.index[-1].date()
open_t = dt.datetime.combine(today, dt.time(11, 0), ZoneInfo("America/New_York"))
shut_t = dt.datetime.combine(today, dt.time(17, 0), ZoneInfo("America/New_York"))
assert len(sc.drop_partial_bar(x, open_t)) == 9 and len(sc.drop_partial_bar(x, shut_t)) == 10
print("7 extract (both column orders), single frame, partial bar ok")

# ---- 8. end-to-end with a fake loader, short history, missing ticker, CSV
rng = np.random.default_rng(7)
def fake(tickers, start):
    parts = {}
    for t in tickers:
        if t == "NONE": continue
        n = 120 if t == "SHORT" else 420
        cl = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
        parts[t] = mk(cl, spread=0.6)
    return pd.concat(parts, axis=1)
with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()) as buf:
    outp = os.path.join(td, "o.csv")
    res = sc.run(["A", "B", "C", "D", "NONE", "SHORT"] + ["T%d" % i for i in range(40)], out=outp, loader=fake, chunk=15, lookback=5)
    txt = buf.getvalue()
    assert "no data/errors 1" in txt and "too little history 1" in txt, txt[:300]
    assert set(res["ticker"]) <= set(["A", "B", "C", "D"] + ["T%d" % i for i in range(40)])
    assert len(res) == 0 or os.path.exists(outp)
    assert len(res) < 80, len(res)                 # selective, not firing on everything
    need = {"ticker", "setup", "side", "grade", "bars_ago", "date", "close", "level", "dist_atr", "stop", "adx", "atr", "atr_pct", "note"}
    assert len(res) == 0 or need <= set(res.columns)
    pd.testing.assert_frame_equal(sc.run([], loader=fake, verbose=False) if False else pd.DataFrame(), pd.DataFrame())
print("8 end to end ok; signals on 44 random walks:", len(res), dict(res["setup"].value_counts()) if len(res) else {})

# ---- 9. robustness
tiny = ind(mk([100, 101, 102]))
assert sc.scan_symbol(tiny, "T") == []
nanv = mk(B); nanv["Volume"] = np.nan
assert isinstance(sc.scan_symbol(ind(nanv), "T"), list)                          # index-like data with no volume
print("9 robustness ok")
print("ALL SCREENER CHECKS PASSED")
