"""Tests for scipy-free backtest, GEX/DEX trend cards, OHLC parsing and the candlestick spec.
Usage: python3 test_features.py <folder_with_the_8_csvs>
scipy is deliberately blocked, because the app must not need it."""
import io
import json
import sys

sys.modules["scipy"] = None          # any `import scipy` / pandas spearman now raises ImportError
sys.modules["scipy.stats"] = None
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import core  # noqa: E402

folder = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
data = core.load_folder(folder)
assert set(data) == {"flow", "unusual", "chain", "history", "dex_exp", "gex_exp", "gex_strike", "exp_move"}, sorted(data)
print("8 files still classified correctly; no false 'ohlc' match")

# ---- 1. backtest without scipy
h = core.parse_history(data["history"])
hc = core.history_components(h)
bt = core.backtest_tables(hc, {"price": 60, "flow": 25, "dex": 15})
assert bt["ic"]["IC_5d"].notna().all() and bt["n"] > 300
print("backtest ran without scipy; composite IC5 =", bt["ic"].set_index("signal").loc["score", "IC_5d"], bt["stability"])
x = pd.Series([1, 2, 3, 4, 5, 6.0]); y = pd.Series([2, 1, 4, 3, 6, 5.0])
assert abs(core.spearman(x, y) - 0.8286) < 1e-3, core.spearman(x, y)      # hand value: 1 - 6*6/(6*35)... checked below
assert core.spearman(x, x) == 1.0 and core.spearman(x, -x) == -1.0
assert np.isnan(core.spearman(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.0])))

# ---- 2. trend cards on synthetic series
rng = np.random.default_rng(1)
idx = pd.bdate_range("2026-01-01", periods=150)
def mk(a, b, noise=0.05e9): return pd.Series(np.linspace(a, b, 150) + rng.normal(0, noise, 150), index=idx)
cases = {
    "neg_strengthening": (mk(-1e9, -6e9), ("Negative", "strengthening")),
    "neg_weakening": (mk(-6e9, -1e9), ("Negative", "weakening")),
    "pos_strengthening": (mk(1e9, 6e9), ("Positive", "strengthening")),
    "pos_weakening": (mk(6e9, 1e9), ("Positive", "weakening")),
}
for name, (s, want) in cases.items():
    tc = core.trend_card(s)
    assert (tc["sign"], tc["trend"]) == want, (name, tc["sign"], tc["trend"], tc["z"])
flat = pd.Series(-3e9 + rng.normal(0, 1e9, 150), index=idx)
flat.iloc[-6:] = flat.iloc[-6]            # no change over last 5 sessions
assert core.trend_card(flat)["trend"] == "steady"
assert core.trend_card(pd.Series([1.0] * 10)) is None
crossing = pd.concat([mk(-2e9, -1e8).iloc[:120], pd.Series([5e8] * 30, index=idx[120:])])
tc = core.trend_card(crossing)
assert tc["sign"] == "Positive" and tc["days_in_sign"] == 30, (tc["sign"], tc["days_in_sign"])
print("trend_card: 4 directions, steady, short-history and sign-run checks passed")

# ---- 3. OHLC parsing for the three common export formats
dates = pd.bdate_range("2026-01-02", periods=80)
px = 600 + np.cumsum(rng.normal(0, 3, 80))
o = px + rng.normal(0, 1, 80); hi = np.maximum(o, px) + 2; lo = np.minimum(o, px) - 2
yahoo = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Open": o, "High": hi, "Low": lo, "Close": px,
                      "Adj Close": px, "Volume": 1_000_000})
tv = pd.DataFrame({"time": ((dates - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)), "open": o, "high": hi, "low": lo, "close": px, "Volume": 5})
bar = pd.DataFrame({"Time": dates.strftime("%m/%d/%Y"), "Open": [f"{v:,.2f}" for v in o], "High": [f"{v:,.2f}" for v in hi],
                    "Low": [f"{v:,.2f}" for v in lo], "Last": [f"{v:,.2f}" for v in px], "Volume": "1,234"})
footer = pd.DataFrame([["Downloaded from Barchart.com as of 11-02-2026"] + [None] * 5], columns=bar.columns)
bar = pd.concat([bar, footer], ignore_index=True)
for label, df in (("yahoo", yahoo), ("tradingview", tv), ("barchart", bar)):
    buf = io.StringIO(); df.to_csv(buf, index=False)
    kind, raw = core.read_source(buf.getvalue().encode())
    assert kind == "ohlc", (label, kind)
    parsed = core.parse_ohlc(raw)
    assert len(parsed) == 80 and (parsed["high"] >= parsed["low"]).all(), (label, len(parsed))
    assert abs(parsed["close"].iloc[-1] - px[-1]) < 0.01, label
    assert parsed["date"].iloc[0] == dates[0], (label, parsed["date"].iloc[0])
print("OHLC detection and parsing: yahoo, tradingview (unix time), barchart (commas + footer) ok")

# ---- 4. candle frames and spec
mode_frame, mode = core.candle_frame(hc, None)
assert mode == "close-to-close" and (mode_frame["high"] >= mode_frame["low"]).all()
assert (mode_frame["open"].iloc[1:].values == mode_frame["close"].iloc[:-1].values).all()
real_dates = h["Date"].iloc[-80:]
ohlc_df = pd.DataFrame({"date": real_dates.values, "open": h["Close Price"].iloc[-80:].values - 1,
                        "high": h["Close Price"].iloc[-80:].values + 3, "low": h["Close Price"].iloc[-80:].values - 3,
                        "close": h["Close Price"].iloc[-80:].values})
f2, mode2 = core.candle_frame(hc, ohlc_df)
assert mode2 == "ohlc" and f2["ema200"].notna().all() and f2["hi20"].iloc[19:].notna().all()
assert f2["ema200"].iloc[-1] == hc["ema200"].iloc[-1]            # EMA taken from the longer history file
for fr in (mode_frame, f2):
    for ov in (list(core.OVERLAYS), ["200 EMA"], []):
        spec = core.candle_spec(fr, ov, days=126)
        json.dumps(spec, allow_nan=False)
        assert spec["$schema"].endswith("v5.json") and len(spec["layer"]) == 2 + sum(len(core.OVERLAYS[n][0]) for n in ov)
        rec = spec["data"]["values"]
        assert len(rec) == min(126, len(fr)) and set(rec[0]) >= {"date", "open", "high", "low", "close"}
        dom = spec["layer"][0]["encoding"]["y"]["scale"]["domain"]
        assert dom[0] < min(r["low"] for r in rec) and dom[1] > max(r["high"] for r in rec)
        assert spec["layer"][0]["encoding"]["y"]["scale"]["zero"] is False
        assert all(isinstance(r["date"], str) for r in rec)
spec = core.candle_spec(mode_frame, list(core.OVERLAYS), days=len(mode_frame))     # includes warm-up NaNs
assert any(r.get("hi55") is None for r in spec["data"]["values"]), "NaN warm-up values must become null"
print("candle_frame / candle_spec: both modes, all overlay combos, JSON-safe, y-domain excludes zero")

# ---- 5. analyse with and without an OHLC file, and the new cards on real data
opts = {"flow_contrarian": False, "flow_min_dte": 1, "t_mild": 0.1, "t_strong": 0.3, "reg_min_dte": 1, "reg_max_dte": 45,
        "flip_buffer": 0.01, "edge_threshold": 1.5, "top_n": 5,
        "structure": {"vert_dte": 18, "em_fraction": 0.75, "width": 10, "cal_front_dte": 10, "cal_gap": 30,
                      "diag_front_dte": 5, "diag_gap": 7}}
R = core.analyse(data, core.DEFAULT_WEIGHTS, opts)
assert R["candle_mode"] == "close-to-close"
d2 = dict(data); d2["ohlc"] = ohlc_df
R2 = core.analyse(d2, core.DEFAULT_WEIGHTS, opts)
assert R2["candle_mode"] == "ohlc"
g = R["regime"]
print("GEX card:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R["gex_trend"].items() if k not in ("tail",)})
print("DEX card:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R["dex_trend"].items() if k not in ("tail",)})
print("DEX window:", round(R["dex_window"] / 1e9, 2), "B | flip", round(g["flip"], 1), "range", round(g["flip_min"], 1), "-", round(g["flip_max"], 1),
      "| nearest", g["flip_nearest_date"].date(), round(g["flip_nearest"], 1), "| spot", round(R["spot"], 2))
assert g["flip_min"] <= g["flip"] <= g["flip_max"]

# ---- 6. all-expiry (GEX profile) flip
af = R["agg_flip"]
assert af is not None and abs(af["flip"] - 715.54) < 0.05 and af["up_crossing"] is True, af
assert af["state_at_spot"] == "Negative" and af["value_at_spot"] < -5e9 and af["crossings"] == [af["flip"]], af
two = pd.DataFrame({"strike": [600, 650, 700, 750, 800], "gex_profile": [-1.0, -2.0, 1.0, 3.0, -1.0]})
a2 = core.profile_flip(two, 690.0)               # crossings near 666.7 (up) and 775 (down); 666.7 is nearer
assert abs(a2["flip"] - (650 + 2 / 3 * 50)) < 1e-6 and len(a2["crossings"]) == 2, a2
a3 = core.profile_flip(pd.DataFrame({"strike": [600, 650, 700], "gex_profile": [1.0, 2.0, 3.0]}), 650.0)
assert np.isnan(a3["flip"]) and a3["state_at_spot"] == "Positive", a3
assert core.profile_flip(None, 650.0) is None
assert core.profile_flip(pd.DataFrame({"strike": [600, 650, 700], "call_gex": [1, 2, 3]}), 650.0) is None
print("profile_flip: real file (715.54, negative at spot), two crossings, none, missing column")
print("ALL FEATURE CHECKS PASSED")
