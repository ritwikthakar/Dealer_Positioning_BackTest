"""Tests for ichimoku_tab.py using a stand-in `streamlit` (the real package is not needed) and a fake yfinance."""
import json, os, sys, types
sys.modules["scipy"] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

LOG = {"errors": [], "warnings": [], "infos": [], "frames": [], "vega": [], "charts": 0, "metrics": {}, "dl": [], "texts": [], "progress": []}
OV = {}
class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return getattr(st, n)
st = types.ModuleType("streamlit")
st.session_state = {}
def _cache(*a, **k):
    if a and callable(a[0]): return a[0]
    return lambda f: f
st.cache_data = _cache
for n in ("subheader", "caption", "write", "markdown"):
    setattr(st, n, (lambda n: lambda *a, **k: LOG["texts"].append(str(a[0]) if a else ""))(n))
st.error = lambda m, *a, **k: LOG["errors"].append(str(m))
st.warning = lambda m, *a, **k: LOG["warnings"].append(str(m))
st.info = lambda m, *a, **k: LOG["infos"].append(str(m))
st.columns = lambda n, *a, **k: [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
st.expander = lambda *a, **k: Ctx()
st.radio = lambda label, options, **k: OV.get(k["key"], options[0])
st.number_input = lambda label, mn, mx, val, **k: OV.get(k["key"], val)
st.button = lambda label, **k: OV.get(k["key"], False)
st.text_area = lambda label, value="", **k: OV.get(k["key"], value)
st.multiselect = lambda label, options, default=None, **k: OV.get(k["key"], list(default or []))
st.selectbox = lambda label, options, **k: options[OV.get(k["key"], 0)]
class Bar:
    def progress(self, v, text=None): LOG["progress"].append(v)
    def empty(self): pass
st.progress = lambda v, text=None, **k: (LOG["progress"].append(v), Bar())[1]
def _metric(label, value, delta=None, **k):
    assert delta is None; LOG["metrics"][label] = value
st.metric = _metric
def _df(x, *a, **k):
    assert isinstance(x, pd.DataFrame) and len(x) > 0; LOG["frames"].append(x)
st.dataframe = _df
def _chart(x, *a, **k):
    assert len(x) > 0 and all(pd.api.types.is_numeric_dtype(t) for t in x.dtypes); LOG["charts"] += 1
st.line_chart = _chart
def _vega(spec=None, **k):
    json.dumps(spec, allow_nan=False); assert spec["data"]["values"] and len(spec["layer"]) == 7; LOG["vega"].append(spec)
st.vega_lite_chart = _vega
def _dl(label, data, **k):
    assert isinstance(data, bytes) and k["file_name"] == "ichimoku_setups.csv"; LOG["dl"].append(data)
st.download_button = _dl
sys.modules["streamlit"] = st

# fake yfinance: random walks, plus a planted reversal on ticker TURN (long downtrend, then a steady rally through the cloud)
def mk(closes, spread=0.6):
    c = np.asarray(closes, float); o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=3), periods=len(c))
    return pd.DataFrame({"Open": o, "High": np.maximum(c, o) + spread, "Low": np.minimum(c, o) - spread, "Close": c,
                         "Volume": 1e6}, index=idx)
rng = np.random.default_rng(11)
TURN = [300 - 0.3 * i for i in range(300)] + [210 + 0.5 * i for i in range(1, 26)]      # cloud break lands 5 bars before the end
def download(tickers, start=None, **k):
    parts = {}
    for t in tickers:
        if t == "NODATA": continue
        if t == "TURN":
            parts[t] = mk(TURN)
        elif t == "SHORTY":
            parts[t] = mk(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 100))))
        else:
            parts[t] = mk(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 420))))
    return pd.concat(parts, axis=1)
yf = types.ModuleType("yfinance"); yf.download = download; sys.modules["yfinance"] = yf

import donchian_screener as dsc
import ichimoku_screener as isc
import ichimoku_tab as tab

# ---- 1. nothing run yet: friendly info, no crash; ticker box inherits the Donchian tab's list when there is one
tab.render(); assert LOG["infos"] and not LOG["errors"] and not LOG["frames"]
st.session_state["scr_tickers"] = "TURN, SPY"
seen = {}
orig = st.text_area; st.text_area = lambda label, value="", **k: seen.setdefault("v", value)
OV["ich_universe"] = "My tickers"; tab.render(); assert seen["v"] == "TURN, SPY"
st.text_area = orig; st.session_state.pop("scr_tickers"); OV.clear()
print("1 empty state ok; ticker list carried over from the Donchian tab")

# ---- 2. run on a custom list
OV.update(ich_run=True, ich_universe="My tickers",
          ich_tickers="TURN, NODATA, SHORTY, " + ", ".join("R%d" % i for i in range(30)), ich_lookback=6)
tab.render()
assert not LOG["errors"], LOG["errors"]
res = st.session_state["ich_res"]
assert res.attrs["n_tickers"] == 33 and res.attrs["failed"] == ["NODATA"] and res.attrs["short"] == ["SHORTY"]
assert LOG["progress"][-1] == 1.0 and LOG["progress"][0] == 0
cb = res[(res.ticker == "TURN") & (res.setup == "cloud_break")]
assert len(cb) == 1 and cb.iloc[0]["side"] == "long" and cb.iloc[0]["bars_ago"] == 5, res[res.ticker == "TURN"]
assert LOG["frames"] and len(LOG["dl"]) == 1 and len(LOG["vega"]) == 1 and LOG["charts"] == 1
assert sum(LOG["metrics"].values()) == len(res) and set(LOG["metrics"]) == set(tab.SETUP_LABEL.values())
assert any("too little history (<150 bars): 1" in t for t in LOG["texts"]), [t for t in LOG["texts"] if "Scanned" in t]
spec = LOG["vega"][0]
assert len(spec["data"]["values"]) == 120 and spec["data"]["values"][-1]["tenkan"] is not None
print("2 run: cloud break found, NODATA and short history reported, chart + table + csv ok; signals:", len(res))

# ---- 3. chart spec for the planted reversal: projected cloud sits after the last candle, on weekdays, with a colour per point
d_ = isc.add_indicators(mk(TURN))
spec = tab.candle_spec(d_)
json.dumps(spec, allow_nan=False)
cloud = spec["layer"][0]["data"]["values"]
last_candle = pd.Timestamp(spec["data"]["values"][-1]["date"])
assert last_candle == d_.index[-1]
fut = [r for r in cloud if pd.Timestamp(r["date"]) > last_candle]
assert len(fut) == 26 and all(pd.Timestamp(r["date"]).weekday() < 5 for r in fut)
assert all(r["lo"] <= r["hi"] and r["cloud"] in ("bull", "bear") for r in cloud)
assert any(r["cloud"] == "bear" for r in cloud) and any(r["cloud"] == "bull" for r in cloud), "reversal should show both colours"
assert abs(fut[0]["sa"] - d_["sa_raw"].iloc[-26]) < 1e-9 and abs(fut[-1]["sb"] - d_["sb_raw"].iloc[-1]) < 1e-9
hist = [r for r in cloud if pd.Timestamp(r["date"]) <= last_candle]
assert abs(hist[-1]["sa"] - d_["sa"].iloc[-1]) < 1e-9 and abs(hist[-1]["sb"] - d_["sb"].iloc[-1]) < 1e-9
print("3 chart spec: 26 projected sessions, un-shifted spans, bull/bear colouring")

# ---- 4. filters and selection
LOG["frames"].clear(); LOG["vega"].clear(); OV["ich_run"] = False
OV.update(ich_f_setup=["cloud_break"], ich_f_grade=["A", "B", "C"], ich_f_side=["long"], ich_pick=0)
tab.render()
shown = LOG["frames"][0]
assert set(shown["setup"]) == {"cloud_break"} and set(shown["side"]) == {"long"} and not LOG["errors"] and len(LOG["vega"]) == 1
LOG["infos"].clear(); LOG["frames"].clear(); OV.update(ich_f_setup=[])
tab.render(); assert any("matches" in t for t in LOG["infos"]) and not LOG["frames"]
print("4 filters and empty-filter message ok")

# ---- 4b. 1-hour timeframe: the selector reaches the downloader, results and chart stay on the scanned timeframe
import re
def mk_hourly(closes, spread=0.3):
    c = np.asarray(closes, float); o = np.r_[c[0], c[:-1]]
    days = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=3), periods=len(c) // 7 + 2)
    stamps = [d + pd.Timedelta(hours=9.5 + k) for d in days for k in range(7)][-len(c):]
    idx = pd.DatetimeIndex(stamps).tz_localize("America/New_York")
    return pd.DataFrame({"Open": o, "High": np.maximum(c, o) + spread, "Low": np.minimum(c, o) - spread, "Close": c,
                         "Volume": 1e6}, index=idx)
CALLS = []
def download_h(tickers, start=None, interval="1d", **k):
    CALLS.append((list(tickers), start, interval))
    if interval != "1h":
        return download(tickers, start, **k)
    return pd.concat({t: mk_hourly(TURN if t == "TURN" else 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.004, 450))))
                      for t in tickers if t != "NODATA"}, axis=1)
yf.download = download_h
OV.clear(); st.session_state.clear()
for k_ in ("errors", "infos", "frames", "vega", "texts", "dl", "progress"): LOG[k_].clear()
OV.update(ich_run=True, ich_universe="My tickers", ich_tickers="TURN, NODATA, " + ", ".join("H%d" % i for i in range(10)),
          ich_tf=1, ich_lookback=6)
tab.render()
assert not LOG["errors"], LOG["errors"]
res = st.session_state["ich_res"]
assert res.attrs["interval"] == "1h" and res.attrs["failed"] == ["NODATA"]
assert CALLS[0][2] == "1h" and CALLS[0][1] == dsc.default_start("1h"), CALLS[0]
cb = res[(res.ticker == "TURN") & (res.setup == "cloud_break")]
assert len(cb) == 1 and cb.iloc[0]["side"] == "long" and cb.iloc[0]["bars_ago"] == 5, res[res.ticker == "TURN"]
assert all(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", s) for s in res["date"]), res["date"].head()
assert CALLS[-1][2] == "1h" and len(CALLS[-1][0]) == 1, CALLS[-1]          # the drill-down chart asked for hourly bars too
assert any(t.startswith("1 hour bars |") for t in LOG["texts"]), [t for t in LOG["texts"] if "Scanned" in t or "scanned" in t]
spec = LOG["vega"][-1]
xe = spec["layer"][3]["encoding"]["x"]
assert xe["type"] == "ordinal" and len(spec["data"]["values"]) == 200 and len(xe["scale"]["domain"]) == 226
dom = xe["scale"]["domain"]
assert dom == sorted(dom, key=lambda s: dom.index(s)) and len(set(dom)) == 226
assert [r["date"] for r in spec["data"]["values"]] == dom[:200]
cl = spec["layer"][0]["data"]["values"]
assert all(r["date"] in dom for r in cl) and sum(1 for r in cl if r["date"] in dom[200:]) == 26
yr, prev, fut_ts = mk_hourly(TURN).index[-1].year, 0, []          # labels carry no year: rebuild it, rolling over in January
for s in dom[200:]:
    mo = int(s[:2]); yr += mo < prev; prev = mo
    fut_ts.append(pd.Timestamp(f"{yr}-{s}"))
assert all(t.weekday() < 5 and 9 <= t.hour <= 15 and t.minute == 30 for t in fut_ts) and fut_ts == sorted(fut_ts)
# changing the selector without pressing Run must not change what the existing results/chart use
n_calls = len(CALLS); OV.update(ich_run=False, ich_tf=0); LOG["vega"].clear()
tab.render()
assert CALLS[n_calls:] and all(c[2] == "1h" for c in CALLS[n_calls:]) and len(LOG["vega"]) == 1
# daily still works from the same selector
OV.update(ich_run=True, ich_tf=0); LOG["vega"].clear(); tab.render()
assert st.session_state["ich_res"].attrs["interval"] == "1d" and CALLS[-1][2] == "1d" and not LOG["errors"], CALLS[-1]
assert LOG["vega"][-1]["layer"][3]["encoding"]["x"]["type"] == "temporal"
yf.download = download
print("4b 1-hour: interval reaches the downloader, hourly stamps, ordinal chart with 26 projected session bars, daily unaffected")

# ---- 4c. Liquid list universe (the Gamma screener's list), first option and default
import gamma_screener as gs
assert tab.dtab.UNIVERSES == ["Liquid list", "S&P 500", "My tickers"]
REQ = []
def download_rec(tickers, start=None, **k):
    REQ.extend(tickers); return download(tickers, start, **k)
yf.download = download_rec
OV.clear(); st.session_state.clear()
for k_ in ("errors", "infos", "frames", "vega", "texts", "dl", "progress"): LOG[k_].clear()
OV.update(ich_run=True)                                                        # no universe override: radio default
tab.render()
assert not LOG["errors"], LOG["errors"]
assert sorted(set(REQ)) == sorted(gs.LIQUID) and st.session_state["ich_res"].attrs["n_tickers"] == len(gs.LIQUID), len(REQ)
# a missing gamma_screener module is reported, not a crash
saved = sys.modules.pop("gamma_screener"); sys.modules["gamma_screener"] = None
st.session_state.clear()                                                        # no stale results, so no drill-down download either
OV.update(ich_universe="Liquid list"); LOG["errors"].clear(); REQ.clear(); tab.render()
assert any("Liquid list universe" in e for e in LOG["errors"]) and not REQ, LOG["errors"]
sys.modules["gamma_screener"] = saved; yf.download = download
print("4c liquid list: first/default option, uses gamma_screener.LIQUID, missing module reported")

# ---- 5. empty result, S&P path failure, missing package
OV.clear(); st.session_state.clear(); LOG["errors"].clear(); LOG["infos"].clear()
OV.update(ich_run=True, ich_universe="My tickers", ich_tickers="NODATA")
tab.render(); assert st.session_state["ich_res"].empty and any("No setups" in t for t in LOG["infos"]), LOG
OV.update(ich_universe="S&P 500"); LOG["errors"].clear()
dsc.get_sp500_tickers = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
tab.render(); assert any("S&P 500 universe" in e for e in LOG["errors"]), LOG["errors"]
OV.update(ich_universe="My tickers", ich_tickers="R1, R2, R3"); LOG["errors"].clear()
del sys.modules["yfinance"]
real_import = __import__
def blocked(name, *a, **k):
    if name == "yfinance": raise ImportError("No module named 'yfinance'")
    return real_import(name, *a, **k)
import builtins; builtins.__import__ = blocked
try:
    tab.render()
finally:
    builtins.__import__ = real_import
assert any("Missing package" in e for e in LOG["errors"]), LOG["errors"]
print("5 empty result, offline S&P list, missing yfinance reported cleanly")
print("ALL ICHIMOKU TAB CHECKS PASSED")
