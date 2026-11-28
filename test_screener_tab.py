"""Tests for screener_tab.py using a stand-in `streamlit` (the real package is not needed) and a fake yfinance."""
import json, os, sys, types
sys.modules["scipy"] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

LOG = {"errors": [], "warnings": [], "infos": [], "frames": [], "vega": [], "charts": 0, "metrics": {}, "dl": 0, "texts": [], "progress": []}
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
    json.dumps(spec, allow_nan=False); assert spec["data"]["values"] and len(spec["layer"]) == 5; LOG["vega"].append(spec)
st.vega_lite_chart = _vega
def _dl(label, data, **k):
    assert isinstance(data, bytes) and k["file_name"].endswith(".csv"); LOG["dl"] += 1
st.download_button = _dl
sys.modules["streamlit"] = st

# fake yfinance: random walks, with a planted failed breakout on ticker FAIL
def mk(closes, spread=0.6):
    c = np.asarray(closes, float); o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=3), periods=len(c))
    return pd.DataFrame({"Open": o, "High": np.maximum(c, o) + spread, "Low": np.minimum(c, o) - spread, "Close": c,
                         "Volume": 1e6}, index=idx)
rng = np.random.default_rng(11)
def download(tickers, start=None, **k):
    parts = {}
    for t in tickers:
        if t == "NODATA": continue
        if t == "FAIL":
            base = [100 + 0.2 * (i % 3) for i in range(400)]
            parts[t] = mk(base + [104, 99.0])
        else:
            parts[t] = mk(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 420))))
    return pd.concat(parts, axis=1)
yf = types.ModuleType("yfinance"); yf.download = download; sys.modules["yfinance"] = yf

import donchian_screener as sc
import screener_tab as tab

# ---- parse_tickers
assert tab.parse_tickers("qqq, spy  AAPL;brk.b\nqqq bad$ticker") == ["QQQ", "SPY", "AAPL", "BRK-B"]
assert tab.parse_tickers("") == [] and tab.parse_tickers(None) == []

# ---- 1. nothing run yet: friendly info, no crash
tab.render(); assert LOG["infos"] and not LOG["errors"] and not LOG["frames"]

# ---- 2. run on a custom list
OV.update(scr_run=True, scr_universe="My tickers", scr_tickers="FAIL, NODATA, " + ", ".join("R%d" % i for i in range(30)),
          scr_lookback=3)
tab.render()
assert not LOG["errors"], LOG["errors"]
res = st.session_state["scr_res"]
assert res.attrs["n_tickers"] == 32 and res.attrs["failed"] == ["NODATA"] and res.attrs["short"] == []
assert LOG["progress"][-1] == 1.0 and LOG["progress"][0] == 0
fb = res[(res.ticker == "FAIL") & (res.setup == "failed_break")]
assert len(fb) == 1 and fb.iloc[0]["side"] == "short" and fb.iloc[0]["grade"] == "A", res[res.ticker == "FAIL"]
assert LOG["frames"] and LOG["dl"] == 1 and len(LOG["vega"]) == 1 and LOG["charts"] == 1
assert sum(LOG["metrics"].values()) == len(res)
spec = LOG["vega"][0]
assert len(spec["data"]["values"]) == 120 and spec["data"]["values"][-1]["close"] is not None
print("2 run: failed breakout found, NODATA reported, chart + table + csv ok; signals:", len(res))

# ---- 3. filters and selection
LOG["frames"].clear(); LOG["vega"].clear(); OV["scr_run"] = False
OV.update(scr_f_setup=["failed_break"], scr_f_grade=["A"], scr_f_side=["short"], scr_pick=0)
tab.render()
shown = LOG["frames"][0]
assert set(shown["setup"]) == {"failed_break"} and set(shown["grade"]) == {"A"} and set(shown["side"]) == {"short"}
assert not LOG["errors"] and len(LOG["vega"]) == 1
LOG["infos"].clear(); LOG["frames"].clear(); OV.update(scr_f_setup=[])
tab.render(); assert any("matches" in t for t in LOG["infos"]) and not LOG["frames"]
print("3 filters and empty-filter message ok")

# ---- 3b. 1-hour timeframe: the selector reaches the downloader, results and chart stay on the scanned timeframe
import re
def mk_hourly(closes, spread=0.6):
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
    return pd.concat({t: mk_hourly([100 + 0.2 * (i % 3) for i in range(400)] + [104, 99.0] if t == "FAIL"
                                   else 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.004, 420))))
                      for t in tickers if t != "NODATA"}, axis=1)
yf.download = download_h
OV.clear(); st.session_state.clear()
for k_ in ("errors", "infos", "frames", "vega", "texts", "progress"): LOG[k_].clear()
OV.update(scr_run=True, scr_universe="My tickers", scr_tickers="FAIL, NODATA, " + ", ".join("H%d" % i for i in range(10)),
          scr_tf=1, scr_lookback=3)
tab.render()
assert not LOG["errors"], LOG["errors"]
res = st.session_state["scr_res"]
assert res.attrs["interval"] == "1h" and res.attrs["failed"] == ["NODATA"]
assert CALLS[0][2] == "1h" and CALLS[0][1] == sc.default_start("1h"), CALLS[0]
fb = res[(res.ticker == "FAIL") & (res.setup == "failed_break")]
assert len(fb) == 1 and fb.iloc[0]["side"] == "short", res[res.ticker == "FAIL"]
assert all(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", s) for s in res["date"]), res["date"].head()
assert CALLS[-1][2] == "1h" and len(CALLS[-1][0]) == 1, CALLS[-1]          # the drill-down chart asked for hourly bars too
assert any(t.startswith("1 hour bars |") for t in LOG["texts"]), [t for t in LOG["texts"] if "canned" in t]
spec = LOG["vega"][-1]
xe = spec["layer"][0]["encoding"]["x"]
assert xe["type"] == "ordinal" and len(spec["data"]["values"]) == 200 and len(set(xe["scale"]["domain"])) == 200
assert [r["date"] for r in spec["data"]["values"]] == xe["scale"]["domain"]
n_calls = len(CALLS); OV.update(scr_run=False, scr_tf=0); LOG["vega"].clear()          # selector change alone must not change the chart
tab.render()
assert CALLS[n_calls:] and all(c[2] == "1h" for c in CALLS[n_calls:]) and len(LOG["vega"]) == 1
OV.update(scr_run=True, scr_tf=0); LOG["vega"].clear(); tab.render()
assert st.session_state["scr_res"].attrs["interval"] == "1d" and CALLS[-1][2] == "1d" and not LOG["errors"], CALLS[-1]
assert LOG["vega"][-1]["layer"][0]["encoding"]["x"]["type"] == "temporal"
yf.download = download
print("3b 1-hour: interval reaches the downloader, hourly stamps, ordinal chart, daily unaffected")

# ---- 3c. Liquid list universe (the Gamma screener's list), first option and default
import gamma_screener as gs
assert tab.UNIVERSES == ["Liquid list", "S&P 500", "My tickers"] and len(set(gs.LIQUID)) == len(gs.LIQUID) > 50
REQ = []
def download_rec(tickers, start=None, **k):
    REQ.extend(tickers); return download(tickers, start, **k)
yf.download = download_rec
OV.clear(); st.session_state.clear()
for k_ in ("errors", "infos", "frames", "vega", "texts", "progress"): LOG[k_].clear()
OV.update(scr_run=True)                                                        # no universe override: radio default
tab.render()
assert not LOG["errors"], LOG["errors"]
assert sorted(set(REQ)) == sorted(gs.LIQUID) and st.session_state["scr_res"].attrs["n_tickers"] == len(gs.LIQUID), len(REQ)
assert tab.universe_tickers("Liquid list") == list(gs.LIQUID) and tab.universe_tickers("Liquid list") is not gs.LIQUID
assert tab.universe_tickers("My tickers", "aapl msft") == ["AAPL", "MSFT"]
# a missing gamma_screener module is reported, not a crash
saved = sys.modules.pop("gamma_screener"); sys.modules["gamma_screener"] = None
st.session_state.clear()                                                        # no stale results, so no drill-down download either
OV.update(scr_universe="Liquid list"); LOG["errors"].clear(); REQ.clear(); tab.render()
assert any("Liquid list universe" in e for e in LOG["errors"]) and not REQ, LOG["errors"]
sys.modules["gamma_screener"] = saved; yf.download = download
print("3c liquid list: first/default option, uses gamma_screener.LIQUID, missing module reported")

# ---- 4. empty result, S&P path failure, missing package, drill-down failure
OV.clear(); st.session_state.clear(); LOG["errors"].clear(); LOG["infos"].clear()
OV.update(scr_run=True, scr_universe="My tickers", scr_tickers="NODATA")
tab.render(); assert st.session_state["scr_res"].empty and any("No setups" in t for t in LOG["infos"]), LOG
OV.update(scr_universe="S&P 500"); LOG["errors"].clear()
sc.get_sp500_tickers = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
tab.render(); assert any("S&P 500 universe" in e for e in LOG["errors"]), LOG["errors"]
OV.update(scr_universe="My tickers", scr_tickers="R1, R2, R3"); LOG["errors"].clear()
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
print("4 empty result, offline S&P list, missing yfinance reported cleanly")
print("ALL SCREENER TAB CHECKS PASSED")
