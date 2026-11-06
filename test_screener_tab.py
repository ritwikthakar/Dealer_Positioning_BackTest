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

# ---- 4. empty result, S&P path failure, missing package, drill-down failure
OV.clear(); st.session_state.clear(); LOG["errors"].clear(); LOG["infos"].clear()
OV.update(scr_run=True, scr_universe="My tickers", scr_tickers="NODATA")
tab.render(); assert st.session_state["scr_res"].empty and any("No setups" in t for t in LOG["infos"]), LOG
OV.update(scr_universe="S&P 500"); LOG["errors"].clear()
sc.get_sp500_tickers = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
tab.render(); assert any("S&P 500 list" in e for e in LOG["errors"]), LOG["errors"]
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
