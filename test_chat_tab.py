"""Tests for chat_tab.py using a stand-in `streamlit` and a fake Anthropic client (neither real package is needed).
Run:  python3 test_chat_tab.py"""
import os, sys, types
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

LOG = {"errors": [], "infos": [], "texts": [], "code": [], "chat": []}
OV = {}
class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return getattr(st, n)
st = types.ModuleType("streamlit")
st.session_state = {}
st.secrets = {}
for n in ("subheader", "caption", "write", "markdown"):
    setattr(st, n, lambda *a, **k: LOG["texts"].append(str(a[0]) if a else ""))
st.error = lambda m, *a, **k: LOG["errors"].append(str(m))
st.info = lambda m, *a, **k: LOG["infos"].append(str(m))
st.code = lambda m, *a, **k: LOG["code"].append(str(m))
st.columns = lambda spec, *a, **k: [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
st.expander = lambda *a, **k: Ctx()
st.chat_message = lambda role: Ctx()
st.text_input = lambda label, **k: OV.get(k["key"], "")
st.button = lambda label, **k: OV.get(k["key"], False)
st.chat_input = lambda label, **k: OV.get(k["key"])
st.write_stream = lambda gen: "".join(gen)
sys.modules["streamlit"] = st

CALLS = []
class FakeStream:
    def __init__(self, kw): self.kw = kw
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def text_stream(self): return iter(["Hello ", "there."])
class FakeClient:
    def __init__(self, api_key): self.api_key = api_key; self.messages = self
    def stream(self, **kw): CALLS.append(kw); return FakeStream(kw)
anth = types.ModuleType("anthropic"); anth.Anthropic = FakeClient
sys.modules["anthropic"] = anth

import chat_tab as ct

fails = 0
def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1

# ---- a minimal stand-in for core.analyse() output
leg = NS(qty=1, exp=pd.Timestamp("2026-12-04"), cp="P", strike=600.0, bid=5.1, ask=5.3, iv=0.21)
leg2 = NS(qty=-1, exp=pd.Timestamp("2026-11-27"), cp="P", strike=590.0, bid=3.0, ask=3.2, iv=0.23)
strat = NS(name="Put calendar", eligible=True, source="matrix", legs=[leg, leg2], debit_nat=1.2, max_profit=300.0,
           max_loss=120.0, model_based=True, breakevens=[585.5, 596.0], pop=0.41, reason="", warnings=["Flat IV assumed"])
blocked = NS(name="Iron condor", eligible=False, source="adjacent", legs=[leg], debit_nat=-2.0, max_profit=200.0,
             max_loss=800.0, model_based=False, breakevens=[], pop=0.6, reason="Front-back IV edge below threshold", warnings=[])
R = {"asof": pd.Timestamp("2026-11-16"), "spot": 598.4,
     "direction": {"band": "Mild bullish", "score": 0.14, "used": {"price": 0.6, "flow": 0.25, "dex": 0.15}, "agreement": 0.67},
     "components": {"price": 0.3, "flow": -0.1, "dex": float("nan")}, "flow_source": "history put/call proxy",
     "regime": {"effective": "Negative", "near_flip": True, "net": -1.2e9, "source": "expiration file", "flip": 597.0,
                "flip_min": 590.0, "flip_max": 604.0},
     "agg_flip": {"flip": 600.0, "value_at_spot": -2.0e8, "state_at_spot": "negative"},
     "walls": {"put_wall": 590.0, "call_wall": 610.0},
     "gex_trend": {"sign": "Negative", "trend": "strengthening", "value": -1.2e9, "short": 5, "long": 20, "d_short": -3e8,
                   "d_long": -5e8, "pctile": 0.12, "days_in_sign": 4},
     "dex_trend": None,
     "edge": {"front_iv": 0.22, "back_iv": 0.20, "front_dte": 10, "back_dte": 45, "edge_pts": 2.0}, "ivrank": 31.7,
     "flow": {"aggr": 0.12, "unusual": float("nan"), "mid_share": 0.55},
     "oi": None, "oi_walls": [], "stay_out": False, "strategies": [strat, blocked]}

ctx = ct.build_context(R)
check("context has direction band", "Mild bullish" in ctx and "+0.14" in ctx)
check("nan component rendered as n/a, not 'nan'", "dex: score n/a" in ctx and "nan" not in ctx.lower().replace("unavailable", ""))
check("near-flip noted", "inside near-flip buffer" in ctx)
check("flip side computed (spot below flip)", "spot is below it" in ctx)
check("walls present", "Put wall 590" in ctx and "call wall 610" in ctx)
check("strategy legs and numbers present", "Put calendar" in ctx and "Buy 1 2026-12-04" in ctx and "max loss 120" in ctx)
check("blocked strategy shows reason", "NOT ELIGIBLE" in ctx and "edge below threshold" in ctx)
check("missing OI handled", "OI changes are unavailable" in ctx)
check("None dex_trend skipped silently", "DEX trend" not in ctx)

bad = dict(R); bad["edge"] = {}
ctx_bad = ct.build_context(bad)
check("one broken section does not break the rest", "(unavailable: KeyError)" in ctx_bad and "Put calendar" in ctx_bad)

ctx_none = ct.build_context(None)
check("no-data context says files not loaded", "not loaded yet" in ctx_none and "Ranked strategies" not in ctx_none)

scr = pd.DataFrame({"ticker": ["AAPL", "NVDA"], "setup": ["breakout", "pullback"], "side": ["long", "long"], "grade": ["A", "B"],
                    "close": [230.123, 140.5], "extra_col": [1, 2]})
ctx_s = ct.build_context(None, scr_res=scr, scr_when="2026-11-17 09:00")
check("screener rows included, extra columns dropped", "AAPL" in ctx_s and "230.12" in ctx_s and "extra_col" not in ctx_s)
check("empty screener result noted", "no setups" in ct.build_context(None, scr_res=scr.iloc[0:0], scr_when="x"))
big = pd.concat([scr] * 30, ignore_index=True)
check("long screener result is truncated", "showing first 40 of 60" in ct.build_context(None, scr_res=big))

# ---- history trimming
m = [{"role": "assistant" if i % 2 == 0 else "user", "content": str(i)} for i in range(30)]
t = ct._trim(m)
check("trim keeps <= 20 and starts on user", len(t) <= 20 and t[0]["role"] == "user")

# ---- render flows
st.session_state.clear(); LOG.update({k: [] for k in LOG})
ct.render(R)
check("no key -> asks for key, no crash", any("API key" in s for s in LOG["infos"]))

st.session_state.clear(); LOG.update({k: [] for k in LOG}); CALLS.clear()
os.environ["ANTHROPIC_API_KEY"] = "sk-test"
OV["chat_input"] = "Why is the first strategy ranked first?"
ct.render(R)
check("one API call made", len(CALLS) == 1)
kw = CALLS[0] if CALLS else {}
check("model default used", kw.get("model") == "claude-sonnet-5-5")
check("system prompt carries app state", "Put calendar" in kw.get("system", "") and "APP STATE" in kw.get("system", ""))
check("first message is the user's question", kw.get("messages", [{}])[0] == {"role": "user", "content": OV["chat_input"]})
check("reply stored in history", st.session_state["chat_msgs"][-1] == {"role": "assistant", "content": "Hello there."})

# second turn appends and alternates
OV["chat_input"] = "And the second one?"
ct.render(R)
roles = [x["role"] for x in CALLS[-1]["messages"]]
check("second call sends alternating history", roles == ["user", "assistant", "user"])

# API failure drops the unanswered question
class Boom(FakeClient):
    def stream(self, **kw): raise RuntimeError("rate limited")
anth.Anthropic = Boom
n_before = len(st.session_state["chat_msgs"])
OV["chat_input"] = "third"
ct.render(R)
check("failure shown and history unchanged", any("rate limited" in e for e in LOG["errors"]) and len(st.session_state["chat_msgs"]) == n_before)

# clear button
anth.Anthropic = FakeClient
OV["chat_input"] = None; OV["chat_clear"] = True
ct.render(R)
check("clear chat empties history", st.session_state["chat_msgs"] == [])

print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
