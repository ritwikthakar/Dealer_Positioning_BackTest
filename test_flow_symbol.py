"""Flow / unusual files must be matched to QQQ even when Symbol has odd case, spaces or a '$'.  Run: python3 test_flow_symbol.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import core

fails = 0
def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1

def flow(sym):
    return pd.DataFrame({"Symbol": [sym, sym, "SPY"], "Type": ["Call", "Put", "Call"], "Side": ["ask", "ask", "ask"],
                         "DTE": [5, 5, 5], "Premium": [100000.0, 40000.0, 999999.0], "Bid x Size": ["1x2"] * 3})

def unusual(sym):
    return pd.DataFrame({"Symbol": [sym, sym], "Type": ["Call", "Put"], "DTE": [10, 10], "Volume": ["1,000", "500"],
                         "Latest": [2.0, 2.0], "Vol/OI": [3, 3], "Moneyness": ["1%", "1%"]})

for sym in ["QQQ", "qqq", " QQQ ", "$QQQ", "Qqq\t"]:
    r = core.flow_live(flow(sym), unusual(sym))
    check(f"symbol {sym!r}: flow and unusual both score", r["aggr"] == (100000 - 40000) / 140000 and r["unusual"] > 0)

check("SPY rows are still excluded", len(core.underlying_rows(flow("QQQ"))) == 2)
check("no QQQ rows -> empty, flow n/a (no crash)", core.underlying_rows(flow("SPY")).shape[0] == 0
      and core.flow_live(flow("IWM"), unusual("IWM"))["aggr"] != core.flow_live(flow("IWM"), unusual("IWM"))["aggr"])
check("missing Symbol column is returned unchanged", len(core.underlying_rows(pd.DataFrame({"a": [1, 2]}))) == 2)
check("core version stamp matches app.py", f'EXPECTED_CORE = "{core.CORE_VERSION}"' in open("app.py", encoding="utf-8").read())
print("\nALL PASSED" if not fails else f"\n{fails} FAILED"); sys.exit(1 if fails else 0)
