"""Smoke/regression test for core.py. Usage: python3 -I test_core.py <folder_with_the_8_csvs>"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import core  # noqa: E402

folder = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
data = core.load_folder(folder)
print("files detected:", sorted(data))
assert set(data) == {"flow", "unusual", "chain", "history", "dex_exp", "gex_exp", "gex_strike", "exp_move"}, "all 8 files should be detected"

opts = {"flow_contrarian": False, "flow_min_dte": 1, "t_mild": 0.10, "t_strong": 0.30, "reg_min_dte": 1,
        "reg_max_dte": 45, "flip_buffer": 0.01, "edge_threshold": 1.5, "top_n": 5,
        "structure": {"vert_dte": 18, "em_fraction": 0.75, "width": 10, "cal_front_dte": 10, "cal_gap": 30,
                      "diag_front_dte": 5, "diag_gap": 7}}
r = core.analyse(data, core.DEFAULT_WEIGHTS, opts)

print(f"as-of {r['asof'].date()}  spot {r['spot']:.2f}")
print("components", {k: round(v, 3) for k, v in r["components"].items()})
d = r["direction"]
print("score", round(d["score"], 3), d["band"], "agreement", round(d["agreement"], 2), "weights", {k: round(v, 2) for k, v in d["used"].items()})
g = r["regime"]
print("regime", g["state"], "effective", g["effective"], "net GEX", f"{g['net']/1e6:.0f}M", "flip", round(g["flip"], 1), "near", g["near_flip"])
print("IV rank", r["ivrank"], " term edge", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["edge"].items() if k.endswith(("pts", "iv", "dte"))})
print("walls", r["walls"]["put_wall"], r["walls"]["call_wall"])
print("flow", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["flow"].items()}, r["flow_source"])
print("stay_out", r["stay_out"])
for s in r["strategies"]:
    legs = " | ".join(f"{'+' if l.qty > 0 else '-'}{abs(l.qty)} {str(l.exp.date())[5:]} {l.strike:g}{l.cp}" for l in s.legs)
    print(f"- {s.name:28s} {legs:62s} debit_nat {s.debit_nat:6.2f} maxP {s.max_profit:8.0f} maxL {s.max_loss:8.0f} "
          f"BE {s.breakevens} POP {s.pop:.0%} elig {s.eligible} [{s.source}] {s.reason}")

# ---- regression checks against the earlier hand analysis
assert 685 < r["spot"] < 687, r["spot"]
assert d["band"] in ("Bearish", "Mild Bearish"), d["band"]
assert g["state"] == "Negative"
assert r["walls"]["put_wall"] in (680.0, 675.0, 670.0, 685.0) and r["walls"]["call_wall"] in (700.0, 725.0, 720.0, 710.0)
assert len(r["strategies"]) >= 3
for s in r["strategies"]:
    assert np.isfinite(s.debit_nat) and np.isfinite(s.max_loss) and 0 <= s.pop <= 1
    assert s.max_loss <= 0 + 1e-6 and s.max_profit >= 0 - 1e-6, (s.name, s.max_profit, s.max_loss)

# builder spot checks: bear put spread should reproduce the earlier 685/660 11/20 structure at vert_dte=18, em_fraction=0.75
b = r["builder"]
s = b.bear_put_debit()
print("bear put debit:", [(l.strike, str(l.exp.date())) for l in s.legs], round(s.debit_nat, 2))
assert [l.strike for l in s.legs] == [685.0, 660.0] and str(s.legs[0].exp.date()) == "2026-11-20"
assert abs(s.debit_nat - 7.94) < 0.05, s.debit_nat

# backtest tables
bt = core.backtest_tables(r["hist"], {"price": 60, "flow": 25, "dex": 15})
print(bt["ic"].to_string(index=False)); print(bt["bands"].to_string(index=False)); print(bt["regime"]); print(bt["stability"], bt["n"])

# weights renormalise when components are missing
x = core.direction_score({"price": 0.5, "flow": np.nan, "dex": -0.2}, core.DEFAULT_WEIGHTS)
assert abs(sum(x["used"].values()) - 1) < 1e-9 and "flow" not in x["used"]

# partial data: without optional files it should still run
mini = {k: data[k] for k in ("history", "chain")}
r2 = core.analyse(mini, core.DEFAULT_WEIGHTS, opts)
print("minimal-files run ->", r2["direction"]["band"], r2["regime"]["state"], r2["regime"]["source"], r2["flow_source"], len(r2["strategies"]), "strategies")
print("ALL CHECKS PASSED")
