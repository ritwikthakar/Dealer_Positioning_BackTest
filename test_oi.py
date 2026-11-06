"""Tests for the open-interest-change features. Usage: python3 test_oi.py <folder_with_the_8_csvs>
scipy is blocked on purpose: the app must run without it."""
import sys

sys.modules["scipy"] = None
sys.modules["scipy.stats"] = None
sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import core  # noqa: E402

folder = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
data = core.load_folder(folder)
assert set(data) == {"flow", "unusual", "chain", "history", "dex_exp", "gex_exp", "gex_strike", "exp_move"}, sorted(data)

# ---- 1. parser
s = pd.Series(["+9 (+40.91%)", "-17 (-100.00%)", "+1,223 (+79.10%)", "\u2014", "", "+5", "-2,345 (-3.1%)", "n/a"])
chg, pct = core.parse_oi_chg(s)
assert chg.tolist()[:3] == [9.0, -17.0, 1223.0] and chg.iloc[5] == 5.0 and chg.iloc[6] == -2345.0
assert chg.isna().tolist() == [False, False, False, True, True, False, False, True], chg.tolist()
assert abs(pct.iloc[0] - 40.91) < 1e-9 and pct.iloc[1] == -100.0 and abs(pct.iloc[6] + 3.1) < 1e-9 and np.isnan(pct.iloc[5])
raw = data["chain"]
ch = core.parse_chain(raw)
known = ch["oi_chg_n"].notna()
prior = ch["open_interest"] - ch["oi_chg_n"]
assert (prior[known] >= 0).all(), "OI minus change must never be negative if the change is vs the prior day"
m = known & (prior > 0)
err = (ch.loc[m, "oi_chg_pct"] - 100 * ch.loc[m, "oi_chg_n"] / prior[m]).abs()
assert err.median() < 0.01 and err.quantile(0.99) < 0.05, (err.median(), err.quantile(0.99))
print(f"parser ok: {known.mean():.0%} of contracts have a change value; pct matches change/prior")

# ---- 2. tags
t = lambda chg, prior: core.oi_tag(chg, prior)
assert t(1000, 4000) == "strong build" and t(1000, 8000) == "build" and t(1000, 20000) == "flat"
assert t(-1000, 4000) == "strong erode" and t(-1000, 8000) == "erode" and t(100, 100) == "flat"      # < 500 contracts: flat
assert t(2000, 0) == "strong build" and t(np.nan, 5) == "n/a" and t(5, None) == "n/a"
assert [core.oi_arrow(x) for x in ("strong build", "flat", "erode", "n/a")] == ["\u25b2\u25b2", "\u2013", "\u25bc", "?"]
print("tags ok")

# ---- 3. synthetic window: sums, signs, unknown handling
exp1, exp2 = pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-20")
asof = pd.Timestamp("2026-11-02")
rows = [  # exp, cp, strike, oi, chg(nan=unknown), gamma
    (exp1, "C", 700.0, 10000, 5000, 0.01), (exp1, "P", 700.0, 4000, -1000, 0.01),
    (exp2, "C", 700.0, 5000, np.nan, 0.01), (exp2, "P", 700.0, 3000, np.nan, 0.01),   # all unknown at this expiry
    (exp1, "P", 680.0, 20000, 200, 0.02), (exp2, "P", 680.0, 10000, 6000, 0.02),
    (exp1, "C", 900.0, 100, 50, 0.01),                                                # outside +/-8% band
]
syn = pd.DataFrame(rows, columns=["exp", "cp", "strike", "open_interest", "oi_chg_n", "gamma"])
oi = core.oi_change_tables(syn, 690.0, asof, 1, 45, {"min_contracts": 500})
bs = oi["by_strike"].set_index("strike")
assert 900.0 not in bs.index and set(bs.index) == {680.0, 700.0}                       # +/-8% band drops 900
# 700 calls: expiry-1 known (OI 10000, +5000), expiry-2 unknown (OI 5000)
assert bs.loc[700, "call_chg"] == 5000 and bs.loc[700, "call_oi"] == 15000
assert bs.loc[700, "call_prior"] == 5000                                                 # unknown row excluded from prior
assert abs(bs.loc[700, "call_pct"] - 1.0) < 1e-12                                        # 5000/5000, not diluted by unknowns
assert bs.loc[700, "call_tag"] == "strong build"                                         # 67% of OI known >= 50%
# 700 puts: known -1000 on prior 5000 = -20% -> erode (not strong); 57% of OI known
assert bs.loc[700, "put_oi"] == 7000 and bs.loc[700, "put_chg"] == -1000 and bs.loc[700, "put_prior"] == 5000
assert abs(bs.loc[700, "put_pct"] + 0.2) < 1e-12 and bs.loc[700, "put_tag"] == "erode"
# 680 puts: +6200 on prior 19800+4000 = 23800 -> +26.05% -> strong build
assert bs.loc[680, "put_chg"] == 6200 and bs.loc[680, "put_prior"] == 23800
assert abs(bs.loc[680, "put_pct"] - 6200 / 23800) < 1e-12 and bs.loc[680, "put_tag"] == "strong build"
# d_gex: exact formula, calls positive when added, puts negative when added, removing puts is positive
S2 = 690.0 * 690.0
assert abs(bs.loc[700, "call_dgex"] - 0.01 * 5000 * 100 * S2 * 0.01) < 1e-3
assert abs(bs.loc[700, "put_dgex"] - (-1) * 0.01 * (-1000) * 100 * S2 * 0.01) < 1e-3 and bs.loc[700, "put_dgex"] > 0
assert abs(bs.loc[680, "put_dgex"] - (-1) * 0.02 * 6200 * 100 * S2 * 0.01) < 1e-3 and bs.loc[680, "put_dgex"] < 0
be = oi["by_exp"].set_index("exp")
assert be.loc[exp1, "dte"] == 11 and be.loc[exp2, "dte"] == 18
assert be["call_chg"].sum() == bs["call_chg"].sum() and be["put_chg"].sum() == bs["put_chg"].sum()
assert abs(be["d_gex"].sum() - bs["d_gex"].sum()) < 1e-3
sm = oi["summary"]
assert sm["call_adds"] == 5000 and sm["put_adds"] == 6200 and sm["put_cuts"] == 1000 and sm["call_cuts"] == 0
assert abs(sm["put_add_share"] - 6200 / 11200) < 1e-12
assert abs(sm["known_share"] - 44000 / 52000) < 1e-12 and sm["n_contracts"] == 6
assert oi["top"].iloc[0]["chg"] == 6000 and len(oi["top"]) == 4                         # unknown rows are not listed
assert (oi["top"]["chg"].abs().diff().dropna() <= 0).all()
# a strike whose open interest is mostly unknown is n/a, never guessed
syn2 = syn.copy(); syn2.loc[0, "oi_chg_n"] = np.nan                                      # 700 calls: no known change at all
b2 = core.oi_change_tables(syn2, 690.0, asof, 1, 45, {"min_contracts": 500})["by_strike"].set_index("strike")
assert b2.loc[700, "call_tag"] == "n/a", b2.loc[700, "call_tag"]
assert core.oi_change_tables(syn.assign(oi_chg_n=np.nan), 690.0, asof) is None
assert core.oi_change_tables(syn.drop(columns="oi_chg_n"), 690.0, asof) is None
print("synthetic window: sums, unknown handling, band filter, signs, d_gex formula, top ordering ok")

# ---- 4. real data: reproduces the earlier hand analysis, conservation, GEX unit convention
opts = {"flow_contrarian": False, "flow_min_dte": 1, "t_mild": 0.1, "t_strong": 0.3, "reg_min_dte": 1, "reg_max_dte": 45,
        "flip_buffer": 0.01, "edge_threshold": 1.5, "top_n": 5,
        "structure": {"vert_dte": 18, "em_fraction": 0.75, "width": 10, "cal_front_dte": 10, "cal_gap": 30,
                      "diag_front_dte": 5, "diag_gap": 7}}
R = core.analyse(data, core.DEFAULT_WEIGHTS, opts)
oi = R["oi"]; bs = oi["by_strike"].set_index("strike"); sm = oi["summary"]
assert bs.loc[700, "call_chg"] == 31994 and bs.loc[685, "put_chg"] == 41595 and bs.loc[650, "put_chg"] == 46337
assert oi["frame"]["chg"].sum() == 379803 and sm["n_contracts"] == 1474
assert abs(sm["gross_turnover"] - 0.225) < 0.001 and 0.95 < sm["known_share"] < 1.0
assert abs(oi["by_exp"]["d_gex"].sum() - oi["by_strike"]["d_gex"].sum()) < 1.0
assert oi["by_strike"][["call_oi", "put_oi"]].sum().sum() == oi["frame"]["open_interest"].sum()
gs = data["gex_strike"].set_index("strike")
spot = R["spot"]
for cp, col in (("C", "call_gex"), ("P", "put_gex")):
    lv = ch[(ch["cp"] == cp) & (ch["exp"] >= asof)].assign(v=lambda x: x["gamma"] * x["open_interest"] * 100 * spot * spot * 0.01)
    j = pd.concat([lv.groupby("strike")["v"].sum(), gs[col].abs()], axis=1, keys=["m", "f"]).dropna()
    j = j[(j["f"] > 1e5) & (j["m"] > 1e5)]
    ratio = (j["f"] / j["m"]).median()
    assert 0.97 < ratio < 1.03 and j["m"].rank().corr(j["f"].rank()) > 0.99, (cp, ratio)
print("real data: matches earlier hand numbers, totals conserved, GEX-file unit convention confirmed")

# ---- 5. walls, legs, snapshot
tags = {w["label"]: w for w in R["oi_walls"]}
assert tags["Put wall"]["strike"] == 680.0 and tags["Put wall"]["available"] and tags["Put wall"]["tag"] == "flat"
assert tags["Call wall"]["strike"] == 700.0 and tags["Call wall"]["tag"] == "strong build" and tags["Call wall"]["chg"] == 31994
assert tags["Call wall"]["other_tag"] == "erode" and tags["Call wall"]["other_chg"] == -12995
leg = R["strategies"][0].legs
assert all(np.isfinite(l.oi) and np.isfinite(l.oi_chg) for l in leg), [(l.oi, l.oi_chg) for l in leg]
assert leg[0].oi == 24472 and leg[0].oi_chg == -1268 and leg[1].oi == 29261 and leg[1].oi_chg == 4590
row = core.snapshot_row(R)
assert list(row) == core.SNAP_COLS and row["asof"] == "2026-11-02" and row["put_wall"] == 680.0
assert row["call_wall_call_chg"] == 31994 and row["top_put_add_strike"] == 650.0 and row["top_call_add_strike"] == 700.0
df1 = core.merge_snapshot(None, row)
row2 = dict(row, asof="2026-11-03", spot=700.0)
df2 = core.merge_snapshot(df1, row2)
df3 = core.merge_snapshot(df2, dict(row, spot=111.0))                              # same date as first row: replaced
assert len(df1) == 1 and len(df2) == 2 and len(df3) == 2
assert df3["asof"].tolist() == ["2026-11-02", "2026-11-03"] and df3.loc[0, "spot"] == 111.0 and list(df3.columns) == core.SNAP_COLS
rt = core.merge_snapshot(pd.read_csv(__import__("io").StringIO(df3.to_csv(index=False))), dict(row, asof="2026-11-04"))
assert len(rt) == 3 and rt["asof"].tolist()[-1] == "2026-11-04"                       # survives a CSV round trip
print("walls, leg-level OI, snapshot row and merge (dedupe, order, CSV round trip) ok")

# ---- 6. chain without a change column, and without the strike file
d2 = dict(data); d2["chain"] = raw.drop(columns=["oi_chg"])
R2 = core.analyse(d2, core.DEFAULT_WEIGHTS, opts)
assert R2["oi"] is None and R2["oi_walls"] == [] and all(np.isnan(l.oi_chg) for l in R2["strategies"][0].legs)
row = core.snapshot_row(R2); assert np.isnan(row["win_net_put_chg"]) and row["asof"] == "2026-11-02"
d3 = {k: v for k, v in data.items() if k != "gex_strike"}
R3 = core.analyse(d3, core.DEFAULT_WEIGHTS, opts)
assert R3["oi"] is not None and R3["oi_walls"] == [] and np.isfinite(core.snapshot_row(R3)["win_net_put_chg"])
R4 = core.analyse(data, core.DEFAULT_WEIGHTS, dict(opts, oi_min_contracts=50000))      # absurd min: everything flat
assert set(R4["oi"]["by_strike"]["put_tag"]) <= {"flat", "n/a"} and R4["oi"]["summary"]["net_put_chg"] == R["oi"]["summary"]["net_put_chg"]
print("no-change-column, no-strike-file and parameter overrides ok")
print("ALL OI CHECKS PASSED")
