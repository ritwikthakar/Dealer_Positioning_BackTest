"""Tests for vanna_charm.py.  Usage: python3 -I test_vanna_charm.py [folder_with_the_csvs]   (or: pytest test_vanna_charm.py)
The synthetic tests need no data; the last one runs on the real files if the folder exists."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import vanna_charm as vc  # noqa: E402

ASOF = pd.Timestamp("2026-11-20")


def _chain(rows):
    d = pd.DataFrame(rows, columns=["exp", "cp", "strike", "open_interest", "vanna", "charm"])
    d["exp"] = pd.to_datetime(d["exp"])
    return d


def test_signs_and_exclusions():
    ch = _chain([
        ("2026-12-18", "P", 500, 1000, -0.002, 0.003),   # OTM put: counted, dealers short
        ("2026-12-18", "C", 700, 1000, 0.001, -0.004),   # OTM call: ambiguous side
        ("2026-12-18", "P", 700, 5000, -0.009, 0.009),   # ITM put: ignored
        ("2026-12-18", "C", 500, 5000, -0.009, 0.009),   # ITM call: ignored
        ("2027-12-17", "P", 500, 9999, -0.002, 0.003),   # outside the DTE window
        ("2026-12-18", "P", 510, 0, -0.002, 0.003),      # no open interest
    ])
    r = vc.positioning(ch, 613.0, ASOF, {"max_dte": 120})
    assert r["ok"]
    # short puts: IV down 1pt -> dealer delta falls by vanna*OI*100 * (-1) ... hedge = -(-0.002)*1000*100 = +200 shares bought
    assert abs(r["put_vanna"] - 200.0) < 1e-9, r["put_vanna"]
    assert abs(r["put_charm"] - 300.0) < 1e-9, r["put_charm"]      # OTM put delta decays toward 0 -> dealers buy back hedge
    assert abs(r["call_vanna"] - 100.0) < 1e-9 and abs(r["call_charm"] - 400.0) < 1e-9   # if dealers were long the calls
    assert r["net_vanna"] == r["put_vanna"], "default convention ignores calls"
    r2 = vc.positioning(ch, 613.0, ASOF, {"convention": "long_calls"})
    assert abs(r2["net_vanna"] - 300.0) < 1e-9 and abs(r2["net_charm"] - 700.0) < 1e-9
    r3 = vc.positioning(ch, 613.0, ASOF, {"convention": "short_calls"})
    assert abs(r3["net_vanna"] - 100.0) < 1e-9 and abs(r3["net_charm"] - (300.0 - 400.0)) < 1e-9
    s = r["scenarios"].set_index("IV change (vol pts)")
    assert abs(s.loc[-3.0, "put-side shares (M)"] - 3 * 200.0 / 1e6) < 1e-12
    assert abs(s.loc[-5.0, "put-side $ (B)"] - 5 * 200.0 * 613.0 / 1e9) < 1e-12


def test_missing_columns():
    ch = _chain([("2026-12-18", "P", 500, 1000, -0.002, 0.003)]).drop(columns=["vanna"])
    assert not vc.positioning(ch, 613.0, ASOF)["ok"]


def _hist(kind):
    """~127 sessions: quiet, then a washout, then (depending on kind) a vol rollover."""
    n = 127
    idx = pd.bdate_range("2026-05-01", periods=n)
    px = np.full(n, 100.0) + 0.2 * np.sin(np.arange(n))
    iv = np.full(n, 20.0) + 0.3 * np.cos(np.arange(n))
    dex = 10e9 + 3e9 * np.sin(np.arange(n) / 3.0)
    ivr = np.full(n, 30.0)
    px[120:125] = [99, 98, 96.5, 95, 94]
    iv[120:125] = [22, 25, 28, 30, 32]
    dex[120:125] = [-20e9, -60e9, -90e9, -110e9, -120e9]
    ivr[120:125] = [50, 70, 85, 95, 100]
    if kind == "none":
        px[120:125], iv[120:125], dex[120:125], ivr[120:125] = 100, 20, 10e9, 30
        px[125:], iv[125:] = 100, 20
    elif kind == "watch":          # washout continues, IV still rising
        iv[125], iv[126], px[125], px[126] = 33, 34, 93, 92.5
        ivr[125], ivr[126] = 100, 100
    elif kind == "armed":          # IV falls two sessions in a row but price keeps dropping
        iv[125], iv[126], px[125], px[126] = 31, 29.5, 93.5, 93.0
        ivr[125], ivr[126] = 95, 90
    elif kind == "alert":
        iv[125], iv[126], px[125], px[126] = 31, 29.5, 94.5, 95.0
        ivr[125], ivr[126] = 95, 90
    return pd.DataFrame({"Date": idx, "Close Price": px, "IV 30d": iv, "IV Rank": ivr, "DEX Net OI": dex})


def _vc_ok(put_v=1.0e6, put_c=1.0e6):
    return {"ok": True, "put_vanna": put_v, "put_charm": put_c}


def test_alert_levels():
    for kind, level in [("none", "none"), ("watch", "watch"), ("armed", "armed"), ("alert", "alert")]:
        r = vc.squeeze_alert(_hist(kind), _vc_ok(), "Negative", 95.0, 99.0, 102.0)
        assert r["ok"] and r["level"] == level, (kind, r["level"], r["checks"][["condition", "met"]].to_dict("records"))
    # without dealer put-side support the alert stays at "armed"
    r = vc.squeeze_alert(_hist("alert"), _vc_ok(0.0, 0.0), "Negative", 95.0, 99.0, 102.0)
    assert r["level"] == "armed", r["level"]
    # chain without vanna/charm columns: the support rows fail, the alert degrades but does not crash
    r = vc.squeeze_alert(_hist("alert"), {"ok": False}, "Negative", 95.0, None, None)
    assert r["ok"] and r["level"] == "armed"
    # too little history
    assert not vc.squeeze_alert(_hist("alert").head(10), _vc_ok(), "Negative", 95.0, None, None)["ok"]


def test_history_stats_shape():
    r = vc.squeeze_alert(_hist("alert"), _vc_ok(), "Negative", 95.0, 99.0, 102.0)
    h = r["history"]
    assert set(h) >= {"baseline", "trigger", "washout_then_trigger", "n_days"}
    for k in ("trigger", "washout_then_trigger"):
        assert h[k]["n_events"] == len(h[k]["events"])
        assert set(h[k]["stats"]) == {"f5", "f10", "f20"}


def test_real_files():
    folder = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else "/mnt/user-data/uploads"
    if not os.path.isdir(folder):
        return
    import core
    data = core.load_folder(folder)
    if "chain" not in data or "vanna" not in data["chain"].columns:
        return
    chain = core.parse_chain(data["chain"])
    spot = core.estimate_spot(chain)
    # all-expiry OTM-put flows reproduce the hand analysis (0.83M shares per vol point, 0.86M per day at spot 613)
    r = vc.positioning(chain, 613.0, pd.Timestamp("2026-11-21"), {"min_dte": 1, "max_dte": 5000})
    assert abs(r["put_vanna"] / 1e6 - 0.825) < 0.02, r["put_vanna"]
    assert abs(r["put_charm"] / 1e6 - 0.857) < 0.02, r["put_charm"]
    assert 600 < spot < 625, spot


def test_snapshot_schema():
    """The daily snapshot log carries the put- and call-side readings the trend charts use."""
    import core
    for k in ("vanna_put_sh_per_pt", "charm_put_sh_per_day", "vanna_call_sh_per_pt", "charm_call_sh_per_day"):
        assert k in core.SNAP_COLS, k
    old = {c: 1 for c in core.SNAP_COLS[:21]}
    old["asof"] = "2026-11-02"                       # a log written before these columns existed
    new = {c: float("nan") for c in core.SNAP_COLS}
    new.update(asof="2026-11-20", vanna_put_sh_per_pt=5e5, vanna_call_sh_per_pt=1.7e6)
    m = core.merge_snapshot(core.merge_snapshot(None, old), new)
    assert list(m.columns)[:len(core.SNAP_COLS)] == core.SNAP_COLS and len(m) == 2
    assert m["vanna_put_sh_per_pt"].isna().iloc[0] and m["vanna_put_sh_per_pt"].iloc[1] == 5e5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("ALL CHECKS PASSED")
