"""Tests for vanna_charm.empirical_vanna.  Usage: python3 -I test_empirical_vanna.py   (or: pytest test_empirical_vanna.py)
Synthetic data with a known IV sensitivity; no data files needed."""
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import vanna_charm as vc  # noqa: E402

TRUE_BETA = -2.0e8        # DEX ($) added per +1 vol point of IV  ->  dealer buying of +2e8 $ per 1 pt IV fall


def make_history(n=500, seed=0, lag=0, noise=3e7, beta=TRUE_BETA):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)
    d_iv = rng.normal(0, 1.0, n)
    ret = -0.5 * d_iv + rng.normal(0, 0.9, n)                      # price and IV move against each other
    px = 600 * np.cumprod(1 + ret / 100)
    iv = 20 + np.cumsum(d_iv)
    gex = 1.0e9 + 2.0e8 * rng.standard_normal(n)
    gex_prev = np.concatenate([[gex[0]], gex[:-1]])
    d_dex = beta * d_iv + 0.9 * gex_prev * ret + 3.0e8 * ret + 1e7 + rng.normal(0, noise, n)
    dex = 5.0e8 + np.cumsum(d_dex)
    if lag:                                                         # history row reports the previous day's DEX
        dex = np.concatenate([np.full(lag, dex[0]), dex[:-lag]])
    return pd.DataFrame({"Date": dates, "Close Price": px, "IV 30d": iv, "GEX Net OI": gex, "DEX Net OI": dex})


def test_recovers_planted_beta():
    r = vc.empirical_vanna(make_history(), window=250)
    assert r["ok"] and r["lag"] == 0, r.get("reason")
    assert abs(r["full"]["beta"] - TRUE_BETA) / abs(TRUE_BETA) < 0.05, r["full"]
    lt = r["latest"]
    assert abs(lt["buy_usd_per_pt"] - (-TRUE_BETA)) / (-TRUE_BETA) < 0.25, lt          # rolling window is noisier
    assert lt["buy_usd_per_pt"] > 0 and abs(lt["t"]) > 2 and lt["r2"] > 0.8, lt
    assert abs(lt["buy_sh_per_pt"] * lt["px"] - lt["buy_usd_per_pt"]) < 1e-3          # shares = dollars / price
    s = r["series"]
    assert len(s) == 250, len(s)                                   # 499 usable rows, one regression per window end
    assert s.index.is_monotonic_increasing and np.isfinite(s["beta"]).all()


def test_sign_follows_the_data():
    pos = vc.empirical_vanna(make_history(beta=+2.0e8), window=250)["latest"]
    neg = vc.empirical_vanna(make_history(beta=-2.0e8), window=250)["latest"]
    assert pos["buy_usd_per_pt"] < 0 < neg["buy_usd_per_pt"], (pos, neg)    # delta FALLING when IV falls = dealer selling


def test_picks_the_one_row_lag():
    r = vc.empirical_vanna(make_history(lag=1), window=250)
    assert r["ok"] and r["lag"] == 1, (r["lag"], r["lag_scores"])
    assert abs(r["full"]["beta"] - TRUE_BETA) / abs(TRUE_BETA) < 0.10, r["full"]
    assert r["lag_scores"][1]["r2"] > r["lag_scores"][0]["r2"]


def test_no_relationship_is_flagged_weak():
    h = make_history(seed=3)
    rng = np.random.default_rng(9)
    h["DEX Net OI"] = 5e8 + np.cumsum(rng.normal(0, 5e8, len(h)))                       # pure noise
    r = vc.empirical_vanna(h, window=250)
    assert r["ok"]
    assert abs(r["latest"]["t"]) < 3, r["latest"]                                      # no strong spurious signal
    assert r["latest"]["r2"] < 0.2, r["latest"]


def test_window_changes_series_length_and_is_capped():
    h = make_history(n=300)
    a, b = vc.empirical_vanna(h, window=60), vc.empirical_vanna(h, window=120)
    assert len(a["series"]) > len(b["series"]) > 0
    big = vc.empirical_vanna(h, window=5000)                       # larger than the data: uses what exists
    assert big["ok"] and big["window"] <= 299 and len(big["series"]) >= 1


def test_nans_and_infs_are_dropped():
    h = make_history()
    h.loc[10, "IV 30d"] = np.nan
    h.loc[20, "Close Price"] = np.nan
    h.loc[30, "DEX Net OI"] = np.inf
    r = vc.empirical_vanna(h, window=250)
    assert r["ok"] and np.isfinite(r["latest"]["beta"])


def test_date_index_or_column_both_work():
    h = make_history()
    a = vc.empirical_vanna(h, window=120)["latest"]["beta"]
    b = vc.empirical_vanna(h.set_index("Date"), window=120)["latest"]["beta"]
    assert abs(a - b) < 1e-6


def test_graceful_failures():
    assert not vc.empirical_vanna(None)["ok"]
    assert not vc.empirical_vanna(make_history(n=40))["ok"]                                 # too little history
    h = make_history().drop(columns=["GEX Net OI"])
    r = vc.empirical_vanna(h)
    assert not r["ok"] and "GEX Net OI" in r["reason"]


def test_core_wires_it_in_and_versions_match():
    import core
    src = open((__file__.rsplit("/", 1)[0] if "/" in __file__ else ".") + "/app.py").read()
    assert re.search(rf'EXPECTED_CORE = "{re.escape(core.CORE_VERSION)}"', src), "bump EXPECTED_CORE in app.py"
    assert "emp_vanna" in open(core.__file__).read()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("ALL CHECKS PASSED")
