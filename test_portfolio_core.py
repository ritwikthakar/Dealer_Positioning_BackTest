"""Tests for portfolio_core.py. Usage: python3 -m pytest test_portfolio_core.py -q   (or python3 test_portfolio_core.py)"""
import datetime as dt
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import portfolio_core as pc  # noqa: E402

TODAY = dt.date(2026, 11, 25)
EXP1 = TODAY + dt.timedelta(days=14)
EXP2 = TODAY + dt.timedelta(days=21)
COLS = pc.POSITION_COLS


def legs_from(rows, spot=100.0, iv=0.25, r=0.04, q=0.0):
    """rows: (type, strike, expiry, qty, entry_price[, mark]) -> resolved legs for one underlying 'QQQ'."""
    recs = []
    for row in rows:
        mark = row[5] if len(row) > 5 else np.nan
        recs.append({"trade": "t", "underlying": "QQQ", "type": row[0], "strike": row[1], "expiry": row[2],
                     "qty": row[3], "entry_price": row[4], "iv_pct": np.nan, "mark": mark})
    df, issues = pc.normalize_positions(pd.DataFrame(recs, columns=COLS))
    assert not issues
    return pc.resolve_legs(df, {"QQQ": spot}, {"QQQ": iv}, TODAY, r, q)


def test_bs_reference_and_parity():
    c = float(pc.bs_price("C", 100, 100, 1.0, 0.2, 0.05, 0.0))
    p = float(pc.bs_price("P", 100, 100, 1.0, 0.2, 0.05, 0.0))
    assert abs(c - 10.4506) < 1e-3
    assert abs((c - p) - (100 - 100 * math.exp(-0.05))) < 1e-9
    assert float(pc.bs_price("C", 110, 100, 0, 0.2)) == 10.0
    assert float(pc.bs_price("P", 110, 100, 0, 0.2)) == 0.0


def test_greeks_match_finite_differences():
    S, K, T, iv, r, q = 100.0, 102.0, 30 / 365, 0.22, 0.04, 0.01
    for kind in "CP":
        g = pc.bs_greeks(kind, S, K, T, iv, r, q)
        h = 0.01
        fd_delta = (float(pc.bs_price(kind, S + h, K, T, iv, r, q)) - float(pc.bs_price(kind, S - h, K, T, iv, r, q))) / (2 * h)
        fd_gamma = (float(pc.bs_price(kind, S + h, K, T, iv, r, q)) - 2 * float(pc.bs_price(kind, S, K, T, iv, r, q))
                    + float(pc.bs_price(kind, S - h, K, T, iv, r, q))) / h ** 2
        fd_vega = (float(pc.bs_price(kind, S, K, T, iv + 0.005, r, q)) - float(pc.bs_price(kind, S, K, T, iv - 0.005, r, q))) / 0.01 / 100
        fd_theta = (float(pc.bs_price(kind, S, K, T - 1 / 365, iv, r, q)) - float(pc.bs_price(kind, S, K, T + 1 / 365, iv, r, q))) / 2
        assert abs(g["delta"] - fd_delta) < 1e-4
        assert abs(g["gamma"] - fd_gamma) < 1e-4
        assert abs(g["vega"] - fd_vega) < 1e-4
        assert abs(g["theta"] - fd_theta) < 5e-4


def test_implied_vol_roundtrip_and_bounds():
    for kind in "CP":
        px = float(pc.bs_price(kind, 100, 105, 45 / 365, 0.31, 0.04, 0.0))
        assert abs(pc.implied_vol(kind, px, 100, 105, 45 / 365, 0.04, 0.0) - 0.31) < 1e-6
    assert pc.implied_vol("C", 0.0, 100, 105, 0.1) is None
    assert pc.implied_vol("C", 1.0, 100, 50, 0.1) is None  # below intrinsic


def test_normalize_positions():
    df = pd.DataFrame([
        {"trade": "", "underlying": "qqq", "type": "put", "strike": 600, "expiry": "2026-12-11", "qty": -1, "entry_price": 3.1},
        {"trade": "x", "underlying": "QQQ", "type": "P", "strike": 600, "expiry": "2026-12-18", "qty": 1, "entry_price": 4.0},
        {"trade": "bad", "underlying": "QQQ", "type": "C", "strike": None, "expiry": "2026-12-18", "qty": 1, "entry_price": 1},
        {"trade": "bad2", "underlying": "QQQ", "type": "C", "strike": 600, "expiry": "2026-12-18", "qty": 0, "entry_price": 1},
        {"trade": "sh", "underlying": "AAPL", "type": "stock", "strike": None, "expiry": None, "qty": 100, "entry_price": 190},
    ])
    out, issues = pc.normalize_positions(df)
    assert len(out) == 3 and len(issues) == 2
    assert out.loc[0, "underlying"] == "QQQ" and out.loc[0, "type"] == "P"
    assert out.loc[0, "trade"].startswith("QQQ P600")
    assert out.loc[2, "type"] == "S" and pd.isna(out.loc[2, "strike"])
    assert isinstance(out.loc[0, "expiry"], dt.date)
    assert pc.normalize_positions(pd.DataFrame())[0].empty


def test_long_call_breakeven_and_risk():
    legs = legs_from([("C", 100, EXP2, 1, 2.50)], spot=100)
    m = pc.analyze_trade(legs, 100, TODAY, r=0.04)
    assert m["kind"] == "debit" and abs(m["cost"] - 250) < 1e-9
    assert m["unlimited_profit"] and m["stats_first"]["max_profit"] == math.inf
    assert abs(m["risk"] - 250) < 1e-6
    assert m["stats_first"]["breakevens"] == [102.5] or abs(m["stats_first"]["breakevens"][0] - 102.5) < 0.1


def test_call_debit_spread_defined_both_ways():
    legs = legs_from([("C", 100, EXP1, 1, 3.0), ("C", 105, EXP1, -1, 1.0)])
    m = pc.analyze_trade(legs, 100, TODAY, r=0.04)
    assert abs(m["cost"] - 200) < 1e-9
    assert not m["unlimited_profit"] and not m["unlimited_loss"]
    assert abs(m["stats_first"]["max_profit"] - 300) < 1e-6      # 5 wide * 100 - 200 debit
    assert abs(m["stats_first"]["max_loss"] + 200) < 1e-6
    assert abs(m["stats_first"]["breakevens"][0] - 102.0) < 0.1


def test_put_credit_spread_kind_and_risk():
    legs = legs_from([("P", 95, EXP1, -1, 1.5), ("P", 90, EXP1, 1, 0.5)])
    m = pc.analyze_trade(legs, 100, TODAY, r=0.04)
    assert m["kind"] == "credit" and abs(m["cost"] + 100) < 1e-9
    assert abs(m["risk"] - 400) < 1e-6                          # 5 wide * 100 - 100 credit
    assert abs(m["stats_first"]["max_profit"] - 100) < 1e-6
    assert abs(m["stats_first"]["breakevens"][0] - 94.0) < 0.1


def test_naked_short_put_risk_uses_zero_tail():
    legs = legs_from([("P", 100, EXP1, -1, 2.0)])
    m = pc.analyze_trade(legs, 100, TODAY, r=0.04)
    assert abs(m["risk"] - (100 * 100 - 200)) < 5                # strike*100 minus credit
    naked_call = pc.analyze_trade(legs_from([("C", 100, EXP1, -1, 2.0)]), 100, TODAY, r=0.04)
    assert naked_call["unlimited_loss"] and abs(naked_call["risk"] - 400) < 1e-6  # 2x credit convention


def test_calendar_shape():
    # put calendar at the money: short front, long back
    short_px = float(pc.bs_price("P", 100, 100, 14 / 365, 0.25, 0.04))
    long_px = float(pc.bs_price("P", 100, 100, 21 / 365, 0.25, 0.04))
    legs = legs_from([("P", 100, EXP1, -1, round(short_px, 2)), ("P", 100, EXP2, 1, round(long_px, 2))])
    m = pc.analyze_trade(legs, 100, TODAY, r=0.04)
    assert m["kind"] == "calendar" and m["cost"] > 0
    assert not m["unlimited_profit"] and not m["unlimited_loss"]
    be = m["stats_first"]["breakevens"]
    assert len(be) == 2 and be[0] < 100 < be[1]
    center = int(np.argmin(abs(m["grid"] - 100)))
    assert m["pnl_first"][center] == m["pnl_first"].max() or m["pnl_first"][center] > 0.9 * m["pnl_first"].max()
    assert m["stats_first"]["max_profit"] > 0
    assert m["risk"] <= m["cost"] + 1e-6                         # loss never exceeds the debit
    assert abs(m["pnl_now"]) < 1.0                               # marked at model = entry prices
    assert m["greeks"]["theta"] > 0                              # long-back/short-front: positive carry
    assert m["pop_first"] is not None and 0 < m["pop_first"] < 1
    # one honest horizon: today and the first expiry. No bogus "last expiry" curve for the long leg.
    assert list(m["curves"]) == ["Today", f"At {EXP1:%d %b} expiry"]
    assert not hasattr(m, "stats_last") and "stats_last" not in m and "pnl_last" not in m


def test_diagonal_and_call_calendar_are_bounded_but_naked_call_is_not():
    short_c = float(pc.bs_price("C", 100, 100, 14 / 365, 0.25, 0.04))
    long_c = float(pc.bs_price("C", 100, 100, 60 / 365, 0.25, 0.04))
    cal = pc.analyze_trade(legs_from([("C", 100, EXP1, -1, round(short_c, 2)), ("C", 100, TODAY + dt.timedelta(days=60), 1, round(long_c, 2))]),
                           100, TODAY, r=0.04, q=0.005)
    assert not cal["unlimited_loss"] and not cal["unlimited_profit"]
    assert cal["risk"] <= cal["cost"] + 1e-6 or cal["risk"] < 1.1 * cal["cost"]
    naked = pc.analyze_trade(legs_from([("C", 100, EXP1, -1, 2.0)]), 100, TODAY, r=0.04, q=0.005)
    assert naked["unlimited_loss"]


def test_marks_override_and_iv_from_mark():
    legs = legs_from([("C", 100, EXP2, 1, 2.50, 3.10)], spot=100)
    assert abs(legs.loc[0, "mark_eff"] - 3.10) < 1e-12
    expected = pc.implied_vol("C", 3.10, 100, 100, 21 / 365, 0.04, 0.0)
    assert abs(legs.loc[0, "iv"] - expected) < 1e-9
    m = pc.analyze_trade(legs, 100, TODAY, r=0.04)
    assert abs(m["pnl_now"] - 60.0) < 1e-6


def test_expiry_curve_ignores_iv_shift_for_expired_legs():
    legs = legs_from([("C", 100, EXP1, 1, 3.0), ("C", 105, EXP1, -1, 1.0)])
    a = pc.analyze_trade(legs, 100, TODAY, r=0.04, iv_shift=0.0)
    b = pc.analyze_trade(legs, 100, TODAY, r=0.04, iv_shift=0.10)
    assert np.allclose(a["pnl_first"], b["pnl_first"])
    assert not np.allclose(a["pnl_today"], b["pnl_today"])


# ------------------------------------------------------------------------------------------ advisor
def _long_call(mark, dte=40, spot=100, entry=2.50):
    exp = TODAY + dt.timedelta(days=dte)
    legs = legs_from([("C", 100, exp, 1, entry, mark)], spot=spot)
    return pc.evaluate_trade(legs, spot, TODAY, r=0.04)


def test_advisor_debit_stop_target_hold():
    assert _long_call(1.20)["advice"]["action"] == "CUT LOSS"          # -52% of 250
    assert _long_call(3.90)["advice"]["action"] == "BOOK PROFIT"       # +56%
    hold = _long_call(2.60)["advice"]
    assert hold["action"] == "HOLD" and hold["reasons"]
    lock = _long_call(3.50)["advice"]                                   # +40%: >60% of the way to +50%
    assert lock["action"] == "ADJUST"


def test_advisor_time_exit_and_expiry():
    assert _long_call(2.0, dte=4)["advice"]["action"] == "CUT LOSS"
    assert _long_call(3.0, dte=4)["advice"]["action"] == "BOOK PROFIT"
    assert _long_call(2.5, dte=0)["advice"]["action"] in ("CUT LOSS", "BOOK PROFIT")
    exp_past = TODAY - dt.timedelta(days=2)
    legs = legs_from([("C", 100, exp_past, 1, 2.5, 0.0)])
    assert pc.evaluate_trade(legs, 100, TODAY)["advice"]["action"] == "CLOSE (EXPIRED)"


def _calendar(spot_now, short_mark=None, long_mark=None, front_dte=14, back_dte=21):
    f, b = TODAY + dt.timedelta(days=front_dte), TODAY + dt.timedelta(days=back_dte)
    sp = round(float(pc.bs_price("P", 100, 100, front_dte / 365, 0.25, 0.04)), 2)
    lp = round(float(pc.bs_price("P", 100, 100, back_dte / 365, 0.25, 0.04)), 2)
    rows = [("P", 100, f, -1, sp) + ((short_mark,) if short_mark is not None else ()),
            ("P", 100, b, 1, lp) + ((long_mark,) if long_mark is not None else ())]
    if short_mark is None and long_mark is not None:
        rows[0] = ("P", 100, f, -1, sp, np.nan)
    legs = legs_from(rows, spot=spot_now)
    return pc.evaluate_trade(legs, spot_now, TODAY, r=0.04)


def test_advisor_calendar_hold_adjust_and_close():
    flat = _calendar(100.0)["advice"]
    assert flat["action"] == "HOLD"
    moved = _calendar(104.0)["advice"]                                   # ~0.85 EM away from the strike
    assert moved["action"] in ("ADJUST", "CUT LOSS")
    near_exp = _calendar(100.0, front_dte=2, back_dte=9)["advice"]
    assert near_exp["action"] in ("BOOK PROFIT", "CUT LOSS") and "Front leg" in near_exp["reasons"][0]


def test_advisor_credit_spread_adjust_and_cut():
    f = TODAY + dt.timedelta(days=21)
    legs = legs_from([("P", 98, f, -1, 1.8), ("P", 93, f, 1, 0.6)], spot=97.0)   # spot through the short strike
    r = pc.evaluate_trade(legs, 97.0, TODAY, r=0.04)["advice"]
    assert r["action"] in ("ADJUST", "CUT LOSS")
    ok = legs_from([("P", 90, f, -1, 0.8), ("P", 85, f, 1, 0.3)], spot=100.0)
    assert pc.evaluate_trade(ok, 100.0, TODAY, r=0.04)["advice"]["action"] in ("HOLD", "BOOK PROFIT")


# ---------------------------------------------------------------------------------------- portfolio
def test_portfolio_summary_cash_and_return():
    rows = [{"cost": 250.0, "value_now": 300.0, "pnl_now": 50.0, "risk": 250.0, "kind": "debit", "dollar_delta": 20.0,
             "greeks": {"delta": 0.2, "gamma": 0.01, "theta": -1.0, "vega": 5.0}},
            {"cost": -100.0, "value_now": -60.0, "pnl_now": 40.0, "risk": 400.0, "kind": "credit", "dollar_delta": -5.0,
             "greeks": {"delta": -0.05, "gamma": 0.0, "theta": 0.5, "vega": -1.0}}]
    s = pc.portfolio_summary(rows, realized=200.0, capital=10_000.0, deposits=0.0)
    assert abs(s["cash"] - (10_000 + 200 - 150)) < 1e-9
    assert abs(s["market_value"] - 240) < 1e-9
    assert abs(s["equity"] - (10_050 + 240)) < 1e-9
    assert abs(s["total_pnl"] - 290) < 1e-9
    assert abs(s["total_return_pct"] - 2.9) < 1e-9
    assert abs(s["free_cash"] - (s["cash"] - 400)) < 1e-9
    assert abs(s["greeks"]["theta"] + 0.5) < 1e-9
    assert pc.portfolio_summary(rows, 0, 10_000, cash_override=5000)["cash"] == 5000
    empty = pc.portfolio_summary([], 0.0, 10_000.0)
    assert empty["equity"] == 10_000 and empty["total_return_pct"] == 0.0


def test_closed_stats():
    df = pd.DataFrame({"realized_pnl": [100, -50, 200, -25]})
    s = pc.closed_stats(df)
    assert s["n"] == 4 and s["win_rate"] == 0.5 and abs(s["profit_factor"] - 4.0) < 1e-9
    assert s["total"] == 225 and abs(s["expectancy"] - 56.25) < 1e-9
    assert pc.closed_stats(pc.empty_closed())["n"] == 0


def test_size_trade():
    m = {"risk": 400.0}
    assert pc.size_trade(m, equity=50_000, cash=50_000, risk_pct=2.0)[0] == 2
    assert pc.size_trade(m, equity=50_000, cash=500, risk_pct=2.0)[0] == 1
    assert pc.size_trade({"risk": math.inf}, 1, 1, 1)[0] == 0


def test_templates_and_model_prices():
    for name in pc.TEMPLATES:
        d = pc.build_template(name, "QQQ", 600, 5, 10, EXP1, EXP2, qty=2)
        assert list(d.columns) == COLS and len(d) >= 1
    dc = pc.build_template("Double calendar", "QQQ", 600, 5, 10, EXP1, EXP2)
    assert len(dc) == 4 and dc["qty"].sum() == 0 and set(dc["strike"]) == {590.0, 610.0}
    filled = pc.fill_model_prices(pc.build_template("Put calendar", "QQQ", 600, 5, 10, EXP1, EXP2), 600.0, 0.22, TODAY, 0.04)
    assert filled["entry_price"].notna().all()
    assert filled.loc[1, "entry_price"] > filled.loc[0, "entry_price"]    # back month costs more
    assert pc.next_weekday(TODAY, 4, 7).weekday() == 4 and pc.next_weekday(TODAY, 4, 7) >= TODAY + dt.timedelta(days=7)


def test_fetch_quotes_with_fake_loader():
    legs, _ = pc.normalize_positions(pd.DataFrame([
        {"underlying": "QQQ", "type": "P", "strike": 600, "expiry": EXP1, "qty": -1, "entry_price": 3},
        {"underlying": "QQQ", "type": "P", "strike": 601, "expiry": EXP1, "qty": 1, "entry_price": 3},
        {"underlying": "QQQ", "type": "C", "strike": 600, "expiry": EXP2, "qty": 1, "entry_price": 3},
        {"underlying": "ZZZ", "type": "C", "strike": 10, "expiry": EXP1, "qty": 1, "entry_price": 1},
    ]))

    def loader(und, expiries):
        if und == "ZZZ":
            raise ValueError("no price history")
        ch = pd.DataFrame({"strike": [600.0, 600.0], "cp": ["P", "C"], "bid": [2.0, 0.0], "ask": [2.2, 0.0], "last": [2.1, 5.0]})
        return 601.5, {EXP1: ch}

    spots, marks, errors = pc.fetch_quotes(legs, loader)
    assert spots == {"QQQ": 601.5}
    assert abs(marks.iloc[0] - 2.1) < 1e-9                       # mid of 2.0 / 2.2
    assert np.isnan(marks.iloc[1]) and np.isnan(marks.iloc[2])   # strike 601 missing; EXP2 not listed
    assert "ZZZ" in errors and any("expiry not listed" in v for v in errors.values())


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
