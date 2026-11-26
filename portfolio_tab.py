"""Streamlit tab: option portfolio analyzer. Needs portfolio_core.py next to it. Self-contained (no option CSVs needed).
Call render() inside a tab.

Three sections: Positions & advice (cash, per-trade BOOK PROFIT / HOLD / CUT LOSS / ADJUST, breakevens, payoff curves),
Evaluate a trade (build a structure, see max profit / loss, breakevens, probability of profit, sizing, add it to the
portfolio), and Closed trades & return (realised results, win rate, portfolio return).
"""
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

import portfolio_core as pc

ICON = {"BOOK PROFIT": "🟢", "HOLD": "⚪", "CUT LOSS": "🔴", "ADJUST": "🟡", "CLOSE (EXPIRED)": "⚫"}
POS_CONFIG = {
    "trade": st.column_config.TextColumn("Trade", help="Legs sharing a name are analysed as one trade. Blank = one trade per leg."),
    "underlying": st.column_config.TextColumn("Underlying"),
    "type": st.column_config.SelectboxColumn("Type", options=["C", "P", "S"], help="C call, P put, S shares"),
    "strike": st.column_config.NumberColumn("Strike", format="%.2f", step=0.5),
    "expiry": st.column_config.DateColumn("Expiry", format="YYYY-MM-DD"),
    "qty": st.column_config.NumberColumn("Qty", step=1, help="Contracts (shares for S). + long, - short."),
    "entry_price": st.column_config.NumberColumn("Entry price", format="%.2f", step=0.01, help="Per share, what you paid / received."),
    "iv_pct": st.column_config.NumberColumn("IV % (optional)", format="%.1f", help="Overrides the IV backed out of the mark."),
    "mark": st.column_config.NumberColumn("Mark (optional)", format="%.2f", step=0.01, help="Current price per share. Blank = model price."),
}
CLOSED_CONFIG = {
    "trade": st.column_config.TextColumn("Trade"),
    "underlying": st.column_config.TextColumn("Underlying"),
    "close_date": st.column_config.DateColumn("Closed", format="YYYY-MM-DD"),
    "realized_pnl": st.column_config.NumberColumn("Realised P&L ($)", format="%.2f", step=1.0),
}


# ------------------------------------------------------------------------------------------------ helpers
def _money(x, plus=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "unlimited" if x > 0 else "unlimited loss"
    s = f"${abs(x):,.0f}"
    return ("-" + s) if round(x) < 0 else (("+" + s) if plus and round(x) > 0 else s)


def _pct(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}%"


def _be_text(be):
    return ", ".join(f"{b:,.2f}" for b in be) if be else "none"


def _for_editor(df, date_cols):
    d = df.copy()
    for c in date_cols:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    return d


def _init():
    ss = st.session_state
    ss.setdefault("pf_base", pc.empty_positions())
    ss.setdefault("pf_ver", 0)
    ss.setdefault("pf_cur", pc.empty_positions())
    ss.setdefault("pf_closed_base", pc.empty_closed())
    ss.setdefault("pf_closed_ver", 0)
    ss.setdefault("pf_closed_cur", pc.empty_closed())
    ss.setdefault("pf_ev_base", pc.empty_positions())
    ss.setdefault("pf_ev_ver", 0)
    ss.setdefault("pf_msgs", [])
    ss.setdefault("pf_imported", set())


def _payoff_chart(m, spot, title=None):
    """P&L vs spot: today and at each expiry, zero line and the current spot. Cropped to a sensible range."""
    half = float(np.clip(3.0 * (m["em_first"] or 0.0) / spot if spot else 0.1, 0.06, 0.8))
    grid = m["grid"]
    keep = np.abs(grid / spot - 1.0) <= half
    rows = []
    for name, y in m["curves"].items():
        rows += list(zip(grid[keep], np.asarray(y)[keep], [name] * int(keep.sum())))
    df = pd.DataFrame(rows, columns=["spot", "pnl", "curve"])
    try:
        import altair as alt
    except ImportError:
        st.line_chart(df.pivot(index="spot", columns="curve", values="pnl"), height=320)
        return
    line = alt.Chart(df).mark_line().encode(
        x=alt.X("spot:Q", title="Underlying price", scale=alt.Scale(zero=False)),
        y=alt.Y("pnl:Q", title="P&L ($)"), color=alt.Color("curve:N", title=None),
        tooltip=[alt.Tooltip("spot:Q", format=",.2f"), alt.Tooltip("pnl:Q", format=",.0f"), "curve:N"])
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
    here = alt.Chart(pd.DataFrame({"x": [spot]})).mark_rule(color="#e45756").encode(x="x:Q")
    st.altair_chart((line + zero + here).properties(height=320, title=title or ""), use_container_width=True)


# ---------------------------------------------------------------------------------------------- callbacks
def _fetch_cb():
    ss = st.session_state
    cur, _ = pc.normalize_positions(ss["pf_cur"])
    if cur.empty:
        ss["pf_msgs"] = [("warning", "Add some positions before fetching prices.")]
        return
    try:
        spots, marks, errors = pc.fetch_quotes(cur)
    except ImportError as e:
        ss["pf_msgs"] = [("error", f"Missing package: {e}. Add yfinance to requirements.txt and reboot the app.")]
        return
    for u, s in spots.items():
        ss[f"pf_spot_{u}"] = float(s)
    new = cur.copy()
    new["mark"] = marks.where(marks.notna(), cur["mark"])
    ss["pf_base"], ss["pf_ver"] = new, ss["pf_ver"] + 1
    msgs = [("success", f"Fetched {int(marks.notna().sum())} of {len(cur)} marks and {len(spots)} spot price(s) from Yahoo "
                        f"(delayed; check against your broker).")]
    if errors:
        msgs.append(("warning", "Not found: " + "; ".join(f"{k}: {v}" for k, v in list(errors.items())[:6])))
    ss["pf_msgs"] = msgs


def _example_cb():
    ss = st.session_state
    today = ss.get("pf_today", dt.date.today())
    spot = float(ss.get("pf_spot_QQQ", 0.0)) or 600.0
    step = pc.default_step(spot)
    f1 = pc.next_weekday(today, 4, 14)
    legs = pc.build_template("Double calendar", "QQQ", pc.round_strike(spot, step), 0, max(step, pc.round_strike(spot * 0.02, step)),
                             f1, f1 + dt.timedelta(days=3), 1)
    legs = pc.fill_model_prices(legs, spot, 0.22, today, ss.get("pf_r", 4.0) / 100.0, ss.get("pf_q", 0.5) / 100.0)
    legs["trade"] = "QQQ double calendar (example)"
    ss["pf_base"], ss["pf_ver"] = legs, ss["pf_ver"] + 1
    ss["pf_spot_QQQ"] = spot
    ss["pf_msgs"] = [("info", f"Loaded an illustrative QQQ double calendar around {spot:,.2f} with model prices. "
                              f"Replace the spot, prices and marks with yours, or delete the rows.")]


def _close_cb(name, pnl, und):
    ss = st.session_state
    cur, _ = pc.normalize_positions(ss["pf_cur"])
    ss["pf_base"], ss["pf_ver"] = cur[cur["trade"] != name].reset_index(drop=True), ss["pf_ver"] + 1
    closed = pc.normalize_closed(ss["pf_closed_cur"])
    row = pd.DataFrame([{"trade": name, "underlying": und, "close_date": ss.get("pf_today", dt.date.today()),
                         "realized_pnl": round(float(pnl), 2)}])
    ss["pf_closed_base"] = pd.concat([closed, row], ignore_index=True)
    ss["pf_closed_ver"] += 1
    ss["pf_msgs"] = [("success", f"Moved '{name}' to closed trades at {_money(pnl)}. Edit the realised P&L there to match your actual fill.")]


def _eval_spot_cb():
    ss = st.session_state
    und = ss.get("pf_ev_und", "QQQ").strip().upper()
    try:
        ss["pf_ev_spot"] = float(pc.fetch_spot(und))
    except Exception as e:
        ss["pf_msgs"] = [("warning", f"Could not fetch {und}: {str(e)[:100]}")]


def _build_cb():
    ss = st.session_state
    und = ss.get("pf_ev_und", "QQQ").strip().upper() or "QQQ"
    spot = float(ss.get("pf_ev_spot", 0.0))
    step = pc.default_step(spot or 100.0)
    center = float(ss.get("pf_ev_center", 0.0)) or pc.round_strike(spot, step)
    legs = pc.build_template(ss["pf_ev_tpl"], und, center, float(ss.get("pf_ev_width", 5.0)), float(ss.get("pf_ev_offset", 5.0)),
                             ss["pf_ev_exp1"], ss["pf_ev_exp2"], int(ss.get("pf_ev_qty", 1)))
    ss["pf_ev_base"], ss["pf_ev_ver"] = legs, ss["pf_ev_ver"] + 1


def _add_cb(legs, name):
    ss = st.session_state
    cur, _ = pc.normalize_positions(ss["pf_cur"])
    taken = set(cur["trade"]) if not cur.empty else set()
    final, k = name, 2
    while final in taken:
        final, k = f"{name} #{k}", k + 1
    new = legs.copy()
    new["trade"], new["mark"] = final, np.nan
    ss["pf_base"], ss["pf_ver"] = pd.concat([cur, new[pc.POSITION_COLS]], ignore_index=True), ss["pf_ver"] + 1
    ss["pf_msgs"] = [("success", f"Added '{final}' to your positions (entry prices as shown; replace model prices with your fills).")]


# ------------------------------------------------------------------------------------------- settings block
def _settings():
    with st.expander("Account, model and rules", expanded=False):
        a1, a2, a3, a4 = st.columns(4)
        capital = a1.number_input("Starting capital ($)", min_value=0.0, value=10_000.0, step=500.0, key="pf_capital")
        deposits = a2.number_input("Net deposits / withdrawals ($)", value=0.0, step=500.0, key="pf_deposits",
                                   help="Money added (+) or taken out (-) since the starting capital, so the return is not inflated by deposits.")
        use_cash = a3.checkbox("Use my broker cash balance", key="pf_use_cash",
                               help="Off: cash = capital + deposits + realised P&L - cost of open trades.")
        cash_in = a3.number_input("Cash balance ($)", value=0.0, step=100.0, key="pf_cash", disabled=not use_cash)
        today = a4.date_input("Valuation date", value=dt.date.today(), key="pf_today")
        b1, b2, b3 = st.columns(3)
        r = b1.number_input("Risk-free rate (%)", value=4.0, step=0.25, key="pf_r") / 100.0
        q = b2.number_input("Dividend yield (%)", value=0.5, step=0.1, key="pf_q") / 100.0
        risk_pct = b3.number_input("Risk per new trade (% of equity)", min_value=0.1, max_value=100.0, value=2.0, step=0.5, key="pf_risk_pct")
        st.markdown("**Management rules** (the advisor applies these in order: stop, target, time exit, tested / adjust, hold)")
        c = st.columns(4)
        rules = {
            "stop_loss": c[0].number_input("Stop-loss (% of risk)", 5.0, 100.0, 50.0, 5.0, key="pf_stop"),
            "calendar_target": c[1].number_input("Calendar target (% of debit)", 5.0, 200.0, 25.0, 5.0, key="pf_tgt_cal"),
            "credit_target": c[2].number_input("Credit target (% of credit)", 5.0, 100.0, 50.0, 5.0, key="pf_tgt_cr"),
            "debit_target": c[3].number_input("Long-premium target (% of risk)", 5.0, 500.0, 50.0, 5.0, key="pf_tgt_db"),
        }
        d = st.columns(4)
        rules.update({
            "calendar_exit_dte": d[0].number_input("Calendar: close at DTE <=", 0, 30, 2, key="pf_dte_cal"),
            "credit_exit_dte": d[1].number_input("Credit: manage at DTE <=", 0, 45, 7, key="pf_dte_cr"),
            "debit_exit_dte": d[2].number_input("Long premium: exit at DTE <=", 0, 45, 5, key="pf_dte_db"),
            "tested_em": d[3].number_input("Tested inside (expected moves)", 0.05, 2.0, 0.35, 0.05, key="pf_tested",
                                           help="Adjust when spot is this close to a breakeven, measured in 1-sigma expected moves to the first expiry."),
        })
    return {"capital": capital, "deposits": deposits, "cash_override": cash_in if use_cash else None, "today": today,
            "r": r, "q": q, "risk_pct": risk_pct, "rules": rules}


# ------------------------------------------------------------------------------------- positions & advice
def _positions_tab(S):
    ss = st.session_state
    st.caption("One row per leg. Legs that share a trade name are analysed together (calendar, spread, condor...). "
               "Marks are optional: leave blank for model prices or press Fetch live prices. Delete rows to remove legs.")
    b1, b2, b3 = st.columns([1, 1, 2])
    b1.button("Load example", key="pf_ex", on_click=_example_cb, help="Illustrative QQQ double calendar with model prices.")
    b2.button("Fetch live prices & marks", key="pf_fetch", on_click=_fetch_cb, help="Yahoo Finance, delayed. Fills spot and each leg's mark.")
    up = b3.file_uploader("Import positions CSV", type="csv", key="pf_up", label_visibility="collapsed")
    if up is not None:
        sig = (up.name, up.size)
        if sig not in ss["pf_imported"]:
            try:
                df, issues = pc.normalize_positions(pd.read_csv(up))
                ss["pf_imported"].add(sig)
                ss["pf_base"], ss["pf_ver"] = df, ss["pf_ver"] + 1
                ss["pf_msgs"] = [("success", f"Imported {len(df)} legs from {up.name}.")] + [("warning", i) for i in issues]
                st.rerun()
            except Exception as e:
                ss["pf_imported"].add(sig)
                st.error(f"Could not read that CSV: {e}")

    for kind, text in ss.get("pf_msgs", []):
        getattr(st, kind)(text)
    ss["pf_msgs"] = []

    edited = st.data_editor(_for_editor(ss["pf_base"], ["expiry"]), num_rows="dynamic", column_config=POS_CONFIG,
                            hide_index=True, key=f"pf_pos_ed_{ss['pf_ver']}")
    ss["pf_cur"] = edited
    clean, issues = pc.normalize_positions(edited)
    for i in issues:
        st.warning(i)
    d1, d2 = st.columns(2)
    d1.download_button("Download positions CSV", clean.to_csv(index=False).encode("utf-8"), file_name="positions.csv",
                       mime="text/csv", key="pf_dl", disabled=clean.empty)
    tmpl = pd.DataFrame([{"trade": "QQQ put calendar", "underlying": "QQQ", "type": "P", "strike": 600, "expiry": "2026-12-11", "qty": -1,
                          "entry_price": 3.10, "iv_pct": "", "mark": ""},
                         {"trade": "QQQ put calendar", "underlying": "QQQ", "type": "P", "strike": 600, "expiry": "2026-12-14", "qty": 1,
                          "entry_price": 3.60, "iv_pct": "", "mark": ""}])
    d2.download_button("Download CSV template", tmpl.to_csv(index=False).encode("utf-8"), file_name="positions_template.csv",
                       mime="text/csv", key="pf_dl_t")
    if clean.empty:
        st.info("No positions yet. Add rows above, import a CSV, or press Load example.")
        return [], {}

    unds = sorted(clean["underlying"].unique())
    st.markdown("**Underlying prices**")
    cols = st.columns(min(len(unds), 4))
    spots, ivs = {}, {}
    for i, u in enumerate(unds):
        ss.setdefault(f"pf_spot_{u}", 0.0)
        ss.setdefault(f"pf_iv_{u}", 22.0)
        spots[u] = cols[i % len(cols)].number_input(f"{u} spot", min_value=0.0, step=0.5, format="%.2f", key=f"pf_spot_{u}")
        ivs[u] = cols[i % len(cols)].number_input(f"{u} default IV %", min_value=1.0, max_value=300.0, step=0.5, key=f"pf_iv_{u}",
                                                  help="Used for legs with no mark and no IV, and for probability of profit.") / 100.0
    missing = [u for u in unds if spots[u] <= 0]
    if missing:
        st.warning(f"Enter a spot price for {', '.join(missing)} (or press Fetch live prices) to analyse those trades.")
    clean = clean[clean["underlying"].isin([u for u in unds if spots[u] > 0])]
    if clean.empty:
        return [], {}
    legs = pc.resolve_legs(clean, spots, ivs, S["today"], S["r"], S["q"])
    n_model = int(legs.loc[legs["type"] != "S", "mark"].isna().sum())
    if n_model:
        st.caption(f"{n_model} leg(s) have no mark, so their P&L uses model prices at the IVs above. Type marks or fetch live prices for real P&L.")

    rows, table = [], []
    for name, g in legs.groupby("trade", sort=False):
        if g["underlying"].nunique() > 1:
            st.warning(f"'{name}' mixes underlyings; give each underlying its own trade name.")
            continue
        u = g["underlying"].iloc[0]
        try:
            m = pc.evaluate_trade(g, spots[u], S["today"], S["rules"], S["r"], S["q"])
        except Exception as e:
            st.warning(f"Could not analyse '{name}': {e}")
            continue
        m["name"], m["und"] = name, u
        rows.append(m)
        a = m["advice"]
        table.append({"Trade": name, "Action": f"{ICON.get(a['action'], '')} {a['action']}", "P&L $": round(m["pnl_now"]),
                      "P&L % of risk": round(a["pnl_pct"]), "Risk $": round(m["risk"]), "DTE": m["dte_first"],
                      "Breakevens": _be_text((m["stats_first"] or {}).get("breakevens", m["breakevens_today"])),
                      "Delta $": round(m["dollar_delta"]), "Theta $/day": round(m["greeks"]["theta"], 1),
                      "Vega $/pt": round(m["greeks"]["vega"], 1), "Why": a["reasons"][0]})
    if not rows:
        return [], {}
    st.subheader("Trade management")
    st.dataframe(pd.DataFrame(table), hide_index=True)
    st.caption("Breakevens are at the first expiry (calendars: short leg gone, long leg valued with IVs held fixed). "
               "P&L % of risk uses the worst case on the modelled curves (undefined-risk trades use 2x the credit).")

    by_name = {m["name"]: m for m in rows}
    st.subheader("Trade detail")
    if ss.get("pf_sel") not in by_name:      # the selected trade was closed / removed: fall back to the first one
        ss["pf_sel"] = next(iter(by_name))
    pick = st.selectbox("Trade", list(by_name), key="pf_sel")
    m = by_name[pick]
    _detail(m, S, legs[legs["trade"] == pick])
    return rows, by_name


def _detail(m, S, g):
    a = m["advice"]
    banner = {"BOOK PROFIT": st.success, "CUT LOSS": st.error, "ADJUST": st.warning}.get(a["action"], st.info)
    banner(f"**{ICON.get(a['action'], '')} {a['action']}**: {m['name']}")
    for t in a["reasons"]:
        st.markdown(f"- {t}")
    st.markdown(f"**What to do:** {a['how']}")
    st.button("Close this trade at current P&L", key="pf_close_btn", on_click=_close_cb, args=(m["name"], m["pnl_now"], m["und"]),
              help="Moves the legs to Closed trades with today's P&L as the realised result. Edit it there to match your actual fill.")

    c = st.columns(5)
    c[0].metric("P&L", _money(m["pnl_now"], plus=True), _pct(a["pnl_pct"], 0) + " of risk")
    c[1].metric("Cost basis", _money(m["cost"]), "debit" if m["cost"] > 0 else "credit")
    c[2].metric("At risk", _money(m["risk"]))
    c[3].metric("Profit target", _money(a["target_amt"]))
    c[4].metric("Stop", "-" + _money(a["stop_amt"]))
    sf = m["stats_first"]
    c = st.columns(5)
    if m["first_exp"] is not None:
        c[0].metric(f"Max profit @ {m['first_exp']:%d %b}", _money(sf["max_profit"]))
        c[1].metric(f"Breakevens @ {m['first_exp']:%d %b}", _be_text(sf["breakevens"]))
        c[2].metric("Probability of profit", _pct(100 * m["pop_first"], 0), help="Lognormal at the default IV, zero skew. A model number, not a forecast.")
        c[3].metric("Expected move (1σ)", f"±{m['em_first']:,.2f}")
        c[4].metric("DTE (first / last)", f"{m['dte_first']} / {m['dte_last']}")
    gk = m["greeks"]
    c = st.columns(4)
    c[0].metric("Net delta (shares)", f"{gk['delta']:+,.1f}")
    c[1].metric("Theta $/day", f"{gk['theta']:+,.2f}")
    c[2].metric("Vega $/vol pt", f"{gk['vega']:+,.2f}")
    c[3].metric("Gamma (shares per $1)", f"{gk['gamma']:+,.3f}")

    shift = st.slider("What-if: shift all IVs (vol points)", -15, 15, 0, key="pf_shift",
                      help="Moves today's curve and the first-expiry curve of calendars; expiry curves of fully expired legs do not change.")
    mm = pc.analyze_trade(g, m["spot"], S["today"], S["r"], S["q"], iv_shift=shift / 100.0) if shift else m
    _payoff_chart(mm, m["spot"], title=f"{m['name']}  |  IV {shift:+d} pts" if shift else m["name"])


# --------------------------------------------------------------------------------------- evaluate a trade
def _evaluate_tab(S, summary):
    ss = st.session_state
    today = S["today"]
    st.caption("Build a structure, see what it costs, where it makes and loses money, and how big to trade it. "
               "Prices you leave blank are filled from the model at the IV below, so type your real quotes where you have them.")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    und = c1.text_input("Underlying", value="QQQ", key="pf_ev_und").strip().upper() or "QQQ"
    known = float(ss.get(f"pf_spot_{und}", 0.0))
    ss.setdefault("pf_ev_spot", known)
    spot = c2.number_input("Spot", min_value=0.0, step=0.5, format="%.2f", key="pf_ev_spot")
    c2.button("Fetch live spot", key="pf_ev_fetch", on_click=_eval_spot_cb)
    iv = c3.number_input("Default IV %", min_value=1.0, max_value=300.0, value=22.0, step=0.5, key="pf_ev_iv") / 100.0
    shift = c4.slider("IV shift (vol pts)", -15, 15, 0, key="pf_ev_shift", help="Applied to every leg in the curves and statistics.")
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    tpl = t1.selectbox("Structure", pc.TEMPLATES, index=pc.TEMPLATES.index("Put calendar"), key="pf_ev_tpl")
    t2.number_input("Center strike (0 = at the money)", min_value=0.0, step=0.5, key="pf_ev_center", value=0.0)
    t3.number_input("Wing width ($)", min_value=0.5, step=0.5, value=5.0, key="pf_ev_width", disabled=tpl not in pc.NEEDS_WIDTH)
    t4.number_input("Offset from center ($)", min_value=0.0, step=0.5, value=5.0, key="pf_ev_offset", disabled=tpl not in pc.NEEDS_OFFSET)
    t5.date_input("Front / only expiry", value=pc.next_weekday(today, 4, 7), key="pf_ev_exp1")
    t6.date_input("Back expiry", value=pc.next_weekday(today, 4, 7) + dt.timedelta(days=7), key="pf_ev_exp2", disabled=tpl not in pc.NEEDS_BACK_EXPIRY)
    q1, q2 = st.columns([1, 3])
    q1.number_input("Contracts", min_value=1, step=1, value=1, key="pf_ev_qty")
    q2.write("")
    q2.button("Build legs", key="pf_ev_build", on_click=_build_cb, help="Fills the leg table below from the structure. Edit any cell afterwards.")

    edited = st.data_editor(_for_editor(ss["pf_ev_base"], ["expiry"]), num_rows="dynamic", column_config=POS_CONFIG, hide_index=True,
                            column_order=["type", "strike", "expiry", "qty", "entry_price", "iv_pct"], key=f"pf_ev_ed_{ss['pf_ev_ver']}")
    ed = edited.copy()
    ed["underlying"], ed["trade"] = und, tpl
    legs, issues = pc.normalize_positions(ed, require_price=False)
    for i in issues:
        st.warning(i)
    if legs.empty:
        st.info("Pick a structure and press Build legs (or add legs by hand).")
        return
    if spot <= 0:
        st.warning("Enter the spot price (or press Fetch live spot) to evaluate the trade.")
        return
    if not legs["type"].isin(["C", "P"]).any():
        st.warning("Add at least one option leg (type C or P) to evaluate.")
        return
    n_blank = int(legs["entry_price"].isna().sum())
    legs = pc.fill_model_prices(legs, spot, iv, today, S["r"], S["q"])
    resolved = pc.resolve_legs(legs, {und: spot}, {und: iv}, today, S["r"], S["q"])
    try:
        m = pc.analyze_trade(resolved, spot, today, S["r"], S["q"], iv_shift=shift / 100.0)
    except Exception as e:
        st.error(f"Could not evaluate this structure: {e}")
        return
    if n_blank:
        st.caption(f"{n_blank} leg price(s) were blank and use model prices at {iv * 100:.1f}% IV. Real quotes will differ, mostly by skew and bid/ask.")

    sf = m["stats_first"]
    multi = m["first_exp"] != m["last_exp"]
    st.subheader("Evaluation")
    c = st.columns(5)
    c[0].metric("Net debit" if m["cost"] >= 0 else "Net credit", _money(abs(m["cost"])))
    c[1].metric("Max loss", _money(sf["max_loss"]))
    c[2].metric(f"Max profit @ {m['first_exp']:%d %b}", _money(sf["max_profit"]))
    risk_reward = (sf["max_profit"] / m["risk"]) if np.isfinite(sf["max_profit"]) and m["risk"] > 0 else None
    c[3].metric("Reward / risk", f"{risk_reward:.2f}" if risk_reward is not None else "n/a")
    c[4].metric("Expected value", _money(m["ev_first"], plus=True),
                help="Probability-weighted P&L at the first expiry under a lognormal at the default IV, zero skew, held IVs. Model number.")
    c = st.columns(5)
    c[0].metric(f"Breakevens @ {m['first_exp']:%d %b}", _be_text(sf["breakevens"]))
    c[1].metric("Breakevens today", _be_text(m["breakevens_today"]))
    c[2].metric("Probability of profit", _pct(100 * m["pop_first"], 0))
    c[3].metric("Expected move (1σ)", f"±{m['em_first']:,.2f}")
    c[4].metric("Days to first expiry", m["dte_first"])
    gk = m["greeks"]
    c = st.columns(4)
    c[0].metric("Net delta $", _money(m["dollar_delta"], plus=True))
    c[1].metric("Theta $/day", f"{gk['theta']:+,.2f}")
    c[2].metric("Vega $/vol pt", f"{gk['vega']:+,.2f}")
    c[3].metric("Gamma (shares per $1)", f"{gk['gamma']:+,.3f}")

    units = float(np.abs(resolved["qty"]).max()) or 1.0          # contracts entered; sizing is per 1-contract set
    n_allowed, why = pc.size_trade({**m, "risk": m["risk"] / units}, summary["equity"], summary["cash"], S["risk_pct"])
    after = summary["cash"] - m["cost"]
    collateral = "" if m["cost"] >= 0 else " ({} free after holding the max loss as collateral)".format(_money(after - m["risk"]))
    risk_share = _pct(100 * m["risk"] / summary["equity"]) if summary["equity"] else "n/a"
    st.markdown("**Sizing:** as entered this risks {} = {} of equity ({}). Your {:g}% rule allows **{}** contract set(s) of this size, {}. "
                "Cash after the trade: {}{}.".format(_money(m["risk"]), risk_share, _money(summary["equity"]), S["risk_pct"],
                                                      n_allowed, why, _money(after), collateral))
    if m["unlimited_loss"]:
        st.warning("This structure has undefined risk (net short calls). Risk is shown as 2x the credit by convention; the real loss can be larger.")
    if multi:
        st.caption(f"Calendar / diagonal numbers are at the first expiry ({m['first_exp']:%d %b}): the short leg settles, the long leg is still "
                   "alive and valued with its IV held fixed. What happens to the long leg after that depends on what you do next (close or roll). "
                   "A drop in back-month IV, or a rise in front-month IV, moves the real result; use the IV shift to stress it.")
    _payoff_chart(m, spot, title=f"{tpl} on {und}")
    st.button("Add to portfolio", key="pf_ev_add", on_click=_add_cb, args=(legs[pc.POSITION_COLS], f"{und} {tpl.lower()} {m['first_exp']:%d%b}"),
              help="Copies these legs (with the prices shown) into your positions.")


# --------------------------------------------------------------------------------------- closed trades
def _closed_tab(S):
    ss = st.session_state
    st.caption("Realised results. Pressing 'Close this trade at current P&L' on a position adds a row here; edit the amount to match your real fill.")
    up = st.file_uploader("Import closed trades CSV (trade, underlying, close_date, realized_pnl)", type="csv", key="pf_cl_up")
    if up is not None:
        sig = ("closed", up.name, up.size)
        if sig not in ss["pf_imported"]:
            try:
                ss["pf_closed_base"] = pc.normalize_closed(pd.read_csv(up))
                ss["pf_closed_ver"] += 1
                ss["pf_imported"].add(sig)
                st.rerun()
            except Exception as e:
                ss["pf_imported"].add(sig)
                st.error(f"Could not read that CSV: {e}")
    edited = st.data_editor(_for_editor(ss["pf_closed_base"], ["close_date"]), num_rows="dynamic", column_config=CLOSED_CONFIG,
                            hide_index=True, key=f"pf_cl_ed_{ss['pf_closed_ver']}")
    ss["pf_closed_cur"] = edited
    closed = pc.normalize_closed(edited)
    if not closed.empty:
        st.download_button("Download closed trades CSV", closed.to_csv(index=False).encode("utf-8"), file_name="closed_trades.csv",
                           mime="text/csv", key="pf_cl_dl")
    return closed


def _closed_stats_block(closed):
    s = pc.closed_stats(closed)
    if s["n"] == 0:
        st.info("No closed trades yet.")
        return
    c = st.columns(6)
    c[0].metric("Closed trades", s["n"])
    c[1].metric("Realised P&L", _money(s["total"], plus=True))
    c[2].metric("Win rate", _pct(100 * s["win_rate"], 0))
    c[3].metric("Avg win / loss", f"{_money(s['avg_win']) if s['avg_win'] is not None else 'n/a'} / {_money(s['avg_loss']) if s['avg_loss'] is not None else 'n/a'}")
    pf = s["profit_factor"]
    c[4].metric("Profit factor", "n/a" if pf is None else ("inf" if not np.isfinite(pf) else f"{pf:.2f}"))
    c[5].metric("Expectancy / trade", _money(s["expectancy"], plus=True))
    dated = closed.dropna(subset=["close_date"]).sort_values("close_date")
    if len(dated) > 1:
        curve = dated.assign(cumulative=dated["realized_pnl"].cumsum()).set_index("close_date")["cumulative"]
        st.line_chart(curve, height=220)
        st.caption("Cumulative realised P&L by close date.")


# ------------------------------------------------------------------------------------------- header block
def _header(summary):
    c = st.columns(6)
    c[0].metric("Cash", _money(summary["cash"]), _pct(summary["cash_pct"], 0) + " of equity" if summary["cash_pct"] is not None else None, delta_color="off")
    c[1].metric("Open P&L", _money(summary["unrealized"], plus=True))
    c[2].metric("Equity", _money(summary["equity"]))
    ret = summary["total_return_pct"]
    c[3].metric("Portfolio return", _pct(ret, 2) if ret is not None else "n/a",
                _money(summary["total_pnl"], plus=True) + " total P&L", delta_color="off",
                help="(Equity - capital - deposits) / (capital + deposits). Simple, not time-weighted.")
    c[4].metric("Capital at risk", _money(summary["at_risk"]), _pct(summary["at_risk_pct"], 0) + " of equity" if summary["at_risk_pct"] is not None else None, delta_color="off")
    c[5].metric("Free cash", _money(summary["free_cash"]), "after credit-trade collateral", delta_color="off",
                help="Cash minus the max loss of open credit trades (a stand-in for broker margin; check your broker's real requirement).")
    with st.expander("Portfolio detail: realised vs open, return on risk, net Greeks"):
        g = summary["greeks"]
        d = st.columns(5)
        d[0].metric("Realised P&L", _money(summary["realized"], plus=True))
        d[1].metric("Open P&L", _money(summary["unrealized"], plus=True))
        d[2].metric("Return on capital at risk", _pct(summary["return_on_risk_pct"], 1) if summary["return_on_risk_pct"] is not None else "n/a")
        d[3].metric("Open trades", summary["n_trades"])
        d[4].metric("Market value of positions", _money(summary["market_value"]))
        e = st.columns(4)
        e[0].metric("Net delta $", _money(summary["dollar_delta"], plus=True))
        e[1].metric("Theta $/day", f"{g['theta']:+,.2f}")
        e[2].metric("Vega $/vol pt", f"{g['vega']:+,.2f}")
        e[3].metric("Gamma (shares per $1)", f"{g['gamma']:+,.3f}")
        if summary["cash_calc"] != summary["cash"]:
            st.caption(f"Using your broker cash. The tab's own calculation gives {_money(summary['cash_calc'])}.")


def _notes():
    with st.expander("How this works and what to trust"):
        st.markdown("**Method.** Each trade is priced leg by leg with Black-Scholes (floored at intrinsic as a cheap American-exercise "
                    "stand-in). If you give a mark, the IV is backed out of it so the curve passes through today's price; otherwise the "
                    "default IV is used. Payoff curves are drawn today, at the first expiry (calendars: the short leg has settled, the long "
                    "leg is valued with its IV held fixed) and at the last expiry. Breakevens are the zero crossings. Probability of profit and "
                    "expected value use a lognormal at the default IV with no skew.")
        st.markdown("**Advisor.** A fixed checklist: stop-loss, profit target, time exit by trade type, tested / adjust, then hold. "
                    "Thresholds are the ones you set under Account, model and rules; the defaults (calendars +25% of debit, credit "
                    "trades 50% of credit, long premium +50%, stop at -50% of risk) are common rules of thumb, not tested edges. "
                    "It does not know about earnings, dividends, assignment risk or your view.")
        st.markdown("**Cash and return.** Cash = starting capital + deposits + realised P&L - cost of open trades (credits add cash). "
                    "It is not broker buying power. Return is simple, not time-weighted, and open P&L is mark-to-model unless you typed or fetched marks. "
                    "Yahoo quotes are delayed and its IVs are often stale after hours, so check marks against your broker.")
        st.markdown("**Persistence.** Positions live in this browser session. Use Download positions CSV / Download closed trades CSV to keep them, "
                    "and import them next time. Research tool, not financial advice.")


# ------------------------------------------------------------------------------------------------ render
def render():
    try:
        import portfolio_core  # noqa: F401
    except ImportError:
        st.error("portfolio_core.py is missing. Copy it into the same folder as app.py and reboot the app.")
        return
    _init()
    st.subheader("Option portfolio analyzer")
    st.caption("Cash, per-trade BOOK PROFIT / HOLD / CUT LOSS / ADJUST calls, breakevens, trade evaluation and portfolio return. "
               "Model-based and rules-based: a discipline aid, not a signal.")
    S = _settings()
    header = st.container()
    tab_pos, tab_eval, tab_closed = st.tabs(["Positions & advice", "Evaluate a trade", "Closed trades & return"])
    with tab_closed:
        closed = _closed_tab(S)
    with tab_pos:
        rows, _by = _positions_tab(S)
    realized = pc.closed_stats(closed)["total"]
    summary = pc.portfolio_summary(rows, realized, S["capital"], S["deposits"], S["cash_override"])
    with header:
        _header(summary)
    with tab_eval:
        _evaluate_tab(S, summary)
    with tab_closed:
        st.subheader("Results")
        _closed_stats_block(closed)
        st.markdown(f"**Portfolio return so far:** {_pct(summary['total_return_pct'], 2)} on "
                    f"{_money(summary['base'])} of capital (realised {_money(summary['realized'], plus=True)}, "
                    f"open {_money(summary['unrealized'], plus=True)}).")
    _notes()
