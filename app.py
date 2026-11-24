"""Streamlit UI for the layered QQQ regime scorer.  Run:  streamlit run app.py
All logic lives in core.py; this file only collects inputs and displays results."""
import datetime as dt
import importlib
import inspect
import io
import os

import numpy as np
import pandas as pd
import streamlit as st

import core
import vanna_charm
import screener_tab
import ichimoku_tab
import gamma_tab
import chat_tab

st.set_page_config(page_title="Regime & Strategy Selector", layout="wide")

# app.py and core.py must come from the same release. A stale cached module is reloaded once; a genuinely
# old core.py file stops the app with a clear message instead of a KeyError deep inside a tab.
EXPECTED_CORE = "2026-11-23.1"
if getattr(core, "CORE_VERSION", None) != EXPECTED_CORE:
    core = importlib.reload(core)
if getattr(core, "CORE_VERSION", None) != EXPECTED_CORE:
    st.error(f"core.py is out of date: found version {getattr(core, 'CORE_VERSION', 'none (pre-2026-11-05)')}, "
             f"but app.py needs {EXPECTED_CORE}. Copy BOTH app.py and core.py from the same zip into the app "
             "folder (or repo), commit/save, then reboot the app so Streamlit drops its cached copy.")
    st.stop()

st.title("Regime & Strategy Selector")
st.caption("Layered scorer: direction (price / flow / DEX) -> gates (IV rank, term structure) -> "
           "gamma regime -> strategy matrix. Research tool, not financial advice.")

KIND_LABELS = {
    "history": "Option history", "chain": "Option chain", "flow": "Options flow",
    "unusual": "Unusual activity", "dex_exp": "DEX by expiration", "gex_exp": "GEX by expiration",
    "gex_strike": "GEX by strike", "exp_move": "Expected move", "ohlc": "Price OHLC (optional)",
}


def fmt_big(x: float) -> str:
    if x is None or np.isnan(x):
        return "n/a"
    return f"{x / 1e9:+.2f}B" if abs(x) >= 1e9 else f"{x / 1e6:+,.0f}M"


def fmt_contracts(x: float) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:+,.0f}"


def fmt_sh(x: float) -> str:
    """Dealer hedge flow in millions of shares, signed (+ = dealers buy)."""
    return "n/a" if x is None or np.isnan(x) else f"{x / 1e6:+.2f}M sh"


VC_TREND_COLS = {"vanna_put_sh_per_pt": "Put-side vanna (M shares per vol pt)",
                 "charm_put_sh_per_day": "Put-side charm (M shares per day)",
                 "vanna_call_sh_per_pt": "Call-side vanna, if dealers are long the calls (M shares per vol pt)",
                 "charm_call_sh_per_day": "Call-side charm, if dealers are long the calls (M shares per day)"}


def vc_trend_ui(full) -> None:
    """Trend of the put-side vanna / charm readings, from the daily snapshot log (one point per as-of date)."""
    if full is None or not {"vanna_put_sh_per_pt", "charm_put_sh_per_day"} <= set(full.columns):
        st.caption("Trend: no vanna / charm readings in the snapshot log yet.")
        return
    cols = [k for k in VC_TREND_COLS if k in full.columns]
    t = full[["asof", *cols]].copy()
    t["asof"] = pd.to_datetime(t["asof"], errors="coerce")
    for k in cols:
        t[k] = pd.to_numeric(t[k], errors="coerce") / 1e6
    t = t.dropna(subset=["asof"]).dropna(subset=cols, how="all").set_index("asof").sort_index()
    if len(t) < 2:
        st.caption("Trend: one chain file is one day, so there is only one reading so far. Save a snapshot "
                   "(Dealer map tab, Daily snapshot log) after each new export and the chart fills in.")
        return
    st.markdown("**Trend of the vanna / charm readings (daily snapshot log)**")
    slots = []
    for i in range(0, len(cols), 2):
        slots += list(st.columns(2))
    for col, k in zip(slots, cols):
        label = VC_TREND_COLS[k]
        v = t[k].dropna()
        with col:
            st.caption(label)
            st.line_chart(v.rename(label).to_frame(), height=160)
            if len(v) >= 2:
                st.caption(f"latest {v.iloc[-1]:+.2f}, previous reading {v.iloc[-2]:+.2f} "
                           f"({v.iloc[-1] - v.iloc[-2]:+.2f}); {len(v)} readings")


def tag_text(tag: str) -> str:
    return f"{core.oi_arrow(tag)} {tag}"


def tag_color(tag: str) -> str:
    return "blue" if "build" in tag else ("orange" if "erode" in tag else "gray")


def leg_trend(leg, p: dict) -> str:
    if np.isnan(leg.oi_chg) or np.isnan(leg.oi):
        return "unknown"
    return tag_text(core.oi_tag(leg.oi_chg, leg.oi - leg.oi_chg, p["build_pct"], p["strong_pct"], p["min_contracts"]))


def show_vega(spec: dict) -> None:
    """Render a Vega-Lite spec across Streamlit versions (width API changed over time)."""
    try:
        params = inspect.signature(st.vega_lite_chart).parameters
    except (TypeError, ValueError):
        params = {}
    if "width" in params:
        try:
            st.vega_lite_chart(spec=spec, width="stretch")
            return
        except Exception:
            pass
    if "use_container_width" in params:
        st.vega_lite_chart(spec=spec, use_container_width=True)
    else:
        st.vega_lite_chart(spec=spec)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    folder = st.text_input("Folder with the CSV files", value=os.environ.get("QQQ_DATA_DIR", ""), key="folder",
                           help="Files are recognised by their columns, so file names don't matter.")
    uploads = st.file_uploader("...or upload the CSVs", type=["csv"], accept_multiple_files=True, key="uploads")
    flow_sym = st.text_input("Flow / unusual-activity symbol", value="", key="flow_symbol",
                             help="Leave blank to use the most common ticker in those files. Set it (for example QQQ) "
                                  "if a file mixes several underlyings. It should match the underlying of your chain files.")
    st.caption("Optional: add a daily QQQ OHLC export (TradingView, Yahoo or Barchart) for true candlesticks.")

    st.header("Direction weights")
    w_price = st.slider("Price structure (200 EMA, Donchian)", 0, 100, int(core.DEFAULT_WEIGHTS["price"]), key="w_price")
    w_flow = st.slider("Flow", 0, 100, int(core.DEFAULT_WEIGHTS["flow"]), key="w_flow")
    w_dex = st.slider("DEX", 0, 100, int(core.DEFAULT_WEIGHTS["dex"]), key="w_dex")
    st.caption("GEX is not part of direction; it selects the regime. Weights renormalise over available components.")
    contrarian = st.checkbox("Treat flow as contrarian", value=False, key="contrarian",
                             help="The 19-month backtest found put/call flow was mildly contrarian.")

    st.header("Bands and regime")
    t_mild = st.number_input("Mild threshold (|score|)", 0.0, 1.0, 0.10, 0.01, key="t_mild")
    t_strong = st.number_input("Strong threshold (|score|)", 0.0, 1.0, 0.30, 0.01, key="t_strong")
    reg_min = st.number_input("Regime window: min DTE", 0, 30, 1, key="reg_min")
    reg_max = st.number_input("Regime window: max DTE", 5, 400, 45, key="reg_max")
    flip_buf = st.number_input("Near-flip buffer (% of spot)", 0.0, 5.0, 1.0, 0.1, key="flip_buf")
    edge_thr = st.number_input("Min front-back IV edge for calendars (vol pts)", 0.0, 10.0, 1.5, 0.1, key="edge_thr")
    top_n = st.number_input("Strategies to show", 1, 8, 5, key="top_n")

    with st.expander("Structure parameters"):
        vert_dte = st.number_input("Verticals / BWB / condor: target DTE", 2, 120, 18, key="vert_dte")
        em_fraction = st.number_input("Short strike at fraction of expected move", 0.1, 1.5, 0.75, 0.05, key="em_fraction")
        width = st.number_input("Credit spread / condor wing width (dollars)", 1, 50, 10, key="width")
        cal_front = st.number_input("Calendar front leg: target DTE", 2, 90, 10, key="cal_front")
        cal_gap = st.number_input("Calendar back leg: days after front", 7, 180, 30, key="cal_gap")
        diag_front = st.number_input("Diagonal short leg: target DTE", 1, 60, 5, key="diag_front")
        diag_gap = st.number_input("Diagonal long leg: days after short", 3, 120, 7, key="diag_gap")

    with st.expander("Open-interest change (display only)"):
        oi_build = st.number_input("Build / erode threshold (% of prior OI)", 1, 100, 10, key="oi_build")
        oi_strong = st.number_input("Strong threshold (% of prior OI)", 5, 500, 25, key="oi_strong")
        oi_min = st.number_input("Minimum change (contracts)", 0, 100000, 500, key="oi_min")
        oi_band = st.number_input("Strike band around spot (%)", 2, 30, 8, key="oi_band")
        st.caption("A move must clear both the % and the contract threshold to be tagged. These are judgment "
                   "calls, not fitted values. OI change is context only and never enters the score.")

    with st.expander("Vanna / charm and squeeze alert"):
        vc_conv = st.selectbox("Dealer positioning assumption", list(vanna_charm.CONVENTIONS), key="vc_conv",
                               format_func=lambda k: vanna_charm.CONVENTIONS[k],
                               help="Open interest does not say who is long or short. Customers owning OTM puts "
                                    "(dealers short) is the defensible assumption; OTM calls could go either way, so "
                                    "they are excluded by default and shown as a +/- range.")
        vc_max = st.number_input("Vanna/charm window: max DTE", 5, 800, int(vanna_charm.DEFAULTS["max_dte"]), key="vc_max",
                                 help="Back-month IV usually moves less than front-month IV, so very long dated "
                                      "open interest is left out by default.")
        vc_band = st.number_input("Near-spot band (%)", 2, 30, int(vanna_charm.DEFAULTS["near_band"] * 100), key="vc_band")
        st.caption("Squeeze alert thresholds (judgment calls, not fitted values):")
        sq_ret5 = st.number_input("Price stretched: 5-session drop (%)", 1.0, 20.0, vanna_charm.DEFAULTS["sq_ret5"] * 100, 0.5, key="sq_ret5")
        sq_ret20 = st.number_input("... or 20-session drop (%)", 2.0, 40.0, vanna_charm.DEFAULTS["sq_ret20"] * 100, 0.5, key="sq_ret20")
        sq_dex = st.number_input("DEX washout: bottom X% of 120 sessions", 1, 50, int(vanna_charm.DEFAULTS["sq_dex_pct"] * 100), key="sq_dex")
        sq_ivr = st.number_input("Vol elevated: IV rank at least", 0, 200, int(vanna_charm.DEFAULTS["sq_ivrank"]), key="sq_ivr")
        sq_recent = st.number_input("Washout must be within the last N sessions", 1, 20, int(vanna_charm.DEFAULTS["sq_recent"]), key="sq_recent")
        sq_min_v = st.number_input("Min put-side vanna (M shares per vol pt)", 0.0, 10.0, vanna_charm.DEFAULTS["sq_min_vanna"] / 1e6, 0.05, key="sq_min_v")
        sq_min_c = st.number_input("Min put-side charm (M shares per day)", 0.0, 10.0, vanna_charm.DEFAULTS["sq_min_charm"] / 1e6, 0.05, key="sq_min_c")
        sq_rng = st.number_input("Vol-turn trigger: 10-session IV30 range at least (pts)", 0.0, 30.0, vanna_charm.DEFAULTS["sq_iv_range"], 0.5, key="sq_rng")

# ------------------------------------------------------------------ load data
data = {}
if folder.strip():
    if os.path.isdir(folder.strip()):
        data.update(core.load_folder(folder.strip()))
    else:
        st.sidebar.error("That folder does not exist.")
if uploads:
    data.update(core.load_uploads(uploads))

cols = st.columns(len(KIND_LABELS))
for c, (k, label) in zip(cols, KIND_LABELS.items()):
    c.caption(("OK  " if k in data else "missing  ") + label)

if "history" not in data or "chain" not in data:
    tab_need, tab_scr0, tab_ich0, tab_gam0, tab_chat0 = st.tabs(["Score & regime", "Donchian screener", "Ichimoku screener", "Gamma screener", "Chat"])
    with tab_need:
        st.info("Point the sidebar at the folder with your CSVs (or upload them). The option history and option chain "
                "files are required; the others refine the score, regime and charts and are used when present.")
    with tab_scr0:                      # the screener needs no option files, so it works before they are loaded
        screener_tab.render(show_vega)
    with tab_ich0:
        ichimoku_tab.render(show_vega)
    with tab_gam0:
        gamma_tab.render()
    with tab_chat0:                     # no option data yet: the chat only knows about any screener results
        chat_tab.render(None)
    st.stop()

weights = {"price": float(w_price), "flow": float(w_flow), "dex": float(w_dex)}
if sum(weights.values()) <= 0:
    st.error("All weights are zero.")
    st.stop()

opts = {
    "flow_contrarian": contrarian, "flow_min_dte": 1, "t_mild": float(t_mild), "t_strong": float(t_strong),
    "reg_min_dte": int(reg_min), "reg_max_dte": int(reg_max), "flip_buffer": float(flip_buf) / 100.0,
    "edge_threshold": float(edge_thr), "top_n": int(top_n),
    "structure": {"vert_dte": int(vert_dte), "em_fraction": float(em_fraction), "width": float(width),
                  "cal_front_dte": int(cal_front), "cal_gap": int(cal_gap), "diag_front_dte": int(diag_front),
                  "diag_gap": int(diag_gap)},
    "oi_build_pct": float(oi_build) / 100.0, "oi_strong_pct": float(oi_strong) / 100.0,
    "oi_min_contracts": float(oi_min), "oi_band": float(oi_band) / 100.0,
    "flow_symbol": flow_sym.strip(),
    "vanna_charm": {
        "convention": vc_conv, "max_dte": int(vc_max), "near_band": float(vc_band) / 100.0,
        "sq_ret5": float(sq_ret5) / 100.0, "sq_ret20": float(sq_ret20) / 100.0, "sq_dex_pct": float(sq_dex) / 100.0,
        "sq_ivrank": float(sq_ivr), "sq_recent": int(sq_recent), "sq_min_vanna": float(sq_min_v) * 1e6,
        "sq_min_charm": float(sq_min_c) * 1e6, "sq_iv_range": float(sq_rng),
    },
}

try:
    R = core.analyse(data, weights, opts)
except Exception as e:  # show a readable error instead of a stack trace
    st.error(f"Could not analyse these files: {e}")
    st.stop()

asof = R["asof"]
age = (pd.Timestamp(dt.date.today()) - asof).days
if age >= 1:
    st.warning(f"Data is as of {asof.date()} ({age} day(s) old). Quotes, strikes and DTEs below reflect that date; "
               "re-export the files before trading.")

d, g = R["direction"], R["regime"]


def trend_card_ui(title, main_text, main_note, tc, kind):
    """One bordered card: headline value, trend verdict (strengthening / weakening / steady), sparkline."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.metric(main_note, main_text)
        if tc is None:
            st.caption("Not enough history to judge the trend.")
            return
        sign, trend = tc["sign"], tc["trend"]
        if trend == "steady":
            color, verdict = "gray", f"{sign}, steady"
        else:
            color = "red" if (sign == "Negative" and trend == "strengthening") else (
                "green" if (sign == "Positive" and trend == "strengthening") else "orange")
            verdict = f"{sign}, {trend}"
        st.markdown(f":{color}[**{verdict}**]")
        if kind == "gex":
            meaning = {
                ("Negative", "strengthening"): "Dealer hedging increasingly amplifies moves.",
                ("Negative", "weakening"): "Amplification is fading; moves feed on themselves less.",
                ("Positive", "strengthening"): "Pinning / mean-reversion pressure is building.",
                ("Positive", "weakening"): "Pinning pressure is fading toward zero.",
            }.get((sign, trend), "No meaningful change over the last week.")
        else:
            meaning = {
                ("Negative", "strengthening"): "Put-heavy (bearish-tilted) delta exposure is growing.",
                ("Negative", "weakening"): "Put-heavy delta exposure is easing toward neutral.",
                ("Positive", "strengthening"): "Call-heavy (bullish-tilted) delta exposure is growing.",
                ("Positive", "weakening"): "Call-heavy delta exposure is fading toward neutral.",
            }.get((sign, trend), "No meaningful change over the last week.")
        st.caption(meaning)
        st.caption(f"{tc['short']}d change {fmt_big(tc['d_short'])}  |  {tc['long']}d change {fmt_big(tc['d_long'])}"
                   f"{'' if tc['agree'] else '  (mixed)'}  |  {tc['pctile']:.0%} percentile of last 120 days  |  "
                   f"{tc['days_in_sign']} session(s) in this sign")
        spark = (tc["tail"] / 1e9).rename("history (billions)").to_frame()
        st.line_chart(spark, height=130)


# ------------------------------------------------------------------ tabs
tab_over, tab_strat, tab_dealer, tab_flow, tab_vol, tab_bt, tab_scr, tab_ich, tab_gam, tab_chat = st.tabs(
    ["Score & regime", "Strategies", "Dealer map", "Flow", "Volatility", "Backtest", "Donchian screener",
     "Ichimoku screener", "Gamma screener", "Chat"])

with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment", d["band"])
    c1.caption(f"score {d['score']:+.2f}")
    c2.metric("Gamma regime", g["effective"] + (" (near flip)" if g["near_flip"] else ""))
    c2.caption(f"net GEX {fmt_big(g['net'])} ({g['source']})")
    c3.metric("Spot (parity-implied)", f"{R['spot']:.2f}")
    c3.caption("from the option chain")
    c4.metric("IV rank", f"{R['ivrank']:.0f}" if R["ivrank"] is not None else "n/a")
    c4.caption(f"front-back IV edge {R['edge']['edge_pts']:+.1f} pts")

    # ---- short-squeeze / relief-rally alert (details in the "Vanna & charm positioning" section below)
    sq = R["squeeze"]
    if sq.get("ok"):
        banner = {"alert": st.success, "armed": st.warning, "watch": st.info}.get(sq["level"])
        if banner is not None:
            banner(f"**{sq['headline']}**")
        else:
            st.caption(sq["headline"])
    else:
        st.caption(f"Squeeze alert unavailable: {sq.get('reason', 'unknown reason')}")

    comp_rows = []
    for k in ("price", "flow", "dex"):
        v = R["components"][k]
        comp_rows.append({"component": k, "score (-1..+1)": None if np.isnan(v) else round(v, 3),
                          "weight used": round(d["used"].get(k, 0.0), 2)})
    st.dataframe(pd.DataFrame(comp_rows), hide_index=True)
    dex_exp_txt = "n/a" if np.isnan(R["dex_exp"]) else f"{R['dex_exp']:+.2f}"
    st.caption(f"Flow source: {R['flow_source']}.  DEX component = average of history z-score ({R['dex_hist']:+.2f}) "
               f"and DEX-by-expiration ({dex_exp_txt}).")
    if not np.isnan(d["agreement"]):
        st.caption(f"Component agreement with the composite sign: {d['agreement']:.0%}.")

    st.warning("Backtest caveat: over the 19 months in the history file the composite score was not a reliable "
               "direction signal (mildly contrarian, unstable between halves). Use it for structure selection and "
               "sizing, not as a standalone call. See the Backtest tab.")

    # ---- dealer-positioning cards
    st.subheader("Dealer positioning")
    k1, k2, k3 = st.columns(3)
    with k1:
        trend_card_ui("Gamma (GEX)", fmt_big(g["net"]), "Net GEX, trade window", R["gex_trend"], "gex")
        st.caption("Headline = net GEX over the regime window. Trend = history file, all expiries, open-interest based.")
    with k2:
        dw = R["dex_window"]
        trend_card_ui("Delta (DEX)", fmt_big(R["dex_trend"]["value"]) if R["dex_trend"] else "n/a",
                      "Net DEX, all expiries", R["dex_trend"], "dex")
        if not np.isnan(dw):
            st.caption(f"Trade-window DEX from the expiration file: {fmt_big(dw)}.")
    with k3:
        with st.container(border=True):
            st.markdown("**Aggregate gamma flip**")
            af = R["agg_flip"]
            use_profile = bool(af) and not np.isnan(af["flip"])
            flip = af["flip"] if use_profile else g["flip"]
            label = "All-expiry flip (GEX profile zero-crossing)" if use_profile else "Gamma-weighted flip level"
            if np.isnan(flip):
                st.metric(label, "n/a")
                st.caption("Add the GEX-by-strike file (profile column) or the GEX-by-expiration file to compute the flip.")
            else:
                dist = R["spot"] - flip
                st.metric(label, f"{flip:.1f}")
                if dist < 0:
                    st.markdown(f":red[**Spot is {abs(dist):.1f} pts ({abs(dist) / R['spot'] * 100:.1f}%) below the flip**]")
                    st.caption("Below the flip dealers are net short gamma: hedging tends to amplify moves.")
                else:
                    st.markdown(f":green[**Spot is {dist:.1f} pts ({dist / R['spot'] * 100:.1f}%) above the flip**]")
                    st.caption("Above the flip dealers are net long gamma: hedging tends to damp moves.")
                if use_profile:
                    st.caption(f"GEX profile at spot: {fmt_big(af['value_at_spot'])} ({af['state_at_spot']} gamma zone).")
                if g["near_flip"]:
                    st.markdown(":orange[**Inside the near-flip buffer: regime is unstable, treated as Negative.**]")
                if not np.isnan(g["flip"]):
                    nd = g["flip_nearest_date"]
                    st.caption(f"Gamma-weighted flip across expiries in the trade window: {g['flip']:.1f} "
                               f"(range {g['flip_min']:.0f} to {g['flip_max']:.0f}; nearest expiry "
                               f"{nd.date() if nd is not None else 'n/a'}: {g['flip_nearest']:.1f}). "
                               "The near-flip check uses this figure. Model estimates.")

    # ---- GEX intensity gauge: where today's net GEX sits inside its own historical range
    with st.container(border=True):
        st.markdown("**GEX intensity gauge**")
        lookbacks = {"60 sessions": 60, "120 sessions": 120, "1 year (252 sessions)": 252, "Full history": None}
        g_left, g_right = st.columns([1, 1])
        with g_right:
            lb_choice = st.selectbox("Lookback window", list(lookbacks), index=2, key="gex_gauge_lookback",
                                     help="History the gauge is scaled against. A shorter window makes it react "
                                          "to recent extremes; a longer one measures against the full range.")
        gi = core.gex_intensity(R["hist"].set_index("Date")["GEX Net OI"], window=lookbacks[lb_choice])
        if gi is None:
            with g_left:
                st.caption("Not enough GEX history in the history file to draw the gauge (needs 30+ sessions).")
        else:
            svg = core.gex_gauge_svg(gi)
            with g_left:
                if hasattr(st, "html"):
                    st.html(f'<div style="color:inherit">{svg}</div>')
                else:
                    st.markdown(svg, unsafe_allow_html=True)
            with g_right:
                st.markdown(f"**{gi['label']}**")
                st.caption(gi["meaning"])
                m1, m2 = st.columns(2)
                m1.metric("Net GEX (history file)", fmt_big(gi["value"]))
                m2.metric("Percentile in window", f"{gi['pctile']:.0%}")
                st.caption(f"Window range {fmt_big(gi['hist_min'])} to {fmt_big(gi['hist_max'])} over "
                           f"{gi['n']} sessions.")
        st.caption("Scale: -100 is the most negative net GEX in the window, +100 the most positive, 0 the zero-gamma "
                   "line (positive and negative sides scaled separately). All-expiry, open-interest based, using the "
                   "standard dealer convention (long calls, short puts), so treat it as a model estimate. Display "
                   "only: it does not feed the direction score or the strategy matrix.")

    # ---- vanna & charm positioning and the short-squeeze checklist
    st.subheader("Vanna & charm positioning")
    vc, sq = R["vanna_charm"], R["squeeze"]
    vc_trend_slot = None
    if not vc.get("ok"):
        st.info(vc.get("reason", "Vanna / charm positioning is unavailable."))
    else:
        lo, hi = vc["window"]
        a1, a2, a3 = st.columns(3)
        a1.metric("Vanna: dealer buying if IV falls 1 pt", fmt_sh(vc["put_vanna"]))
        a1.caption(f"OTM puts, dealers short. About ${vc['put_vanna'] * R['spot'] / 1e9:.2f}B per vol point; "
                   f"{fmt_sh(vc['put_vanna_near'])} of it within {vc['near_band']:.0%} of spot.")
        a2.metric("Charm: dealer buying per day", fmt_sh(vc["put_charm"]))
        a2.caption(f"Put deltas decay toward zero as time passes. A weekend (3 days) is roughly "
                   f"{fmt_sh(3 * vc['put_charm'])}; {fmt_sh(vc['put_charm_near'])}/day within {vc['near_band']:.0%} of spot.")
        a3.metric("Call side (dealer side unknown)", f"\u00b1{abs(vc['call_vanna']) / 1e6:.2f}M sh")
        a3.caption(f"Vanna \u00b1{abs(vc['call_vanna']) / 1e6:.2f}M per vol pt and charm \u00b1{abs(vc['call_charm']) / 1e6:.2f}M/day: "
                   "positive if dealers are long the calls, negative if short. Not in the headline.")
        if vc["convention"] != "puts":
            st.caption(f"Selected assumption: {vc['convention_label']}. Net vanna {fmt_sh(vc['net_vanna'])} per vol pt, "
                       f"net charm {fmt_sh(vc['net_charm'])} per day.")
        st.caption(f"Window {lo}-{hi} DTE, OTM options only, positive = dealers buy the underlying. Vanna and charm only "
                   "become buying when implied vol actually falls or time passes; they are potential flow, not a price forecast.")

        vc_trend_slot = st.container()      # filled further down, once the snapshot log has been loaded

        z1, z2 = st.columns(2)
        with z1:
            st.markdown("**IV-drop scenarios**")
            scen = vc["scenarios"].copy()
            st.dataframe(scen.round(2), hide_index=True)
            st.markdown("**By put strike (distance below spot)**")
            bz = vc["by_zone"].copy()
            bz["vanna (M sh/pt)"] = bz["vanna_sh_per_pt"] / 1e6
            bz["charm (M sh/day)"] = bz["charm_sh_per_day"] / 1e6
            st.dataframe(bz[["put strikes", "vanna (M sh/pt)", "charm (M sh/day)"]].round(3), hide_index=True)
        with z2:
            st.markdown("**By expiry**")
            bd = vc["by_dte"].copy()
            bd["vanna (M sh/pt)"] = bd["vanna_sh_per_pt"] / 1e6
            bd["charm (M sh/day)"] = bd["charm_sh_per_day"] / 1e6
            st.dataframe(bd[["expiry bucket", "vanna (M sh/pt)", "charm (M sh/day)"]].round(3), hide_index=True)
            st.markdown("**Largest put-side vanna strikes**")
            tp = vc["top"].copy()
            tp["exp"] = tp["exp"].dt.date
            tp["vanna (k sh/pt)"] = tp["vflow"] / 1e3
            tp["charm (k sh/day)"] = tp["cflow"] / 1e3
            st.dataframe(tp[["exp", "strike", "open_interest", "vanna (k sh/pt)", "charm (k sh/day)"]].round(1), hide_index=True)

    st.subheader("Short-squeeze checklist")
    st.caption("A dealer-driven relief rally / short-covering move, not a stock short-interest squeeze. It needs a washout, "
               "dealer positioning that supports a bounce, and a trigger: implied vol rolling over while price stabilises.")
    if not sq.get("ok"):
        st.info(sq.get("reason", "The alert could not be evaluated."))
    else:
        chk = sq["checks"].copy()
        chk["met"] = chk["met"].map({True: "\u2713", False: "\u2717"})
        st.dataframe(chk.rename(columns={"group": "stage", "met": "met?"}), hide_index=True)
        st.caption(f"Washout seen in the last {sq['recent_window']} sessions: {'yes' if sq['setup_recent'] else 'no'}. "
                   "Level: none -> watch (washout) -> armed (IV rolling over) -> alert (price stabilising with dealer support).")
        if sq["levels"]:
            st.caption("Levels to watch: " + "  |  ".join(
                f"{n} {v:.1f}" + ("" if n == "Spot" else f" ({v - R['spot']:+.1f} pts)") for n, v in sq["levels"]))
        st.markdown("**30d IV, last 60 sessions** (the trigger needs it to fall two sessions in a row)")
        st.line_chart(sq["series"][["iv30"]].tail(60).rename(columns={"iv30": "IV30"}), height=160)
        with st.expander("How often did this fire in the history file?"):
            hh = sq["history"]
            rows = [{"signal": f"All sessions (baseline, {hh['n_days']} days)", "events": hh["n_days"],
                     **{f"{n[1:]}d mean": hh["baseline"][n] for n in ("f5", "f10", "f20")},
                     **{f"{n[1:]}d up": hh["baseline_hit"][n] for n in ("f5", "f10", "f20")}}]
            for key, label in (("trigger", "IV falling 2 sessions + price up"),
                               ("washout_then_trigger", "Washout, then the trigger (the alert)")):
                st_ = hh[key]["stats"]
                rows.append({"signal": label, "events": hh[key]["n_events"],
                             **{f"{n[1:]}d mean": st_[n]["mean"] for n in ("f5", "f10", "f20")},
                             **{f"{n[1:]}d up": st_[n]["hit"] for n in ("f5", "f10", "f20")}})
            ev = pd.DataFrame(rows)
            for c in ev.columns:
                if c.endswith("mean"):
                    ev[c] = ev[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1%}")
                elif c.endswith("up"):
                    ev[c] = ev[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.0%}")
            st.dataframe(ev, hide_index=True)
            e2 = hh["washout_then_trigger"]["events"].copy()
            if len(e2):
                e2.index = pd.to_datetime(e2.index).date
                e2 = e2.rename(columns={"px": "close", "iv30": "IV30", "ivrank": "IV rank", "f5": "next 5d",
                                        "f10": "next 10d", "f20": "next 20d"})
                for c in ("next 5d", "next 10d", "next 20d"):
                    e2[c] = e2[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.1%}")
                st.dataframe(e2.round(2))
            st.caption("Events are de-clustered (at least 10 sessions apart). Samples this small say little: compare each "
                       "row with the baseline before reading anything into it. Thresholds are judgment calls, not fitted.")

    # ---- open-interest change (display only; not part of the score)
    st.subheader("Open-interest change")
    oi = R["oi"]
    if oi is None:
        st.info("The option chain has no usable oi_chg column, so open-interest changes are unavailable.")
    else:
        sm = oi["summary"]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Net call OI change", fmt_contracts(sm["net_call_chg"]))
        q1.caption(f"adds {sm['call_adds']:,.0f} / cuts {sm['call_cuts']:,.0f}")
        q2.metric("Net put OI change", fmt_contracts(sm["net_put_chg"]))
        q2.caption(f"adds {sm['put_adds']:,.0f} / cuts {sm['put_cuts']:,.0f}")
        q3.metric("Put share of OI adds", "n/a" if np.isnan(sm["put_add_share"]) else f"{sm['put_add_share']:.0%}")
        q3.caption("50% = balanced adds; descriptive, not a direction call")
        q4.metric("One-day turnover", "n/a" if np.isnan(sm["gross_turnover"]) else f"{sm['gross_turnover']:.0%}")
        q4.caption(f"{sm['known_share']:.0%} of OI has a change figure")
        if R["oi_walls"]:
            wcols = st.columns(2)
            for wc, t in zip(wcols, R["oi_walls"]):
                with wc, st.container(border=True):
                    if not t["available"]:
                        st.markdown(f"**{t['label']}**")
                        st.caption("No OI-change data at this strike (outside the band or window).")
                        continue
                    st.markdown(f"**{t['label']} {t['strike']:.0f}**")
                    st.markdown(f":{tag_color(t['tag'])}[**{tag_text(t['tag'])}**]")
                    pct = "" if np.isnan(t["pct"]) else f", {t['pct']:+.0%}"
                    st.caption(f"{t['side']} OI {t['oi']:,.0f} ({t['chg']:+,.0f}{pct})")
                    st.caption(f"{t['other_side']} OI at the same strike {t['other_oi']:,.0f} "
                               f"({t['other_chg']:+,.0f}) {tag_text(t['other_tag'])}")
        else:
            st.caption("Add the GEX-by-strike file to tag the put and call walls.")
        st.caption(f"Change versus the previous session's open interest (published after the close, so it describes "
                   f"the prior day's trading), summed over expiries {sm['min_dte']}-{sm['max_dte']} DTE within "
                   f"{sm['band']:.0%} of spot. It cannot say who bought or sold. Display only: not in the score. "
                   "Strike and expiration detail is on the Dealer map tab.")

    # ---- price chart
    st.subheader("Price, 200 EMA and Donchian channels")
    o1, o2, o3 = st.columns([1, 1, 2])
    style = o1.radio("Style", ["Candlestick", "Line"], horizontal=True, key="chart_style")
    window = o2.selectbox("Window", ["3M", "6M", "1Y", "All"], index=1, key="chart_window")
    overlays = o3.multiselect("Overlays", list(core.OVERLAYS), default=list(core.OVERLAYS), key="overlays")
    days = {"3M": 63, "6M": 126, "1Y": 252}.get(window, len(R["candles"]))
    frame = R["candles"]
    if style == "Candlestick":
        show_vega(core.candle_spec(frame, overlays, days=days))
    else:
        line_cols = ["close"] + [c for n in overlays for c in core.OVERLAYS[n][0]]
        lf = frame.tail(days).set_index("date")[line_cols]
        st.line_chart(lf)
    legend = "  ".join(f":{c}[{n}]" for n, c in (("200 EMA", "blue"), ("Donchian 20", "orange"), ("Donchian 55", "violet"))
                       if n in overlays)
    st.caption(f"Overlays: {legend}")
    if R["candle_mode"] == "ohlc":
        st.caption("True OHLC from your price file; Donchian channels use highs and lows. The score itself uses "
                   "the history file's closes.")
    else:
        st.caption(":orange[Close-to-close candles: the history file has closes only, so each body runs from the "
                   "prior close to the close and there are no wicks.] Add a daily OHLC CSV in the sidebar for true "
                   "candlesticks.")

with tab_strat:
    if R["stay_out"]:
        st.error("No eligible trade: every candidate failed a gate (IV edge / IV rank). Stay out.")
    st.write(f"Matrix cell: **{d['band']} / {g['effective']} gamma**. Prices are natural fills "
             "(buy at ask, sell at bid); P&L per 1 contract, evaluated at the earliest expiry; "
             "multi-expiry structures use Black-Scholes with each leg's chain IV held flat (model-based).")
    for i, s in enumerate(R["strategies"]):
        tag = "" if s.eligible else "  [NOT ELIGIBLE]"
        with st.expander(f"{i + 1}. {s.name}{tag}  -  {s.source}", expanded=(i == 0)):
            legs = pd.DataFrame([{
                "side": "Buy" if l.qty > 0 else "Sell", "qty": abs(l.qty), "expiry": str(l.exp.date()),
                "DTE": (l.exp - asof).days, "type": "Call" if l.cp == "C" else "Put", "strike": l.strike,
                "bid": l.bid, "ask": l.ask, "IV %": round(l.iv * 100, 1),
                "OI": None if np.isnan(l.oi) else int(l.oi),
                "OI chg": None if np.isnan(l.oi_chg) else int(l.oi_chg),
                "OI trend": leg_trend(l, R["oi_params"])} for l in s.legs])
            st.dataframe(legs, hide_index=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Net debit (natural)", f"{s.debit_nat:+.2f}")
            m2.metric("Max profit" + (" (model)" if s.model_based else ""), f"{s.max_profit:,.0f} USD")
            m3.metric("Max loss" + (" (model)" if s.model_based else ""), f"{s.max_loss:,.0f} USD")
            m4.metric("Breakeven", ", ".join(f"{b:.2f}" for b in s.breakevens) or "n/a")
            m5.metric("Approx. POP", f"{s.pop:.0%}")
            if s.reason:
                st.error(s.reason)
            for w in s.warnings:
                st.caption("Note: " + w)
            gr = s.grid[(s.grid["spot"] >= R["spot"] * 0.85) & (s.grid["spot"] <= R["spot"] * 1.15)]
            st.line_chart(gr.set_index("spot")["pnl"])
            st.caption(f"P&L at {s.eval_date.date()} vs QQQ price. Net debit negative = credit. "
                       "POP is a lognormal estimate at ATM IV, not a forecast.")

full = None            # the snapshot log, set inside the Dealer map tab when open-interest data exists
with tab_dealer:
    w = R["walls"]
    if not w:
        st.info("Add the GEX-by-strike file to see walls.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Put wall (largest put GEX within 7%)", f"{w['put_wall']:.0f}")
        c2.metric("Call wall (largest call GEX within 7%)", f"{w['call_wall']:.0f}")
        st.bar_chart(w["table"].set_index("strike")["net"])
        st.caption("Net GEX by strike within +/-7% of spot (model estimate; assumes standard dealer-position conventions).")
    if R["gex_exp"] is not None:
        t = R["gex_exp"].copy()
        t["DTE"] = (t["date"] - asof).dt.days
        t = t[(t["DTE"] >= 0) & (t["DTE"] <= 90)]
        st.subheader("Net GEX / DEX by expiration (first 90 days)")
        view = t[["expiration", "DTE", "net_gex", "gamma_flip", "call_wall", "put_wall"]].copy()
        if R["dex_exp_df"] is not None:
            view = view.merge(R["dex_exp_df"][["expiration", "net_dex"]], on="expiration", how="left")
        st.dataframe(view, hide_index=True)

    # ---- open-interest change by strike / expiration / contract
    st.subheader("Open-interest change")
    oi = R["oi"]
    if oi is None:
        st.info("The option chain has no usable oi_chg column.")
    else:
        sm = oi["summary"]
        st.caption(f"Window: expiries {sm['min_dte']}-{sm['max_dte']} DTE, strikes within {sm['band']:.0%} of spot "
                   f"({sm['n_contracts']:,} contracts). Change is versus the previous session. Unknown changes are "
                   f"left out of percentages and tags (known for {sm['known_share']:.0%} of OI). "
                   "d GEX = gamma x change x 100 x spot^2 x 1%, the same convention as the GEX file, a model estimate.")
        bsd = oi["by_strike"]
        ch_plot = bsd.set_index("strike")[["call_chg", "put_chg"]].rename(
            columns={"call_chg": "call OI change", "put_chg": "put OI change"})
        st.bar_chart(ch_plot)
        only = st.checkbox("Only strikes with a build or erode tag", value=True, key="oi_only_moves")
        sv = bsd.copy()
        if only:
            moved = (~sv["call_tag"].isin(["flat", "n/a"])) | (~sv["put_tag"].isin(["flat", "n/a"]))
            sv = sv[moved]
        if sv.empty:
            st.caption("No strike clears the thresholds.")
        else:
            st.dataframe(pd.DataFrame({
                "strike": sv["strike"], "call OI": sv["call_oi"].round(0), "call chg": sv["call_chg"].round(0),
                "call %": (sv["call_pct"] * 100).round(0), "call trend": sv["call_tag"].map(tag_text),
                "put OI": sv["put_oi"].round(0), "put chg": sv["put_chg"].round(0),
                "put %": (sv["put_pct"] * 100).round(0), "put trend": sv["put_tag"].map(tag_text),
                "d GEX ($M)": (sv["d_gex"] / 1e6).round(1)}), hide_index=True)
        st.subheader("By expiration")
        be = oi["by_exp"]
        st.dataframe(pd.DataFrame({
            "expiry": be["exp"].dt.date.astype(str), "DTE": be["dte"], "call OI": be["call_oi"].round(0),
            "call chg": be["call_chg"].round(0), "call trend": be["call_tag"].map(tag_text),
            "put OI": be["put_oi"].round(0), "put chg": be["put_chg"].round(0), "put trend": be["put_tag"].map(tag_text),
            "d GEX ($M)": (be["d_gex"] / 1e6).round(1), "note": be["note"]}), hide_index=True)
        st.subheader("Largest single-contract changes")
        tp = oi["top"]
        st.dataframe(pd.DataFrame({
            "expiry": tp["exp"].dt.date.astype(str), "DTE": tp["dte"], "type": tp["cp"].map({"C": "Call", "P": "Put"}),
            "strike": tp["strike"], "OI": tp["open_interest"].round(0), "change": tp["chg"].round(0),
            "change %": (tp["pct"] * 100).round(0), "trend": tp["tag"].map(tag_text),
            "d GEX ($M)": (tp["d_gex"] / 1e6).round(1)}), hide_index=True)

        # ---- snapshot log: lets you build (and later test) a history of these readings
        st.subheader("Daily snapshot log")
        default_snap = os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__", "app.py"))),
                                    "oi_snapshots.csv")
        snap_path = st.text_input("Snapshot file", value=os.environ.get("QQQ_SNAPSHOT_FILE", default_snap),
                                  key="snap_path")
        log = None
        try:
            if os.path.isfile(snap_path):
                log = pd.read_csv(snap_path)
        except Exception as ex:
            st.warning(f"Could not read {snap_path}: {ex}")
        up = st.file_uploader("Or load a log you downloaded earlier", type=["csv"], key="snap_up")
        if up is not None:
            try:
                prior_log = pd.read_csv(io.BytesIO(up.getvalue()))
                for _, rr in prior_log.iterrows():
                    log = core.merge_snapshot(log, rr.to_dict())
            except Exception as ex:
                st.warning(f"Could not read the uploaded log: {ex}")
        today_row = core.snapshot_row(R)
        if st.button("Save this snapshot", key="snap_save"):
            try:
                log = core.merge_snapshot(log, today_row)
                log.to_csv(snap_path, index=False)
                st.success(f"Saved the {today_row['asof']} snapshot ({len(log)} rows in {snap_path}).")
            except Exception as ex:
                st.error(f"Could not write {snap_path}: {ex}")
        try:
            full = core.merge_snapshot(log, today_row)
        except Exception as ex:
            st.warning(f"{snap_path} does not look like a snapshot log ({ex}); showing today's row only.")
            full = core.merge_snapshot(None, today_row)
        st.caption("One row per as-of date; saving the same date again replaces it. Hosted apps (for example "
                   "Streamlit Cloud) lose local files on reboot, so download the log and keep it. After several "
                   "weeks it can be tested against forward returns; one day of data cannot.")
        st.dataframe(full.tail(30), hide_index=True)
        st.download_button("Download log (including today's row)", full.to_csv(index=False).encode("utf-8"),
                           file_name="oi_snapshots.csv", mime="text/csv", key="snap_dl")

if vc_trend_slot is not None:
    with vc_trend_slot:
        vc_trend_ui(full)

with tab_flow:
    f = R["flow"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Aggressor imbalance", "n/a" if np.isnan(f["aggr"]) else f"{f['aggr']:+.2f}")
    c1.caption("(call at ask + put at bid) vs (put at ask + call at bid)")
    c2.metric("Unusual-activity imbalance (2+ DTE)", "n/a" if np.isnan(f["unusual"]) else f"{f['unusual']:+.2f}")
    c3.metric("Premium at mid (low confidence)", "n/a" if np.isnan(f["mid_share"]) else f"{f['mid_share']:.0%}")
    st.caption("A large mid-price share means most prints cannot be classified as buyer- or seller-initiated; "
               "treat flow as weak evidence.")
    fsym = R["flow_symbol"]
    if "flow" in data or "unusual" in data:
        st.caption(f"Flow and unusual activity are scored for **{fsym or 'n/a'}** "
                   f"({'set in the sidebar' if flow_sym.strip() else 'detected from the files'}). "
                   "Make sure this is the same underlying as your option chain and history files.")
    for kind_key, label in (("flow", "Options flow"), ("unusual", "Unusual activity")):
        raw = data.get(kind_key)
        if raw is not None and core.underlying_rows(raw, fsym).empty:      # say why the numbers above are n/a
            clean = core._symbols(raw)
            seen = clean[clean != ""].value_counts().head(8).to_dict() if "Symbol" in raw.columns else {}
            st.warning(f"{label}: no rows for {fsym or 'any ticker'} in this file ({len(raw)} rows loaded). "
                       f"Symbols found: {seen or 'no Symbol column'}. Clear or change the sidebar symbol, or "
                       "re-export the file for the right underlying.")
    if "flow" in data:
        fl = core.underlying_rows(data["flow"], fsym)
        fl["Premium"] = pd.to_numeric(fl["Premium"], errors="coerce")
        st.subheader("Largest prints")
        st.dataframe(fl.sort_values("Premium", ascending=False).head(25)[
            ["Exp Date", "Type", "Strike", "DTE", "Trade", "Size", "Side", "Premium", "*"]], hide_index=True)

with tab_vol:
    e = R["edge"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Front IV", f"{e['front_iv'] * 100:.1f}%")
    c1.caption(f"{e['front_dte']} DTE")
    c2.metric("Back IV", f"{e['back_iv'] * 100:.1f}%")
    c2.caption(f"{e['back_dte']} DTE")
    c3.metric("Front minus back", f"{e['edge_pts']:+.1f} pts")
    c3.caption("meets threshold" if e["edge_pts"] >= edge_thr else "below threshold")
    it = R["ivtab"][R["ivtab"]["dte"] >= 1].copy()
    it["atm_iv_pct"] = it["atm_iv"] * 100
    st.subheader("ATM IV term structure")
    st.line_chart(it.set_index("dte")["atm_iv_pct"])
    if R["em"] is not None:
        em = R["em"].copy()
        em["DTE"] = (em["date"] - asof).dt.days
        pick = em[em["DTE"].isin([1, 4, 11, 18, 25, 46])]
        st.subheader("Expected move")
        st.dataframe(pick[["date", "DTE", "expected_move_amt", "expected_move_percentage", "lower_price", "upper_price"]],
                     hide_index=True)

with tab_bt:
    st.write("Backtest of the history-file components against forward QQQ returns (signals at close, "
             "returns start from that close). Flow here is the put/call-ratio proxy, since flow files are snapshots.")
    try:
        hc = core.history_components(core.parse_history(data["history"]), flow_contrarian=contrarian)
        bt = core.backtest_tables(hc, weights, float(t_mild), float(t_strong))
        st.caption(f"Sample {bt['start'].date()} to {bt['end'].date()}, n={bt['n']}. Rank correlation (IC) vs forward return; "
                   "negative = bullish scores preceded lower returns. Overlapping windows overstate significance.")
        st.dataframe(bt["ic"], hide_index=True)
        st.subheader("Forward returns by band")
        st.dataframe(bt["bands"], hide_index=True)
        st.subheader("Gamma regime vs next-5-day realised move")
        st.dataframe(bt["regime"])
        st.caption(f"Composite IC stability (5d): first half {bt['stability']['first half']}, "
                   f"second half {bt['stability']['second half']}. Small bands (n < 30) are noise.")
    except Exception as ex:
        st.error(f"Backtest unavailable: {ex}")

with tab_scr:
    screener_tab.render(show_vega)

with tab_ich:
    ichimoku_tab.render(show_vega)

with tab_gam:
    gamma_tab.render()

with tab_chat:
    chat_tab.render(R)
