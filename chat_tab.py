"""Streamlit tab: chat with Claude about what the app is currently showing.
Self-contained like screener_tab.py: needs only the `anthropic` package and an API key. It works before the option
files are loaded (it then just knows about the screeners). Call render(R) inside a tab, where R is core.analyse() output
or None.  build_context() is pure (no Streamlit) so it can be tested on its own."""
import math
import os

import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TURNS_SENT = 20          # most recent chat messages sent to the API (older ones stay on screen only)
MAX_TOKENS = 1500
SCREENER_COLS = ["ticker", "setup", "side", "grade", "bars_ago", "date", "close", "level", "dist_atr",
                 "stop", "adx", "atr_pct", "note"]

SYSTEM_PROMPT = """You are the assistant built into a Streamlit research app called "Regime & Strategy Selector". \
The app scores QQQ option data (price structure, flow, DEX), gates on IV rank and term structure, picks a gamma \
regime, and proposes option structures. It also has Donchian and Ichimoku stock screeners.

Rules:
- Ground every number in the APP STATE block below. If a figure is not there, say it is not in the current view \
instead of guessing or using memory. Never invent prices, strikes or levels. Before saying something is missing, check \
every section: per-expiry GEX/DEX, GEX by strike, OI change tables, IV term structure and expected moves are all included.
- Not included: the raw CSV files, the largest-flow-prints table, the Backtest tab tables and the price chart history. \
If asked about those, say they are not in what you were given.
- Explain what the app is showing and why: what a reading means, why a strategy was ranked or blocked, what the gates \
and gamma regime imply, how to interpret OI build/erode tags, flip levels, walls, and screener grades.
- Be concise and plain. Use short paragraphs or a few bullets; skip preamble.
- GEX, DEX, walls and multi-expiry P&L are model estimates; POP is a lognormal estimate; the 19-month backtest found \
the composite direction score unreliable on its own. Mention these limits when they matter to the answer.
- This is a research tool. Do not give personalised financial advice or tell the user to place a trade; frame answers \
as how the signals read and what the trade-offs are.
- If the user asks how to change a setting, point to the sidebar control or tab by name."""


def _fnum(x, nd=2, plus=False):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(x):
        return "n/a"
    return f"{x:+,.{nd}f}" if plus else f"{x:,.{nd}f}"


def _big(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(x):
        return "n/a"
    return f"{x / 1e9:+.2f}B" if abs(x) >= 1e9 else f"{x / 1e6:+,.0f}M"


def _section(lines, title, fn):
    """Run one context section; a failure in one section must never break the chat."""
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        out = [f"(unavailable: {type(e).__name__})"]
    if out:
        lines.append(f"## {title}")
        lines.extend(out)


def _direction(R):
    d, comps = R["direction"], R["components"]
    out = [f"Sentiment band: {d['band']} (composite score {_fnum(d['score'], 2, True)}, -1..+1)."]
    for k in ("price", "flow", "dex"):
        out.append(f"- {k}: score {_fnum(comps.get(k), 3, True)}, weight used {_fnum(d['used'].get(k, 0.0))}")
    out.append(f"Flow source: {R.get('flow_source')}. Component agreement with composite sign: "
               f"{_fnum(d.get('agreement'), 2)}.")
    return out


def _regime(R):
    g = R["regime"]
    out = [f"Effective gamma regime: {g['effective']}{' (inside near-flip buffer)' if g.get('near_flip') else ''}.",
           f"Net GEX over the trade window: {_big(g.get('net'))} (source: {g.get('source')}).",
           f"Spot (parity-implied from the chain): {_fnum(R['spot'])}. Data as of {pd.Timestamp(R['asof']).date()}."]
    if not _isnan(g.get("flip")):
        out.append(f"Gamma-weighted flip across trade-window expiries: {_fnum(g['flip'], 1)} "
                   f"(range {_fnum(g.get('flip_min'), 0)} to {_fnum(g.get('flip_max'), 0)}).")
    af = R.get("agg_flip")
    if af and not _isnan(af.get("flip")):
        side = "below" if R["spot"] < af["flip"] else "above"
        out.append(f"All-expiry GEX-profile flip: {_fnum(af['flip'], 1)}; spot is {side} it "
                   f"({_fnum(abs(R['spot'] - af['flip']), 1)} pts). GEX profile at spot: {_big(af.get('value_at_spot'))} "
                   f"({af.get('state_at_spot')} gamma zone).")
    w = R.get("walls")
    if w:
        out.append(f"Put wall {_fnum(w['put_wall'], 0)}, call wall {_fnum(w['call_wall'], 0)} (largest GEX within 7% of spot).")
    for name, key in (("GEX", "gex_trend"), ("DEX", "dex_trend")):
        tc = R.get(key)
        if tc:
            out.append(f"{name} trend: {tc['sign']}, {tc['trend']}; value {_big(tc.get('value'))}, "
                       f"{tc['short']}d change {_big(tc['d_short'])}, {tc['long']}d change {_big(tc['d_long'])}, "
                       f"{_fnum(tc.get('pctile', float('nan')) * 100, 0)}th percentile of last 120 days, "
                       f"{tc.get('days_in_sign')} sessions in this sign.")
    return out


def _vol(R):
    e = R["edge"]
    return [f"IV rank: {_fnum(R['ivrank'], 0) if R['ivrank'] is not None else 'n/a'}.",
            f"Front IV {_fnum(e['front_iv'] * 100, 1)}% ({e['front_dte']} DTE), back IV {_fnum(e['back_iv'] * 100, 1)}% "
            f"({e['back_dte']} DTE), front minus back {_fnum(e['edge_pts'], 1, True)} vol pts."]


def _flow(R):
    f = R["flow"]
    sym = R.get("flow_symbol")
    return [f"Flow and unusual activity are scored for {sym or 'n/a'} (the underlying in those files; "
            "it should match the option chain's underlying).",
            f"Aggressor imbalance {_fnum(f.get('aggr'), 2, True)}; unusual-activity imbalance (2+ DTE) "
            f"{_fnum(f.get('unusual'), 2, True)}; premium at mid {_fnum(f.get('mid_share', float('nan')) * 100, 0)}% "
            "(high mid share = flow is weak evidence)."]


def _oi(R):
    oi = R.get("oi")
    if oi is None:
        return ["No usable oi_chg column in the chain, so OI changes are unavailable."]
    s = oi["summary"]
    out = [f"Window: expiries {s['min_dte']}-{s['max_dte']} DTE within {s['band']:.0%} of spot (versus previous session).",
           f"Net call OI change {_fnum(s['net_call_chg'], 0, True)} (adds {_fnum(s['call_adds'], 0)}, cuts {_fnum(s['call_cuts'], 0)}); "
           f"net put OI change {_fnum(s['net_put_chg'], 0, True)} (adds {_fnum(s['put_adds'], 0)}, cuts {_fnum(s['put_cuts'], 0)}).",
           f"Put share of OI adds {_fnum(s['put_add_share'] * 100, 0)}%; one-day turnover {_fnum(s['gross_turnover'] * 100, 0)}%; "
           f"{_fnum(s['known_share'] * 100, 0)}% of OI has a change figure. Display only: never in the score."]
    for t in R.get("oi_walls") or []:
        if t.get("available"):
            out.append(f"{t['label']} {_fnum(t['strike'], 0)}: {t['tag']} ({t['side']} OI {_fnum(t['oi'], 0)}, "
                       f"change {_fnum(t['chg'], 0, True)}).")
    return out


def _strategies(R):
    out = []
    if R.get("stay_out"):
        out.append("STAY OUT: every candidate failed a gate (IV edge / IV rank).")
    out.append(f"Matrix cell: {R['direction']['band']} / {R['regime']['effective']} gamma. Prices are natural fills; "
               "P&L is per 1 contract; multi-expiry structures are Black-Scholes with flat chain IV (model-based).")
    asof = pd.Timestamp(R["asof"])
    for i, s in enumerate(R["strategies"], 1):
        legs = "; ".join(f"{'Buy' if l.qty > 0 else 'Sell'} {abs(l.qty)} {pd.Timestamp(l.exp).date()} "
                         f"({(pd.Timestamp(l.exp) - asof).days} DTE) {_fnum(l.strike, 0)}{l.cp} "
                         f"bid {_fnum(l.bid)} ask {_fnum(l.ask)} IV {_fnum(l.iv * 100, 1)}%" for l in s.legs)
        be = ", ".join(_fnum(b) for b in s.breakevens) or "n/a"
        out.append(f"{i}. {s.name} [{'eligible' if s.eligible else 'NOT ELIGIBLE'}] ({s.source}). Legs: {legs}. "
                   f"Net debit {_fnum(s.debit_nat, 2, True)} (negative = credit), max profit {_fnum(s.max_profit, 0)} USD, "
                   f"max loss {_fnum(s.max_loss, 0)} USD{' (model)' if s.model_based else ''}, breakevens {be}, "
                   f"approx POP {_fnum(s.pop * 100, 0)}%.")
        if s.reason:
            out.append(f"   Why not eligible: {s.reason}")
        for wmsg in s.warnings:
            out.append(f"   Note: {wmsg}")
    return out


def _isnan(x):
    try:
        return x is None or math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _screener(res, when, label, max_rows=40):
    if res is None:
        return None
    if res.empty:
        return [f"{label} (run {when or 'earlier'}): no setups at the latest bar."]
    cols = [c for c in SCREENER_COLS if c in res.columns]
    shown = res[cols].head(max_rows)
    counts = res["setup"].value_counts().to_dict() if "setup" in res.columns else {}
    body = shown.to_csv(index=False, float_format="%.2f")
    more = f"\n(showing first {max_rows} of {len(res)} rows)" if len(res) > max_rows else ""
    return [f"{label} (run {when or 'earlier'}): {len(res)} setups; counts by setup {counts}.", body + more]


def _csv(df, cols, max_rows):
    """Compact CSV of the chosen columns, with a truncation note."""
    cols = [c for c in cols if c in df.columns]
    body = df[cols].head(max_rows).to_csv(index=False, float_format="%.2f")
    more = f"\n(showing first {max_rows} of {len(df)} rows)" if len(df) > max_rows else ""
    return body.rstrip("\n") + more


def _by_expiration(R):
    """The 'Net GEX / DEX by expiration' table on the Dealer map tab, one row per expiry (first 90 days)."""
    g = R.get("gex_exp")
    if g is None:
        return ["No GEX-by-expiration file loaded, so there is no per-expiry GEX breakdown."]
    asof = pd.Timestamp(R["asof"])
    t = g.copy()
    t["DTE"] = (t["date"] - asof).dt.days
    t = t[(t["DTE"] >= 0) & (t["DTE"] <= 90)]
    if R.get("dex_exp_df") is not None:
        t = t.merge(R["dex_exp_df"][["expiration", "net_dex"]], on="expiration", how="left")
    t["expiration"] = pd.to_datetime(t["date"]).dt.strftime("%b %d %Y")
    for c in ("net_gex", "net_dex"):           # millions of dollars keeps the numbers short and readable
        if c in t.columns:
            t[c + "_$M"] = t[c] / 1e6
    return ["One row per expiry, first 90 DTE (this is the app's 'Net GEX / DEX by expiration' table). "
            "Values in millions of dollars per 1% move; gamma_flip, call_wall and put_wall are QQQ price levels.",
            _csv(t, ["expiration", "DTE", "net_gex_$M", "net_dex_$M", "gamma_flip", "call_wall", "put_wall"], 45)]


def _strike_gex(R):
    w = R.get("walls")
    if not w or w.get("table") is None:
        return None
    t = w["table"].copy()
    t["net_gex_$M"] = t["net"] / 1e6
    return ["Net GEX by strike within +/-7% of spot, millions of dollars (model estimate).",
            _csv(t.sort_values("strike"), ["strike", "net_gex_$M"], 45)]


def _oi_detail(R):
    oi = R.get("oi")
    if oi is None:
        return None
    out = []
    be = oi["by_exp"].copy()
    be["expiry"] = be["exp"].dt.date.astype(str)
    be["d_gex_$M"] = be["d_gex"] / 1e6
    out += ["OI change by expiration (vs previous session):",
            _csv(be, ["expiry", "dte", "call_oi", "call_chg", "call_tag", "put_oi", "put_chg", "put_tag", "d_gex_$M"], 30)]
    bs = oi["by_strike"].copy()
    moved = (~bs["call_tag"].isin(["flat", "n/a"])) | (~bs["put_tag"].isin(["flat", "n/a"]))
    bs = bs[moved].copy()
    bs["d_gex_$M"] = bs["d_gex"] / 1e6
    out += ["OI change by strike, only strikes tagged build/erode:",
            _csv(bs, ["strike", "call_oi", "call_chg", "call_tag", "put_oi", "put_chg", "put_tag", "d_gex_$M"], 25)
            if len(bs) else "(no strike clears the thresholds)"]
    tp = oi["top"].copy()
    tp["expiry"] = tp["exp"].dt.date.astype(str)
    tp["d_gex_$M"] = tp["d_gex"] / 1e6
    out += ["Largest single-contract OI changes:",
            _csv(tp, ["expiry", "dte", "cp", "strike", "open_interest", "chg", "tag", "d_gex_$M"], 10)]
    return out


def _term_and_em(R):
    out = []
    it = R["ivtab"][R["ivtab"]["dte"] >= 1].copy()
    it["atm_iv_pct"] = it["atm_iv"] * 100
    out += ["ATM implied vol term structure:", _csv(it, ["dte", "atm_iv_pct"], 30)]
    em = R.get("em")
    if em is not None:
        e = em.copy()
        e["DTE"] = (e["date"] - pd.Timestamp(R["asof"])).dt.days
        e["date"] = e["date"].dt.date.astype(str)
        out += ["Expected move by expiry:",
                _csv(e[(e["DTE"] >= 0) & (e["DTE"] <= 60)],
                     ["date", "DTE", "expected_move_amt", "expected_move_percentage", "lower_price", "upper_price"], 20)]
    return out


def build_context(R=None, scr_res=None, scr_when=None, ich_res=None, ich_when=None) -> str:
    """Plain-text snapshot of what the app is currently showing. Each section degrades independently."""
    lines = ["# APP STATE (what the user is currently looking at)"]
    if R is None:
        lines.append("Option files are not loaded yet, so there is no QQQ score, regime or strategy data.")
    else:
        _section(lines, "Direction", lambda: _direction(R))
        _section(lines, "Gamma regime, flip, walls, dealer trends", lambda: _regime(R))
        _section(lines, "Volatility gates", lambda: _vol(R))
        _section(lines, "Flow", lambda: _flow(R))
        _section(lines, "Open-interest change", lambda: _oi(R))
        _section(lines, "GEX / DEX by expiration (per-expiry breakdown)", lambda: _by_expiration(R))
        _section(lines, "GEX by strike", lambda: _strike_gex(R))
        _section(lines, "Open-interest change detail", lambda: _oi_detail(R))
        _section(lines, "IV term structure and expected move", lambda: _term_and_em(R))
        _section(lines, "Ranked strategies", lambda: _strategies(R))
    for res, when, label in ((scr_res, scr_when, "Donchian screener"), (ich_res, ich_when, "Ichimoku screener")):
        _section(lines, label, lambda res=res, when=when, label=label: _screener(res, when, label))
    return "\n".join(lines)


# ------------------------------------------------------------------ Streamlit UI
def _secret(name):
    try:
        v = st.secrets.get(name)
    except Exception:  # no secrets file configured
        v = None
    return v or os.environ.get(name)


def _client(api_key):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _trim(msgs, n=MAX_TURNS_SENT):
    """Last n messages, starting on a user turn (the API requires the first message to be from the user)."""
    out = list(msgs[-n:])
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def _stream_reply(client, model, system, history):
    """Yield text chunks for st.write_stream."""
    with client.messages.stream(model=model, max_tokens=MAX_TOKENS, system=system, messages=history) as stream:
        for text in stream.text_stream:
            yield text


def render(R=None):
    st.subheader("Ask about this app")
    st.caption("Chat with Claude about the readings, strategies and screener results currently on screen. It sees a text "
               "snapshot of the app's state (not your raw CSV files). Research tool, not financial advice.")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        st.error("The chat needs the `anthropic` package. Add `anthropic>=0.40` to requirements.txt and reboot the app.")
        return

    api_key = _secret("ANTHROPIC_API_KEY")
    model = _secret("CHAT_MODEL") or DEFAULT_MODEL
    if not api_key:
        api_key = st.text_input("Anthropic API key", type="password", key="chat_api_key",
                                help="Kept only in this browser session. To avoid pasting it each time, add "
                                     "ANTHROPIC_API_KEY to Streamlit secrets (Manage app -> Settings -> Secrets).")
        if not api_key:
            st.info("Enter an Anthropic API key to start chatting.")
            return

    context = build_context(R, st.session_state.get("scr_res"), st.session_state.get("scr_when"),
                            st.session_state.get("ich_res"), st.session_state.get("ich_when"))
    system = SYSTEM_PROMPT + "\n\n" + context

    msgs = st.session_state.setdefault("chat_msgs", [])
    top = st.columns([4, 1])
    top[0].caption(f"Model: {model}")
    if top[1].button("Clear chat", key="chat_clear"):
        msgs.clear()
    with st.expander("What the chatbot can see"):
        st.code(context, language="text")

    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if not msgs:
        st.caption("Try: \"Why is the top strategy ranked first?\"  |  \"What does the near-flip buffer mean right now?\"  |  "
                   "\"Summarise the Donchian hits by grade.\"")

    prompt = st.chat_input("Ask about the regime, a strategy, or the screeners...", key="chat_input")
    if not prompt:
        return
    msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(_stream_reply(_client(api_key), model, system, _trim(msgs)))
        except Exception as e:  # noqa: BLE001
            msgs.pop()   # drop the unanswered question so the history keeps alternating user/assistant
            st.error(f"The chat request failed: {e}")
            return
    msgs.append({"role": "assistant", "content": reply})
