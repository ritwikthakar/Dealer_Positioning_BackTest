"""Streamlit tab: Donchian setup screener (breakout / pullback / squeeze / failed breakout) with ADX + ATR grading.
Self-contained: needs only donchian_screener.py next to it (and yfinance + lxml for live data). It does not depend
on core.py, so it works before the QQQ option files are loaded.  Call render() inside a tab."""
import datetime as dt
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

SETUP_LABEL = {"breakout": "Breakout", "pullback": "Pullback", "squeeze": "Squeeze (watch)",
               "squeeze_break": "Squeeze break", "failed_break": "Failed breakout"}
SHOW = ["ticker", "setup", "side", "grade", "bars_ago", "date", "close", "level", "dist_atr", "stop", "adx", "atr_pct", "note"]
HELP = {
    "breakout": "Fresh close beyond the PRIOR 20-day high/low. Grade A/B/C from: 55-day break, 200 EMA trend, volume, "
                "no big gap, ADX >= 20 and rising. Stop = 0.5 ATR back inside the channel.",
    "pullback": "55-day extreme in the last 15 bars, price back at/below the channel midpoint, still on the trend side of "
                "the 200 EMA, ADX >= 20 (A if >= 25). Stop = 0.5 ATR beyond the opposite channel edge.",
    "squeeze": "Channel width in the bottom 15% of the past year. A = ADX < 20 and ATR in the bottom 30%. Alert levels are in the note.",
    "squeeze_break": "First close out of a squeeze. A = true range >= 1.2 x ATR and volume above average.",
    "failed_break": "Close beyond the 20-day level, then the first close back inside within 1-3 bars. A = ADX < 25 or fading "
                    "(mean reversion more plausible), B = strong rising ADX (riskier fade). Stop = 0.5 ATR past the trap extreme; target = channel mid.",
}


def parse_tickers(text):
    seen, out = set(), []
    for t in re.split(r"[,\s;]+", text or ""):
        t = t.strip().upper().replace(".", "-")
        if t and re.fullmatch(r"[A-Z0-9\-\^=]{1,12}", t) and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def candle_spec(d, days=120, height=380):
    """Candles with 20-day Donchian high/low and the 200 EMA, from sc.add_indicators output."""
    x = d.tail(days).copy()
    x["date"] = x.index
    cols = ["date", "Open", "High", "Low", "Close", "hi", "lo", "ema"]
    recs = json.loads(x[cols].rename(columns=str.lower).to_json(orient="records", date_format="iso"))
    vals = x[["Low", "hi", "lo", "ema"]].to_numpy(dtype=float)
    top = x[["High", "hi", "lo", "ema"]].to_numpy(dtype=float)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(top))
    pad = (hi - lo) * 0.03 or 1.0
    dom = [lo - pad, hi + pad]
    xe = {"field": "date", "type": "temporal", "axis": {"title": None, "format": "%b %d", "labelOverlap": True}}
    ye = lambda f: {"field": f, "type": "quantitative", "scale": {"zero": False, "domain": dom}, "axis": {"title": None}}
    color = {"condition": {"test": "datum.close >= datum.open", "value": "#26a69a"}, "value": "#ef5350"}
    tip = [{"field": "date", "type": "temporal", "title": "Date", "format": "%Y-%m-%d"}] + [
        {"field": f, "type": "quantitative", "title": f.title(), "format": ".2f"} for f in ("open", "high", "low", "close")]
    bar = int(max(2, min(9, 650 / max(len(x), 1) * 0.7)))
    layers = [
        {"mark": {"type": "rule"}, "encoding": {"x": xe, "y": ye("low"), "y2": {"field": "high"}, "color": color, "tooltip": tip}},
        {"mark": {"type": "bar", "size": bar}, "encoding": {"x": xe, "y": ye("open"), "y2": {"field": "close"}, "color": color, "tooltip": tip}},
    ]
    for f, col, dash in (("hi", "#fb8c00", None), ("lo", "#fb8c00", [4, 3]), ("ema", "#1e88e5", None)):
        mark = {"type": "line", "color": col, "strokeWidth": 1.4}
        if dash:
            mark["strokeDash"] = dash
        layers.append({"mark": mark, "encoding": {"x": xe, "y": ye(f)}})
    return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "height": height,
            "data": {"values": recs}, "layer": layers}


def _show_vega(spec, show_vega):
    if show_vega is not None:
        show_vega(spec)
    else:
        st.vega_lite_chart(spec=spec)


@st.cache_data(ttl=86400, show_spinner=False)
def _sp500():
    import donchian_screener as sc
    return sc.get_sp500_tickers()


@st.cache_data(ttl=1800, show_spinner=False)
def _one(ticker, start):
    import donchian_screener as sc
    raw = sc.yf_loader([ticker], start)
    return sc.extract(raw, ticker)


def render(show_vega=None):
    try:
        import donchian_screener as sc
    except ImportError:
        st.error("donchian_screener.py is missing. Copy it into the same folder as app.py and reboot the app.")
        return

    st.subheader("Donchian setup screener")
    st.caption("Four setups, both directions, graded with ADX and ATR. Channels use the PRIOR bars only. "
               "Daily data from Yahoo Finance via yfinance; the last bar is dropped while the US market is open. "
               "Candidate list for review, not trade signals.")
    c1, c2, c3 = st.columns([2, 1, 1])
    universe = c1.radio("Universe", ["S&P 500", "My tickers"], horizontal=True, key="scr_universe")
    lookback = c2.number_input("Look back (bars)", 0, 10, 3, key="scr_lookback",
                               help="Report breakouts, squeeze breaks and failures whose trigger was within this many bars.")
    c3.write("")
    run_it = c3.button("Run screener", key="scr_run")
    custom = ""
    if universe == "My tickers":
        custom = st.text_area("Tickers (comma, space or newline separated)", value="QQQ, SPY, AAPL, NVDA", key="scr_tickers")

    if run_it:
        try:
            tickers = _sp500() if universe == "S&P 500" else parse_tickers(custom)
        except Exception as e:
            st.error(f"Could not load the S&P 500 list ({e}). Use 'My tickers' instead.")
            tickers = []
        if not tickers:
            st.warning("No tickers to scan.")
        else:
            bar = st.progress(0.0, text="Downloading and scanning...")

            def cb(done, total):
                bar.progress(min(done / max(total, 1), 1.0), text=f"{done}/{total} tickers")

            try:
                res = sc.run(tickers, lookback=int(lookback), out=None, verbose=False, progress=cb)
                st.session_state["scr_res"] = res
                st.session_state["scr_when"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            except ImportError as e:
                st.error(f"Missing package: {e}. Add yfinance and lxml to requirements.txt and reboot the app.")
            except Exception as e:
                st.error(f"Screener failed: {e}")
            bar.empty()

    res = st.session_state.get("scr_res")
    if res is None:
        st.info("Choose a universe and press Run screener. The S&P 500 scan downloads about 500 tickers and takes a minute or two.")
        _notes()
        return

    meta = res.attrs
    st.caption(f"Scanned {meta.get('n_tickers', '?')} tickers at {st.session_state.get('scr_when', '')} | "
               f"no data/errors: {len(meta.get('failed', []))} | too little history (<260 bars): {len(meta.get('short', []))}")
    if meta.get("failed"):
        st.caption("No data: " + ", ".join(meta["failed"][:20]) + (" ..." if len(meta["failed"]) > 20 else ""))
    if res.empty:
        st.info("No setups found at the latest bar.")
        _notes()
        return

    counts = res["setup"].value_counts()
    cols = st.columns(len(SETUP_LABEL))
    for col, k in zip(cols, SETUP_LABEL):
        col.metric(SETUP_LABEL[k], int(counts.get(k, 0)))

    f1, f2, f3, f4 = st.columns(4)
    pick_setup = f1.multiselect("Setup", list(SETUP_LABEL), default=list(SETUP_LABEL), format_func=lambda k: SETUP_LABEL[k], key="scr_f_setup")
    pick_side = f2.multiselect("Side", ["long", "short", "watch"], default=["long", "short", "watch"], key="scr_f_side")
    pick_grade = f3.multiselect("Grade", ["A", "B", "C"], default=["A", "B", "C"], key="scr_f_grade")
    max_ago = f4.number_input("Max bars ago", 0, 10, 10, key="scr_f_ago")
    view = res[res["setup"].isin(pick_setup) & res["side"].isin(pick_side) & res["grade"].isin(pick_grade)
               & (res["bars_ago"] <= int(max_ago))]
    if view.empty:
        st.info("Nothing matches these filters.")
        _notes()
        return
    st.dataframe(view[SHOW], hide_index=True)
    st.download_button("Download CSV", view[SHOW].to_csv(index=False).encode("utf-8"),
                       file_name="donchian_setups.csv", mime="text/csv", key="scr_dl")
    st.caption("level = the channel level that was broken (or the midpoint for pullbacks); dist_atr = (close - level) / ATR; "
               "stop is a reference level, 0.5 ATR beyond the invalidation point.")

    st.subheader("Chart")
    labels = [f"{r.ticker}  |  {SETUP_LABEL[r.setup]} {r.side} (grade {r.grade})" for r in view.itertuples()]
    choice = st.selectbox("Ticker", labels, key="scr_pick")
    row = view.iloc[labels.index(choice)]
    try:
        start = (dt.date.today() - dt.timedelta(days=900)).isoformat()
        raw = _one(row["ticker"], start)
        if raw is None or raw.empty:
            raise ValueError("no price data returned")
        d = sc.add_indicators(sc.drop_partial_bar(raw))
        _show_vega(candle_spec(d), show_vega)
        st.caption(":orange[Prior 20-day high (solid) and low (dashed)]  :blue[200 EMA]. "
                   f"ADX {d['adx'].iloc[-1]:.1f}, ATR {d['atr'].iloc[-1]:.2f} ({100 * d['atr'].iloc[-1] / d['Close'].iloc[-1]:.2f}% of price).")
        st.line_chart(d["adx"].tail(120).rename("ADX(14)").to_frame(), height=140)
        st.caption(f"Why: {row['note']}")
    except ImportError as e:
        st.error(f"Missing package: {e}. Add yfinance to requirements.txt and reboot the app.")
    except Exception as e:
        st.error(f"Could not draw {row['ticker']}: {e}")
    _notes()


def _notes():
    with st.expander("How each setup is defined and what to trust"):
        for k, v in HELP.items():
            st.markdown(f"**{SETUP_LABEL[k]}.** {v}")
        st.markdown("**Caveat.** Thresholds (ADX 20/25, 1.5 ATR, 2-point ADX fade, 15% squeeze percentile) are starting "
                    "values, not tuned. On the QQQ history, fading failed up-breakouts showed no edge at 5-10 days, and the "
                    "gamma flip was present in almost every failure, so it did not separate good from bad. Treat grades as a ranking of "
                    "how clean the setup looks, not a forecast.")
