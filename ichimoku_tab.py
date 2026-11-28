"""Streamlit tab: Ichimoku setup screener (TK cross / cloud break / Kijun bounce / Kumo twist) with ADX + ATR grading.
Mirrors screener_tab.py (Donchian) and shares its ticker parsing, S&P 500 list and price cache. Needs ichimoku_screener.py,
donchian_screener.py and screener_tab.py next to it; it does not depend on core.py, so it works before the option files are loaded.
Call render() inside a tab."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import streamlit as st

import screener_tab as dtab          # parse_tickers, _show_vega, cached _sp500 / _one (shared with the Donchian tab)

SETUP_LABEL = {"tk_cross": "TK cross", "cloud_break": "Cloud break", "kijun_bounce": "Kijun bounce", "kumo_twist": "Kumo twist"}
SHOW = ["ticker", "setup", "side", "grade", "bars_ago", "date", "close", "level", "dist_atr", "stop", "adx", "atr_pct", "cloud", "note"]
DEFAULT_TICKERS = "QQQ, SPY, AAPL, NVDA"
HELP = {
    "tk_cross": "Tenkan (9) crosses Kijun (26). Grade from five checks: price on the trend side of the cloud, chikou clear "
                "(close beyond the close 26 bars ago), projected cloud colour agrees, volume above its 20-day average, ADX >= 20 and "
                "rising. A = 4+ checks including the cloud one, B = 3, C otherwise. Stop = 0.5 ATR beyond Kijun (or the close, if "
                "that is further out).",
    "cloud_break": "First close above the cloud top / below the cloud bottom. Checks: Tenkan on the right side of Kijun, chikou "
                   "clear, projected cloud colour agrees, volume, ADX >= 20 and rising. A = 4+, B = 3, C otherwise. Stop = 0.5 ATR "
                   "beyond the far side of the cloud, so a thick cloud means a wide stop (thickness is in the note).",
    "kijun_bounce": "Latest bar only. Price on the trend side of the cloud, Kijun sloping with the trend (vs 5 bars ago), the bar's "
                    "low (high for shorts) within 0.25 ATR of Kijun and the close held on the trend side of it, ADX >= 20. A = ADX >= 25 "
                    "and chikou clear. Stop = 0.5 ATR beyond the near edge of the cloud.",
    "kumo_twist": "Senkou A crosses Senkou B, so the cloud projected 26 bars ahead changes colour. Checks: price vs the current cloud, "
                  "Tenkan vs Kijun, chikou, ADX >= 20 and rising. A = 3+ including price on the trend side of the cloud, B = 2, C "
                  "otherwise. Level = the near edge of the current cloud; stop = 0.5 ATR beyond the cloud or the close, whichever is further.",
}


def _future_slots(last, n):
    """The next n hourly US session bars after `last` (09:30 .. 15:30 starts, weekends skipped, exchange holidays ignored)."""
    if last.tzinfo is not None:
        last = last.tz_convert("America/New_York")
    out, day = [], last.normalize()
    while len(out) < n:
        if day.weekday() < 5:
            for k in range(7):
                t = day.replace(hour=9, minute=30) + pd.Timedelta(hours=k)
                if t > last:
                    out.append(t)
        day = (day + pd.Timedelta(hours=36)).normalize()
    return pd.DatetimeIndex(out[:n])


def candle_spec(d, days=120, height=380, shift=26):
    """Candles + Tenkan, Kijun and the shaded cloud (with the cloud projected `shift` bars ahead), from ichimoku_screener.add_indicators.
    Daily frames get a date axis; intraday frames get an ordinal bar axis (no overnight / weekend gaps)."""
    import donchian_screener as dsc
    x = d.tail(days).copy()
    intraday = dsc.is_intraday(x.index)
    last = d.iloc[-shift:]
    if intraday:
        fut_dates = dtab.stamp_labels(_future_slots(x.index[-1], len(last)))
        x["date"] = dtab.stamp_labels(x.index)
    else:
        x["date"] = x.index
        # projected cloud: today's un-shifted spans belong to the next `shift` sessions (weekends skipped, exchange holidays ignored)
        fut_dates = pd.bdate_range(x.index[-1] + pd.Timedelta(days=1), periods=len(last))
    fut = pd.DataFrame({"date": fut_dates, "sa": last["sa_raw"].to_numpy(), "sb": last["sb_raw"].to_numpy()})
    cl = pd.concat([x[["date", "sa", "sb"]], fut], ignore_index=True).dropna()
    cl["lo"], cl["hi"] = cl[["sa", "sb"]].min(axis=1), cl[["sa", "sb"]].max(axis=1)
    cl["cloud"] = np.where(cl["sa"] >= cl["sb"], "bull", "bear")

    def recs(frame, cols):
        return json.loads(frame[cols].to_json(orient="records", date_format="iso"))

    candles = recs(x.rename(columns=str.lower), ["date", "open", "high", "low", "close", "tenkan", "kijun"])
    lows = np.r_[x["Low"].to_numpy(float), x["tenkan"].to_numpy(float), x["kijun"].to_numpy(float), cl["lo"].to_numpy(float)]
    highs = np.r_[x["High"].to_numpy(float), x["tenkan"].to_numpy(float), x["kijun"].to_numpy(float), cl["hi"].to_numpy(float)]
    lo, hi = float(np.nanmin(lows)), float(np.nanmax(highs))
    pad = (hi - lo) * 0.03 or 1.0
    dom = [lo - pad, hi + pad]
    xe, tip_date = dtab.x_encoding(list(x["date"]) + list(fut_dates) if intraday else None)
    ye = lambda f, **kw: {"field": f, "type": "quantitative", "scale": {"zero": False, "domain": dom}, "axis": {"title": None}, **kw}
    color = {"condition": {"test": "datum.close >= datum.open", "value": "#26a69a"}, "value": "#ef5350"}
    tip = [tip_date] + [
        {"field": f, "type": "quantitative", "title": f.title(), "format": ".2f"} for f in ("open", "high", "low", "close")]
    bar = int(max(2, min(9, 650 / max(len(x) + len(fut), 1) * 0.7)))
    cloud_data = {"values": recs(cl, ["date", "sa", "sb", "lo", "hi", "cloud"])}
    layers = [
        {"data": cloud_data, "mark": {"type": "area", "opacity": 0.22},
         "encoding": {"x": xe, "y": ye("lo", stack=None), "y2": {"field": "hi"},
                      "color": {"field": "cloud", "type": "nominal", "legend": None,
                                "scale": {"domain": ["bull", "bear"], "range": ["#26a69a", "#ef5350"]}}}},
        {"data": cloud_data, "mark": {"type": "line", "color": "#43a047", "strokeWidth": 1, "opacity": 0.8}, "encoding": {"x": xe, "y": ye("sa")}},
        {"data": cloud_data, "mark": {"type": "line", "color": "#e53935", "strokeWidth": 1, "opacity": 0.8}, "encoding": {"x": xe, "y": ye("sb")}},
        {"mark": {"type": "rule"}, "encoding": {"x": xe, "y": ye("low"), "y2": {"field": "high"}, "color": color, "tooltip": tip}},
        {"mark": {"type": "bar", "size": bar}, "encoding": {"x": xe, "y": ye("open"), "y2": {"field": "close"}, "color": color, "tooltip": tip}},
        {"mark": {"type": "line", "color": "#1e88e5", "strokeWidth": 1.4}, "encoding": {"x": xe, "y": ye("tenkan")}},
        {"mark": {"type": "line", "color": "#fb8c00", "strokeWidth": 1.4}, "encoding": {"x": xe, "y": ye("kijun")}},
    ]
    return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "height": height,
            "data": {"values": candles}, "layer": layers}


def render(show_vega=None):
    try:
        import ichimoku_screener as isc
        import donchian_screener as dsc
    except ImportError:
        st.error("ichimoku_screener.py (and donchian_screener.py) must be in the same folder as app.py. Copy them in and reboot the app.")
        return

    st.subheader("Ichimoku setup screener")
    st.caption("Four setups, both directions, graded with ADX and ATR. Classic 9 / 26 / 52 settings; the cloud under a bar is the one "
               "computed 26 bars earlier, so nothing looks ahead. Daily or 1-hour data from Yahoo Finance via yfinance; the latest bar "
               "is dropped until it has finished. Candidate list for review, not trade signals.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    universe = c1.radio("Universe", dtab.UNIVERSES, horizontal=True, key="ich_universe",
                        help="Liquid list: about 90 names with liquid options (the Gamma screener's list). Fast, and every name "
                             "can also be checked for gamma.")
    interval, tf_label = dtab.pick_timeframe(c2, "ich_tf")
    lookback = c3.number_input("Look back (bars)", 0, 10, 3, key="ich_lookback",
                               help="Report TK crosses, cloud breaks and twists whose trigger was within this many bars "
                                    "of the selected timeframe.")
    c4.write("")
    run_it = c4.button("Run screener", key="ich_run")
    custom = ""
    if universe == "My tickers":
        custom = st.text_area("Tickers (comma, space or newline separated)",
                              value=st.session_state.get("scr_tickers") or DEFAULT_TICKERS, key="ich_tickers")

    if run_it:
        try:
            tickers = dtab.universe_tickers(universe, custom)
        except Exception as e:
            st.error(f"Could not load the {universe} universe ({e}). Use 'My tickers' instead.")
            tickers = []
        if not tickers:
            st.warning("No tickers to scan.")
        else:
            bar = st.progress(0.0, text="Downloading and scanning...")

            def cb(done, total):
                bar.progress(min(done / max(total, 1), 1.0), text=f"{done}/{total} tickers")

            try:
                res = isc.run(tickers, lookback=int(lookback), out=None, verbose=False, progress=cb, interval=interval)
                st.session_state["ich_res"] = res
                st.session_state["ich_when"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state["ich_tf_label"] = tf_label
            except ImportError as e:
                st.error(f"Missing package: {e}. Add yfinance and lxml to requirements.txt and reboot the app.")
            except Exception as e:
                st.error(f"Screener failed: {e}")
            bar.empty()

    res = st.session_state.get("ich_res")
    if res is None:
        st.info("Choose a universe and press Run screener. The liquid list (~90 names) is quick; the S&P 500 scan downloads "
                "about 500 tickers and takes a minute or two.")
        _notes()
        return

    meta = res.attrs
    res_interval = meta.get("interval", "1d")           # the timeframe these results were scanned on (not the selector's current value)
    res_label = st.session_state.get("ich_tf_label", "Daily")
    st.caption(f"{res_label} bars | scanned {meta.get('n_tickers', '?')} tickers at {st.session_state.get('ich_when', '')} | "
               f"no data/errors: {len(meta.get('failed', []))} | too little history (<{meta.get('min_rows', isc.MIN_ROWS)} bars): "
               f"{len(meta.get('short', []))}")
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
    pick_setup = f1.multiselect("Setup", list(SETUP_LABEL), default=list(SETUP_LABEL), format_func=lambda k: SETUP_LABEL[k], key="ich_f_setup")
    pick_side = f2.multiselect("Side", ["long", "short"], default=["long", "short"], key="ich_f_side")
    pick_grade = f3.multiselect("Grade", ["A", "B", "C"], default=["A", "B", "C"], key="ich_f_grade")
    max_ago = f4.number_input("Max bars ago", 0, 10, 10, key="ich_f_ago")
    view = res[res["setup"].isin(pick_setup) & res["side"].isin(pick_side) & res["grade"].isin(pick_grade)
               & (res["bars_ago"] <= int(max_ago))]
    if view.empty:
        st.info("Nothing matches these filters.")
        _notes()
        return
    st.dataframe(view[SHOW], hide_index=True)
    st.download_button("Download CSV", view[SHOW].to_csv(index=False).encode("utf-8"),
                       file_name="ichimoku_setups.csv", mime="text/csv", key="ich_dl")
    st.caption("level = Kijun for TK crosses and bounces, the cloud edge for cloud breaks and twists; dist_atr = (close - level) / ATR; "
               "cloud = where the close sits relative to the cloud; stop is a reference level, 0.5 ATR beyond the invalidation point.")

    st.subheader("Chart")
    labels = [f"{r.ticker}  |  {SETUP_LABEL[r.setup]} {r.side} (grade {r.grade})" for r in view.itertuples()]
    choice = st.selectbox("Ticker", labels, key="ich_pick")
    row = view.iloc[labels.index(choice)]
    try:
        raw = dtab._one(row["ticker"], dsc.default_start(res_interval), res_interval)
        if raw is None or raw.empty:
            raise ValueError("no price data returned")
        d = isc.add_indicators(dsc.drop_partial_bar(raw, interval=res_interval))
        shown = 120 if res_interval == "1d" else 200
        dtab._show_vega(candle_spec(d, days=shown), show_vega)
        ahead = "26 sessions ahead (weekends skipped, exchange holidays ignored)" if res_interval == "1d" else \
            "26 hourly bars ahead (weekends skipped, exchange holidays ignored)"
        st.caption(":blue[Tenkan (9)]  :orange[Kijun (26)]  :green[Senkou A]  :red[Senkou B]; shaded = cloud (green bullish, red bearish), "
                   f"extended {ahead}. "
                   f"ADX {d['adx'].iloc[-1]:.1f}, ATR {d['atr'].iloc[-1]:.2f} ({100 * d['atr'].iloc[-1] / d['Close'].iloc[-1]:.2f}% of price).")
        st.line_chart(d["adx"].tail(shown).rename("ADX(14)").to_frame(), height=140)
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
        st.markdown("**Caveat.** The 9 / 26 / 52 periods are the classic defaults, and the ADX 20/25, 0.25 ATR touch and 1.5 ATR "
                    "extension thresholds are starting values, not tuned. Unlike the Donchian screener's notes, nothing here has "
                    "been backtested yet, so treat grades as a ranking of how many conditions line up, not a forecast. Cloud "
                    "signals lag by construction, and a break out of a thick cloud comes with a wide stop.")
        st.markdown("**1-hour mode.** Every period (9 / 26 / 52, ATR, ADX, the 20-bar volume average, 'bars ago', 'look back') then "
                    "counts hourly bars, about 7 per session, so the cloud spans roughly 4 sessions ahead and Kijun about 4 sessions "
                    "back. Hourly volume is lumpy (heavy at the open and close), so the volume check is noisier than on daily bars. "
                    "Bars run 09:30-15:30 New York time, the forming hour is dropped, and Yahoo limits hourly history, so this "
                    "app requests about 150 days.")
