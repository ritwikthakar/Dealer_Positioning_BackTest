"""Streamlit tab: gamma (GEX) screener on free yfinance option chains. Needs gamma_screener.py next to it.
Self-contained like screener_tab.py (no option CSVs needed). Call render() inside a tab."""
import datetime as dt

import pandas as pd
import streamlit as st

import screener_tab as dtab  # parse_tickers and the cached S&P 500 list

SHOW = ["ticker", "regime", "spot", "net_gex_m", "gex_pct_adv", "flip", "spot_vs_flip_pct",
        "call_wall", "put_wall", "put_call_oi", "total_oi", "n_expiries"]


def render():
    try:
        import gamma_screener as gs
    except ImportError:
        st.error("gamma_screener.py is missing. Copy it into the same folder as app.py and reboot the app.")
        return

    st.subheader("Gamma screener")
    st.caption("Net dealer gamma exposure from Yahoo Finance option chains (end-of-day open interest). Calls count positive, "
               "puts negative. gex_pct_adv = net GEX per 1% move as % of 20-day average dollar volume, so names compare. "
               "Most negative first. Candidate list for review, not trade signals.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    universe = c1.radio("Universe", ["Liquid list", "S&P 500", "My tickers"], horizontal=True, key="gam_universe")
    max_dte = c2.number_input("Max DTE", 7, 120, 45, key="gam_dte", help="Only expiries within this many days.")
    cap = c3.number_input("Max tickers", 10, 500, 100, key="gam_cap", help="Caps the S&P 500 scan (each ticker is several Yahoo calls).")
    c4.write("")
    run_it = c4.button("Run screener", key="gam_run")
    custom = ""
    if universe == "My tickers":
        custom = st.text_area("Tickers (comma, space or newline separated)", value="SPY, QQQ, NVDA, TSLA", key="gam_tickers")

    if run_it:
        try:
            if universe == "Liquid list":
                tickers = list(gs.LIQUID)
            elif universe == "S&P 500":
                tickers = dtab._sp500()[:int(cap)]
            else:
                tickers = dtab.parse_tickers(custom)
        except Exception as e:
            st.error(f"Could not load the S&P 500 list ({e}). Use 'Liquid list' or 'My tickers' instead.")
            tickers = []
        if not tickers:
            st.warning("No tickers to scan.")
        else:
            bar = st.progress(0.0, text="Downloading option chains...")

            def cb(done, total):
                bar.progress(min(done / max(total, 1), 1.0), text=f"{done}/{total} tickers")

            try:
                res, strikes, failed = gs.run(tickers, max_dte=int(max_dte), progress=cb)
                st.session_state.update(gam_res=res, gam_strikes=strikes, gam_failed=failed,
                                        gam_when=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
            except ImportError as e:
                st.error(f"Missing package: {e}. Add yfinance to requirements.txt and reboot the app.")
            except Exception as e:
                st.error(f"Screener failed: {e}")
            bar.empty()

    res = st.session_state.get("gam_res")
    if res is None:
        st.info("Choose a universe and press Run screener. The liquid list (~90 names) takes about a minute.")
        _notes()
        return
    failed = st.session_state.get("gam_failed", {})
    st.caption(f"Scanned at {st.session_state.get('gam_when', '')} | usable: {len(res)} | failed: {len(failed)}")
    if failed:
        st.caption("Failed: " + ", ".join(f"{k} ({v})" for k, v in list(failed.items())[:8]) + (" ..." if len(failed) > 8 else ""))
    if res.empty:
        st.info("No usable chains returned.")
        _notes()
        return

    counts = res["regime"].value_counts()
    cols = st.columns(3)
    for col, k in zip(cols, ("Negative", "Near flip", "Positive")):
        col.metric(k, int(counts.get(k, 0)))

    f1, f2, f3 = st.columns(3)
    pick = f1.multiselect("Regime", ["Negative", "Near flip", "Positive"], default=["Negative", "Near flip"], key="gam_f_reg")
    min_oi = f2.number_input("Min total OI", 0, 10_000_000, 5000, key="gam_f_oi")
    max_pct = f3.number_input("Max gex_pct_adv", -1000.0, 1000.0, 1000.0, key="gam_f_pct",
                              help="e.g. 0 keeps only names with negative normalized GEX.")
    view = res[res["regime"].isin(pick) & (res["total_oi"] >= int(min_oi))
               & (res["gex_pct_adv"].fillna(1e9) <= float(max_pct))]
    if view.empty:
        st.info("Nothing matches these filters.")
        _notes()
        return
    st.dataframe(view[SHOW], hide_index=True)
    st.download_button("Download CSV", view[SHOW].to_csv(index=False).encode("utf-8"),
                       file_name="gamma_screen.csv", mime="text/csv", key="gam_dl")
    st.caption("net_gex_m = $ millions per 1% move. spot_vs_flip_pct > 0 means spot is above the zero-gamma flip. "
               "Walls are the strikes with the largest call / put gamma.")

    st.subheader("Strike detail")
    choice = st.selectbox("Ticker", list(view["ticker"]), key="gam_pick")
    row = view[view["ticker"] == choice].iloc[0]
    tbl = st.session_state.get("gam_strikes", {}).get(choice)
    if tbl is not None and not tbl.empty:
        near = tbl[(tbl["strike"] >= row["spot"] * 0.85) & (tbl["strike"] <= row["spot"] * 1.15)]
        st.bar_chart((near.set_index("strike")[["call", "put"]] / 1e6), height=300)
        flip = "none within +/-15%" if pd.isna(row["flip"]) else f"{row['flip']:.2f}"
        st.caption(f"{choice}: spot {row['spot']:.2f}, flip {flip}, call wall {row['call_wall']:.2f}, put wall "
                   f"{row['put_wall']:.2f}. Bars are $M GEX per 1% move by strike (calls up, puts down).")
    _notes()


def _notes():
    with st.expander("How this is computed and what to trust"):
        st.markdown("**Method.** GEX per strike = Black-Scholes gamma x open interest x 100 x spot^2 x 0.01, calls positive, "
                    "puts negative. The flip is where net GEX crosses zero when spot is moved +/-15% with IV and OI held fixed. "
                    "Strikes beyond +/-30% of spot, zero-OI rows and IVs outside 3%-300% are dropped.")
        st.markdown("**Caveats.** Open interest is end-of-day; the dealers-long-calls/short-puts assumption is weak on single "
                    "stocks (better on SPY/QQQ); Yahoo IVs can be stale outside market hours; 0DTE and index options "
                    "carry most gamma on SPX, which Yahoo does not cover well. Use it to shortlist names, then check the strike detail.")
