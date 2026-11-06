# Regime & Strategy Selector

Layered scorer for QQQ-style option data (OptionCharts + Barchart exports).

    pip install -r requirements.txt
    set QQQ_DATA_DIR=C:\path\to\csvs      (or paste the folder in the sidebar, or upload the files)
    streamlit run app.py

Files are recognised by their columns, so names don't matter. Required: option history + option chain.
Optional (used when present): flow, unusual activity, DEX/GEX by expiration, GEX by strike, expected move.

Layers: direction score (price 60 / flow 25 / DEX 15, editable) -> gates (IV rank, front-back IV edge)
-> gamma regime (sign of net GEX over the trade's expiry window, near-flip buffer) -> matrix -> strikes
from expected-move bands and GEX walls. Calendars and condors are blocked when their gate fails.

Price chart: candlesticks need open/high/low. The history file only has closes, so by default the app draws
close-to-close candles (no wicks) and says so. Add a daily QQQ OHLC export (TradingView, Yahoo or Barchart CSV) to
the same folder / uploader for true candles; it is detected automatically.

Dealer cards: GEX and DEX trend (strengthening / weakening / steady, from the history file), and the aggregate gamma
flip (zero-crossing of the GEX-by-strike profile, plus the gamma-weighted per-expiry flip used for the near-flip check).

Open-interest change (display only, never in the score): the chain's oi_chg column is parsed ('+9 (+40.91%)'; a dash
means unknown, not zero). The Score tab shows net call/put OI change, put share of adds, one-day turnover and a
build / erode tag on the put and call walls. The Dealer map tab has the change by strike, by expiration and by
contract, with d GEX in the same units as the GEX file. The Strategies tab shows OI, change and trend for every leg.
A daily snapshot log (one row per as-of date) can be saved or downloaded so the signal can be tested once there is
enough history; one day of data cannot be backtested.

Tests (no Streamlit needed):  python3 test_core.py <folder>, python3 test_features.py <folder>, python3 test_oi.py <folder>
(test_features blocks scipy on purpose: the app must run without it.)
`backtest_standalone.py` is the corrected standalone backtest (fixes a vol-score scaling bug in the earlier copy).

Caveats: backtest covers ~19 months in one bull regime; GEX/DEX/walls are model estimates; multi-expiry P&L is
Black-Scholes with flat IV; POP is a lognormal estimate; this is a research tool, not financial advice.

Updating: always replace app.py AND core.py together (they are versioned as a pair; app.py stops with a clear
message if core.py is out of date). On Streamlit Cloud, commit both files, then use Manage app -> Reboot.


## Donchian screener tab
Seventh tab ("Donchian screener"); it also appears before any option files are loaded. It scans the S&P 500 (or your own
tickers) for breakout, pullback, squeeze and failed-breakout setups, grades them with ADX and ATR, and charts any hit.
Files: `screener_tab.py` (UI) and `donchian_screener.py` (logic, also runnable alone: `python3 donchian_screener.py AAPL MSFT`).
Needs `yfinance` and `lxml` (see requirements.txt) and internet access from wherever the app runs.
Tests: `python3 test_screener_tab.py` (tab, stand-in Streamlit) and `python3 test_screener.py` (screener logic).

## Link
https://dealerpositioningbacktest-vydwrk6jfuj4ysbptutjxa.streamlit.app/
