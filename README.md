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

Tests (no Streamlit needed):  python3 test_core.py <folder>   and   python3 test_features.py <folder>
(test_features blocks scipy on purpose: the app must run without it.)
`backtest_standalone.py` is the corrected standalone backtest (fixes a vol-score scaling bug in the earlier copy).

Caveats: backtest covers ~19 months in one bull regime; GEX/DEX/walls are model estimates; multi-expiry P&L is
Black-Scholes with flat IV; POP is a lognormal estimate; this is a research tool, not financial advice.

Updating: always replace app.py AND core.py together (they are versioned as a pair; app.py stops with a clear
message if core.py is out of date). On Streamlit Cloud, commit both files, then use Manage app -> Reboot.

### Link
https://dealerpositioningbacktest-vydwrk6jfuj4ysbptutjxa.streamlit.app/

