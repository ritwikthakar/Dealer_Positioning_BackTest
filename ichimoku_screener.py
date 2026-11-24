"""Ichimoku setup screener (S&P 500 by default) using ADX and ATR, built to match donchian_screener.py.

Four setups, both directions:
  1. tk_cross       Tenkan (9) crosses Kijun (26); graded by cloud position, chikou, future cloud colour, volume, ADX
  2. cloud_break    first close out of the cloud (above Senkou top / below Senkou bottom); same style of grading
  3. kijun_bounce   trend continuation: price on the trend side of the cloud, Kijun sloping with it, low tags Kijun (latest bar)
  4. kumo_twist     Senkou A crosses Senkou B, i.e. the cloud projected 26 bars ahead changes colour

Colab:   run()                      -> DataFrame, also saved to ichimoku_setups.csv
         run(["AAPL", "MSFT"])      -> your own list
Shell:   python3 ichimoku_screener.py [TICKER ...] [--lookback 3] [--out file.csv]

No look-ahead: the cloud drawn at a bar is the Senkou pair computed 26 bars earlier, chikou is judged as today's close vs
the close 26 bars ago, and the "future cloud" is only read at the latest bar, never shifted backwards.
ATR/ADX (Wilder) and the data helpers come from donchian_screener.py. Output is a candidate list, not validated trade signals.
"""
import argparse
import functools
import sys

import numpy as np
import pandas as pd

import donchian_screener as dsc

FIELDS = dsc.FIELDS
MIN_ROWS = 150                       # 52 + 26 bars for the cloud, plus room for ADX to settle
SHIFT = 26


# ---------------------------------------------------------------- indicators
def add_indicators(df, tenkan_n=9, kijun_n=26, senkou_n=52, shift=SHIFT):
    # ATR / ADX / volume ratio from the Donchian module (same Wilder maths); its Donchian columns are left in place and
    # sq_win is kept small so the unused percentile columns stay cheap.
    d = dsc.add_indicators(df, sq_win=60)
    h, l, c = d["High"], d["Low"], d["Close"]

    def mid(n):
        return (h.rolling(n).max() + l.rolling(n).min()) / 2

    d["tenkan"], d["kijun"] = mid(tenkan_n), mid(kijun_n)
    d["sa_raw"] = (d["tenkan"] + d["kijun"]) / 2            # un-shifted spans: the cloud 26 bars into the future
    d["sb_raw"] = mid(senkou_n)
    d["sa"], d["sb"] = d["sa_raw"].shift(shift), d["sb_raw"].shift(shift)     # the cloud as it sits under today's bar
    d["cloud_top"] = d[["sa", "sb"]].max(axis=1, skipna=False)
    d["cloud_bot"] = d[["sa", "sb"]].min(axis=1, skipna=False)
    d["fut_bull"] = d["sa_raw"] > d["sb_raw"]               # colour of the cloud projected ahead
    d["past"] = c.shift(shift)                              # chikou test: today's close vs the close `shift` bars ago
    return d


# ---------------------------------------------------------------- setups
def compute_flags(d, adx_min=20, touch=0.25):
    c, a = d["Close"], d["atr"]
    top, bot = d["cloud_top"], d["cloud_bot"]
    f = pd.DataFrame(index=d.index)
    f["above"], f["below"] = c > top, c < bot
    prev_ok = top.shift(1).notna()
    f["above_new"] = f["above"] & ~f["above"].shift(1, fill_value=False) & prev_ok
    f["below_new"] = f["below"] & ~f["below"].shift(1, fill_value=False) & prev_ok
    tk = d["tenkan"] - d["kijun"]
    f["tk_up_new"] = (tk > 0) & (tk.shift(1) <= 0)
    f["tk_dn_new"] = (tk < 0) & (tk.shift(1) >= 0)
    sp = d["sa_raw"] - d["sb_raw"]
    f["twist_up"] = (sp > 0) & (sp.shift(1) <= 0)
    f["twist_dn"] = (sp < 0) & (sp.shift(1) >= 0)
    kr = d["kijun"] - d["kijun"].shift(5)
    f["kij_up"], f["kij_dn"] = kr > 0, kr < 0
    f["bounce_up"] = f["above"] & f["kij_up"] & (d["Low"] <= d["kijun"] + touch * a) & (c >= d["kijun"]) & (d["adx"] >= adx_min)
    f["bounce_dn"] = f["below"] & f["kij_dn"] & (d["High"] >= d["kijun"] - touch * a) & (c <= d["kijun"]) & (d["adx"] >= adx_min)
    return f


def _cloud_pos(r):
    return "above" if r["Close"] > r["cloud_top"] else ("below" if r["Close"] < r["cloud_bot"] else "inside")


def scan_symbol(d, sym, lookback=3, adx_min=20, touch=0.25):
    """Signals at the latest bar, plus TK crosses / cloud breaks / twists whose trigger was within `lookback` bars."""
    if len(d) < 3 or not np.isfinite(d["atr"].iloc[-1]):
        return []
    f = compute_flags(d, adx_min, touch)
    rows = []
    intraday = dsc.is_intraday(d.index)

    def base(i, setup, side, grade, level, stop, note):
        r = d.iloc[i]
        rows.append(dict(ticker=sym, setup=setup, side=side, grade=grade, bars_ago=len(d) - 1 - i,
                         date=dsc.bar_stamp(d.index[i], intraday), close=round(r["Close"], 2),
                         level=round(level, 2) if np.isfinite(level) else np.nan,
                         dist_atr=round((r["Close"] - level) / r["atr"], 2) if np.isfinite(level) else np.nan,
                         stop=round(stop, 2) if np.isfinite(stop) else np.nan, adx=round(r["adx"], 1),
                         atr=round(r["atr"], 2), atr_pct=round(100 * r["atr"] / r["Close"], 2),
                         cloud=_cloud_pos(r), note=note))

    def fails(checks):
        miss = [k for k, v in checks.items() if not v]
        return "fails: " + ", ".join(miss) if miss else "all filters pass"

    for k in range(lookback + 1):
        i = len(d) - 1 - k
        r = d.iloc[i]
        a = r["atr"]
        if not (np.isfinite(a) and np.isfinite(r["cloud_top"]) and np.isfinite(r["kijun"])):
            continue
        adx_rising = bool(r["adx"] > d["adx"].iloc[i - 3]) if i >= 3 else False
        vol_ok = bool(r["vol_ratio"] >= 1)
        adx_ok = bool(r["adx"] >= 20 and adx_rising)
        thick = (r["cloud_top"] - r["cloud_bot"]) / a

        for side in ("up", "dn"):
            L = side == "up"
            sd = "long" if L else "short"
            chikou = bool(r["Close"] > r["past"]) if L else bool(r["Close"] < r["past"])
            fut = bool(r["fut_bull"]) if L else not bool(r["fut_bull"])
            tk_ok = bool(r["tenkan"] > r["kijun"]) if L else bool(r["tenkan"] < r["kijun"])
            pos = bool(f["above"].iloc[i]) if L else bool(f["below"].iloc[i])

            # 1. Tenkan / Kijun cross
            if f["tk_" + side + "_new"].iloc[i]:
                checks = {"price vs cloud": pos, "chikou": chikou, "future cloud": fut, "volume": vol_ok,
                          "ADX>=20 rising": adx_ok}
                score = sum(checks.values())
                grade = "A" if score >= 4 and pos else ("B" if score >= 3 else "C")
                lvl = r["kijun"]
                stop = min(lvl, r["Close"]) - 0.5 * a if L else max(lvl, r["Close"]) + 0.5 * a
                ext = abs(r["Close"] - lvl) / a
                base(i, "tk_cross", sd, grade, lvl, stop,
                     f"cloud {_cloud_pos(r)}; {fails(checks)}" + ("; extended >1.5 ATR from Kijun" if ext > 1.5 else ""))

            # 2. cloud break (fresh close out of the cloud)
            if f[("above" if L else "below") + "_new"].iloc[i]:
                checks = {"TK aligned": tk_ok, "chikou": chikou, "future cloud": fut, "volume": vol_ok,
                          "ADX>=20 rising": adx_ok}
                score = sum(checks.values())
                grade = "A" if score >= 4 else ("B" if score >= 3 else "C")
                lvl = r["cloud_top"] if L else r["cloud_bot"]
                stop = r["cloud_bot"] - 0.5 * a if L else r["cloud_top"] + 0.5 * a
                ext = abs(r["Close"] - lvl) / a
                base(i, "cloud_break", sd, grade, lvl, stop,
                     f"{fails(checks)}; cloud {thick:.1f} ATR thick" + ("; extended >1.5 ATR" if ext > 1.5 else ""))

            # 4. kumo twist (projected cloud changes colour)
            if f["twist_" + side].iloc[i]:
                checks = {"price vs cloud": pos, "TK aligned": tk_ok, "chikou": chikou, "ADX>=20 rising": adx_ok}
                score = sum(checks.values())
                grade = "A" if score >= 3 and pos else ("B" if score >= 2 else "C")
                lvl = r["cloud_top"] if L else r["cloud_bot"]
                stop = min(r["Close"], r["cloud_bot"]) - 0.5 * a if L else max(r["Close"], r["cloud_top"]) + 0.5 * a
                base(i, "kumo_twist", sd, grade, lvl, stop,
                     f"projected cloud turned {'bullish' if L else 'bearish'}; cloud {_cloud_pos(r)}; {fails(checks)}")

    # 3. Kijun bounce: latest bar only
    i = len(d) - 1
    r, a = d.iloc[i], d["atr"].iloc[i]
    for side, flag in (("long", "bounce_up"), ("short", "bounce_dn")):
        if f[flag].iloc[i] and np.isfinite(a):
            L = side == "long"
            chikou = bool(r["Close"] > r["past"]) if L else bool(r["Close"] < r["past"])
            grade = "A" if (r["adx"] >= 25 and chikou) else "B"
            reach = ((r["Low"] - r["kijun"]) if L else (r["kijun"] - r["High"])) / a
            stop = r["cloud_top"] - 0.5 * a if L else r["cloud_bot"] + 0.5 * a
            base(i, "kijun_bounce", side, grade, r["kijun"], stop,
                 f"{'low' if L else 'high'} {reach:+.2f} ATR vs Kijun; ADX {r['adx']:.0f}; chikou {'clear' if chikou else 'not clear'}")

    seen, out = set(), []
    for row in rows:                                   # most recent first; keep one per (setup, side)
        key = (row["setup"], row["side"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


# ---------------------------------------------------------------- driver
def run(tickers=None, lookback=3, out="ichimoku_setups.csv", start=None, chunk=80, loader=None, min_rows=MIN_ROWS,
        verbose=True, progress=None, interval="1d"):
    tickers = tickers or dsc.get_sp500_tickers()
    start = start or dsc.default_start(interval)
    loader = loader or functools.partial(dsc.yf_loader, interval=interval)   # a custom loader is still called as loader(part, start)
    sigs, failed, short = [], [], []
    for s in range(0, len(tickers), chunk):
        part = tickers[s:s + chunk]
        if progress:
            progress(s, len(tickers))
        try:
            raw = loader(part, start)
        except ImportError:
            raise                                    # yfinance not installed: say so instead of "no data"
        except Exception as e:
            failed += part
            print(f"download failed for {part[0]}..{part[-1]}: {e}")
            continue
        for t in part:
            try:
                df = dsc.extract(raw, t)
                if df is None or df.empty:
                    failed.append(t)
                    continue
                df = dsc.drop_partial_bar(df, interval=interval)
                if len(df) < min_rows:
                    short.append(t)
                    continue
                sigs += scan_symbol(add_indicators(df), t, lookback=lookback)
            except Exception as e:
                failed.append(t)
                print(f"{t} skipped: {e}")
    if progress:
        progress(len(tickers), len(tickers))
    res = pd.DataFrame(sigs)
    res.attrs.update(failed=failed, short=short, n_tickers=len(tickers), min_rows=min_rows, interval=interval)
    if not res.empty:
        order = {"tk_cross": 0, "cloud_break": 1, "kijun_bounce": 2, "kumo_twist": 3}
        res["_o"] = res["setup"].map(order)
        res = res.sort_values(["_o", "grade", "bars_ago", "ticker"]).drop(columns="_o").reset_index(drop=True)
        if out:
            res.to_csv(out, index=False)
    if verbose:
        print(f"\n{len(tickers)} tickers | {len(res)} signals | no data/errors {len(failed)} | too little history {len(short)}")
        if failed:
            print("  no data:", ", ".join(failed[:25]) + (" ..." if len(failed) > 25 else ""))
        if not res.empty:
            for setup, g in res.groupby("setup", sort=False):
                print(f"\n== {setup} ({len(g)}) ==")
                cols = ["ticker", "side", "grade", "bars_ago", "close", "level", "dist_atr", "stop", "adx", "atr_pct", "cloud", "note"]
                print(g[cols].head(25).to_string(index=False))
            if out:
                print(f"\nsaved {out}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--lookback", type=int, default=3)
    ap.add_argument("--out", default="ichimoku_setups.csv")
    ap.add_argument("--interval", default="1d", choices=sorted(dsc.INTERVALS.values()), help="bar size: 1d (default) or 1h")
    a, _ = ap.parse_known_args(argv)                  # tolerate Colab/Jupyter's extra -f argument
    run(a.tickers or None, lookback=a.lookback, out=a.out, interval=a.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
