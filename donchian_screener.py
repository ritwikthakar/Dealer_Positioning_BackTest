"""Donchian setup screener (S&P 500 by default) using ADX and ATR.

Four setups, both directions:
  1. breakout       close beyond the PRIOR 20-day high/low (grade uses 55-day, 200 EMA, volume, gap, ADX)
  2. pullback       55-day extreme in the last 15 bars, then back to the channel midpoint, trend intact
  3. squeeze        narrow channel + low ADX (watch), and the first break out of it
  4. failed_break   close beyond the 20-day level, then first close back inside within 1-3 bars

Colab:   run()                      -> DataFrame, also saved to donchian_setups.csv
         run(["AAPL", "MSFT"])      -> your own list
Shell:   python3 donchian_screener.py [TICKER ...] [--lookback 3] [--out file.csv]

Channels use the PRIOR bars only (shift(1)); a channel that includes today can never be broken.
Wilder ATR/ADX. Output is a candidate list, not validated trade signals.
"""
import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------- indicators
def wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_indicators(df, n=20, m=55, ema_n=200, adx_n=14, sq_win=252):
    d = df[FIELDS].astype(float).copy()
    o, h, l, c, v = (d[k] for k in FIELDS)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = wilder(tr, adx_n)

    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index).where(up.notna())
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index).where(up.notna())
    pdi = 100 * wilder(pdm, adx_n) / atr
    mdi = 100 * wilder(mdm, adx_n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)

    d["tr"], d["atr"], d["pdi"], d["mdi"], d["adx"] = tr, atr, pdi, mdi, wilder(dx, adx_n)
    d["hi"], d["lo"] = h.rolling(n).max().shift(1), l.rolling(n).min().shift(1)
    d["hiM"], d["loM"] = h.rolling(m).max().shift(1), l.rolling(m).min().shift(1)
    d["mid"] = (d["hi"] + d["lo"]) / 2
    d["ema"] = c.ewm(span=ema_n, adjust=False, min_periods=ema_n).mean()
    d["ema_up"] = d["ema"] > d["ema"].shift(20)
    d["vol_ratio"] = v / v.rolling(20).mean().shift(1)
    d["gap_atr"] = (o - pc).abs() / atr.shift(1)
    mp = max(sq_win // 2, 30)
    d["w_pct"] = ((d["hi"] - d["lo"]) / c).rolling(sq_win, min_periods=mp).rank(pct=True)
    d["atr_pct"] = (atr / c).rolling(sq_win, min_periods=mp).rank(pct=True)
    return d


# ---------------------------------------------------------------- setups
def _failed(c, brk_new, level, lags=(1, 2, 3), above=True):
    """lag of the first close back inside the level after a fresh break (0 = none); smallest lag wins."""
    out = pd.Series(0, index=c.index)
    for L in sorted(lags, reverse=True):
        lv = level.shift(L)
        cond = brk_new.shift(L, fill_value=False) & ((c <= lv) if above else (c >= lv))
        for k in range(1, L):
            cond &= (c.shift(k) > lv) if above else (c.shift(k) < lv)
        out[cond] = L
    return out


def compute_flags(d, pb_bars=15, sq_pct=0.15, adx_min=20):
    c, h, l = d["Close"], d["High"], d["Low"]
    f = pd.DataFrame(index=d.index)
    f["up"], f["dn"] = c > d["hi"], c < d["lo"]
    f["up_new"] = f["up"] & ~f["up"].shift(1, fill_value=False)
    f["dn_new"] = f["dn"] & ~f["dn"].shift(1, fill_value=False)
    f["upM"], f["dnM"] = c > d["hiM"], c < d["loM"]
    f["pb_up"] = (f["upM"].rolling(pb_bars).max() > 0) & (c <= d["mid"]) & (c > d["lo"]) \
        & (c > d["ema"]) & (d["adx"] >= adx_min)
    f["pb_dn"] = (f["dnM"].rolling(pb_bars).max() > 0) & (c >= d["mid"]) & (c < d["hi"]) \
        & (c < d["ema"]) & (d["adx"] >= adx_min)
    f["squeeze"] = d["w_pct"] <= sq_pct
    sq_prev = f["squeeze"].shift(1, fill_value=False)
    f["sq_up"], f["sq_dn"] = sq_prev & f["up"], sq_prev & f["dn"]
    f["fb_up_lag"] = _failed(c, f["up_new"], d["hi"], above=True)       # failed up-break
    f["fb_dn_lag"] = _failed(c, f["dn_new"], d["lo"], above=False)      # failed down-break
    return f


def scan_symbol(d, sym, lookback=3, pb_bars=15, sq_pct=0.15, adx_min=20):
    """Signals at the latest bar, plus breakouts / squeeze breaks / failures whose trigger was within `lookback` bars."""
    if len(d) < 3 or not np.isfinite(d["atr"].iloc[-1]):
        return []
    f = compute_flags(d, pb_bars, sq_pct, adx_min)
    c = d["Close"]
    rows = []

    def base(i, setup, side, grade, level, stop, note, extra=None):
        r = d.iloc[i]
        row = dict(ticker=sym, setup=setup, side=side, grade=grade, bars_ago=len(d) - 1 - i,
                   date=d.index[i].date(), close=round(r["Close"], 2), level=round(level, 2) if np.isfinite(level) else np.nan,
                   dist_atr=round((r["Close"] - level) / r["atr"], 2) if np.isfinite(level) else np.nan,
                   stop=round(stop, 2) if np.isfinite(stop) else np.nan, adx=round(r["adx"], 1),
                   atr=round(r["atr"], 2), atr_pct=round(100 * r["atr"] / r["Close"], 2), note=note)
        if extra:
            row.update(extra)
        rows.append(row)

    for k in range(lookback + 1):
        i = len(d) - 1 - k
        r = d.iloc[i]
        a = r["atr"]
        if not np.isfinite(a) or not np.isfinite(r["hi"]):
            continue
        adx_rising = r["adx"] > d["adx"].iloc[i - 3] if i >= 3 else False
        # 1. breakout (fresh break only)
        for side in ("up", "dn"):
            if not f[side + "_new"].iloc[i]:
                continue
            L = side == "up"
            checks = {
                "55d": bool(f["upM" if L else "dnM"].iloc[i]),
                "200EMA trend": bool((r["Close"] > r["ema"] and r["ema_up"]) if L else (r["Close"] < r["ema"] and not r["ema_up"])),
                "volume": bool(r["vol_ratio"] >= 1),
                "no big gap": bool(r["gap_atr"] <= 1),
                "ADX>=20 rising": bool(r["adx"] >= 20 and adx_rising),
            }
            score = sum(checks.values())
            grade = "A" if score >= 4 and checks["200EMA trend"] else ("B" if score >= 3 else "C")
            lvl = r["hi"] if L else r["lo"]
            ext = abs(r["Close"] - lvl) / a
            miss = [kk for kk, vv in checks.items() if not vv]
            note = ("fails: " + ", ".join(miss) if miss else "all filters pass") + ("; extended >1.5 ATR" if ext > 1.5 else "")
            base(i, "breakout", "long" if L else "short", grade, lvl, lvl - 0.5 * a if L else lvl + 0.5 * a, note)
        # 3b. squeeze break
        for side in ("up", "dn"):
            if not f["sq_" + side].iloc[i]:
                continue
            L = side == "up"
            prev_atr = d["atr"].iloc[i - 1] if i >= 1 else a
            expand = bool(r["tr"] >= 1.2 * prev_atr)
            grade = "A" if expand and r["vol_ratio"] >= 1 else "B"
            lvl = r["hi"] if L else r["lo"]
            note = f"TR/ATR {r['tr'] / prev_atr:.2f}, vol x{r['vol_ratio']:.2f}"
            base(i, "squeeze_break", "long" if L else "short", grade, lvl, r["mid"], note)
        # 4. failed breakout
        for side in ("up", "dn"):
            lag = int(f["fb_" + side + "_lag"].iloc[i])
            if not lag:
                continue
            L = side == "up"                      # failed UP-break -> fade short
            j = i - lag
            lvl = d["hi"].iloc[j] if L else d["lo"].iloc[j]
            seg = d.iloc[j:i + 1]
            trap = seg["High"].max() if L else seg["Low"].min()
            weak = bool(r["adx"] < 25 or r["adx"] <= d["adx"].iloc[i - 3] - 2)       # fading = down 2+ points in 3 bars
            grade = "A" if weak else "B"
            trap_atr = abs(trap - lvl) / a
            note = (f"lag {lag}; ADX {'<25 or fading' if weak else '>=25 and rising (strong trend, riskier fade)'}; "
                    f"trap extent {trap_atr:.2f} ATR; target = channel mid {r['mid']:.2f}")
            base(i, "failed_break", "short" if L else "long", grade, lvl, trap + 0.5 * a if L else trap - 0.5 * a, note)

    # latest-bar states
    i = len(d) - 1
    r, a = d.iloc[i], d["atr"].iloc[i]
    for side, flag in (("long", "pb_up"), ("short", "pb_dn")):
        if f[flag].iloc[i] and np.isfinite(a):
            L = side == "long"
            peak = c.iloc[-pb_bars:].max() if L else c.iloc[-pb_bars:].min()
            depth = abs(peak - r["Close"]) / a
            stop = r["lo"] - 0.5 * a if L else r["hi"] + 0.5 * a
            grade = "A" if r["adx"] >= 25 else "B"
            base(i, "pullback", side, grade, r["mid"], stop, f"{depth:.1f} ATR off the recent extreme; ADX {r['adx']:.0f}")
    if f["squeeze"].iloc[i] and not (f["sq_up"].iloc[i] or f["sq_dn"].iloc[i]):
        grade = "A" if (r["adx"] < 20 and r["atr_pct"] <= 0.30) else "B"
        base(i, "squeeze", "watch", grade, np.nan, np.nan,
             f"channel width pct {r['w_pct']:.2f}, ATR pct {r['atr_pct']:.2f}; alert above {r['hi']:.2f} / below {r['lo']:.2f}")
    # near-level watch is intentionally not included: the four setups above are the screen
    seen, out = set(), []
    for row in rows:                                   # most recent first; keep one per (setup, side)
        key = (row["setup"], row["side"])
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


# ---------------------------------------------------------------- data
def get_sp500_tickers():
    import requests
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    for tbl in pd.read_html(r.text):
        cols = [str(x).lower().strip() for x in tbl.columns]
        if "symbol" in cols:
            s = tbl.iloc[:, cols.index("symbol")].astype(str).str.strip().str.replace(".", "-", regex=False)
            return s.tolist()
    raise ValueError("S&P 500 table not found")


def extract(raw, t):
    """One ticker's OHLCV from a yfinance frame, whichever column level holds the ticker."""
    if isinstance(raw.columns, pd.MultiIndex):
        if t in raw.columns.get_level_values(0):
            df = raw[t]
        elif t in raw.columns.get_level_values(1):
            df = raw.xs(t, axis=1, level=1)
        else:
            return None
    else:
        df = raw
    if not {"High", "Low", "Close"} <= set(df.columns):
        return None
    df = df.copy()
    for k in FIELDS:
        if k not in df.columns:
            df[k] = np.nan
    if df["Open"].isna().all():
        df["Open"] = df["Close"].shift(1)
    if df["Volume"].isna().all():
        df["Volume"] = np.nan
    df = df[FIELDS].dropna(subset=["High", "Low", "Close"])
    return df[~df.index.duplicated()].sort_index()


def drop_partial_bar(df, now=None):
    """Drop today's bar if the US market is still open (daily bar not final)."""
    try:
        from zoneinfo import ZoneInfo
        now = now or dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return df
    if len(df) and df.index[-1].date() == now.date() and (now.weekday() < 5 and now.time() < dt.time(16, 10)):
        return df.iloc[:-1]
    return df


def yf_loader(tickers, start):
    import yfinance as yf
    return yf.download(tickers, start=start, auto_adjust=True, group_by="ticker", threads=True, progress=False)


# ---------------------------------------------------------------- driver
def run(tickers=None, lookback=3, out="donchian_setups.csv", start=None, chunk=80, loader=None, min_rows=260, verbose=True,
        progress=None):
    tickers = tickers or get_sp500_tickers()
    start = start or (dt.date.today() - dt.timedelta(days=900)).isoformat()
    loader = loader or yf_loader
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
                df = extract(raw, t)
                if df is None or df.empty:
                    failed.append(t)
                    continue
                df = drop_partial_bar(df)
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
    res.attrs.update(failed=failed, short=short, n_tickers=len(tickers))
    if not res.empty:
        order = {"breakout": 0, "pullback": 1, "squeeze_break": 2, "squeeze": 3, "failed_break": 4}
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
                cols = ["ticker", "side", "grade", "bars_ago", "close", "level", "dist_atr", "stop", "adx", "atr_pct", "note"]
                print(g[cols].head(25).to_string(index=False))
            if out:
                print(f"\nsaved {out}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--lookback", type=int, default=3)
    ap.add_argument("--out", default="donchian_setups.csv")
    a, _ = ap.parse_known_args(argv)                  # tolerate Colab/Jupyter's extra -f argument
    run(a.tickers or None, lookback=a.lookback, out=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
