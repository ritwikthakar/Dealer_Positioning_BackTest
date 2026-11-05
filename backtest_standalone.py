"""Backtest of the proposed market-sentiment scoring on QQQ option_history.csv.

Signals use only data available at each day's close; forward returns start from that close.
Usage: python3 -I backtest.py [path_to_option_history_csv]
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PATH = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/option_history_qqq.csv'
HORIZONS = (1, 5, 10)
WARMUP = 200          # drop rows before the 200 EMA is meaningful
BANDS = ['Bearish', 'Mild Bearish', 'Neutral', 'Mild Bullish', 'Bullish']

# weights from the proposed table; flow/walls are not in this file so flow uses a PCR proxy
W = dict(price=0.25, gex=0.30, flow=0.20, vol=0.15, dex=0.10)


def num(s):
    """Parse '1.2B', '875.91M', '754.93K', '21.23%', '—' into floats."""
    s = s.astype(str).str.replace(',', '').str.strip()
    mult = s.str[-1].map({'K': 1e3, 'M': 1e6, 'B': 1e9}).fillna(1.0)
    base = pd.to_numeric(s.str.replace(r'[KMB%]$', '', regex=True), errors='coerce')
    return base * mult


def clip(x):
    return x.clip(-1, 1)


def zscore_roll(x, n=120):
    return (x - x.rolling(n, min_periods=60).mean()) / x.rolling(n, min_periods=60).std()


d = pd.read_csv(PATH)
d['Date'] = pd.to_datetime(d['Date'])
d = d.sort_values('Date').reset_index(drop=True)
c = d['Close Price']
for col in ['GEX Net OI', 'DEX Net OI', 'IV 7d', 'IV 30d', 'IV 90d']:
    d[col] = num(d[col])

# ---------- components, each in [-1, 1], + = bullish ----------
ema200 = c.ewm(span=200, adjust=False).mean()
hi20, lo20 = c.rolling(20).max(), c.rolling(20).min()
hi55, lo55 = c.rolling(55).max(), c.rolling(55).min()
pos20 = (c - lo20) / (hi20 - lo20)          # 0 = at 20d low, 1 = at 20d high
pos55 = (c - lo55) / (hi55 - lo55)
dist = (c / ema200 - 1) / 0.04               # +-4% from 200 EMA = full score
slope = (ema200 - ema200.shift(20)) / ema200.shift(20) / 0.02
d['price'] = clip(0.35 * clip(dist) + 0.25 * (2 * pos20 - 1) + 0.25 * (2 * pos55 - 1) + 0.15 * clip(slope))

d['dex'] = clip(zscore_roll(d['DEX Net OI']) / 2)
# GEX: sign of net gamma is a regime, not direction. Direction input = change in net GEX z-score
d['gex_z'] = zscore_roll(d['GEX Net OI'])
d['gex'] = clip(d['gex_z'] / 2)              # as proposed: positive GEX scores bullish
d['flow'] = clip(-(zscore_roll(d['Option Volume Put-Call Ratio'])) / 2)   # high PCR = bearish proxy

# own IV rank from IV 30d (the file's rank is blank for the first ~240 rows)
ivr = (d['IV 30d'] - d['IV 30d'].rolling(252, min_periods=120).min()) / (
    d['IV 30d'].rolling(252, min_periods=120).max() - d['IV 30d'].rolling(252, min_periods=120).min())
term = (d['IV 7d'] - d['IV 90d'])            # inverted (>0) = stress
d['vol'] = clip(-(0.5 * (2 * ivr - 1) + 0.5 * clip(term / 5)))   # high/inverted vol = bearish

d['score'] = sum(W[k] * d[k] for k in W)
# bands on the composite: fixed thresholds so they are not fit to the sample
d['band'] = pd.cut(d['score'], [-9, -0.30, -0.10, 0.10, 0.30, 9], labels=BANDS)

# gamma regime: net GEX OI sign
d['gamma'] = np.where(d['GEX Net OI'] < 0, 'Negative', 'Positive')

for h in HORIZONS:
    d[f'f{h}'] = c.shift(-h) / c - 1
    d[f'mfe{h}'] = (c[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1) / c - 1)   # worst drawdown in window

t = d.iloc[WARMUP:].dropna(subset=['score']).copy()
print(f'Sample: {t.Date.iloc[0].date()} to {t.Date.iloc[-1].date()}  n={len(t)}')
print(f'Unconditional fwd return (%): ' + ', '.join(f'{h}d {t[f"f{h}"].mean()*100:.2f}' for h in HORIZONS))
print(f'Unconditional % up days over: ' + ', '.join(f'{h}d {(t[f"f{h}"]>0).mean()*100:.0f}' for h in HORIZONS))

# ---------- 1. information coefficient per component and composite ----------
print('\n== Spearman IC vs forward return (p-values naive; overlapping windows overstate significance) ==')
rows = []
for k in ['price', 'gex', 'flow', 'vol', 'dex', 'score']:
    row = {'signal': k}
    for h in HORIZONS:
        s = t[[k, f'f{h}']].dropna()
        ic, p = spearmanr(s[k], s[f'f{h}'])
        row[f'IC_{h}d'] = round(ic, 3)
        row[f'p_{h}d'] = round(p, 3)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ---------- 2. band outcomes ----------
print('\n== Forward returns by sentiment band (mean %, hit-rate % up, count) ==')
out = []
for b in BANDS:
    s = t[t.band == b]
    r = {'band': b, 'n': len(s)}
    for h in HORIZONS:
        r[f'{h}d_mean'] = round(s[f'f{h}'].mean() * 100, 2) if len(s) else np.nan
        r[f'{h}d_up%'] = round((s[f'f{h}'] > 0).mean() * 100) if len(s) else np.nan
    out.append(r)
print(pd.DataFrame(out).to_string(index=False))

# ---------- 3. band x gamma regime (the strategy-selection grid) ----------
print('\n== Band x gamma regime, 5d forward (mean %, 5d down>1.5% frequency, n) ==')
g = []
for b in BANDS:
    for gr in ('Positive', 'Negative'):
        s = t[(t.band == b) & (t.gamma == gr)]
        if len(s) == 0:
            g.append((b, gr, 0, np.nan, np.nan, np.nan)); continue
        g.append((b, gr, len(s), round(s.f5.mean() * 100, 2), round(s.f5.std() * 100, 2),
                  round((s.f5 < -0.015).mean() * 100)))
print(pd.DataFrame(g, columns=['band', 'gamma', 'n', 'mean5d%', 'std5d%', 'P(5d<-1.5%)%']).to_string(index=False))

# ---------- 4. does gamma regime itself separate realised vol? ----------
t['absf5'] = t.f5.abs()
print('\n== Gamma regime vs next-5d realised move ==')
print(t.groupby('gamma').agg(n=('f5', 'size'), mean5=('f5', 'mean'), abs5=('absf5', 'mean'), std5=('f5', 'std')).round(4).to_string())

# ---------- 5. stability: first half vs second half ----------
print('\n== IC stability, composite score vs 5d return, by half ==')
half = len(t) // 2
for name, s in (('first half', t.iloc[:half]), ('second half', t.iloc[half:])):
    s = s[['score', 'f5']].dropna()
    print(name, round(spearmanr(s.score, s.f5)[0], 3), 'n', len(s))

# ---------- 6. simple weight check: price-only vs full composite ----------
print('\n== Alternative composites, IC vs 5d / 10d ==')
alts = {
    'as proposed': t.score,
    'price only': t.price,
    'price+dex+vol (no GEX double count)': (0.25 * t.price + 0.10 * t.dex + 0.15 * t.vol) / 0.5,
    'equal weight': (t.price + t.gex + t.flow + t.vol + t.dex) / 5,
}
for k, v in alts.items():
    r5 = spearmanr(v[t.f5.notna()], t.f5.dropna())[0]
    r10 = spearmanr(v[t.f10.notna()], t.f10.dropna())[0]
    print(f'{k:40s} IC5={r5:+.3f}  IC10={r10:+.3f}')

# ---------- latest reading ----------
last = d.iloc[-1]
print('\n== Latest (', last.Date.date(), ') ==')
print({k: round(float(last[k]), 2) for k in ['price', 'gex', 'flow', 'vol', 'dex', 'score']}, last.band, last.gamma)

t.to_csv('backtest_rows.csv', index=False)
