"""Tests for the GEX intensity gauge (core.gex_intensity / core.gex_gauge_svg). No data files needed.
Usage: python3 -I test_gex_intensity.py"""
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import core  # noqa: E402

rng = np.random.default_rng(0)


def series(vals):
    return pd.Series(vals, index=pd.date_range("2025-01-01", periods=len(vals), freq="B"))


# --- too little history -> None
assert core.gex_intensity(series(np.arange(10.0))) is None

# --- lopsided history: min -4e9, max +1e9
base = np.concatenate([rng.uniform(-4e9, 1e9, 99)])
base[0], base[1] = -4e9, 1e9                      # pin the extremes

# today at the most negative value -> -100, Extreme negative
g = core.gex_intensity(series(np.append(base, -4e9)))
assert abs(g["score"] + 100) < 1e-9 and g["level"] == "Extreme" and g["sign"] == "Negative", g
assert g["label"] == "Extreme negative"

# today at half the worst negative -> -50, Strong (>= 50)
g = core.gex_intensity(series(np.append(base, -2e9)))
assert abs(g["score"] + 50) < 1e-9 and g["level"] == "Strong", g

# positive side is scaled by the positive max (1e9), not the negative min: +0.5e9 -> +50
g = core.gex_intensity(series(np.append(base, 0.5e9)))
assert abs(g["score"] - 50) < 1e-9 and g["sign"] == "Positive", g

# near zero -> Neutral
g = core.gex_intensity(series(np.append(base, 1e6)))
assert g["level"] == "Neutral" and g["label"] == "Neutral" and abs(g["score"]) < 15, g
g = core.gex_intensity(series(np.append(base, 0.0)))
assert g["score"] == 0.0 and g["level"] == "Neutral", g

# new all-time high / low clip at +-100 and a bigger value never exceeds the scale
g = core.gex_intensity(series(np.append(base, 9e9)))
assert g["score"] == 100.0 and g["level"] == "Extreme", g
g = core.gex_intensity(series(np.append(base, -9e9)))
assert g["score"] == -100.0, g

# all-positive history with no negatives: still defined
g = core.gex_intensity(series(np.linspace(1e8, 2e9, 60)))
assert g["score"] == 100.0 and g["hist_min"] == 1e8

# NaNs ignored; string-ish values coerced
s = series(np.append(base, -2e9))
s.iloc[5] = np.nan
assert len(s) == 100 and core.gex_intensity(s)["n"] == 99      # the NaN is dropped

# lookback window: an old extreme outside the window no longer sets the scale
long = np.concatenate([[-8e9], rng.uniform(-1e9, 1e9, 299), [-1e9]])
full = core.gex_intensity(series(long), window=None)
recent = core.gex_intensity(series(long), window=60)
assert full["n"] == 301 and recent["n"] == 60
assert abs(full["score"]) < abs(recent["score"]) or recent["score"] <= -99, (full["score"], recent["score"])

# percentile: today's value is the lowest in the window -> ~0
g = core.gex_intensity(series(np.append(base, -4e9)))
assert g["pctile"] < 0.02, g["pctile"]

# --- SVG is well-formed XML, has all 7 zones, shows the score and the label
g = core.gex_intensity(series(np.append(base, -3e9)))
svg = core.gex_gauge_svg(g)
root = ET.fromstring(svg)
ns = "{http://www.w3.org/2000/svg}"
assert len(root.findall(f"{ns}path")) == len(core.GEX_GAUGE_ZONES) == 7
assert f'{g["score"]:+.0f}' in svg and g["label"] in svg
assert "nan" not in svg.lower()

# needle points left for negative, right for positive, straight up at zero
def needle_x(gi):
    r = ET.fromstring(core.gex_gauge_svg(gi))
    ln = r.find(f"{ns}line")
    return float(ln.get("x2"))

assert needle_x(core.gex_intensity(series(np.append(base, -3e9)))) < 150
assert needle_x(core.gex_intensity(series(np.append(base, 0.8e9)))) > 150
assert abs(needle_x(core.gex_intensity(series(np.append(base, 0.0)))) - 150) < 0.1

# --- the version stamp in core.py matches the one app.py demands
import re
app_src = open(__file__.rsplit("/", 1)[0] + "/app.py" if "/" in __file__ else "app.py").read()
assert re.search(rf'EXPECTED_CORE = "{re.escape(core.CORE_VERSION)}"', app_src), "bump EXPECTED_CORE in app.py"

print("test_gex_intensity: all checks passed")
