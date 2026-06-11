# =============================================================================
# File    : scripts/build_manifest.py
# Purpose : One-time script to scan all .npy files in data/, extract metadata
#           (shape, band, station, date, freq range, PSD stats), and write
#           outputs/manifest_v2.csv used by all downstream pipeline steps.
# Run     : python scripts/build_manifest.py
# =============================================================================

import numpy as np
import os
import re
import csv
from collections import Counter

DATA_DIR = "data"
OUT_CSV  = "outputs/manifest_v2.csv"

BAND_MAP = {
    "fm": "FM", "gsm": "GSM", "lte": "LTE",
    "dab": "DAB", "dvbt": "DVBT", "tetra": "TETRA"
}

def parse_band(filename):
    for tag, name in BAND_MAP.items():
        if f"_{tag}_" in filename.lower():
            return name
    return "UNKNOWN"

def parse_freq_range(filename):
    m = re.search(r'_(\d+)_(\d+)_', filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

rows = []
errors = []

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".npy"):
        continue

    fpath = os.path.join(DATA_DIR, fname)

    try:
        arr = np.load(fpath, mmap_mode='r')
        if arr.size == 0:
            raise ValueError("zero-size array")
        n_time, n_freq = arr.shape
        val_min  = float(arr.min())
        val_max  = float(arr.max())
        val_mean = float(arr.mean())
    except Exception as e:
        errors.append((fname, str(e)))
        continue

    parts   = fname.split("__")
    station = parts[0] if len(parts) >= 3 else "unknown"
    date    = parts[1] if len(parts) >= 3 else "unknown"

    band            = parse_band(fname)
    freq_lo, freq_hi = parse_freq_range(fname)
    bw_mhz          = (freq_hi - freq_lo) if (freq_lo and freq_hi) else None

    rows.append({
        "filename":    fname,
        "station":     station,
        "date":        date,
        "band":        band,
        "freq_lo_mhz": freq_lo,
        "freq_hi_mhz": freq_hi,
        "bw_mhz":      bw_mhz,
        "n_time":      n_time,
        "n_freq":      n_freq,
        "val_min":     round(val_min, 4),
        "val_max":     round(val_max, 4),
        "val_mean":    round(val_mean, 4),
    })

os.makedirs("outputs", exist_ok=True)
fieldnames = ["filename","station","date","band","freq_lo_mhz","freq_hi_mhz",
              "bw_mhz","n_time","n_freq","val_min","val_max","val_mean"]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Manifest written: {len(rows)} files → {OUT_CSV}")

if errors:
    print(f"\n⚠️  {len(errors)} files failed to load:")
    for fname, err in errors:
        print(f"  {fname}: {err}")

print("\nFiles per band:")
for band, count in sorted(Counter(r["band"] for r in rows).items()):
    print(f"  {band:6s}: {count}")

print("\nShape range:")
print(f"  n_time : {min(r['n_time'] for r in rows)} – {max(r['n_time'] for r in rows)}")
print(f"  n_freq : {min(r['n_freq'] for r in rows)} – {max(r['n_freq'] for r in rows)}")