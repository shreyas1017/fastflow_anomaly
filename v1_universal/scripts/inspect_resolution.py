# =============================================================================
# File    : scripts/inspect_resolution.py
# Purpose : For each file in manifest_v2.csv, compute Hz-per-bin resolution
#           and patches-at-32-bins to inform patch size and stride decisions.
# Run     : python scripts/inspect_resolution.py
# =============================================================================

import csv
from collections import defaultdict

MANIFEST = "outputs/manifest_v2.csv"

band_stats = defaultdict(list)

with open(MANIFEST) as f:
    for row in csv.DictReader(f):
        bw    = float(row["bw_mhz"]) if row["bw_mhz"] else None
        nfreq = int(row["n_freq"])
        band  = row["band"]
        if bw and nfreq:
            hz_per_bin = (bw * 1e6) / nfreq
            band_stats[band].append((hz_per_bin, nfreq, bw, row["filename"]))

print(f"{'Band':<8} {'Hz/bin min':>12} {'Hz/bin max':>12} {'n_freq min':>12} {'n_freq max':>12}")
print("-" * 60)
for band in sorted(band_stats):
    entries = band_stats[band]
    hz_vals = [e[0] for e in entries]
    nf_vals = [e[1] for e in entries]
    print(f"{band:<8} {min(hz_vals):>12.1f} {max(hz_vals):>12.1f} {min(nf_vals):>12} {max(nf_vals):>12}")

print("\nOutliers (Hz/bin > 50000 or n_freq < 100):")
for band, entries in sorted(band_stats.items()):
    for hz, nfreq, bw, fname in entries:
        if hz > 50000 or nfreq < 100:
            print(f"  {band:<6} {hz:>10.0f} Hz/bin  n_freq={nfreq:<6} bw={bw} MHz  {fname}")