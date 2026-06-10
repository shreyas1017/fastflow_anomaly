import numpy as np
import os

DATA_DIR = "data"
files = sorted(os.listdir(DATA_DIR))

# Sample a few files from different bands
bands = {}
for f in files:
    if not f.endswith(".npy"):
        continue
    for tag in ["fm", "gsm", "lte", "dab", "dvbt", "tetra"]:
        if f"_{tag}_" in f.lower():
            bands.setdefault(tag, []).append(f)
            break

print("=== Band distribution ===")
for band, flist in sorted(bands.items()):
    print(f"  {band.upper():6s}: {len(flist)} files")

print("\n=== Sample files per band (shape, min, max, mean, std) ===")
for band, flist in sorted(bands.items()):
    sample = flist[:2]  # first 2 per band
    for f in sample:
        arr = np.load(os.path.join(DATA_DIR, f), mmap_mode='r')
        flat = np.array(arr).flatten()
        print(f"  [{band.upper():6s}] {f[:60]:60s}  shape={arr.shape}  min={flat.min():.2f}  max={flat.max():.2f}  mean={flat.mean():.2f}  std={flat.std():.2f}")
