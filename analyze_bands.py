import re, numpy as np
from pathlib import Path

files = list(Path("data").glob("*.npy"))
bands = {}
for f in files:
    m = re.search(r"_(fm|gsm|lte|dab|dvbt|tetra)_", f.name.lower())
    band = m.group(1).upper() if m else "UNKNOWN"
    arr = np.load(f)
    if band not in bands:
        bands[band] = {"count": 0, "shapes": [], "total_size_mb": 0}
    bands[band]["count"] += 1
    bands[band]["shapes"].append(arr.shape)
    bands[band]["total_size_mb"] += f.stat().st_size / 1e6

for b in sorted(bands):
    info = bands[b]
    shapes = info["shapes"]
    freqs = [s[1] for s in shapes]
    times = [s[0] for s in shapes]
    count = info["count"]
    size = info["total_size_mb"]
    print(f"{b}: {count} files, {size:.1f} MB")
    print(f"  Time: min={min(times)}, max={max(times)}, median={np.median(times):.0f}")
    print(f"  Freq: min={min(freqs)}, max={max(freqs)}, median={np.median(freqs):.0f}")
    print()
