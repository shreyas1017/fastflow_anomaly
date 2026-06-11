import numpy as np
from pathlib import Path
import re

files = Path("test_set").rglob("*.npy")
band_shapes = {}

for f in files:
    arr = np.load(f)
    m = re.search(r'_(fm|gsm|lte|dab|dvbt|tetra)_', f.name.lower())
    if not m:
        continue
    band = m.group(1).upper()
    F = arr.shape[1]
    
    if band not in band_shapes:
        band_shapes[band] = set()
    band_shapes[band].add(F)

for band, F_set in band_shapes.items():
    print(f"{band}: F={F_set}")
