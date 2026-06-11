import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Let's check a few anomalous files to see where the anomaly actually starts.
data_dir = Path("test_set/anomalous")
files = list(data_dir.rglob("*.npy"))[:3]

for f in files:
    arr = np.load(f)
    print(f"{f.name}: shape {arr.shape}")
    
    # Plot the mean energy across frequencies over time
    energy = np.mean(arr, axis=1)
    
    # Simple check: where does energy jump?
    diff = np.abs(np.diff(energy))
    if len(diff) > 0:
        jump_idx = np.argmax(diff)
        print(f"  Max energy jump around time step {jump_idx} (which is {jump_idx/arr.shape[0]*100:.1f}%)")
