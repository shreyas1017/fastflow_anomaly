import os
import re
import json
from pathlib import Path

def extract_band(filename):
    m = re.search(r'_(fm|gsm|lte|dab|dvbt|tetra)_', filename.lower())
    if m:
        return m.group(1).upper()
    return "UNKNOWN"

def generate_manifests():
    data_dir = Path("data")
    test_normal_dir = Path("test_set/normal")
    test_anomalous_dir = Path("test_set/anomalous")
    out_dir = Path("outputs/band_manifests")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    
    # Process Training Data
    for p in data_dir.glob("*.npy"):
        band = extract_band(p.name)
        if band not in manifests:
            manifests[band] = {"train": [], "test_normal": [], "test_anomalous": []}
        manifests[band]["train"].append(p.name)
        
    # Process Test Normal
    for p in test_normal_dir.glob("*.npy"):
        band = extract_band(p.name)
        if band not in manifests:
            manifests[band] = {"train": [], "test_normal": [], "test_anomalous": []}
        manifests[band]["test_normal"].append(p.name)
        
    # Process Test Anomalous
    for p in test_anomalous_dir.glob("*.npy"):
        band = extract_band(p.name)
        if band not in manifests:
            manifests[band] = {"train": [], "test_normal": [], "test_anomalous": []}
        manifests[band]["test_anomalous"].append(p.name)

    # Save to JSON
    for band, data in manifests.items():
        if band == "UNKNOWN":
            print(f"Warning: {len(data['train']) + len(data['test_normal']) + len(data['test_anomalous'])} files matched UNKNOWN band.")
            continue
            
        out_file = out_dir / f"{band}.json"
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"[{band}] Train: {len(data['train'])}, Test Normal: {len(data['test_normal'])}, Test Anomalous: {len(data['test_anomalous'])}")
        print(f"  -> Saved to {out_file}")

if __name__ == "__main__":
    generate_manifests()
