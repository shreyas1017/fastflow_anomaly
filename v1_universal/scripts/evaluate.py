# =============================================================================
# File    : scripts/evaluate.py
# Purpose : Evaluate FastFlow model on test_set/normal and test_set/anomalous.
#           Uses the robust WarmupNormalizer and scores patches.
# =============================================================================

import sys, os, re, gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastflow.model import FastFlow

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_PATH = BASE_DIR / "outputs" / "checkpoints" / "best.pt"
TEST_DIR = BASE_DIR / "test_set"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Preprocessing Helpers ─────────────────────────────────────────────────
import json

stats_path = OUT_DIR / "norm_stats.json"
if stats_path.exists():
    with open(stats_path) as f:
        band_stats = json.load(f)
else:
    print("Warning: norm_stats.json not found! Falling back to per-file.")
    band_stats = {}

def extract_band(fname):
    m = re.search(r'_(fm|gsm|lte|dab|dvbt|tetra)_', fname.lower())
    if m:
        return m.group(1).upper()
    return "UNKNOWN"

class GlobalNormalizer:
    def __init__(self, band_stats, clip_val=5.0):
        self.band_stats = band_stats
        self.clip_val = clip_val

    def transform(self, arr, band):
        if band in self.band_stats:
            med = self.band_stats[band]["median"]
            iqr = self.band_stats[band]["iqr"]
        else:
            med = np.median(arr)
            iqr = np.percentile(arr, 75) - np.percentile(arr, 25) + 1e-6
            
        normed = (arr - med) / iqr
        return np.clip(normed, -self.clip_val, self.clip_val).astype(np.float32)

def extract_patches(arr, patch_h=32, patch_w=32, stride_h=16, stride_w=16, max_edge=0.20, clip_val=5.0):
    T, F = arr.shape
    if T < patch_h or F < patch_w:
        return np.empty((0, patch_h, patch_w), dtype=np.float32)
    patches = []
    for i in range((T - patch_h) // stride_h + 1):
        for j in range((F - patch_w) // stride_w + 1):
            p = arr[i*stride_h : i*stride_h+patch_h, j*stride_w : j*stride_w+patch_w]
            if (np.abs(p) >= clip_val - 1e-4).sum() / p.size <= max_edge:
                patches.append(p.copy())
    return np.stack(patches) if patches else np.empty((0, patch_h, patch_w), dtype=np.float32)
 
# ── Load Model ────────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=device)

# Provide fallbacks if missing (from kaggle.txt arch dict)
flow_layers = ckpt.get("arch", {}).get("flow_layers", 8)
flow_hidden_ratio = ckpt.get("arch", {}).get("flow_hidden_ratio", 1.0)

model = FastFlow(flow_layers=flow_layers, flow_hidden_ratio=flow_hidden_ratio)
state_dict = ckpt["model_state"]
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("model.", "") if k.startswith("model.") else k
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# ── Score Files ───────────────────────────────────────────────────────────
results = []
normalizer = GlobalNormalizer(band_stats)

def score_dir(folder, label):
    files = sorted(folder.rglob("*.npy"))
    total_files = len(files)
    print(f"\nScoring {total_files} files in {folder.name}...")
    for i, f in enumerate(files):
        print(f"  [{i+1}/{total_files}] Processing {f.name}...", end=" ", flush=True)
        arr = np.load(f).astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] < 10 or arr.shape[1] < 2:
            continue
            
        band = extract_band(f.name)
        normed_arr = normalizer.transform(arr, band)
        patches = extract_patches(normed_arr)
        if len(patches) == 0:
            continue
            
        patches_t = torch.from_numpy(patches).unsqueeze(1).to(device)
        
        # Process patches in batches to prevent OOM on CPU/GPU
        batch_size = 512
        scores_list = []
        with torch.no_grad():
            for i in range(0, len(patches_t), batch_size):
                batch = patches_t[i:i+batch_size]
                batch_scores = model.anomaly_score(batch).cpu().numpy()
                scores_list.append(batch_scores)
                
        scores = np.concatenate(scores_list)
        
        # Robust pooling: Mean of the Top 1% most anomalous patches (minimum 5 patches)
        k = max(5, int(0.01 * len(scores)))
        file_score = float(np.mean(np.sort(scores)[-k:]))
        
        # Parse metadata
        fname = f.name
        print(f"Done. Score: {file_score:.2f} ({len(patches_t)} patches)")
        
        injector = "none"
        severity = "none"
        if label == 1:
            m = re.search(r"_(imposter|flooding)_(\w+)_", fname)
            if m:
                injector, severity = m.groups()
                
        results.append({
            "filename": fname,
            "label": label,
            "score": file_score,
            "injector": injector,
            "severity": severity,
            "n_patches": len(patches)
        })

score_dir(TEST_DIR / "normal", 0)
score_dir(TEST_DIR / "anomalous", 1)

# ── Metrics & Output ──────────────────────────────────────────────────────
if not results:
    print("No files were scored. Check your test_set directory.")
    sys.exit(0)

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "eval_scores.csv", index=False)

if df["label"].nunique() > 1:
    auroc = roc_auc_score(df["label"], df["score"])
    ap = average_precision_score(df["label"], df["score"])
    
    normal_scores = df[df["label"] == 0]["score"]
    threshold = np.percentile(normal_scores, 95)  # 5% FPR threshold
    df["pred"] = (df["score"] > threshold).astype(int)
    f1 = f1_score(df["label"], df["pred"])
    
    print("\n=== Evaluation Metrics ===")
    print(f"AUROC: {auroc:.4f}")
    print(f"AP:    {ap:.4f}")
    print(f"F1:    {f1:.4f} (at 95th percentile normal threshold = {threshold:.2f})")
    
    with open(OUT_DIR / "eval_metrics.txt", "w") as f:
        f.write(f"AUROC: {auroc:.4f}\nAP: {ap:.4f}\nF1: {f1:.4f}\nThreshold: {threshold:.4f}\n")
else:
    print("\nSkipping AUROC/AP: need both normal and anomalous files.")

print(f"\nDone! Scores saved to {OUT_DIR / 'eval_scores.csv'}")
