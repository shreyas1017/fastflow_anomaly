"""
Fast diagnostic: score just 2 normal + 2 anomalous files with warmup=1.0 
to check if there's any signal. Uses PYTORCH_NO_CUDA_MEMORY_CACHING to 
reduce memory, and loads the model once.
"""
import sys, os
import numpy as np
import torch
from pathlib import Path

# Ensure output is not buffered
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastflow.model import FastFlow

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_PATH = BASE_DIR / "outputs" / "checkpoints" / "best.pt"
TEST_DIR = BASE_DIR / "test_set"
device = torch.device("cpu")

# ── Load model ─────────────────────────────────────────────────────────────
print("Loading model...", flush=True)
ckpt = torch.load(CKPT_PATH, map_location=device)
model = FastFlow(flow_layers=8, flow_hidden_ratio=1.0)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()
print(f"Loaded epoch {ckpt['epoch']}, val_nll={ckpt['val_nll']:.2f}", flush=True)

# ── Helpers ────────────────────────────────────────────────────────────────
class WarmupNormalizer:
    def __init__(self, warmup_fraction=1.0, clip_val=5.0, eps=1e-6):
        self.wf = warmup_fraction
        self.clip = clip_val
        self.eps = eps
    def fit_transform(self, arr):
        n = max(1, int(arr.shape[0] * self.wf))
        w = arr[:n]
        med = np.median(w, axis=0).astype(np.float32)
        iqr = (np.percentile(w, 75, axis=0) - np.percentile(w, 25, axis=0)).astype(np.float32)
        normed = (arr - med) / (iqr + self.eps)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

def extract_patches(arr, ph=32, pw=32, sh=16, sw=16, max_edge=0.20, cv=5.0):
    T, F = arr.shape
    if T < ph or F < pw:
        return np.empty((0, ph, pw), dtype=np.float32)
    patches = []
    for i in range((T - ph) // sh + 1):
        for j in range((F - pw) // sw + 1):
            p = arr[i*sh:i*sh+ph, j*sw:j*sw+pw]
            if (np.abs(p) >= cv - 1e-4).sum() / p.size <= max_edge:
                patches.append(p.copy())
    return np.stack(patches) if patches else np.empty((0, ph, pw), dtype=np.float32)

def score_file(fpath, normalizer, model):
    arr = np.load(fpath).astype(np.float32)
    normed = normalizer.fit_transform(arr)
    patches = extract_patches(normed)
    if len(patches) == 0:
        return None, None, None
    patches_t = torch.from_numpy(patches).unsqueeze(1)
    scores_list = []
    with torch.no_grad():
        for i in range(0, len(patches_t), 256):
            s = model.anomaly_score(patches_t[i:i+256]).numpy()
            scores_list.append(s)
    scores = np.concatenate(scores_list)
    k = max(5, int(0.01 * len(scores)))
    file_score = float(np.mean(np.sort(scores)[-k:]))
    return file_score, scores, len(patches)

# ── Score ALL files ────────────────────────────────────────────────────────
normalizer = WarmupNormalizer(warmup_fraction=1.0)

normal_files = sorted((TEST_DIR / "normal").rglob("*.npy"))
anom_files = sorted((TEST_DIR / "anomalous").rglob("*.npy"))

print(f"\n{'='*70}", flush=True)
print(f"Scoring {len(normal_files)} normal + {len(anom_files)} anomalous files", flush=True)
print(f"Using WarmupNormalizer(warmup_fraction=1.0) to match training", flush=True)
print(f"{'='*70}\n", flush=True)

all_results = []

for label_name, files, label in [("NORMAL", normal_files, 0), ("ANOMALOUS", anom_files, 1)]:
    print(f"\n--- {label_name} ({len(files)} files) ---", flush=True)
    for i, f in enumerate(files):
        print(f"  [{i+1}/{len(files)}] {f.name}...", end=" ", flush=True)
        fs, scores, np_ = score_file(f, normalizer, model)
        if fs is None:
            print("SKIPPED", flush=True)
            continue
        print(f"score={fs:.1f} (patches={np_}, mean={scores.mean():.1f}, std={scores.std():.1f})", flush=True)
        all_results.append({"name": f.name, "label": label, "score": fs, 
                           "label_name": label_name, "n_patches": np_,
                           "mean": scores.mean(), "std": scores.std()})

# ── Analysis ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print("ANALYSIS", flush=True)
print(f"{'='*70}", flush=True)

normal_scores = [r["score"] for r in all_results if r["label"] == 0]
anom_scores = [r["score"] for r in all_results if r["label"] == 1]

ns = np.array(normal_scores)
as_ = np.array(anom_scores)

print(f"\nNormal  ({len(ns)} files): mean={ns.mean():.1f}, std={ns.std():.1f}, "
      f"min={ns.min():.1f}, max={ns.max():.1f}", flush=True)
print(f"Anomalous ({len(as_)} files): mean={as_.mean():.1f}, std={as_.std():.1f}, "
      f"min={as_.min():.1f}, max={as_.max():.1f}", flush=True)
print(f"Gap (anom_mean - normal_mean): {as_.mean() - ns.mean():.1f}", flush=True)

from sklearn.metrics import roc_auc_score, average_precision_score
labels = [0]*len(ns) + [1]*len(as_)
all_s = np.concatenate([ns, as_])
auroc = roc_auc_score(labels, all_s)
ap = average_precision_score(labels, all_s)
print(f"\nAUROC: {auroc:.4f}", flush=True)
print(f"AP:    {ap:.4f}", flush=True)

# Show per-file classification at optimal threshold
from sklearn.metrics import f1_score as f1_fn
# Try multiple thresholds
best_f1 = 0
best_thresh = 0
for p in range(1, 100):
    t = np.percentile(all_s, p)
    preds = (all_s > t).astype(int)
    f1 = f1_fn(labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best F1: {best_f1:.4f} (threshold={best_thresh:.1f})", flush=True)

# Per-file verdict
print(f"\n{'='*70}", flush=True)
print("PER-FILE VERDICT (threshold={:.1f})".format(best_thresh), flush=True)
print(f"{'='*70}", flush=True)
print(f"{'File':<80} {'True':>6} {'Pred':>6} {'Score':>10} {'Correct':>8}", flush=True)
print("-" * 120, flush=True)

correct = 0
total = len(all_results)
for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
    pred = "ANOM" if r["score"] > best_thresh else "NORM"
    true = "ANOM" if r["label"] == 1 else "NORM"
    is_correct = pred == true
    if is_correct:
        correct += 1
    marker = "OK" if is_correct else "MISS"
    name = r["name"][:78]
    print(f"{name:<80} {true:>6} {pred:>6} {r['score']:>10.1f} {marker:>8}", flush=True)

print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)", flush=True)
print("DONE.", flush=True)
