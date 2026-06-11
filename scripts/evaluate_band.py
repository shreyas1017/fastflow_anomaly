#!/usr/bin/env python3
"""
Evaluate a single per-band FastFlow model.

Usage:
    python scripts/evaluate_band.py --band FM
    python scripts/evaluate_band.py --band GSM --ckpt outputs/models/GSM/best.pt

Loads the checkpoint (which has embedded norm_stats), scores all test files
for that band, and reports AUROC and AP.
"""

import argparse
import json
import re
import sys

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fastflow.model import FastFlow


# ═══════════════════════════════════════════════════════════════════════════
# Config (must match training)
# ═══════════════════════════════════════════════════════════════════════════
PATCH_H, PATCH_W = 32, 32
STRIDE_H, STRIDE_W = 16, 16
MAX_EDGE_FRAC = 0.20
CLIP_VAL = 5.0


def normalize(arr, stats):
    normed = (arr - stats["median"]) / stats["iqr"]
    return np.clip(normed, -CLIP_VAL, CLIP_VAL).astype(np.float32)


def extract_patches(arr):
    T, F = arr.shape
    if T < PATCH_H or F < PATCH_W:
        return np.empty((0, PATCH_H, PATCH_W), dtype=np.float32)
    patches = []
    for i in range((T - PATCH_H) // STRIDE_H + 1):
        for j in range((F - PATCH_W) // STRIDE_W + 1):
            p = arr[i * STRIDE_H: i * STRIDE_H + PATCH_H,
                    j * STRIDE_W: j * STRIDE_W + PATCH_W]
            if (np.abs(p) >= CLIP_VAL - 1e-4).sum() / p.size <= MAX_EDGE_FRAC:
                patches.append(p.copy())
    if patches:
        return np.stack(patches)
    return np.empty((0, PATCH_H, PATCH_W), dtype=np.float32)


def score_file(model, fpath, norm_stats, device, batch_size=512):
    """Score a single .npy file. Returns (file_score, n_patches) or None."""
    arr = np.load(fpath).astype(np.float32)
    if arr.ndim != 2 or arr.shape[0] < 10 or arr.shape[1] < 2:
        return None

    normed = normalize(arr, norm_stats)
    patches = extract_patches(normed)
    if len(patches) == 0:
        return None

    patches_t = torch.from_numpy(patches).unsqueeze(1).to(device)

    scores_list = []
    with torch.no_grad():
        for b in range(0, len(patches_t), batch_size):
            batch = patches_t[b:b + batch_size]
            batch_scores = model.anomaly_score(batch).cpu().numpy()
            scores_list.append(batch_scores)

    scores = np.concatenate(scores_list)

    # Top-1% pooling (minimum 5 patches)
    k = max(5, int(0.01 * len(scores)))
    file_score = float(np.mean(np.sort(scores)[-k:]))

    return file_score, len(patches)


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-band FastFlow model")
    parser.add_argument("--band", type=str, required=True,
                        choices=["FM", "GSM", "LTE", "DAB", "DVBT", "TETRA"])
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint (default: outputs/models/{BAND}/best.pt)")
    args = parser.parse_args()

    band = args.band
    ckpt_path = Path(args.ckpt) if args.ckpt else PROJECT_ROOT / "outputs" / "models" / band / "best.pt"
    manifest_path = PROJECT_ROOT / "outputs" / "band_manifests" / f"{band}.json"
    test_dir = PROJECT_ROOT / "test_set"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load checkpoint ──
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    flow_layers = ckpt.get("arch", {}).get("flow_layers", 16)
    flow_hidden = ckpt.get("arch", {}).get("flow_hidden_ratio", 128.0)
    norm_stats = ckpt["norm_stats"]

    print(f"  Band: {ckpt.get('band', band)}, Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Val NLL: {ckpt.get('val_nll', '?')}")
    print(f"  Norm stats: median={norm_stats['median']:.2f}, iqr={norm_stats['iqr']:.2f}")

    model = FastFlow(flow_layers=flow_layers, flow_hidden_ratio=flow_hidden)

    # Strip 'model.' prefix from DataParallel checkpoints
    state_dict = ckpt["model_state"]
    clean = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        clean[new_key] = v
    model.load_state_dict(clean)
    model.to(device)
    model.eval()

    # ── Load manifest ──
    with open(manifest_path) as f:
        manifest = json.load(f)

    # ── Score files ──
    results = []

    # Normal files
    normal_files = manifest["test_normal"]
    print(f"\nScoring {len(normal_files)} normal files...")
    for fname in normal_files:
        fpath = test_dir / "normal" / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        result = score_file(model, fpath, norm_stats, device)
        if result is None:
            print(f"  SKIP (bad data): {fname}")
            continue
        score, n_patches = result
        results.append({"filename": fname, "label": 0, "score": score,
                        "n_patches": n_patches, "type": "normal", "severity": "none"})
        print(f"  [{len(results)}] {fname}: {score:.2f} ({n_patches} patches)")

    # Anomalous files
    anom_files = manifest["test_anomalous"]
    print(f"\nScoring {len(anom_files)} anomalous files...")
    for fname in anom_files:
        fpath = test_dir / "anomalous" / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        result = score_file(model, fpath, norm_stats, device)
        if result is None:
            print(f"  SKIP (bad data): {fname}")
            continue
        score, n_patches = result

        # Extract attack type and severity
        atype, severity = "unknown", "unknown"
        m = re.search(r"__(barrage_jammer|narrowband_jammer|rogue_carrier|gradual_drift)__(subtle|moderate|obvious)__", fname)
        if m:
            atype, severity = m.group(1), m.group(2)

        results.append({"filename": fname, "label": 1, "score": score,
                        "n_patches": n_patches, "type": atype, "severity": severity})
        print(f"  [{len(results)}] {fname}: {score:.2f} ({n_patches} patches) [{atype}/{severity}]")

    # ── Compute metrics ──
    if not results:
        print("\nNo files scored!")
        return

    labels = np.array([r["label"] for r in results])
    scores = np.array([r["score"] for r in results])

    n_normal = (labels == 0).sum()
    n_anom = (labels == 1).sum()

    if n_normal == 0 or n_anom == 0:
        print(f"\nCannot compute AUROC: {n_normal} normal, {n_anom} anomalous files.")
        return

    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    print(f"\n{'=' * 50}")
    print(f"[{band}] Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Normal files:    {n_normal}")
    print(f"  Anomalous files: {n_anom}")
    print(f"  AUROC:           {auroc:.4f}")
    print(f"  AP:              {ap:.4f}")

    # Per-severity breakdown
    print(f"\n  Per-severity AUROC:")
    for sev in ["subtle", "moderate", "obvious"]:
        sev_mask = np.array([r["severity"] == sev for r in results])
        normal_mask = labels == 0
        subset = sev_mask | normal_mask
        if sev_mask.sum() > 0 and normal_mask.sum() > 0:
            sev_auroc = roc_auc_score(labels[subset], scores[subset])
            print(f"    {sev:10}: {sev_auroc:.4f} ({sev_mask.sum()} files)")

    # Save results
    out_path = PROJECT_ROOT / "outputs" / "eval" / f"{band}_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "score", "n_patches", "type", "severity"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Scores saved to: {out_path}")

    return {"band": band, "auroc": auroc, "ap": ap, "n_normal": n_normal, "n_anom": n_anom}


if __name__ == "__main__":
    main()
