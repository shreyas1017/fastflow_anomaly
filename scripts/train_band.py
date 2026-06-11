#!/usr/bin/env python3
"""
Per-band FastFlow training script.

Usage:
    python scripts/train_band.py --band FM
    python scripts/train_band.py --band GSM --epochs 3  # quick debug run

Loads only the files belonging to the specified band from data/,
computes per-band normalization stats, trains a FastFlow model,
and saves the checkpoint with embedded norm_stats so evaluation
never needs a separate file.
"""

import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Add project root to path so we can import fastflow ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fastflow.model import FastFlow


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
PATCH_H, PATCH_W = 32, 32
STRIDE_H, STRIDE_W = 16, 16
MAX_EDGE_FRAC = 0.20        # reject patches with >20% clipped values
CLIP_VAL = 5.0
MAX_PATCHES_PER_FILE = 2000  # cap to avoid memory blow-up on huge files
VAL_FRACTION = 0.1
SEED = 42

# Architecture
FLOW_LAYERS = 16
FLOW_HIDDEN = 128

# Training
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SAVE_EVERY = 10  # save periodic checkpoint every N epochs


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════
def compute_band_stats(file_paths):
    """Compute robust median/IQR stats across all training files for one band."""
    all_samples = []
    for p in file_paths:
        arr = np.load(p).astype(np.float32)
        # Subsample every 4th value for speed
        all_samples.append(arr.flatten()[::4])
    
    all_vals = np.concatenate(all_samples)
    median = float(np.median(all_vals))
    q25, q75 = np.percentile(all_vals, [25, 75])
    iqr = float(q75 - q25) if (q75 - q25) > 0 else 1.0
    
    print(f"  Band stats: median={median:.2f}, iqr={iqr:.2f}")
    return {"median": median, "iqr": iqr}


def normalize(arr, stats):
    """Robust normalization: (x - median) / iqr, clipped to [-5, 5]."""
    normed = (arr - stats["median"]) / stats["iqr"]
    return np.clip(normed, -CLIP_VAL, CLIP_VAL).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Patch extraction
# ═══════════════════════════════════════════════════════════════════════════
def extract_patches(arr):
    """Extract 32×32 patches with stride 16, filtering edge-heavy patches."""
    T, F = arr.shape
    if T < PATCH_H or F < PATCH_W:
        return np.empty((0, PATCH_H, PATCH_W), dtype=np.float32)
    
    patches = []
    for i in range((T - PATCH_H) // STRIDE_H + 1):
        for j in range((F - PATCH_W) // STRIDE_W + 1):
            p = arr[i * STRIDE_H: i * STRIDE_H + PATCH_H,
                    j * STRIDE_W: j * STRIDE_W + PATCH_W]
            # Reject patches where too many values are at the clip boundary
            if (np.abs(p) >= CLIP_VAL - 1e-4).sum() / p.size <= MAX_EDGE_FRAC:
                patches.append(p.copy())
    
    if patches:
        return np.stack(patches)
    return np.empty((0, PATCH_H, PATCH_W), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# DataParallel wrapper
# ═══════════════════════════════════════════════════════════════════════════
class FastFlowDP(nn.Module):
    """Thin wrapper so DataParallel calls forward() which returns per-sample NLL."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.flow.nll(x)  # (B,)


def unwrap(m):
    """Strip nn.DataParallel wrapper if present."""
    return m.module if isinstance(m, nn.DataParallel) else m


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train per-band FastFlow model")
    parser.add_argument("--band", type=str, required=True,
                        choices=["FM", "GSM", "LTE", "DAB", "DVBT", "TETRA"],
                        help="Band to train on")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 for Windows, 4 for Linux/Kaggle)")
    args = parser.parse_args()

    band = args.band
    data_dir = PROJECT_ROOT / "data"
    manifest_path = PROJECT_ROOT / "outputs" / "band_manifests" / f"{band}.json"
    out_dir = PROJECT_ROOT / "outputs" / "models" / band
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Device: {device} | GPUs: {n_gpus}")

    # ── Load manifest ──
    with open(manifest_path) as f:
        manifest = json.load(f)
    train_files = manifest["train"]
    print(f"\n[{band}] Loading {len(train_files)} training files...")

    # ── Compute normalization stats ──
    file_paths = [data_dir / fname for fname in train_files]
    norm_stats = compute_band_stats(file_paths)

    # ── Load and patch all training files ──
    rng = np.random.default_rng(SEED)
    all_patches = []
    skipped = 0

    for i, fpath in enumerate(file_paths):
        arr = np.load(fpath).astype(np.float32)
        if arr.ndim != 2 or arr.shape[0] < 10 or arr.shape[1] < 2:
            skipped += 1
            continue
        if not np.isfinite(arr).all():
            skipped += 1
            continue

        normed = normalize(arr, norm_stats)
        patches = extract_patches(normed)

        if patches.shape[0] == 0:
            skipped += 1
            continue
        # Cap patches per file to avoid memory blow-up
        if patches.shape[0] > MAX_PATCHES_PER_FILE:
            idx = rng.choice(patches.shape[0], MAX_PATCHES_PER_FILE, replace=False)
            patches = patches[idx]

        all_patches.append(patches)

    if not all_patches:
        print("ERROR: No patches loaded! Check your data directory.")
        sys.exit(1)

    all_patches = np.concatenate(all_patches, axis=0)
    print(f"  Total patches: {all_patches.shape[0]:,} | Skipped: {skipped}")

    # ── Train/Val split ──
    tensor = torch.from_numpy(all_patches)
    del all_patches; gc.collect()

    n_val = max(1, int(VAL_FRACTION * len(tensor)))
    ds = TensorDataset(tensor)
    train_ds, val_ds = random_split(
        ds, [len(tensor) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    batch_size = args.batch
    if n_gpus >= 2:
        batch_size = max(batch_size, 512)  # scale up for multi-GPU

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                              drop_last=True, persistent_workers=args.workers > 0,
                              prefetch_factor=3 if args.workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=args.workers > 0,
                            prefetch_factor=3 if args.workers > 0 else None)

    print(f"  Train: {len(train_ds):,} patches | Val: {len(val_ds):,} patches")
    print(f"  Batch: {batch_size} | Batches/epoch: {len(train_loader)}")

    # ── Model ──
    raw_model = FastFlow(flow_layers=FLOW_LAYERS, flow_hidden_ratio=float(FLOW_HIDDEN))
    raw_model.to(device)
    model = FastFlowDP(raw_model).to(device)

    if n_gpus >= 2:
        model = nn.DataParallel(model)
        print(f"  DataParallel active — {n_gpus} GPUs")

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,} total, {train_params:,} trainable")

    # ── Optimizer & Scheduler ──
    raw = unwrap(model).model  # the actual FastFlow instance
    flow_params = list(raw.parameters())
    optimizer = torch.optim.AdamW(flow_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0
    )

    # ── Training loop ──
    best_val = float("inf")
    history = {"train_nll": [], "val_nll": [], "lr": [], "epoch_time": []}

    print(f"\nTraining: {args.epochs} epochs, LR={args.lr:.1e}, batch={batch_size}")
    print(f"Band: {band}")
    print(f"\n Ep   Train NLL    Val NLL         LR     Time  Note")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss, n_train = 0.0, 0
        for (patches,) in train_loader:
            patches = patches.unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model(patches).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(flow_params, GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
            n_train += 1

        scheduler.step()

        # ── Validation ──
        model.eval()
        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for (patches,) in val_loader:
                patches = patches.unsqueeze(1).to(device, non_blocking=True)
                v = model(patches).mean()
                if torch.isfinite(v):
                    val_loss += v.item()
                    n_val_batches += 1

        train_nll = train_loss / max(n_train, 1)
        val_nll = val_loss / max(n_val_batches, 1)
        epoch_time = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        note = ""

        # ── Checkpoint ──
        ckpt_data = {
            "band": band,
            "epoch": epoch,
            "val_nll": val_nll,
            "train_nll": train_nll,
            "model_state": raw.state_dict(),
            "arch": {"flow_layers": FLOW_LAYERS, "flow_hidden_ratio": float(FLOW_HIDDEN)},
            "norm_stats": norm_stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        if val_nll < best_val:
            best_val = val_nll
            torch.save(ckpt_data, out_dir / "best.pt")
            note = "* best"

        if epoch % SAVE_EVERY == 0:
            torch.save(ckpt_data, out_dir / f"ckpt_ep{epoch:03d}.pt")

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["lr"].append(cur_lr)
        history["epoch_time"].append(epoch_time)

        print(f"{epoch:>3}  {train_nll:>10.4f}  {val_nll:>9.4f}  "
              f"{cur_lr:>9.2e}  {epoch_time:>6.1f}s  {note}")

    # ── Save history ──
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\nDone. Best val NLL: {best_val:.4f}")
    print(f"Checkpoint saved to: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
