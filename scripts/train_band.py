#!/usr/bin/env python3
"""
Per-band CAE training script.

Usage:
    python scripts/train_band.py --band FM
    python scripts/train_band.py --band GSM --epochs 3  # quick debug run

Loads only the files belonging to the specified band from data/,
computes per-band normalization stats, trains a CAE model,
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
from fastflow.model import CAE, ReconstructionLoss


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
ENC_CHANNELS = (16, 32, 64, 32)

# Training
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
SSIM_WEIGHT = 0.5
SAVE_EVERY = 10  # save periodic checkpoint every N epochs

# Early stopping & LR schedule
EARLY_STOP_PATIENCE = 15
LR_PATIENCE = 15
LR_FACTOR = 0.5
MIN_LR = 1e-6


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


def unwrap(m):
    """Strip nn.DataParallel wrapper if present."""
    return m.module if isinstance(m, nn.DataParallel) else m


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train per-band CAE model")
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
    tensor = torch.from_numpy(all_patches).unsqueeze(1)  # (N, 1, 32, 32)
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
                              num_workers=args.workers, pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=False)

    print(f"  Train: {len(train_ds):,} patches | Val: {len(val_ds):,} patches")
    print(f"  Batch: {batch_size} | Batches/epoch: {len(train_loader)}")

    # ── Model ──
    model = CAE(enc_channels=ENC_CHANNELS).to(device)

    if n_gpus >= 2:
        model = nn.DataParallel(model)
        print(f"  DataParallel active — {n_gpus} GPUs")

    counts = unwrap(model).param_count()
    print(f"  Params: {counts['total']:,} total "
          f"(enc: {counts['encoder']:,}, dec: {counts['decoder']:,})")

    # ── Loss, Optimizer & Scheduler ──
    criterion = ReconstructionLoss(ssim_weight=SSIM_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=MIN_LR
    )

    # ── Training loop ──
    best_val = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time": []}

    print(f"\nTraining: {args.epochs} epochs, LR={args.lr:.1e}, batch={batch_size}")
    print(f"Band: {band} | Architecture: CAE{ENC_CHANNELS}")
    print(f"Loss: MSE({1-SSIM_WEIGHT:.1f}) + SSIM({SSIM_WEIGHT:.1f})")
    print(f"\n Ep   Train Loss   Val Loss        LR     Time  Note")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss, n_train = 0.0, 0
        for (patches,) in train_loader:
            patches = patches.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            x_hat, _ = model(patches)
            loss = criterion(x_hat, patches)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
            n_train += 1

        # ── Validation ──
        model.eval()
        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            for (patches,) in val_loader:
                patches = patches.to(device, non_blocking=True)
                x_hat, _ = model(patches)
                v = criterion(x_hat, patches)
                if torch.isfinite(v):
                    val_loss += v.item()
                    n_val_batches += 1

        train_avg = train_loss / max(n_train, 1)
        val_avg = val_loss / max(n_val_batches, 1)
        epoch_time = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        note = ""

        # ── LR scheduler step ──
        scheduler.step(val_avg)

        # ── Checkpoint ──
        raw_model = unwrap(model)
        ckpt_data = {
            "band": band,
            "epoch": epoch,
            "val_loss": val_avg,
            "train_loss": train_avg,
            "model_state": raw_model.state_dict(),
            "enc_channels": ENC_CHANNELS,
            "norm_stats": norm_stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        if val_avg < best_val:
            best_val = val_avg
            epochs_no_improve = 0
            torch.save(ckpt_data, out_dir / "best.pt")
            note = "* best"
        else:
            epochs_no_improve += 1

        if epoch % SAVE_EVERY == 0:
            torch.save(ckpt_data, out_dir / f"ckpt_ep{epoch:03d}.pt")

        history["train_loss"].append(train_avg)
        history["val_loss"].append(val_avg)
        history["lr"].append(cur_lr)
        history["epoch_time"].append(epoch_time)

        print(f"{epoch:>3}  {train_avg:>10.6f}  {val_avg:>9.6f}  "
              f"{cur_lr:>9.2e}  {epoch_time:>6.1f}s  {note}")

        # ── Early stopping ──
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping — no improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save history ──
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Checkpoint saved to: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
