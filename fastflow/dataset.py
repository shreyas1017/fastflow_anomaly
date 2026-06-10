# =============================================================================
# File    : fastflow/dataset.py
# Purpose : Patch extraction, per-band robust normalization, and PyTorch
#           Dataset / DataLoader for multi-domain spectrogram training.
#           Reads all .npy files listed in manifest_v2.csv, slices them into
#           (PATCH_TIME × PATCH_FREQ) windows, normalizes using per-band
#           median/IQR computed on training files only, and returns patches
#           as single-channel tensors ready for the FastFlow model.
# =============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import os
import json

# ---------------------------------------------------------------------------
# Patch config — fixed at ~9.3 kHz/bin resolution
# ---------------------------------------------------------------------------
PATCH_TIME = 32
PATCH_FREQ = 32
STRIDE_TIME = 16
STRIDE_FREQ = 16


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def compute_band_stats(manifest_path: str, data_dir: str, split_files: set) -> dict:
    """
    Compute per-band robust normalization stats (median, IQR) from training files only.
    Returns dict: {band: {"median": float, "iqr": float}}
    """
    band_values = {}

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        fname = row["filename"]
        if fname not in split_files:
            continue
        band = row["band"]
        arr = np.load(os.path.join(data_dir, fname)).astype(np.float32)
        sample = arr.flatten()[::4]
        band_values.setdefault(band, []).append(sample)

    stats = {}
    for band, samples in band_values.items():
        all_vals = np.concatenate(samples)
        q25, q75 = np.percentile(all_vals, [25, 75])
        stats[band] = {
            "median": float(np.median(all_vals)),
            "iqr":    float(q75 - q25) if (q75 - q25) > 0 else 1.0
        }
    return stats


def normalize_patch(patch: np.ndarray, median: float, iqr: float) -> np.ndarray:
    return (patch - median) / iqr


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(arr: np.ndarray) -> np.ndarray:
    """
    Slide a (PATCH_TIME × PATCH_FREQ) window over a 2D spectrogram array.
    Returns shape (N, PATCH_TIME, PATCH_FREQ).
    """
    T, F = arr.shape
    patches = []
    for t in range(0, T - PATCH_TIME + 1, STRIDE_TIME):
        for f in range(0, F - PATCH_FREQ + 1, STRIDE_FREQ):
            patches.append(arr[t:t + PATCH_TIME, f:f + PATCH_FREQ])
    return np.stack(patches, axis=0) if patches else np.empty((0, PATCH_TIME, PATCH_FREQ))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpectrogramPatchDataset(Dataset):
    """
    Multi-domain patch dataset for unsupervised training.
    Loads all .npy files from the provided file list, extracts patches,
    applies per-band normalization, and returns (patch_tensor, band_str) tuples.
    """

    def __init__(self, manifest_path: str, data_dir: str, file_list: list,
                 band_stats: dict):
        self.samples = []

        with open(manifest_path) as f:
            rows = {r["filename"]: r for r in csv.DictReader(f)}

        for fname in file_list:
            row  = rows[fname]
            band = row["band"]
            arr  = np.load(os.path.join(data_dir, fname)).astype(np.float32)

            patches = extract_patches(arr)
            if patches.shape[0] == 0:
                continue

            med = band_stats[band]["median"]
            iqr = band_stats[band]["iqr"]

            for p in patches:
                norm = normalize_patch(p, med, iqr).astype(np.float32)
                self.samples.append((norm, band))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, band = self.samples[idx]
        return torch.from_numpy(patch).unsqueeze(0), band


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(manifest_path: str, data_dir: str, stats_path: str,
                      test_normal_dir: str, batch_size: int = 128,
                      val_fraction: float = 0.1, seed: int = 42,
                      num_workers: int = 0):
    """
    Splits training files into train/val, computes band stats from train split,
    saves stats to stats_path, and returns (train_loader, val_loader, band_stats).

    num_workers=0  -> safe default for Windows local dev (no multiprocessing).
    num_workers=2  -> pass explicitly on Kaggle (Linux, GPU available).
    """
    with open(manifest_path) as f:
        all_files = [r["filename"] for r in csv.DictReader(f)]

    rng = np.random.default_rng(seed)
    all_files = list(all_files)
    rng.shuffle(all_files)
    n_val       = max(1, int(len(all_files) * val_fraction))
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]

    print(f"Train: {len(train_files)} files | Val: {len(val_files)} files")

    band_stats = compute_band_stats(manifest_path, data_dir, set(train_files))
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(band_stats, f, indent=2)
    print(f"Band stats saved -> {stats_path}")

    train_ds = SpectrogramPatchDataset(manifest_path, data_dir, train_files, band_stats)
    val_ds   = SpectrogramPatchDataset(manifest_path, data_dir, val_files,   band_stats)

    print(f"Train patches: {len(train_ds):,} | Val patches: {len(val_ds):,}")

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_pin)

    return train_loader, val_loader, band_stats