# =============================================================================
# File    : scripts/test_dataset.py
# Purpose : Smoke test for fastflow/dataset.py — verifies patch extraction,
#           normalization, and DataLoader iteration on a small subset.
# Run     : python scripts/test_dataset.py
# =============================================================================

import sys
sys.path.insert(0, ".")

def main():
    from fastflow.dataset import build_dataloaders

    train_loader, val_loader, band_stats = build_dataloaders(
        manifest_path   = "outputs/manifest_v2.csv",
        data_dir        = "data",
        stats_path      = "outputs/norm_stats.json",
        test_normal_dir = "test_set/normal",
        batch_size      = 64,
        num_workers     = 0,   # safe for Windows; use 2 on Kaggle
    )

    print("\nBand stats:")
    for band, s in sorted(band_stats.items()):
        print(f"  {band:<6} median={s['median']:>8.3f}  iqr={s['iqr']:>7.3f}")

    batch, bands = next(iter(train_loader))
    print(f"\nFirst batch shape : {batch.shape}")
    print(f"Dtype             : {batch.dtype}")
    print(f"Value range       : [{batch.min():.3f}, {batch.max():.3f}]")
    print(f"Bands in batch    : {set(bands)}")

if __name__ == "__main__":
    main()