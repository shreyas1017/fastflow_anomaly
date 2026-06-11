#!/usr/bin/env python3
"""
Evaluate all 6 per-band FastFlow models and print a combined report.

Usage:
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --bands FM GSM   # only specific bands

Expects checkpoints at outputs/models/{BAND}/best.pt for each band.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the single-band evaluator
from evaluate_band import main as evaluate_single_band


ALL_BANDS = ["FM", "GSM", "LTE", "DAB", "DVBT", "TETRA"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate all per-band models")
    parser.add_argument("--bands", nargs="+", default=ALL_BANDS,
                        choices=ALL_BANDS,
                        help="Bands to evaluate (default: all)")
    args = parser.parse_args()

    results = []
    missing = []

    for band in args.bands:
        ckpt_path = PROJECT_ROOT / "outputs" / "models" / band / "best.pt"
        if not ckpt_path.exists():
            missing.append(band)
            print(f"\n[{band}] SKIPPED — no checkpoint found at {ckpt_path}")
            continue

        print(f"\n{'═' * 60}")
        print(f"  Evaluating {band}")
        print(f"{'═' * 60}")

        # Temporarily override sys.argv for the single-band evaluator
        original_argv = sys.argv
        sys.argv = ["evaluate_band.py", "--band", band]
        try:
            result = evaluate_single_band()
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            sys.argv = original_argv

    # ── Combined report ──
    if results:
        print(f"\n\n{'═' * 60}")
        print(f"  COMBINED EVALUATION REPORT")
        print(f"{'═' * 60}")
        print(f"\n{'Band':<8} {'AUROC':>7} {'AP':>7} {'Normal':>8} {'Anomalous':>10}")
        print("-" * 45)

        aurocs = []
        aps = []
        total_normal = 0
        total_anom = 0

        for r in results:
            print(f"{r['band']:<8} {r['auroc']:>7.4f} {r['ap']:>7.4f} "
                  f"{r['n_normal']:>8} {r['n_anom']:>10}")
            aurocs.append(r["auroc"])
            aps.append(r["ap"])
            total_normal += r["n_normal"]
            total_anom += r["n_anom"]

        print("-" * 45)
        macro_auroc = sum(aurocs) / len(aurocs)
        macro_ap = sum(aps) / len(aps)
        print(f"{'MACRO':<8} {macro_auroc:>7.4f} {macro_ap:>7.4f} "
              f"{total_normal:>8} {total_anom:>10}")

    if missing:
        print(f"\n⚠ Missing checkpoints for: {', '.join(missing)}")
        print(f"  Train these bands first, then re-run evaluation.")


if __name__ == "__main__":
    main()
