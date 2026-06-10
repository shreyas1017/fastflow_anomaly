# Per-Band FastFlow Anomaly Detection — Full Project Plan

## 1. Project Goal

Build an **unsupervised anomaly detection system for RF spectrum data** that trains one model per frequency band. Each model learns the "normal" distribution of its band using a Normalizing Flow (FastFlow), and flags unseen spectrograms as anomalous if their Negative Log-Likelihood exceeds a learned threshold.

### Phase 1 (This Plan)
Train and evaluate 6 independent per-band FastFlow models (FM, GSM, LTE, DAB, DVB-T, TETRA). Each model only sees data from its own band, eliminating the cross-band confusion that defeated our universal model attempts.

### Phase 2 (Future — After Phase 1 Succeeds)
Evolve the architecture into a **universal adaptive detector** that requires only a short "warmup sequence" of normal data to calibrate itself to any new 2D signal type (e.g., a new RF band, ECG, vibration data). The model would learn general 2D structure features during pretraining across all 6 bands, then use the warmup sequence to set its per-domain normalization and anomaly threshold. No retraining needed — just feed it a reference window of normal data and start detecting.

---

## 2. Why Per-Band Models

Our previous attempts conclusively proved that a single model across all 6 bands cannot work with raw NLL scoring:

| Attempt | Architecture | Failure Mode |
|---------|-------------|-------------|
| 1. Frozen ResNet-18 + Flow | ImageNet features are meaningless for spectrograms → representation collapse |
| 2. Trainable ResNet-18 + Flow | Backbone outputs zeros to cheat the NLL loss → total collapse |
| 3. Raw Spatial Flow (no backbone) | "Simplicity Bias" — jammers look like simple flat blocks, score lower NLL than complex normal signals |

The root cause in all cases: when you pool 6 radically different band types, a "normal" TETRA file (score ~1100) looks more anomalous than an "anomalous" LTE file (score ~-200). Per-band models eliminate this confusion entirely because every file in the training set and every file at evaluation time shares the same spectral characteristics.

---

## 3. Dataset Overview

### 3.1 Original Structure
```
spectrum_bands/
├── alcorcon1/          # Sensor location
│   ├── Feb_1/          # Date folder
│   │   ├── ...SpectrumBands_195_230_dab_alcorcon_195_230.npy
│   │   ├── ...SpectrumBands_791_821_lte_alcorcon_791_821.npy
│   │   └── ...
│   ├── Feb_2/
│   └── Feb_3/
├── Andrew_GVA/
│   └── Aug_1/
├── ...
└── (44 sensor locations total)
```

### 3.2 Flattened Training Data (`data/`)
All 228 `.npy` files were flattened into a single `data/` directory using the naming convention:
```
{sensor}__{date}__{original_filename}.npy
```
Example: `alcorcon1__Feb_2__alcorcon_Feb_2_21SpectrumBands_791_821_lte_alcorcon_791_821.npy`

### 3.3 Band Distribution (Training Data — 228 files total)

| Band  | Files | Total Size | Freq Bins (min–max) | Time Steps (median) |
|-------|-------|-----------|--------------------|--------------------|
| FM    | 41    | 370 MB    | 537 – 4,302        | 449                |
| GSM   | 44    | 143 MB    | 321 – 3,765        | 449                |
| LTE   | 40    | 472 MB    | 3,194 – 6,667      | 448                |
| DAB   | 32    | 838 MB    | 430 – 168,013      | 450                |
| DVB-T | 33    | 630 MB    | 324 – 11,722       | 449                |
| TETRA | 38    | 158 MB    | 215 – 22,154       | 449                |

### 3.4 Test Set (`test_set/`)

**Normal** (30 files):

| Band  | Normal Files |
|-------|-------------|
| FM    | 5           |
| GSM   | 6           |
| LTE   | 6           |
| DAB   | 2           |
| DVB-T | 4           |
| TETRA | 7           |

**Anomalous** (59 files with 4 attack types × 3 severity levels):

| Band  | Anomalous Files |
|-------|----------------|
| FM    | 9              |
| GSM   | 14             |
| LTE   | 8              |
| DAB   | 11             |
| DVB-T | 8              |
| TETRA | 9              |

**Attack Types**: `barrage_jammer`, `narrowband_jammer`, `rogue_carrier`, `gradual_drift`
**Severity Levels**: `subtle`, `moderate`, `obvious`

### 3.5 Band Identification
Every filename contains a band tag that can be extracted via regex:
```python
re.search(r'_(fm|gsm|lte|dab|dvbt|tetra)_', filename.lower())
```
This is the sole mechanism for separating files into per-band groups. It works reliably on all 228 training files and all 89 test files.

---

## 4. Architecture: Per-Band FastFlow

### 4.1 Why FastFlow is Still the Right Choice
FastFlow's core `FlowBlock2D` is an excellent density estimator for 2D patches. The failures we experienced were caused by cross-band confusion and the ImageNet backbone, **not** by the flow itself. Within a single band, the NLL scores will be meaningful because all training and test data share the same spectral characteristics. A jammer on a GSM band will look different from normal GSM patterns, and the model won't be confused by TETRA's higher baseline complexity.

### 4.2 Model Architecture (Per Band)

```
Input: (B, 1, 32, 32) normalized spectrogram patches
  │
  └─→ FlowBlock2D(channels=1, height=32, width=32, n_layers=16, hidden=128)
        │
        └─→ NLL score (B,) — anomaly score per patch
```

- **No backbone.** Raw spatial flow operates directly on the patches.
- **16 coupling layers** with 128 hidden channels — enough capacity for single-band patterns.
- **Checkerboard masking** with alternating patterns for full spatial coverage.
- **Non-persistent mask buffers** for DataParallel compatibility.

> [!NOTE]
> We may tune `n_layers` and `hidden` per band if some bands converge faster than others (e.g., TETRA with its narrow freq range may need fewer layers). But we start with the same architecture for all 6 bands to establish a baseline.

### 4.3 Normalization (Per Band)
Each band gets its own median/IQR statistics computed from its training files only:
```python
normed = (raw_value - band_median) / band_iqr
clipped = np.clip(normed, -5.0, 5.0)
```
These stats are saved alongside each band's model checkpoint.

### 4.4 Patch Extraction
Same as current: `32×32` patches with stride `16×16`, with an edge-clipping filter that rejects patches where >20% of values are at the clip boundary.

---

## 5. File & Directory Reorganization

### 5.1 New Per-Band Data Layout

We will NOT physically copy or move any `.npy` files. Instead, we will use a Python script to create per-band **file lists** (JSON manifests) that point to the existing files in `data/` and `test_set/`. This avoids duplicating ~2.6 GB of data.

```
outputs/
├── band_manifests/
│   ├── FM.json        # {"train": [...], "test_normal": [...], "test_anomalous": [...]}
│   ├── GSM.json
│   ├── LTE.json
│   ├── DAB.json
│   ├── DVBT.json
│   └── TETRA.json
├── models/
│   ├── FM/
│   │   ├── best.pt
│   │   └── norm_stats.json
│   ├── GSM/
│   │   └── ...
│   └── ...
└── eval/
    ├── FM_scores.csv
    ├── GSM_scores.csv
    └── ...
```

### 5.2 Manifest Generation Script

A new script `scripts/generate_manifests.py` will:
1. Scan `data/` for all `.npy` files, classify each by band using the regex.
2. Scan `test_set/normal/` and `test_set/anomalous/` similarly.
3. Write one JSON per band into `outputs/band_manifests/`.

This script runs locally in seconds and produces the definitive file lists for training and evaluation.

---

## 6. Training Pipeline

### 6.1 Training Script: `scripts/train_band.py`

A single script that accepts `--band FM` (or GSM, LTE, etc.) as an argument and:

1. Loads the manifest for that band.
2. Reads only the training files for that band from `data/`.
3. Computes per-band normalization stats (median/IQR) from the training files.
4. Extracts 32×32 patches, applies normalization.
5. Splits into train/val (90/10).
6. Trains a `FastFlow(flow_layers=16, flow_hidden_ratio=128.0)` model.
7. Saves `best.pt` and `norm_stats.json` into `outputs/models/{BAND}/`.

**Training hyperparameters** (same as our successful run):
- Epochs: 50
- Batch size: 512 (2 GPUs via DataParallel) or 256 (1 GPU)
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-5
- Scheduler: CosineAnnealingLR (T_max=50, eta_min=0)
- Gradient clipping: max_norm=1.0

### 6.2 Kaggle Training Script: `kaggle_perband.txt`

A single, self-contained Kaggle notebook cell that:
1. Takes a `BAND = "FM"` variable at the top.
2. Loads only the files matching that band from the attached dataset.
3. Uses the same DataParallel setup (2× T4 GPUs) we've already proven works.
4. Trains the model and saves the checkpoint.
5. Includes a quick-eval cell at the end.

To train all 6 bands, you run the notebook 6 times, changing only the `BAND` variable. Each run takes ~2-3 hours (fewer files per band = fewer patches = faster epochs).

### 6.3 Checkpoint Format
```python
{
    "band":        "FM",
    "epoch":       50,
    "val_nll":     -123.57,
    "train_nll":   -125.04,
    "model_state": raw.state_dict(),
    "arch":        {"flow_layers": 16, "flow_hidden_ratio": 128.0},
    "norm_stats":  {"median": -54.18, "iqr": 16.41},
    "optimizer":   optimizer.state_dict(),
    "scheduler":   scheduler.state_dict(),
}
```

> [!IMPORTANT]
> The `norm_stats` are now embedded directly in the checkpoint. This means evaluation never needs a separate `norm_stats.json` file — it can extract the stats from the checkpoint itself. This was a pain point in our previous runs.

---

## 7. Evaluation Pipeline

### 7.1 Evaluation Script: `scripts/evaluate_band.py`

Accepts `--band FM` and:
1. Loads the band's checkpoint from `outputs/models/{BAND}/best.pt`.
2. Extracts `norm_stats` directly from the checkpoint.
3. Loads the test files for that band (both normal and anomalous) from the manifest.
4. For each file: normalize → extract patches → score all patches → aggregate file-level score.
5. Computes per-band AUROC and AP.

### 7.2 File-Level Scoring Strategy

```python
# Top-1% pooling: mean of the most anomalous patches
k = max(5, int(0.01 * len(patch_scores)))
file_score = np.mean(np.sort(patch_scores)[-k:])
```

This is the same strategy we used before. It works well because anomalies typically affect a subset of patches in the file (especially for subtle attacks). By focusing on the most anomalous patches, we avoid diluting the signal.

### 7.3 Combined Report: `scripts/evaluate_all.py`

Runs evaluation for all 6 bands and prints a combined report:
```
Band   | AUROC | AP    | Normal | Anomalous
-------|-------|-------|--------|----------
FM     | 0.92  | 0.88  | 5      | 9
GSM    | 0.87  | 0.82  | 6      | 14
LTE    | 0.90  | 0.85  | 6      | 8
DAB    | 0.95  | 0.93  | 2      | 11
DVB-T  | 0.88  | 0.84  | 4      | 8
TETRA  | 0.91  | 0.87  | 7      | 9
-------|-------|-------|--------|----------
MACRO  | 0.91  | 0.87  | 30     | 59
```

---

## 8. Implementation Steps (Ordered)

### Step 1: Generate Per-Band Manifests (Local)
- Create `scripts/generate_manifests.py`
- Run it locally → produces 6 JSON files in `outputs/band_manifests/`
- Verify file counts match expectations

### Step 2: Build Per-Band Training Script (Local)
- Create `scripts/train_band.py` that works locally for quick debugging
- Test on a single band (FM, smallest after GSM) with 3 epochs to verify:
  - Correct file loading (only FM files)
  - Correct normalization (FM-specific median/IQR)
  - Model trains without errors
  - Checkpoint saves with embedded norm_stats

### Step 3: Build Kaggle Training Notebook (Local)
- Create `kaggle_perband.txt` — the full Kaggle cell
- Adapted from our proven `kaggle.txt` but with per-band filtering
- Keeps DataParallel, 2× T4 setup, cosine scheduling
- Single `BAND = "FM"` variable at the top

### Step 4: Train All 6 Bands on Kaggle
- Upload the latest `fastflow/model.py` to Kaggle datasets
- Run the notebook 6 times (one per band)
- Download each band's `best.pt` checkpoint
- Place them in `outputs/models/{BAND}/best.pt` locally

### Step 5: Build Evaluation Scripts (Local)
- Create `scripts/evaluate_band.py` (single-band evaluation)
- Create `scripts/evaluate_all.py` (runs all 6 bands, prints combined report)
- Also create a Kaggle-compatible evaluation cell for GPU-accelerated scoring

### Step 6: Evaluate All Bands
- Run evaluation on Kaggle (faster) or locally (slower but works)
- Collect AUROC and AP for each band
- Identify which bands need tuning

### Step 7: Tune & Iterate (If Needed)
- If any band has AUROC < 0.80, investigate:
  - Are there enough training files? (DAB has only 32)
  - Is the freq dimension too variable? (DAB ranges from 430 to 168,013 bins)
  - Does the band need more/fewer flow layers?
- Adjust hyperparameters and retrain only the underperforming bands

---

## 9. Key Design Decisions

### 9.1 State Dict Prefix Stripping
Kaggle wraps the model in `FastFlowDP(FastFlow)` for DataParallel. This adds a `model.` prefix to all state dict keys. Our evaluation scripts must strip this prefix when loading checkpoints:
```python
state_dict = ckpt["model_state"]
clean = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
```

### 9.2 No Physical File Separation
We do NOT create `data/FM/`, `data/GSM/`, etc. directories. The files stay in `data/` and we filter them programmatically using the manifests. This avoids duplicating 2.6 GB of data and keeps the Kaggle dataset unchanged.

### 9.3 Norm Stats in Checkpoint
Previous runs lost the `norm_stats.json` file when Kaggle sessions expired. Embedding the stats directly in the checkpoint prevents this failure mode entirely.

### 9.4 Same FlowBlock2D Architecture for All Bands
We start with identical architecture (16 layers, 128 hidden) for all 6 bands. This gives us a clean baseline. If a band needs tuning, we change only its hyperparameters.

---

## 10. Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `scripts/generate_manifests.py` | Scan data and test_set, produce per-band JSON manifests |
| CREATE | `scripts/train_band.py` | Local per-band training script (for debugging) |
| CREATE | `scripts/evaluate_band.py` | Single-band evaluation with AUROC/AP |
| CREATE | `scripts/evaluate_all.py` | Run all 6 band evaluations, print combined table |
| CREATE | `kaggle_perband.txt` | Self-contained Kaggle notebook for per-band training + eval |
| KEEP   | `fastflow/model.py` | No changes needed — architecture is already correct |
| KEEP   | `fastflow/dataset.py` | Reference only; Kaggle script inlines its own data loading |
| KEEP   | `data/*.npy` | 228 training files, untouched |
| KEEP   | `test_set/` | 89 test files (30 normal + 59 anomalous), untouched |

---

## 11. Kaggle Environment Notes

- **GPUs**: 2× Tesla T4 (15.6 GB each)
- **DataParallel**: Wrap model in `FastFlowDP` → `nn.DataParallel`
- **Batch size**: 512 (256 per GPU)
- **Attached datasets**: `spectrum-npy-data` (the training data), `fastflow` (the model code)
- **Session timeout**: 12 hours
- **Estimated time per band**: 1–3 hours (varies by number of files/patches)
  - GSM (44 files, 143 MB) → ~1 hour
  - DAB (32 files, 838 MB) → ~3 hours (large freq dim = many patches)

---

## 12. Success Criteria

- **AUROC ≥ 0.80** for each of the 6 bands individually
- **Macro-average AUROC ≥ 0.85** across all bands
- Model correctly flags `obvious` severity anomalies with near-100% confidence
- Model detects `subtle` anomalies better than random (AUROC > 0.65 per band)

---

## 13. Future: Phase 2 — Universal Adaptive Detector

Once Phase 1 proves that per-band FastFlow achieves strong AUROC, Phase 2 will unify the models:

1. **Pretrain a shared encoder** across all 6 bands (learns general 2D spectral features).
2. **Attach a lightweight flow head** that can be calibrated with a short warmup sequence.
3. **Warmup protocol**: Feed ~30 seconds of normal data → compute running median/IQR → set anomaly threshold at the p99 of NLL on the warmup data.
4. **Deploy**: The system now detects anomalies on any 2D signal type without retraining — it only needs that initial warmup window.

This Phase 2 architecture bridges the gap between "one model per band" and "one model for everything" by keeping the per-domain normalization adaptive while sharing structural features.

---

*Plan created: 2026-06-10*
*Project root: `c:\Users\ShreyasPatil\fastflow_anomaly`*
