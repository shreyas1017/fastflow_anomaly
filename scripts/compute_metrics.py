import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np

BASE_DIR = Path(r"c:\Users\ShreyasPatil\fastflow_anomaly")
OUT_DIR = BASE_DIR / "outputs"
INJECTORS = ["narrowband_jammer", "barrage_jammer", "rogue_carrier", "gradual_drift"]
SEVERITIES = ["subtle", "moderate", "obvious"]

df = pd.read_csv(OUT_DIR / "eval_scores.csv")
normal_scores = df[df["label"] == 0]["score"].values
anomalous_scores = df[df["label"] == 1]["score"].values

p50_normal = np.percentile(normal_scores, 50)
p95_normal = np.percentile(normal_scores, 95)
p50_anomalous = np.percentile(anomalous_scores, 50)
threshold = float(np.percentile(normal_scores, 95))

y_true = df["label"].values
y_score = df["score"].values
y_pred = (y_score >= threshold).astype(int)

auroc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)
f1 = f1_score(y_true, y_pred)
tp = int(((y_pred == 1) & (y_true == 1)).sum())
fp = int(((y_pred == 1) & (y_true == 0)).sum())
tn = int(((y_pred == 0) & (y_true == 0)).sum())
fn = int(((y_pred == 0) & (y_true == 1)).sum())
prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

metrics_text = f"""
===== FastFlow Evaluation Report =====
Test files  : {len(normal_scores)} normal  |  {len(anomalous_scores)} anomalous
Threshold   : {threshold:.4f}  (p95 of normal scores)

--- Overall Metrics ---
AUROC       : {auroc:.4f}
Avg Precision (AP): {ap:.4f}
F1 Score    : {f1:.4f}
Precision   : {prec:.4f}
Recall      : {rec:.4f}

--- Confusion Matrix ---
TP={tp}  FP={fp}  TN={tn}  FN={fn}

--- Phase 2 Gate ---
Normal  p50 : {p50_normal:.4f}
Anomalous p50 : {p50_anomalous:.4f}
Normal  p95 : {p95_normal:.4f}  (= threshold)
"""
print(metrics_text)

breakdown_rows = []
anom_df = df[df["label"] == 1].copy()
anom_df["predicted"] = (anom_df["score"] >= threshold).astype(int)

for inj in INJECTORS:
    for sev in SEVERITIES:
        sub = anom_df[(anom_df["injector"] == inj) & (anom_df["severity"] == sev)]
        if len(sub) == 0:
            continue
        detected = sub["predicted"].sum()
        breakdown_rows.append({
            "injector":    inj,
            "severity":    sev,
            "n_files":     len(sub),
            "n_detected":  int(detected),
            "recall":      round(detected / len(sub), 3),
            "mean_score":  round(sub["score"].mean(), 4),
        })

bd_df = pd.DataFrame(breakdown_rows)
print("\n--- Breakdown by Injector/Severity ---")
print(bd_df.to_string(index=False))
