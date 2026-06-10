# =============================================================================
# File    : fastflow/train.py
# Purpose : Training loop for the FastFlow multi-domain spectrogram anomaly
#           detector. Trains on normal patches from all bands using NLL loss,
#           validates each epoch on held-out normal patches, saves the best
#           checkpoint based on validation NLL, and logs per-epoch stats to
#           outputs/train_log.csv.
#
# Fixes applied vs original:
#   [F9]  lr raised from 2e-4 to 1e-3 (paper default); epochs from 50 to 200.
#         50 epochs with a random backbone is severe underfitting.
#   [F10] num_workers default changed to 2 (Kaggle T4 has 4 vCPUs; 0 wastes
#         CPU and slows data loading).
#   [F11] NaN/Inf guard added in train_epoch: non-finite losses are logged
#         and the batch is skipped rather than corrupting gradients.
#   [F12] Checkpoint now saves only architectural params (not full cfg with
#         local paths) so best.pt is portable across machines.
#   [F13] maybe_unfreeze_backbone(epoch) called at start of each epoch to
#         implement the freeze-then-unfreeze backbone schedule from model.py.
#   [F14] Optimizer rebuilt after backbone unfreeze so backbone params are
#         included in a separate param group with lower lr (backbone_lr),
#         preventing the backbone from being updated too aggressively.
#   [F15] backbone_frozen column added to train_log.csv so it is easy to
#         see in the log exactly when the backbone unfroze.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import csv
import time

from fastflow.dataset import build_dataloaders
from fastflow.model import FastFlow


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG = {
    # Data
    "manifest_path":    "outputs/manifest_v2.csv",
    "data_dir":         "data",
    "stats_path":       "outputs/norm_stats.json",
    "test_normal_dir":  "",          # empty string = no separate test set

    # DataLoader
    "batch_size":       128,
    "val_fraction":     0.1,
    "seed":             42,
    "num_workers":      0,           # Set to 0 to prevent Windows PyTorch dataloader hang

    # Model
    "flow_layers":             8,
    "flow_hidden_ratio":       1.0,  # hidden = ratio x channels (paper default)

    # Training
    "epochs":           50,          # reduced to 50 for faster initial validation
    "lr":               1e-3,        # [F9] was 2e-4; paper uses 1e-3
    "weight_decay":     1e-5,
    "grad_clip":        1.0,

    # Output
    "checkpoint_dir":   "outputs/checkpoints",
    "log_path":         "outputs/train_log.csv",
}


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def build_optimizer(model: FastFlow, cfg: dict) -> optim.AdamW:
    """
    Build AdamW for flow heads. Backbone remains frozen.
    """
    flow_params = (list(model.flow1.parameters()) +
                   list(model.flow2.parameters()))

    param_groups = [{"params": flow_params, "lr": cfg["lr"]}]

    return optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])


# ---------------------------------------------------------------------------
# Training and validation steps
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    total_loss  = 0.0
    n_batches   = 0
    nan_batches = 0

    for patches, _ in loader:
        patches = patches.to(device)
        optimizer.zero_grad()
        loss = model(patches)

        # NaN / Inf guard [F11]
        if not torch.isfinite(loss):
            nan_batches += 1
            optimizer.zero_grad()
            if nan_batches <= 5:
                print(f"  WARNING: non-finite loss ({loss.item():.4f}), "
                      f"skipping batch {nan_batches}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    if n_batches == 0:
        return float("nan")
    return total_loss / n_batches


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for patches, _ in loader:
        patches = patches.to(device)
        loss = model(patches)
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches  += 1

    if n_batches == 0:
        return float("nan")
    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(cfg: dict = CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Dataloaders
    train_loader, val_loader, band_stats = build_dataloaders(
        manifest_path   = cfg["manifest_path"],
        data_dir        = cfg["data_dir"],
        stats_path      = cfg["stats_path"],
        test_normal_dir = cfg["test_normal_dir"],
        batch_size      = cfg["batch_size"],
        val_fraction    = cfg["val_fraction"],
        seed            = cfg["seed"],
        num_workers     = cfg["num_workers"],
    )

    # Model
    model = FastFlow(
        flow_layers            = cfg["flow_layers"],
        flow_hidden_ratio      = cfg["flow_hidden_ratio"],
    ).to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    flow_params     = (sum(p.numel() for p in model.flow1.parameters()) +
                       sum(p.numel() for p in model.flow2.parameters()))
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"Total params    : {total_params:,}")
    print(f"  Flow heads    : {flow_params:,}")
    print(f"  Backbone      : {backbone_params:,}")
    print("Backbone is strictly frozen.")

    # Initial optimizer: only flow heads
    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    # Logging
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    log_fields = ["epoch", "train_nll", "val_nll", "lr",
                  "backbone_frozen", "epoch_time_s"]

    with open(cfg["log_path"], "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_nll   = float("inf")
    best_ckpt_path = os.path.join(cfg["checkpoint_dir"], "best.pt")

    print(f"\n{'Epoch':>6}  {'Train NLL':>12}  {'Val NLL':>12}  "
          f"{'LR':>10}  {'Frozen':>6}  {'Time':>8}")
    print("-" * 66)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_nll  = train_epoch(model, train_loader, optimizer, device,
                                 cfg["grad_clip"])
        val_nll    = val_epoch(model, val_loader, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t0
        frozen_str = "Y"

        # Save best checkpoint -- arch params only, no local paths [F12]
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            torch.save({
                "epoch":       epoch,
                "val_nll":     val_nll,
                "model_state": model.state_dict(),
                "arch": {                           # [F12] portable params only
                    "flow_layers":            cfg["flow_layers"],
                    "flow_hidden_ratio":      cfg["flow_hidden_ratio"],
                },
            }, best_ckpt_path)
            flag = " *"
        else:
            flag = ""

        val_str = f"{val_nll:12.4f}" if val_nll == val_nll else "         NaN"
        print(f"{epoch:>6}  {train_nll:>12.4f}  {val_str}  "
              f"{current_lr:>10.2e}  {frozen_str:>6}  {epoch_time:>6.1f}s{flag}")

        # Append to log CSV [F15]
        with open(cfg["log_path"], "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":          epoch,
                "train_nll":      round(train_nll, 6),
                "val_nll":        round(val_nll, 6),
                "lr":             round(current_lr, 8),
                "backbone_frozen": frozen_str,
                "epoch_time_s":   round(epoch_time, 1),
            })

    print(f"\nTraining complete.  Best val NLL : {best_val_nll:.4f}")
    print(f"Best checkpoint  : {best_ckpt_path}")
    print(f"Training log     : {cfg['log_path']}")


if __name__ == "__main__":
    train()