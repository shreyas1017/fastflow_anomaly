# =============================================================================
# File    : scripts/test_model.py
# Purpose : Smoke test for fastflow/model.py — verifies forward pass shapes,
#           loss is a finite scalar, and anomaly_score returns correct shape.
# Run     : python scripts/test_model.py
# =============================================================================

import sys
sys.path.insert(0, ".")
import torch
from fastflow.model import FastFlow

def main():
    model = FastFlow(flow_layers=8, flow_hidden=64)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")

    x = torch.randn(16, 1, 32, 32)

    # Training forward
    model.train()
    loss = model(x)
    print(f"Training loss    : {loss.item():.4f}  (finite={torch.isfinite(loss).item()})")

    # Inference scoring
    model.eval()
    scores = model.anomaly_score(x)
    print(f"Score shape      : {scores.shape}")
    print(f"Score range      : [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"All finite       : {torch.isfinite(scores).all().item()}")

if __name__ == "__main__":
    main()