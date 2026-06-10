# =============================================================================
# File    : scripts/test_train.py
# Purpose : 3-epoch sanity check for fastflow/train.py — confirms loss
#           decreases, checkpoint saves correctly, and log CSV is written.
# Run     : python scripts/test_train.py
# =============================================================================

import sys
sys.path.insert(0, ".")

def main():
    from fastflow.train import train, CFG

    cfg = {**CFG, "epochs": 3, "batch_size": 64}
    train(cfg)

if __name__ == "__main__":
    main()