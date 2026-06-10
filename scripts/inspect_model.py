import torch
from fastflow.model import FastFlow

ckpt = torch.load("outputs/checkpoints/best.pt", map_location="cpu")
arch = ckpt.get("arch", {})
model = FastFlow(**arch)

state_dict = ckpt["model_state"]
for k in list(state_dict.keys()):
    if k.endswith(".mask0") or k.endswith(".mask1"):
        del state_dict[k]
model.load_state_dict(state_dict)
model.eval()

# Check for NaNs in weights
has_nan = False
for name, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f"NaN found in parameter: {name}")
        has_nan = True
if not has_nan:
    print("No NaNs found in model weights.")

# Test with random inputs
torch.manual_seed(42)
x1 = torch.randn(1, 1, 32, 32)
x2 = torch.randn(1, 1, 32, 32)

with torch.no_grad():
    feat1_1, feat2_1 = model.backbone(x1)
    feat1_2, feat2_2 = model.backbone(x2)
    
    print(f"Backbone feat1_1 mean: {feat1_1.mean().item():.6f}, std: {feat1_1.std().item():.6f}")
    print(f"Backbone feat1_2 mean: {feat1_2.mean().item():.6f}, std: {feat1_2.std().item():.6f}")
    
    print(f"Backbone feat2_1 mean: {feat2_1.mean().item():.6f}, std: {feat2_1.std().item():.6f}")
    print(f"Backbone feat2_2 mean: {feat2_2.mean().item():.6f}, std: {feat2_2.std().item():.6f}")
    
    score1 = model.anomaly_score(x1)
    score2 = model.anomaly_score(x2)
    
    print(f"Anomaly Score 1: {score1.item():.6f}")
    print(f"Anomaly Score 2: {score2.item():.6f}")
