import torch
import torch.nn as nn
from fastflow.model import FastFlow

# Dummy input
x = torch.randn(2, 1, 32, 32)
model = FastFlow()

# Unfreeze backbone
for p in model.backbone.parameters():
    p.requires_grad = True

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(10):
    opt.zero_grad()
    loss = model(x)
    loss.backward()
    opt.step()
    
    # Check if features are collapsing to 0
    with torch.no_grad():
        f1, f2 = model.backbone(x)
        print(f"Step {i}: Loss {loss.item():.2f}, f1 norm {f1.norm().item():.2f}, f2 norm {f2.norm().item():.2f}")
