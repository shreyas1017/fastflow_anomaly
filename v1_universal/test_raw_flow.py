import torch
from fastflow.model import FlowBlock2D

x = torch.randn(4, 1, 32, 32)
flow = FlowBlock2D(channels=1, height=32, width=32, n_layers=4, hidden=64)

z, log_det = flow(x)
print(f"z shape: {z.shape}")
print(f"log_det shape: {log_det.shape}")

nll = flow.nll(x)
print(f"nll shape: {nll.shape}")
print(f"nll: {nll}")

loss = nll.mean()
loss.backward()
print("Backward pass successful")
