# =============================================================================
# File    : fastflow/model.py
# Purpose : FastFlow anomaly detection model for 2D spectrograms.
#           Backbone: frozen ImageNet-pretrained ResNet-18 (adapted for 1-ch).
#           Flow heads: two FlowBlock2D stacks operating on layer2 and layer3
#           feature maps respectively.
#
# DataParallel safety:
#   Checkerboard masks are registered as non-persistent buffers.
#   nn.DataParallel automatically replicates them to each GPU before
#   forward() runs, so no .to(device) calls are needed inside forward().
# =============================================================================

import math
import torch
import torch.nn as nn
# ---------------------------------------------------------------------------
# Note: The frozen ImageNet ResNet-18 backbone has been removed.
# Spectrogram patches (1x32x32) are fed directly into the spatial FlowBlock2D.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Single affine coupling layer (2D, operates on spatial feature maps)
# ---------------------------------------------------------------------------

class AffineCouplingLayer2D(nn.Module):
    """
    Affine coupling layer for 2D feature maps.

    Splits spatial positions using a checkerboard mask passed at call time.
    Scale/translate net uses alternating 3x3 / 1x1 convolutions (paper-spec).
    """

    def __init__(self, channels: int, hidden: int):
        super().__init__()

        # Alternating 3x3 / 1x1 conv, paper-spec lightweight net
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 1),                    # 1x1 conv
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, channels * 2, 3, padding=1),  # outputs s and t
        )
        # Init last conv to near-zero -> identity transform at epoch 0
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x    : (B, C, H, W)
        mask : (1, 1, H, W) binary -- 1 = pass-through, 0 = transform

        Returns: (z, log_det)
          z       : transformed tensor, same shape as x
          log_det : per-sample scalar, shape (B,)
        """
        inv_mask = 1.0 - mask
        x_masked = x * mask                     # pass-through half fed to net
        st       = self.net(x_masked)
        s, t     = st.chunk(2, dim=1)
        s        = torch.tanh(s) * 2.0          # bound in (-2,2) -> exp(s) safe

        # Transform inv_mask positions; pass-through positions unchanged
        z       = x * mask + inv_mask * (x * torch.exp(s) + t)
        log_det = (s * inv_mask).sum(dim=(1, 2, 3))  # (B,)
        return z, log_det

    def inverse(self, z: torch.Tensor, mask: torch.Tensor):
        inv_mask = 1.0 - mask
        z_masked = z * mask
        st       = self.net(z_masked)
        s, t     = st.chunk(2, dim=1)
        s        = torch.tanh(s) * 2.0
        x        = z * mask + inv_mask * ((z - t) * torch.exp(-s))
        return x


# ---------------------------------------------------------------------------
# Flow block: stack of alternating coupling layers
# ---------------------------------------------------------------------------

class FlowBlock2D(nn.Module):
    """
    Stack of N affine coupling layers with alternating checkerboard masks,
    operating on a 2D feature map of shape (B, C, H, W).

    Masks are registered as non-persistent buffers:
      - nn.DataParallel automatically replicates them to each GPU.
      - They are NOT included in state_dict (no checkpoint issues).
      - No .to(device) calls inside forward(), so fully thread-safe.
    """

    def __init__(self, channels: int, height: int, width: int,
                 n_layers: int = 8, hidden: int = None):
        super().__init__()
        if hidden is None:
            hidden = channels  # paper default: hidden == in_channels

        self.layers = nn.ModuleList([
            AffineCouplingLayer2D(channels, hidden) for _ in range(n_layers)
        ])
        self._expected_hw = (height, width)

        # Build checkerboard masks on CPU; register as non-persistent buffers.
        # DP will replicate them to each GPU before forward() is called.
        xs = torch.arange(height).unsqueeze(1)
        ys = torch.arange(width).unsqueeze(0)
        base = ((xs + ys) % 2).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        self.register_buffer('_mask0', base, persistent=False)
        self.register_buffer('_mask1', 1.0 - base, persistent=False)

    def forward(self, x: torch.Tensor):
        """Returns (z, total_log_det) where total_log_det is (B,)."""
        # Spatial shape guard
        assert x.shape[2:] == torch.Size(self._expected_hw), (
            f"FlowBlock2D got spatial {tuple(x.shape[2:])}, "
            f"expected {self._expected_hw}. "
            "Check backbone spatial output matches declared FEAT_SHAPE."
        )
        # _mask0, _mask1 are already on the correct device (DP replicates
        # all buffers to each GPU before calling forward).
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for i, layer in enumerate(self.layers):
            mask = self._mask1 if (i % 2 == 1) else self._mask0
            z, ld = layer(z, mask)
            log_det_total = log_det_total + ld
        return z, log_det_total

    def nll(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under the flow (per sample)."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi))
        log_pz = log_pz.sum(dim=(1, 2, 3))   # sum over C, H, W
        return -(log_pz + log_det)            # (B,)  NLL per sample


# ---------------------------------------------------------------------------
# FastFlow model — backbone + two parallel flow heads
# ---------------------------------------------------------------------------

class FastFlow(nn.Module):
    """
    FastFlow (Raw Spatial Version) for 2D spectrogram anomaly detection.

    Architecture:
      input (1×32×32 patches)
        -> FlowBlock2D (n_layers=16, hidden=128) -> NLL
        -> anomaly_score = NLL
    """

    def __init__(self, flow_layers: int = 16, flow_hidden_ratio: float = 128.0):
        super().__init__()
        
        # We use flow_hidden_ratio to pass the explicit hidden size
        hidden_dim = int(flow_hidden_ratio)
        
        # Input patches are 1 channel, 32x32
        self.flow = FlowBlock2D(channels=1, height=32, width=32, 
                                n_layers=flow_layers, hidden=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns mean NLL (scalar) — used for training loss."""
        return self.flow.nll(x).mean()

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample anomaly score (B,) — used for inference."""
        return self.flow.nll(x)
