# =============================================================================
# File    : fastflow/model.py
# Purpose : Convolutional Autoencoder (CAE) for 2D spectrogram patch
#           anomaly detection via reconstruction error.
#
# Architecture (for 32×32 input):
#   Input  : (B,  1, 32, 32)
#   Enc 1  : (B, 16, 16, 16)  Conv2d(1→16,  3×3, stride=2, pad=1) + BN + LeakyReLU
#   Enc 2  : (B, 32,  8,  8)  Conv2d(16→32, 3×3, stride=2, pad=1) + BN + LeakyReLU
#   Enc 3  : (B, 64,  4,  4)  Conv2d(32→64, 3×3, stride=2, pad=1) + BN + LeakyReLU
#   Enc 4  : (B, 32,  2,  2)  Conv2d(64→32, 3×3, stride=2, pad=1) + BN + LeakyReLU
#            ── bottleneck: 32×2×2 = 128 values (8:1 compression) ──
#   Dec 1  : (B, 64,  4,  4)  ConvTranspose2d + BN + ReLU
#   Dec 2  : (B, 32,  8,  8)  ConvTranspose2d + BN + ReLU
#   Dec 3  : (B, 16, 16, 16)  ConvTranspose2d + BN + ReLU
#   Dec 4  : (B,  1, 32, 32)  ConvTranspose2d (linear output)
#
# Anomaly scoring:
#   reconstruction_error(x) returns per-patch MSE.
#   Higher MSE = more anomalous.
#
# DataParallel safety:
#   Fully convolutional — no buffers or device-specific state.
#   nn.DataParallel works out of the box.
# =============================================================================

import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


# ══════════════════════════════════════════════════════════════
#  ENCODER BLOCK
# ══════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    """
    One stride-2 encoder step: Conv2d → BatchNorm2d → LeakyReLU.

    Halves both spatial dimensions (H and W) via stride=2.
    Uses padding=1 to ensure clean halving for even-sized inputs.

    LeakyReLU(0.2) is used instead of ReLU to prevent dead neurons
    in the encoder, which can cause flat regions in the latent space
    and reduce anomaly score separation.

    Parameters
    ----------
    in_ch  : Input channel count.
    out_ch : Output channel count.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════
#  DECODER BLOCK
# ══════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """
    One stride-2 decoder step: ConvTranspose2d → BatchNorm2d → ReLU.

    Doubles both spatial dimensions via stride=2.
    output_padding=1 is required to recover the exact spatial size
    after a stride-2 conv on an even-sized input.

    Parameters
    ----------
    in_ch      : Input channel count.
    out_ch     : Output channel count.
    activation : Whether to apply BN + ReLU after the transpose conv.
                 Set to False for the final decoder layer (linear output).
    """

    def __init__(self, in_ch: int, out_ch: int, activation: bool = True):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=False),
        ]

        if activation:
            layers += [
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ══════════════════════════════════════════════════════════════
#  FULL CAE
# ══════════════════════════════════════════════════════════════

class CAE(nn.Module):
    """
    Convolutional Autoencoder for 2D spectrogram patch reconstruction.

    Trained exclusively on normal patches. At inference time,
    anomalous patches produce higher reconstruction error (MSE) because
    the model has only learned the distribution of normal spectral patterns.

    Parameters
    ----------
    enc_channels : Tuple of ints defining the channel count at each
                   encoder block. The decoder mirrors this in reverse.
                   Default: (16, 32, 64, 32)

    Input/Output
    ------------
    Both input and output have shape (B, 1, patch_h, patch_w).
    Output is a linear (unbounded) reconstruction — no final activation.
    The MSE between input and output is the anomaly score.

    Design notes
    ------------
    - No linear/FC layers — the model is fully convolutional.
      This keeps the parameter count low and avoids overfitting
      to specific patch positions.
    - BatchNorm in both encoder and decoder stabilises training
      across the wide range of sensor types and band widths.
    - Works with any even-sized input (32×32, 32×64, etc.)
    """

    def __init__(self, enc_channels: Tuple[int, ...] = (16, 32, 64, 32)):
        super().__init__()

        self.enc_channels = enc_channels

        # ── Encoder ───────────────────────────────────────────
        enc_blocks = [EncoderBlock(1, enc_channels[0])]
        for i in range(1, len(enc_channels)):
            enc_blocks.append(
                EncoderBlock(enc_channels[i - 1], enc_channels[i])
            )
        self.encoder = nn.Sequential(*enc_blocks)

        # ── Decoder ───────────────────────────────────────────
        dec_channels = list(reversed(enc_channels))
        dec_blocks = []
        for i in range(len(dec_channels) - 1):
            dec_blocks.append(
                DecoderBlock(dec_channels[i], dec_channels[i + 1], activation=True)
            )
        # Final block: dec_channels[-1] → 1, no activation (linear output)
        dec_blocks.append(
            DecoderBlock(dec_channels[-1], 1, activation=False)
        )
        self.decoder = nn.Sequential(*dec_blocks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the bottleneck representation z."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs from bottleneck representation z."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns
        -------
        x_hat : Reconstructed tensor, shape (B, 1, H, W).
        z     : Bottleneck tensor (for logging/analysis).
        """
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-patch MSE between input and reconstruction.
        Used at inference time for anomaly scoring.

        Returns
        -------
        errors : 1D tensor of shape (B,), one MSE value per patch.
        """
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            errors = ((x - x_hat) ** 2).mean(dim=[1, 2, 3])
        return errors

    def param_count(self) -> dict:
        """Returns total, encoder, and decoder parameter counts."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total      = enc_params + dec_params
        return {
            "total"  : total,
            "encoder": enc_params,
            "decoder": dec_params,
        }


# ══════════════════════════════════════════════════════════════
#  RECONSTRUCTION LOSS
# ══════════════════════════════════════════════════════════════

class ReconstructionLoss(nn.Module):
    """
    Combined MSE + SSIM loss for CAE training.

    MSE penalises pixel-level amplitude errors uniformly across the patch.
    SSIM penalises structural changes — narrow spikes, wideband noise bursts,
    tone insertions, power-level drifts — which produce only small MSE when
    they affect a small fraction of patch pixels but disrupt local structure.

    loss = (1 - ssim_weight) * MSE + ssim_weight * (1 - SSIM)

    Parameters
    ----------
    ssim_weight : float
        Contribution of the SSIM term. 0.0 = pure MSE, 1.0 = pure SSIM.
    data_range : float
        Value range of the input. For z-scored data clipped at ±5,
        the range is 10.0.
    """

    def __init__(self, ssim_weight: float = 0.5, data_range: float = 10.0):
        super().__init__()
        from pytorch_msssim import SSIM as _SSIM
        self.ssim_weight  = ssim_weight
        self.mse_weight   = 1.0 - ssim_weight
        self.mse          = nn.MSELoss(reduction='mean')
        self.ssim_fn      = _SSIM(
            data_range   = data_range,
            size_average = True,
            channel      = 1,
            win_size     = 7,   # 7×7 fits cleanly in a 32×32 patch
        )

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mse_loss  = self.mse(x_hat, x)
        ssim_loss = 1.0 - self.ssim_fn(x_hat, x)
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss
