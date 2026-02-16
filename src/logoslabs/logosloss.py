"""
LogosLoss V4 reference implementation.

This module provides the LogosLossV4 loss function for computing spectral and phase
differences between predicted and truth tensors.
"""

import torch
import torch.nn as nn


class LogosLossV4(nn.Module):
    """
    LogosLoss V4:
      - Phase is wrap-safe: 1 - cos(angle)
      - Spectral magnitude diff is energy-weighted (truth presence)
      - Mercy focuses on dominant bins by default (power=1.0)
    """
    def __init__(
        self,
        grace_coeff: float = 0.1,
        phase_weight: float = 0.01,
        eps: float = 1e-8,
        freq_power: float = 1.0,
        mercy_power: float = 1.0,
        presence_power: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")

        self.grace_coeff = float(grace_coeff)
        self.phase_weight = float(phase_weight)
        self.eps = float(eps)
        self.freq_power = float(freq_power)
        self.mercy_power = float(mercy_power)
        self.presence_power = float(presence_power)
        self.reduction = reduction

        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if pred.shape != truth.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {truth.shape}")

        device = pred.device
        dtype = pred.dtype

        # 1) Time-domain
        material = self.mse(pred, truth).mean(dim=-1)  # (B,C)

        # 2) FFT with orthonormal scaling
        pred_f = torch.fft.rfft(pred, dim=-1, norm='ortho')
        truth_f = torch.fft.rfft(truth, dim=-1, norm='ortho')
        F = pred_f.shape[-1]

        truth_mag = torch.abs(truth_f).clamp_min(1e-5)
        pred_mag = torch.abs(pred_f).clamp_min(1e-5)

        # 3) Spectral magnitude (log-mag diff), weighted by truth energy and frequency
        log_diff = (torch.log(pred_mag) - torch.log(truth_mag)) ** 2  # (B,C,F)

        # Compute presence and frequency weights, then multiply and normalize once
        w_presence = truth_mag ** self.presence_power
        freq = torch.linspace(0.0, 1.0, F, device=device, dtype=dtype)
        w_freq = (freq ** self.freq_power).view(1, 1, F)
        
        weights = w_presence * w_freq
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        spectral = (log_diff * weights).sum(dim=-1)  # (B,C)

        # 4) Phase (wrap-safe)
        interaction = pred_f * torch.conj(truth_f)
        angle = torch.angle(interaction)
        phase_err = 1.0 - torch.cos(angle)  # [0..2]

        mercy = truth_mag ** self.mercy_power
        mercy = mercy / mercy.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        phase = (phase_err * mercy).sum(dim=-1)  # (B,C)

        total = material + (self.grace_coeff * spectral) + (self.phase_weight * phase)

        if self.reduction == "none":
            return total
        return total.mean()
