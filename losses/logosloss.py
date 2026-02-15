"""
LogosLoss implementation for benchmarking.

Based on LogosLossV4 from the main library, adapted for benchmark evaluation.
"""

import torch
import torch.nn as nn


class LogosLoss(nn.Module):
    """
    LogosLoss: A multi-component loss combining time-domain, spectral, and phase errors.
    
    This loss function is designed to capture both material (time-domain) and structural
    (frequency-domain) properties of signals. It combines three main components:
    
    1. **Material Loss (Time-Domain)**: Standard MSE between predicted and truth signals.
       Measures point-wise agreement in the original space.
    
    2. **Spectral Loss (Frequency Magnitude)**: Log-magnitude difference in frequency domain,
       weighted by the truth signal's energy distribution and frequency importance.
       Penalizes deviations in the spectral content, focusing on frequencies where the
       truth signal has significant energy. This prevents the model from generating
       spurious high-frequency noise or missing important frequency components.
    
    3. **Spectral Noise (Phase Error)**: Wrap-safe phase difference (1 - cos(angle))
       weighted by dominant frequency bins. Measures coherence and structural stability
       in the frequency domain. Phase misalignment indicates structural instability
       even when magnitudes match, which is critical for detecting subtle artifacts
       or hallucinations in generated signals.
    
    The three components are combined as:
        total = material + (grace_coeff × spectral) + (phase_weight × spectral_noise)
    
    Parameters
    ----------
    grace_coeff : float, default=0.5
        Weight for the spectral magnitude component. Controls the relative importance
        of frequency content vs time-domain agreement.
    phase_weight : float, default=0.1
        Weight for the phase error component. Controls the relative importance of
        phase coherence in the loss.
    eps : float, default=1e-8
        Numerical stability epsilon for log and division operations.
    freq_power : float, default=1.0
        Power for frequency weighting. Higher values emphasize high frequencies.
        freq_weight = frequency^freq_power (normalized).
    mercy_power : float, default=1.0
        Power for phase error weighting based on magnitude. Higher values focus
        phase penalties on dominant frequency bins.
    presence_power : float, default=1.0
        Power for spectral weighting based on truth magnitude. Higher values focus
        spectral penalties on high-energy frequencies.
    reduction : str, default="mean"
        Specifies the reduction to apply: "mean" or "none".
        
    Input Shape
    -----------
    pred : torch.Tensor, shape (B, C, T)
        Predicted signal tensor where B is batch size, C is number of channels,
        T is time/sequence dimension.
    truth : torch.Tensor, shape (B, C, T)
        Ground truth signal tensor with the same shape as pred.
    
    Output Shape
    ------------
    torch.Tensor
        If reduction="mean": scalar tensor with mean loss across batch and channels.
        If reduction="none": tensor of shape (B, C) with per-sample, per-channel losses.
        
    Example
    -------
    >>> loss_fn = LogosLoss(grace_coeff=0.5, phase_weight=0.1)
    >>> pred = torch.randn(4, 2, 128)
    >>> truth = torch.randn(4, 2, 128)
    >>> loss = loss_fn(pred, truth)
    >>> print(loss.item())
    """
    
    def __init__(
        self,
        grace_coeff: float = 0.5,
        phase_weight: float = 0.1,
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
        """
        Compute LogosLoss between predicted and truth tensors.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted signal, shape (B, C, T)
        truth : torch.Tensor
            Ground truth signal, shape (B, C, T)
            
        Returns
        -------
        torch.Tensor
            Loss value (scalar if reduction="mean", shape (B,C) if reduction="none")
        """
        if pred.shape != truth.shape:
            raise ValueError(f"pred and truth must match shape. Got {pred.shape} vs {truth.shape}")

        device = pred.device
        dtype = pred.dtype

        # 1) Material loss (time-domain MSE)
        material = self.mse(pred, truth).mean(dim=-1)  # (B, C)

        # 2) FFT for spectral analysis
        pred_f = torch.fft.rfft(pred, dim=-1)
        truth_f = torch.fft.rfft(truth, dim=-1)
        F = pred_f.shape[-1]

        truth_mag = torch.abs(truth_f).clamp_min(self.eps)
        pred_mag = torch.abs(pred_f).clamp_min(self.eps)

        # 3) Spectral loss (log-magnitude difference)
        # Weighted by truth energy distribution and frequency importance
        log_diff = (torch.log(pred_mag) - torch.log(truth_mag)) ** 2  # (B, C, F)

        # Weight by truth signal's energy presence
        presence_w = truth_mag ** self.presence_power
        presence_w = presence_w / presence_w.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # Weight by frequency (emphasize higher frequencies if freq_power > 1)
        freq = torch.linspace(0.0, 1.0, F, device=device, dtype=dtype)
        freq_w = (freq ** self.freq_power).view(1, 1, F)
        freq_w = freq_w / freq_w.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        spectral = (log_diff * presence_w * freq_w).sum(dim=-1)  # (B, C)

        # 4) Spectral noise (phase error - wrap-safe)
        # Measures phase coherence weighted by dominant bins
        interaction = pred_f * torch.conj(truth_f)
        angle = torch.angle(interaction)
        phase_err = 1.0 - torch.cos(angle)  # [0..2], wrap-safe

        # Focus on dominant frequency bins
        mercy = truth_mag ** self.mercy_power
        mercy = mercy / mercy.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        phase = (phase_err * mercy).sum(dim=-1)  # (B, C)

        # Combine all components
        total = material + (self.grace_coeff * spectral) + (self.phase_weight * phase)

        if self.reduction == "none":
            return total
        return total.mean()
