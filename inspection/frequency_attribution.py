"""
Frequency attribution analysis tool.

Computes per-frequency penalty contribution for LogosLoss.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_frequency_attribution(
    pred: torch.Tensor,
    truth: torch.Tensor,
    grace_coeff: float = 0.5,
    phase_weight: float = 0.1,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Compute per-frequency penalty attribution for LogosLoss.
    
    Shows which frequencies contribute most to the loss, helping understand
    where the model struggles.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted signals
    truth : torch.Tensor
        Ground truth signals
    grace_coeff : float
        Spectral weight coefficient
    phase_weight : float
        Phase weight coefficient
    output_dir : Path, optional
        Directory to save results
        
    Returns
    -------
    dict
        Attribution analysis with per-frequency contributions
    """
    eps = 1e-8
    
    # FFT
    pred_f = torch.fft.rfft(pred, dim=-1)
    truth_f = torch.fft.rfft(truth, dim=-1)
    F = pred_f.shape[-1]
    
    truth_mag = torch.abs(truth_f).clamp_min(eps)
    pred_mag = torch.abs(pred_f).clamp_min(eps)
    
    # Spectral component (per frequency)
    log_diff = (torch.log(pred_mag) - torch.log(truth_mag)) ** 2
    presence_w = truth_mag / truth_mag.sum(dim=-1, keepdim=True).clamp_min(eps)
    
    freq = torch.linspace(0.0, 1.0, F, device=pred.device, dtype=pred.dtype)
    freq_w = freq / freq.sum().clamp_min(eps)
    freq_w = freq_w.view(1, 1, F)
    
    spectral_per_freq = log_diff * presence_w * freq_w  # (B, C, F)
    spectral_attribution = spectral_per_freq.mean(dim=(0, 1)).cpu().numpy()
    
    # Phase component (per frequency)
    interaction = pred_f * torch.conj(truth_f)
    angle = torch.angle(interaction)
    phase_err = 1.0 - torch.cos(angle)
    
    mercy = truth_mag / truth_mag.sum(dim=-1, keepdim=True).clamp_min(eps)
    phase_per_freq = phase_err * mercy  # (B, C, F)
    phase_attribution = phase_per_freq.mean(dim=(0, 1)).cpu().numpy()
    
    # Total attribution
    total_attribution = spectral_attribution * grace_coeff + phase_attribution * phase_weight
    
    freq_bins = np.arange(F)
    
    # Plot
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        # Spectral attribution
        axes[0].bar(freq_bins, spectral_attribution, alpha=0.7, color='blue')
        axes[0].set_xlabel('Frequency Bin')
        axes[0].set_ylabel('Spectral Penalty')
        axes[0].set_title('Spectral Loss Attribution (per frequency)')
        axes[0].grid(True, alpha=0.3)
        
        # Phase attribution
        axes[1].bar(freq_bins, phase_attribution, alpha=0.7, color='red')
        axes[1].set_xlabel('Frequency Bin')
        axes[1].set_ylabel('Phase Penalty')
        axes[1].set_title('Phase Loss Attribution (per frequency)')
        axes[1].grid(True, alpha=0.3)
        
        # Total attribution
        axes[2].bar(freq_bins, total_attribution, alpha=0.7, color='green')
        axes[2].set_xlabel('Frequency Bin')
        axes[2].set_ylabel('Total Penalty')
        axes[2].set_title('Total Loss Attribution (per frequency)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "frequency_attribution.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"  Frequency attribution plot saved to {plot_path}")
    
    # Numeric summary
    summary = {
        "total_spectral_penalty": float(spectral_attribution.sum()),
        "total_phase_penalty": float(phase_attribution.sum()),
        "total_combined_penalty": float(total_attribution.sum()),
        "peak_spectral_freq": int(np.argmax(spectral_attribution)),
        "peak_phase_freq": int(np.argmax(phase_attribution)),
        "peak_total_freq": int(np.argmax(total_attribution)),
        "spectral_concentration": float(np.max(spectral_attribution) / (spectral_attribution.sum() + eps)),
        "phase_concentration": float(np.max(phase_attribution) / (phase_attribution.sum() + eps)),
    }
    
    if output_dir is not None:
        summary_path = output_dir / "frequency_attribution.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return summary
