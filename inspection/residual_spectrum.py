"""
Residual spectrum analysis tool.

Compares frequency content of residuals across different loss functions.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def analyze_residual_spectrum(
    pred: torch.Tensor,
    truth: torch.Tensor,
    loss_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Analyze the frequency spectrum of residuals.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted signals
    truth : torch.Tensor
        Ground truth signals
    loss_name : str
        Name of the loss function used
    output_dir : Path
        Directory to save results
        
    Returns
    -------
    dict
        Analysis results with spectral statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute residuals
    residuals = pred - truth
    
    # FFT analysis
    residual_fft = torch.fft.rfft(residuals, dim=-1)
    residual_mag = torch.abs(residual_fft)
    
    truth_fft = torch.fft.rfft(truth, dim=-1)
    truth_mag = torch.abs(truth_fft)
    
    # Statistics
    mean_residual_mag = residual_mag.mean(dim=(0, 1)).cpu().numpy()
    mean_truth_mag = truth_mag.mean(dim=(0, 1)).cpu().numpy()
    
    # Relative error in frequency domain
    rel_error = mean_residual_mag / (mean_truth_mag + 1e-8)
    
    freq_bins = np.arange(len(mean_residual_mag))
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Magnitude spectrum
    axes[0].plot(freq_bins, mean_truth_mag, label='Truth', alpha=0.7)
    axes[0].plot(freq_bins, mean_residual_mag, label='Residual', alpha=0.7)
    axes[0].set_xlabel('Frequency Bin')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title(f'Residual Spectrum - {loss_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Relative error
    axes[1].plot(freq_bins, rel_error, color='red', alpha=0.7)
    axes[1].set_xlabel('Frequency Bin')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_title('Relative Spectral Error')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plot_path = output_dir / f"residual_spectrum_{loss_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # Numeric summary
    summary = {
        "loss_name": loss_name,
        "mean_residual_power": float(np.sum(mean_residual_mag ** 2)),
        "mean_truth_power": float(np.sum(mean_truth_mag ** 2)),
        "residual_to_truth_ratio": float(
            np.sum(mean_residual_mag ** 2) / (np.sum(mean_truth_mag ** 2) + 1e-8)
        ),
        "peak_residual_freq": int(np.argmax(mean_residual_mag)),
        "peak_residual_magnitude": float(np.max(mean_residual_mag)),
        "mean_relative_error": float(np.mean(rel_error)),
        "max_relative_error": float(np.max(rel_error)),
    }
    
    # Save summary
    summary_path = output_dir / f"residual_summary_{loss_name}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Residual spectrum analysis saved to {plot_path}")
    
    return summary


def compare_residual_spectra(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Compare residual spectra across multiple loss functions.
    
    Parameters
    ----------
    results : list
        List of residual analysis results
    output_dir : Path
        Directory to save comparison plot
    """
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for result in results:
        loss_name = result['loss_name']
        ratio = result['residual_to_truth_ratio']
        ax.bar(loss_name, ratio, alpha=0.7)
    
    ax.set_ylabel('Residual/Truth Power Ratio')
    ax.set_title('Residual Power Comparison Across Losses')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "residual_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"  Comparison plot saved to {plot_path}")
