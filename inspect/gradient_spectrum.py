"""
Gradient spectrum analysis tool.

Analyzes the frequency content of gradients during training.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_gradient_spectrum(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    pred: torch.Tensor,
    truth: torch.Tensor,
    output_dir: Path,
    loss_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Analyze the frequency spectrum of gradients.
    
    Computes gradients and analyzes their frequency content to understand
    how different loss functions shape the optimization landscape.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model (not used directly, but for context)
    loss_fn : torch.nn.Module
        Loss function
    pred : torch.Tensor
        Predicted signals (requires_grad=True)
    truth : torch.Tensor
        Ground truth signals
    output_dir : Path
        Directory to save results
    loss_name : str
        Name of the loss function
        
    Returns
    -------
    dict
        Gradient spectrum analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure pred requires grad
    pred = pred.detach().requires_grad_(True)
    
    # Compute loss and gradients
    loss = loss_fn(pred, truth)
    loss.backward()
    
    grad = pred.grad
    
    if grad is None:
        return {"error": "No gradients computed"}
    
    # FFT of gradients
    grad_fft = torch.fft.rfft(grad, dim=-1)
    grad_mag = torch.abs(grad_fft)
    
    # Statistics
    mean_grad_mag = grad_mag.mean(dim=(0, 1)).cpu().numpy()
    freq_bins = np.arange(len(mean_grad_mag))
    
    # Also analyze signal FFT for comparison
    pred_fft = torch.fft.rfft(pred.detach(), dim=-1)
    pred_mag = torch.abs(pred_fft).mean(dim=(0, 1)).cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gradient magnitude spectrum
    axes[0].plot(freq_bins, mean_grad_mag, label='Gradient', alpha=0.7, color='purple')
    axes[0].set_xlabel('Frequency Bin')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title(f'Gradient Spectrum - {loss_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Normalized comparison
    grad_norm = mean_grad_mag / (mean_grad_mag.sum() + 1e-8)
    pred_norm = pred_mag / (pred_mag.sum() + 1e-8)
    
    axes[1].plot(freq_bins, grad_norm, label='Gradient (normalized)', alpha=0.7, color='purple')
    axes[1].plot(freq_bins, pred_norm, label='Signal (normalized)', alpha=0.7, color='blue')
    axes[1].set_xlabel('Frequency Bin')
    axes[1].set_ylabel('Normalized Magnitude')
    axes[1].set_title('Gradient vs Signal Frequency Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"gradient_spectrum_{loss_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # Numeric summary
    summary = {
        "loss_name": loss_name,
        "total_gradient_power": float(np.sum(mean_grad_mag ** 2)),
        "peak_gradient_freq": int(np.argmax(mean_grad_mag)),
        "peak_gradient_magnitude": float(np.max(mean_grad_mag)),
        "gradient_entropy": float(-np.sum(grad_norm * np.log(grad_norm + 1e-10))),
        "high_freq_ratio": float(
            np.sum(mean_grad_mag[len(mean_grad_mag)//2:]) / 
            (np.sum(mean_grad_mag) + 1e-8)
        ),
    }
    
    # Save summary
    summary_path = output_dir / f"gradient_summary_{loss_name}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Gradient spectrum analysis saved to {plot_path}")
    
    return summary


def compare_gradient_spectra(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Compare gradient spectra across multiple loss functions.
    
    Parameters
    ----------
    results : list
        List of gradient analysis results
    output_dir : Path
        Directory to save comparison plot
    """
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # High-frequency ratio comparison
    loss_names = [r['loss_name'] for r in results]
    high_freq_ratios = [r['high_freq_ratio'] for r in results]
    
    axes[0].bar(loss_names, high_freq_ratios, alpha=0.7)
    axes[0].set_ylabel('High-Frequency Gradient Ratio')
    axes[0].set_title('High-Frequency Gradient Content')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Gradient entropy comparison
    entropies = [r['gradient_entropy'] for r in results]
    
    axes[1].bar(loss_names, entropies, alpha=0.7, color='orange')
    axes[1].set_ylabel('Gradient Entropy')
    axes[1].set_title('Gradient Frequency Diversity')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / "gradient_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"  Gradient comparison plot saved to {plot_path}")
