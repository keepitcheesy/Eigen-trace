"""
Time series forecasting task with controllable spectrum.

Provides synthetic time series with specific frequency characteristics.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class TimeSeriesForecastTask:
    """
    Time series forecasting task with controllable spectral properties.
    
    Creates synthetic time series with specific frequency characteristics
    and evaluates forecasting using MAE, MSE, and optional spectral error.
    
    Parameters
    ----------
    seq_length : int, default=128
        Length of time series sequences
    forecast_horizon : int, default=16
        Number of steps to forecast ahead
    num_samples : int, default=100
        Number of synthetic time series to generate
    freq_bands : Optional[list], default=None
        List of (freq, amplitude) tuples for dominant frequencies.
        If None, uses random frequencies.
    noise_level : float, default=0.1
        Standard deviation of additive noise
    seed : Optional[int], default=None
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        seq_length: int = 128,
        forecast_horizon: int = 16,
        num_samples: int = 100,
        freq_bands: Optional[list] = None,
        noise_level: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.num_samples = num_samples
        self.freq_bands = freq_bands
        self.noise_level = noise_level
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate input sequences and forecast targets.
        
        Returns
        -------
        inputs : torch.Tensor, shape (N, 1, seq_length)
            Input time series sequences
        targets : torch.Tensor, shape (N, 1, forecast_horizon)
            Target forecast values
        """
        sequences = []
        
        for _ in range(self.num_samples):
            # Generate full sequence (input + target)
            full_seq = self._generate_sequence(
                self.seq_length + self.forecast_horizon
            )
            sequences.append(full_seq)
        
        sequences = torch.stack(sequences, dim=0)  # (N, 1, total_length)
        
        # Split into input and target
        inputs = sequences[:, :, :self.seq_length]
        targets = sequences[:, :, self.seq_length:]
        
        return inputs, targets
    
    def _generate_sequence(self, length: int) -> torch.Tensor:
        """Generate a single time series with controlled spectrum."""
        t = torch.linspace(0, 1, length)
        signal = torch.zeros(length)
        
        if self.freq_bands is None:
            # Random frequencies
            num_freqs = np.random.randint(1, 4)
            for _ in range(num_freqs):
                freq = np.random.uniform(0.5, 5.0)
                amp = np.random.uniform(0.5, 1.0)
                phase = np.random.uniform(0, 2 * np.pi)
                signal += amp * torch.sin(2 * np.pi * freq * t + phase)
        else:
            # Use specified frequency bands
            for freq, amp in self.freq_bands:
                phase = np.random.uniform(0, 2 * np.pi)
                signal += amp * torch.sin(2 * np.pi * freq * t + phase)
        
        # Add trend (optional)
        if np.random.rand() < 0.3:
            trend_coef = np.random.uniform(-0.5, 0.5)
            signal += trend_coef * t
        
        # Add noise
        noise = torch.randn(length) * self.noise_level
        signal += noise
        
        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        return signal.unsqueeze(0)  # (1, length)
    
    def compute_metrics(
        self, 
        pred: torch.Tensor, 
        truth: torch.Tensor,
        compute_spectral: bool = True,
    ) -> Dict[str, float]:
        """
        Compute MAE, MSE, and optional spectral error.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted forecasts
        truth : torch.Tensor
            Ground truth forecasts
        compute_spectral : bool, default=True
            Whether to compute spectral error
            
        Returns
        -------
        dict
            Dictionary with 'mae', 'mse', and optionally 'spectral_error'
        """
        mae = torch.mean(torch.abs(pred - truth))
        mse = torch.mean((pred - truth) ** 2)
        
        metrics = {
            "mae": mae.item(),
            "mse": mse.item(),
        }
        
        if compute_spectral:
            spectral_err = self._compute_spectral_error(pred, truth)
            metrics["spectral_error"] = spectral_err.item()
        
        return metrics
    
    def _compute_spectral_error(
        self, pred: torch.Tensor, truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral error (magnitude difference in frequency domain).
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted signal
        truth : torch.Tensor
            Ground truth signal
            
        Returns
        -------
        torch.Tensor
            Spectral error (mean absolute difference of magnitudes)
        """
        # FFT
        pred_f = torch.fft.rfft(pred, dim=-1)
        truth_f = torch.fft.rfft(truth, dim=-1)
        
        # Magnitude difference
        pred_mag = torch.abs(pred_f)
        truth_mag = torch.abs(truth_f)
        
        spectral_err = torch.mean(torch.abs(pred_mag - truth_mag))
        
        return spectral_err
