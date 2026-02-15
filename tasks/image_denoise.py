"""
Image denoising task with Gaussian and structured noise.

Provides synthetic image denoising with configurable noise types and metrics.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class ImageDenoiseTask:
    """
    Image denoising task with Gaussian and structured noise patterns.
    
    Creates noisy images and evaluates denoising performance using PSNR and SSIM.
    Supports both simple Gaussian noise and more complex structured noise patterns.
    
    Parameters
    ----------
    image_size : int, default=32
        Size of square images (image_size x image_size)
    noise_type : str, default="gaussian"
        Type of noise to add: "gaussian", "salt_pepper", or "structured"
    noise_level : float, default=0.1
        Standard deviation of noise (for Gaussian) or noise intensity [0, 1]
    num_samples : int, default=100
        Number of synthetic images to generate
    seed : Optional[int], default=None
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        image_size: int = 32,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
        num_samples: int = 100,
        seed: Optional[int] = None,
    ):
        self.image_size = image_size
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate clean and noisy image pairs.
        
        Returns
        -------
        noisy : torch.Tensor, shape (N, 1, H, W)
            Noisy images
        clean : torch.Tensor, shape (N, 1, H, W)
            Clean (ground truth) images
        """
        # Generate synthetic clean images (simple patterns)
        clean = self._generate_clean_images()
        
        # Add noise
        noisy = self._add_noise(clean)
        
        return noisy, clean
    
    def _generate_clean_images(self) -> torch.Tensor:
        """Generate synthetic clean images with various patterns."""
        images = []
        
        for i in range(self.num_samples):
            # Mix of patterns: gradients, circles, rectangles
            pattern_type = i % 4
            img = torch.zeros(1, self.image_size, self.image_size)
            
            if pattern_type == 0:
                # Horizontal gradient
                grad = torch.linspace(0, 1, self.image_size).view(1, -1)
                img[0] = grad.expand(self.image_size, -1)
            elif pattern_type == 1:
                # Vertical gradient
                grad = torch.linspace(0, 1, self.image_size).view(-1, 1)
                img[0] = grad.expand(-1, self.image_size)
            elif pattern_type == 2:
                # Circle
                center = self.image_size // 2
                y, x = torch.meshgrid(
                    torch.arange(self.image_size, dtype=torch.float32),
                    torch.arange(self.image_size, dtype=torch.float32),
                    indexing='ij'
                )
                dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
                img[0] = (dist < self.image_size // 3).float()
            else:
                # Rectangle
                h_start = self.image_size // 4
                h_end = 3 * self.image_size // 4
                w_start = self.image_size // 4
                w_end = 3 * self.image_size // 4
                img[0, h_start:h_end, w_start:w_end] = 1.0
            
            images.append(img)
        
        return torch.stack(images, dim=0)
    
    def _add_noise(self, clean: torch.Tensor) -> torch.Tensor:
        """Add noise to clean images."""
        if self.noise_type == "gaussian":
            noise = torch.randn_like(clean) * self.noise_level
            noisy = clean + noise
        elif self.noise_type == "salt_pepper":
            noisy = clean.clone()
            # Salt (white)
            salt_mask = torch.rand_like(clean) < self.noise_level / 2
            noisy[salt_mask] = 1.0
            # Pepper (black)
            pepper_mask = torch.rand_like(clean) < self.noise_level / 2
            noisy[pepper_mask] = 0.0
        elif self.noise_type == "structured":
            # Structured noise: bands + Gaussian
            noise = torch.randn_like(clean) * self.noise_level * 0.5
            # Add horizontal bands
            for i in range(0, self.image_size, 8):
                noise[:, :, i:i+2, :] += self.noise_level * 0.5
            noisy = clean + noise
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return torch.clamp(noisy, 0, 1)
    
    def compute_metrics(
        self, pred: torch.Tensor, truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute PSNR and SSIM metrics for denoised images.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted/denoised images
        truth : torch.Tensor
            Ground truth clean images
            
        Returns
        -------
        dict
            Dictionary with 'psnr' and 'ssim' metrics
        """
        pred = torch.clamp(pred, 0, 1)
        
        # PSNR
        mse = torch.mean((pred - truth) ** 2)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
        
        # Simplified SSIM (full SSIM would require more computation)
        # This is a fast approximation
        ssim = self._compute_ssim(pred, truth)
        
        return {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
        }
    
    def _compute_ssim(
        self, pred: torch.Tensor, truth: torch.Tensor, window_size: int = 11
    ) -> torch.Tensor:
        """
        Compute simplified SSIM.
        
        This is a lightweight approximation of SSIM that captures the main idea
        without full Gaussian windowing.
        """
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        mu_pred = pred.mean()
        mu_truth = truth.mean()
        
        sigma_pred = pred.var()
        sigma_truth = truth.var()
        sigma_pred_truth = ((pred - mu_pred) * (truth - mu_truth)).mean()
        
        ssim = ((2 * mu_pred * mu_truth + c1) * (2 * sigma_pred_truth + c2)) / \
               ((mu_pred ** 2 + mu_truth ** 2 + c1) * (sigma_pred + sigma_truth + c2))
        
        return ssim
