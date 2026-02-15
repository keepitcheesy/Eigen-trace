"""
Baseline loss functions for comparison.

Includes standard MSE and robust Huber/Charbonnier losses.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Mean Squared Error loss.
    
    Standard L2 loss between predicted and truth signals.
    Serves as a basic baseline for comparison.
    
    Parameters
    ----------
    reduction : str, default="mean"
        Specifies the reduction: "mean", "sum", or "none".
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted signal
        truth : torch.Tensor
            Ground truth signal
            
        Returns
        -------
        torch.Tensor
            MSE loss
        """
        return self.mse(pred, truth)


class HuberLoss(nn.Module):
    """
    Huber loss (also known as Charbonnier loss when delta is small).
    
    Combines L2 loss for small errors and L1 loss for large errors,
    making it more robust to outliers than pure MSE.
    
    The Huber loss is defined as:
        L(x) = 0.5 * x^2                  if |x| <= delta
        L(x) = delta * (|x| - 0.5*delta)  if |x| > delta
    
    When delta is very small (e.g., 0.001), this approximates the Charbonnier loss:
        L(x) = sqrt(x^2 + eps^2) - eps
    
    Parameters
    ----------
    delta : float, default=1.0
        Threshold at which to change between L2 and L1 behavior.
        Smaller values make the loss more robust but less smooth.
    reduction : str, default="mean"
        Specifies the reduction: "mean", "sum", or "none".
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.huber = nn.HuberLoss(delta=delta, reduction=reduction)
    
    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted signal
        truth : torch.Tensor
            Ground truth signal
            
        Returns
        -------
        torch.Tensor
            Huber loss
        """
        return self.huber(pred, truth)


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss: a differentiable approximation of L1 loss.
    
    Defined as: L(x) = sqrt(x^2 + eps^2) - eps
    
    This is equivalent to Huber loss with a very small delta.
    More robust to outliers than MSE while remaining smooth everywhere.
    
    Parameters
    ----------
    eps : float, default=1e-3
        Small constant for numerical stability and smoothness control.
    reduction : str, default="mean"
        Specifies the reduction: "mean", "sum", or "none".
    """
    
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Compute Charbonnier loss.
        
        Parameters
        ----------
        pred : torch.Tensor
            Predicted signal
        truth : torch.Tensor
            Ground truth signal
            
        Returns
        -------
        torch.Tensor
            Charbonnier loss
        """
        diff = pred - truth
        loss = torch.sqrt(diff ** 2 + self.eps ** 2) - self.eps
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
