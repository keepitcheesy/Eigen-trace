"""
Small MLP architecture for simple regression tasks.

A basic feedforward network with a few hidden layers.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Small Multi-Layer Perceptron for regression tasks.
    
    A simple feedforward network with configurable hidden layers.
    Designed for lightweight benchmarking on 1D signals.
    
    Parameters
    ----------
    input_size : int
        Size of input features
    output_size : int
        Size of output features
    hidden_sizes : list, default=[64, 32]
        Sizes of hidden layers
    dropout : float, default=0.1
        Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, C, T) or (B, input_size)
            
        Returns
        -------
        torch.Tensor
            Output tensor, shape (B, output_size) or (B, C, T)
        """
        original_shape = x.shape
        
        # Flatten if needed
        if len(x.shape) == 3:
            batch, channels, time = x.shape
            x = x.permute(0, 2, 1).reshape(batch * time, channels)
            out = self.network(x)
            out = out.reshape(batch, time, -1).permute(0, 2, 1)
        else:
            out = self.network(x)
        
        return out
