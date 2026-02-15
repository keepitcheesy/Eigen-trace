"""
Small LSTM architecture for time series forecasting.

A lightweight LSTM-based model for sequence prediction.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Small LSTM for time series forecasting.
    
    A simple LSTM-based architecture with optional bidirectional processing
    and multiple layers. Designed for lightweight time series benchmarking.
    
    Parameters
    ----------
    input_size : int, default=1
        Number of input features per time step
    hidden_size : int, default=32
        Number of features in hidden state
    num_layers : int, default=2
        Number of LSTM layers
    output_size : int, default=1
        Number of output features
    dropout : float, default=0.1
        Dropout rate between LSTM layers (if num_layers > 1)
    bidirectional : bool, default=False
        If True, use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: tuple = None,
    ) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, C, T) where C is input_size
        hidden : tuple, optional
            Initial hidden state and cell state
            
        Returns
        -------
        torch.Tensor
            Output tensor, shape (B, C_out, T) where C_out is output_size
        """
        # x shape: (B, C, T)
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        
        # Reshape for LSTM: (B, T, C)
        x = x.permute(0, 2, 1)
        
        # LSTM forward
        if hidden is None:
            lstm_out, _ = self.lstm(x)
        else:
            lstm_out, _ = self.lstm(x, hidden)
        
        # Apply output layer: (B, T, hidden) -> (B, T, output_size)
        out = self.fc(lstm_out)
        
        # Reshape back: (B, T, C_out) -> (B, C_out, T)
        out = out.permute(0, 2, 1)
        
        return out
    
    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """
        Initialize hidden and cell states.
        
        Parameters
        ----------
        batch_size : int
            Batch size
        device : torch.device
            Device to create tensors on
            
        Returns
        -------
        tuple
            (hidden_state, cell_state) both of shape 
            (num_layers * num_directions, batch_size, hidden_size)
        """
        h0 = torch.zeros(
            self.num_layers * self.num_directions, 
            batch_size, 
            self.hidden_size,
            device=device,
        )
        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return (h0, c0)
