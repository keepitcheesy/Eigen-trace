"""
Small U-Net architecture for image denoising.

A lightweight U-Net with skip connections, suitable for quick benchmarking.
"""

import torch
import torch.nn as nn


class UNet(nn.Module):
    """
    Small U-Net for image denoising tasks.
    
    A simple encoder-decoder architecture with skip connections.
    Designed to be fast for benchmarking while maintaining reasonable performance.
    
    Parameters
    ----------
    in_channels : int, default=1
        Number of input channels
    out_channels : int, default=1
        Number of output channels
    base_channels : int, default=16
        Number of channels in the first layer (doubled at each level)
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 2, base_channels * 4)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec1 = self._conv_block(base_channels * 4, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = self._conv_block(base_channels * 2, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create a convolutional block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, C, H, W)
            
        Returns
        -------
        torch.Tensor
            Output tensor, shape (B, C, H, W)
        """
        # Encoder
        enc1 = self.enc1(x)
        enc1_pool = self.pool1(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc2_pool)
        
        # Decoder with skip connections
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat([dec1, enc2], dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        # Output
        out = self.out(dec2)
        
        return out
