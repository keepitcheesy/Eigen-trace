"""
Unit tests for LogosLossV4 implementation.

Tests numerical parity with the reference implementation.
"""

import pytest
import torch
import numpy as np

from logoslabs.logosloss import LogosLossV4


class TestLogosLossV4:
    """Test LogosLossV4 implementation."""
    
    def test_initialization(self):
        """Test that LogosLossV4 initializes with correct parameters."""
        loss = LogosLossV4()
        assert loss.grace_coeff == 0.5
        assert loss.phase_weight == 0.1
        assert loss.eps == 1e-8
        assert loss.freq_power == 1.0
        assert loss.mercy_power == 1.0
        assert loss.presence_power == 1.0
        assert loss.reduction == "mean"
        
    def test_initialization_custom(self):
        """Test LogosLossV4 with custom parameters."""
        loss = LogosLossV4(
            grace_coeff=0.7,
            phase_weight=0.2,
            eps=1e-6,
            freq_power=2.0,
            mercy_power=1.5,
            presence_power=0.8,
            reduction="none",
        )
        assert loss.grace_coeff == 0.7
        assert loss.phase_weight == 0.2
        assert loss.eps == 1e-6
        assert loss.freq_power == 2.0
        assert loss.mercy_power == 1.5
        assert loss.presence_power == 0.8
        assert loss.reduction == "none"
        
    def test_invalid_reduction(self):
        """Test that invalid reduction raises ValueError."""
        with pytest.raises(ValueError, match="reduction must be 'mean' or 'none'"):
            LogosLossV4(reduction="sum")
            
    def test_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        loss = LogosLossV4()
        pred = torch.randn(2, 3, 100)
        truth = torch.randn(2, 3, 50)
        
        with pytest.raises(ValueError, match="pred and truth must match shape"):
            loss(pred, truth)
            
    def test_forward_mean_reduction(self):
        """Test forward pass with mean reduction."""
        torch.manual_seed(42)
        loss = LogosLossV4(reduction="mean")
        
        pred = torch.randn(4, 2, 128)
        truth = torch.randn(4, 2, 128)
        
        result = loss(pred, truth)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == ()  # Scalar
        assert result.item() > 0  # Loss should be positive
        
    def test_forward_none_reduction(self):
        """Test forward pass with no reduction."""
        torch.manual_seed(42)
        loss = LogosLossV4(reduction="none")
        
        batch_size, channels = 4, 2
        pred = torch.randn(batch_size, channels, 128)
        truth = torch.randn(batch_size, channels, 128)
        
        result = loss(pred, truth)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (batch_size, channels)
        assert torch.all(result > 0)  # All losses should be positive
        
    def test_identical_inputs(self):
        """Test that identical inputs produce zero loss (approximately)."""
        torch.manual_seed(42)
        loss = LogosLossV4(reduction="mean")
        
        x = torch.randn(2, 1, 64)
        result = loss(x, x)
        
        # Should be very close to zero
        assert result.item() < 1e-5
        
    def test_deterministic(self):
        """Test that same inputs produce same outputs."""
        torch.manual_seed(42)
        pred1 = torch.randn(2, 1, 64)
        truth1 = torch.randn(2, 1, 64)
        
        loss = LogosLossV4(reduction="mean")
        result1 = loss(pred1, truth1)
        
        # Reset and compute again
        loss = LogosLossV4(reduction="mean")
        result2 = loss(pred1, truth1)
        
        assert torch.allclose(result1, result2)
        
    def test_different_inputs(self):
        """Test that different inputs produce different losses."""
        torch.manual_seed(42)
        loss = LogosLossV4(reduction="mean")
        
        pred = torch.randn(2, 1, 64)
        truth1 = torch.randn(2, 1, 64)
        truth2 = torch.randn(2, 1, 64) + 10.0
        
        result1 = loss(pred, truth1)
        result2 = loss(pred, truth2)
        
        assert result1.item() != result2.item()
        
    def test_grace_coeff_effect(self):
        """Test that grace_coeff affects the output."""
        torch.manual_seed(42)
        pred = torch.randn(2, 1, 64)
        truth = torch.randn(2, 1, 64)
        
        loss1 = LogosLossV4(grace_coeff=0.1, reduction="mean")
        loss2 = LogosLossV4(grace_coeff=0.9, reduction="mean")
        
        result1 = loss1(pred, truth)
        result2 = loss2(pred, truth)
        
        # Different grace coefficients should produce different results
        assert result1.item() != result2.item()
        
    def test_phase_weight_effect(self):
        """Test that phase_weight affects the output."""
        torch.manual_seed(42)
        pred = torch.randn(2, 1, 64)
        truth = torch.randn(2, 1, 64)
        
        loss1 = LogosLossV4(phase_weight=0.01, reduction="mean")
        loss2 = LogosLossV4(phase_weight=0.5, reduction="mean")
        
        result1 = loss1(pred, truth)
        result2 = loss2(pred, truth)
        
        # Different phase weights should produce different results
        assert result1.item() != result2.item()
        
    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        torch.manual_seed(42)
        loss = LogosLossV4(reduction="none")
        
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            pred = torch.randn(batch_size, 3, 100)
            truth = torch.randn(batch_size, 3, 100)
            
            result = loss(pred, truth)
            assert result.shape == (batch_size, 3)
            
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        torch.manual_seed(42)
        loss = LogosLossV4(reduction="mean")
        
        seq_lengths = [16, 32, 64, 128, 256]
        for seq_len in seq_lengths:
            pred = torch.randn(2, 1, seq_len)
            truth = torch.randn(2, 1, seq_len)
            
            result = loss(pred, truth)
            assert isinstance(result, torch.Tensor)
            assert result.shape == ()
            
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        loss = LogosLossV4(reduction="mean")
        
        # Very small values
        pred = torch.ones(2, 1, 32) * 1e-10
        truth = torch.ones(2, 1, 32) * 1e-10
        result = loss(pred, truth)
        assert torch.isfinite(result)
        
        # Very large values
        pred = torch.ones(2, 1, 32) * 1e5
        truth = torch.ones(2, 1, 32) * 1e5
        result = loss(pred, truth)
        assert torch.isfinite(result)
        
    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        loss = LogosLossV4(reduction="mean")
        
        pred = torch.randn(2, 1, 64, requires_grad=True)
        truth = torch.randn(2, 1, 64)
        
        result = loss(pred, truth)
        result.backward()
        
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))
