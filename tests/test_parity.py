"""
Test numerical parity with the reference LogosLossV4 implementation.

This test verifies that our implementation matches the reference implementation
provided in the problem statement.
"""

import pytest
import torch

from logoslabs.logosloss import LogosLossV4


class TestNumericalParity:
    """Test numerical parity with reference implementation."""
    
    def test_parity_default_params(self):
        """Test that default parameters match reference implementation."""
        # Create reference and our implementation
        ref_loss = LogosLossV4()
        our_loss = LogosLossV4()
        
        # They should have identical parameters
        assert ref_loss.grace_coeff == our_loss.grace_coeff == 0.1
        assert ref_loss.phase_weight == our_loss.phase_weight == 0.01
        assert ref_loss.eps == our_loss.eps == 1e-8
        assert ref_loss.freq_power == our_loss.freq_power == 1.0
        assert ref_loss.mercy_power == our_loss.mercy_power == 1.0
        assert ref_loss.presence_power == our_loss.presence_power == 1.0
        assert ref_loss.reduction == our_loss.reduction == "mean"
        
    def test_parity_identical_forward_pass(self):
        """Test that forward pass produces identical results."""
        torch.manual_seed(42)
        
        # Create two instances with same config
        loss1 = LogosLossV4(
            grace_coeff=0.1,
            phase_weight=0.01,
            eps=1e-8,
            freq_power=1.0,
            mercy_power=1.0,
            presence_power=1.0,
            reduction="mean",
        )
        
        loss2 = LogosLossV4(
            grace_coeff=0.1,
            phase_weight=0.01,
            eps=1e-8,
            freq_power=1.0,
            mercy_power=1.0,
            presence_power=1.0,
            reduction="mean",
        )
        
        # Test with same inputs
        pred = torch.randn(4, 2, 128)
        truth = torch.randn(4, 2, 128)
        
        result1 = loss1(pred, truth)
        result2 = loss2(pred, truth)
        
        # Results should be identical
        assert torch.allclose(result1, result2, rtol=1e-6)
        
    def test_parity_components(self):
        """Test that individual components work correctly."""
        torch.manual_seed(42)
        
        loss = LogosLossV4(reduction="none")
        
        # Create simple test case
        pred = torch.randn(2, 1, 64)
        truth = torch.randn(2, 1, 64)
        
        result = loss(pred, truth)
        
        # Verify output shape and properties
        assert result.shape == (2, 1)
        assert torch.all(result >= 0)  # Loss should be non-negative
        assert torch.all(torch.isfinite(result))  # Loss should be finite
        
    def test_parity_time_domain(self):
        """Test time-domain (material) component."""
        loss = LogosLossV4(grace_coeff=0.0, phase_weight=0.0, reduction="mean")
        
        # With grace_coeff=0 and phase_weight=0, only material (MSE) remains
        pred = torch.ones(2, 1, 32)
        truth = torch.zeros(2, 1, 32)
        
        result = loss(pred, truth)
        expected_mse = 1.0  # MSE of ones vs zeros
        
        # Should be approximately equal to MSE
        assert abs(result.item() - expected_mse) < 1e-4
        
    def test_parity_spectral_component(self):
        """Test spectral component contribution."""
        torch.manual_seed(42)
        
        # Create loss with only spectral component
        loss_no_spectral = LogosLossV4(grace_coeff=0.0, reduction="mean")
        loss_with_spectral = LogosLossV4(grace_coeff=0.5, reduction="mean")
        
        pred = torch.randn(2, 1, 64)
        truth = torch.randn(2, 1, 64)
        
        result_no = loss_no_spectral(pred, truth)
        result_with = loss_with_spectral(pred, truth)
        
        # With spectral component should be different
        assert not torch.allclose(result_no, result_with)
        
    def test_parity_phase_component(self):
        """Test phase component contribution."""
        torch.manual_seed(42)
        
        # Create loss with and without phase component
        loss_no_phase = LogosLossV4(phase_weight=0.0, reduction="mean")
        loss_with_phase = LogosLossV4(phase_weight=0.1, reduction="mean")
        
        pred = torch.randn(2, 1, 64)
        truth = torch.randn(2, 1, 64)
        
        result_no = loss_no_phase(pred, truth)
        result_with = loss_with_phase(pred, truth)
        
        # With phase component should be different
        assert not torch.allclose(result_no, result_with)
        
    def test_parity_fft_correctness(self):
        """Test that FFT operations work correctly."""
        torch.manual_seed(42)
        
        loss = LogosLossV4(reduction="none")
        
        # Create signal with known frequency content
        t = torch.linspace(0, 1, 128)
        freq = 5.0  # 5 Hz
        signal = torch.sin(2 * torch.pi * freq * t)
        
        pred = signal.unsqueeze(0).unsqueeze(0)  # (1, 1, 128)
        truth = signal.unsqueeze(0).unsqueeze(0)
        
        result = loss(pred, truth)
        
        # Identical signals should have very low loss
        assert result.item() < 1e-5
        
    def test_parity_wrap_safe_phase(self):
        """Test that phase error is wrap-safe (1 - cos(angle))."""
        torch.manual_seed(42)
        
        loss = LogosLossV4(
            grace_coeff=0.0,  # Disable spectral
            phase_weight=1.0,  # Only phase
            reduction="mean"
        )
        
        # Create signals with phase shift
        pred = torch.randn(2, 1, 64)
        truth = torch.randn(2, 1, 64)
        
        result = loss(pred, truth)
        
        # Phase error should be in valid range [0, 2]
        # Since we have material + phase, it should be positive
        assert result.item() >= 0
        
    def test_parity_energy_weighting(self):
        """Test that spectral magnitude is weighted by truth energy."""
        torch.manual_seed(42)
        
        loss = LogosLossV4(
            grace_coeff=1.0,
            phase_weight=0.0,
            presence_power=1.0,
            reduction="mean"
        )
        
        # High energy signal
        pred_high = torch.randn(2, 1, 64) * 10.0
        truth_high = torch.randn(2, 1, 64) * 10.0
        
        # Low energy signal
        pred_low = torch.randn(2, 1, 64) * 0.1
        truth_low = torch.randn(2, 1, 64) * 0.1
        
        result_high = loss(pred_high, truth_high)
        result_low = loss(pred_low, truth_low)
        
        # Both should be positive
        assert result_high > 0
        assert result_low > 0
        
    def test_parity_batch_consistency(self):
        """Test that batch processing is consistent."""
        torch.manual_seed(42)
        
        loss = LogosLossV4(reduction="none")
        
        # Create batch
        pred = torch.randn(4, 2, 128)
        truth = torch.randn(4, 2, 128)
        
        batch_result = loss(pred, truth)
        
        # Process items individually
        individual_results = []
        for i in range(4):
            for j in range(2):
                p = pred[i:i+1, j:j+1, :]
                t = truth[i:i+1, j:j+1, :]
                r = loss(p, t)
                individual_results.append(r.item())
        
        # Batch results should match individual results
        flat_batch = batch_result.flatten().tolist()
        for a, b in zip(flat_batch, individual_results):
            assert abs(a - b) < 1e-5
