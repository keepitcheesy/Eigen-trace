"""
Benchmark tests for LogosLoss AVP.

Run with: RUN_BENCHMARKS=1 pytest tests/test_benchmark.py
"""

import pytest
import os
import time
import torch

from logoslabs.avp import AVPProcessor, compute_instability_score
from logoslabs.logosloss import LogosLossV4


# Skip benchmark tests unless RUN_BENCHMARKS=1
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BENCHMARKS") != "1",
    reason="Benchmark tests only run when RUN_BENCHMARKS=1"
)


class TestBenchmarks:
    """Benchmark tests for performance measurement."""
    
    def test_benchmark_logosloss_forward(self):
        """Benchmark LogosLossV4 forward pass."""
        loss = LogosLossV4(reduction="mean")
        
        # Warm up
        pred = torch.randn(32, 4, 256)
        truth = torch.randn(32, 4, 256)
        _ = loss(pred, truth)
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            pred = torch.randn(32, 4, 256)
            truth = torch.randn(32, 4, 256)
            _ = loss(pred, truth)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_iterations
        
        print(f"\nLogosLossV4 forward pass (batch=32, channels=4, length=256):")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {num_iterations/elapsed:.1f} iterations/s")
        
        # Sanity check: should complete in reasonable time
        assert avg_time < 1.0  # Less than 1 second per iteration
        
    def test_benchmark_instability_scoring(self):
        """Benchmark instability score computation."""
        loss_fn = LogosLossV4()
        
        pred = "This is a sample prediction text that needs to be scored"
        truth = "This is the ground truth text for comparison purposes"
        
        # Warm up
        _ = compute_instability_score(pred, truth, loss_fn)
        
        # Benchmark
        num_iterations = 1000
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = compute_instability_score(pred, truth, loss_fn)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_iterations
        
        print(f"\nInstability score computation:")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {num_iterations/elapsed:.1f} scores/s")
        
        # Sanity check
        assert avg_time < 0.1  # Less than 100ms per score
        
    def test_benchmark_batch_processing(self):
        """Benchmark batch processing."""
        processor = AVPProcessor()
        
        # Create test batch
        items = [
            {
                "prediction": f"Prediction text number {i} for testing",
                "truth": f"Ground truth text number {i} for comparison",
            }
            for i in range(100)
        ]
        
        # Warm up
        _ = processor.process_batch(items[:10])
        
        # Benchmark
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = processor.process_batch(items)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / num_iterations
        items_per_second = len(items) * num_iterations / elapsed
        
        print(f"\nBatch processing (100 items):")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Average batch time: {avg_time:.3f}s")
        print(f"  Throughput: {items_per_second:.1f} items/s")
        
        # Sanity check
        assert avg_time < 10.0  # Less than 10 seconds per batch of 100
        
    def test_benchmark_different_sequence_lengths(self):
        """Benchmark with different sequence lengths."""
        loss = LogosLossV4(reduction="mean")
        
        lengths = [64, 128, 256, 512, 1024]
        results = []
        
        print("\nSequence length scaling:")
        
        for length in lengths:
            # Warm up
            pred = torch.randn(8, 2, length)
            truth = torch.randn(8, 2, length)
            _ = loss(pred, truth)
            
            # Benchmark
            num_iterations = 50
            start_time = time.time()
            
            for _ in range(num_iterations):
                pred = torch.randn(8, 2, length)
                truth = torch.randn(8, 2, length)
                _ = loss(pred, truth)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / num_iterations
            
            results.append((length, avg_time))
            print(f"  Length {length:4d}: {avg_time*1000:6.2f}ms")
        
        # Verify results
        assert len(results) == len(lengths)
