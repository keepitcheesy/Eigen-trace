#!/usr/bin/env python3
"""
Demonstration script for LogosLabs package.

This script demonstrates the key features of the LogosLabs package:
1. LogosLossV4 core functionality
2. Text encoding and instability scoring
3. Batch processing with threshold gating
4. JSONL I/O
5. Summary statistics
"""

import tempfile
import os
import torch
from logoslabs import LogosLossV4
from logoslabs.avp import AVPProcessor, load_jsonl, save_jsonl


def demo_logosloss():
    """Demonstrate LogosLossV4 core functionality."""
    print("=" * 60)
    print("1. LogosLossV4 Core Functionality")
    print("=" * 60)
    
    # Initialize loss function
    loss_fn = LogosLossV4(
        grace_coeff=0.5,
        phase_weight=0.1,
        reduction="mean"
    )
    
    print(f"Parameters:")
    print(f"  grace_coeff: {loss_fn.grace_coeff}")
    print(f"  phase_weight: {loss_fn.phase_weight}")
    print(f"  eps: {loss_fn.eps}")
    
    # Test with tensors
    torch.manual_seed(42)
    pred = torch.randn(4, 2, 128)
    truth = torch.randn(4, 2, 128)
    
    loss = loss_fn(pred, truth)
    print(f"\nLoss value: {loss.item():.6f}")
    
    # Test with identical inputs
    loss_identical = loss_fn(pred, pred)
    print(f"Loss for identical inputs: {loss_identical.item():.6f} (should be ~0)")
    
    print("\n✓ LogosLossV4 demonstration complete\n")


def demo_avp_processor():
    """Demonstrate AVP processor functionality."""
    print("=" * 60)
    print("2. AVP Processor Functionality")
    print("=" * 60)
    
    # Initialize processor
    processor = AVPProcessor(
        threshold=1.0,
        grace_coeff=0.5,
        phase_weight=0.1,
        deterministic=True
    )
    
    print(f"Processor settings:")
    print(f"  threshold: {processor.threshold}")
    print(f"  max_length: {processor.max_length}")
    print(f"  structural_only: {processor.structural_only}")
    
    # Create test items
    items = [
        {"prediction": "The cat sat on the mat", "truth": "The cat sat on the mat"},
        {"prediction": "Hello world", "truth": "Hello world"},
        {"prediction": "Machine learning", "truth": "Deep learning"},
        {"prediction": "Python programming", "truth": "JavaScript development"},
    ]
    
    print(f"\nProcessing {len(items)} items...")
    
    # Process batch
    results = processor.process_batch(items)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"  Item {i+1}:")
        print(f"    Prediction: {result['prediction']}")
        print(f"    Truth: {result['truth']}")
        print(f"    Score: {result['instability_score']:.6f}")
        print(f"    Passed: {result['passed_threshold']}")
    
    # Get summary
    summary = processor.get_summary(results)
    print("\nSummary Statistics:")
    print(f"  Total items: {summary['total_items']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Pass rate: {summary['pass_rate']:.1%}")
    print(f"  Mean score: {summary['mean_score']:.6f}")
    print(f"  Min score: {summary['min_score']:.6f}")
    print(f"  Max score: {summary['max_score']:.6f}")
    
    print("\n✓ AVP processor demonstration complete\n")
    
    return results


def demo_jsonl_io(results):
    """Demonstrate JSONL I/O."""
    print("=" * 60)
    print("3. JSONL Input/Output")
    print("=" * 60)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        filepath = f.name
    
    try:
        # Save results
        print(f"Saving results to: {filepath}")
        save_jsonl(results, filepath)
        
        # Load back
        print(f"Loading results from: {filepath}")
        loaded = load_jsonl(filepath)
        
        print(f"\nLoaded {len(loaded)} items")
        print("First item:")
        print(f"  {loaded[0]}")
        
        # Verify
        assert len(loaded) == len(results)
        print("\n✓ JSONL I/O demonstration complete\n")
        
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.unlink(filepath)


def demo_threshold_gating():
    """Demonstrate threshold gating."""
    print("=" * 60)
    print("4. Threshold Gating")
    print("=" * 60)
    
    # Create test items
    items = [
        {"prediction": "identical", "truth": "identical"},
        {"prediction": "similar text", "truth": "similar text"},
        {"prediction": "somewhat different", "truth": "quite different"},
        {"prediction": "totally unrelated", "truth": "completely other"},
    ]
    
    # Test with different thresholds
    thresholds = [0.001, 0.01, 0.1, 1.0]
    
    print("Testing with different thresholds:")
    for threshold in thresholds:
        processor = AVPProcessor(threshold=threshold)
        results = processor.process_batch(items)
        summary = processor.get_summary(results)
        
        print(f"\n  Threshold {threshold:.3f}:")
        print(f"    Pass rate: {summary['pass_rate']:.1%}")
        print(f"    Passed: {summary['passed']}/{summary['total_items']}")
    
    print("\n✓ Threshold gating demonstration complete\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("LogosLabs Package Demonstration")
    print("=" * 60 + "\n")
    
    # Run demos
    demo_logosloss()
    results = demo_avp_processor()
    demo_jsonl_io(results)
    demo_threshold_gating()
    
    print("=" * 60)
    print("All demonstrations complete!")
    print("=" * 60 + "\n")
    
    print("Next steps:")
    print("  1. Try the CLI: logoslabs input.jsonl output.jsonl --summary")
    print("  2. Run tests: pytest tests/ -v")
    print("  3. Run benchmarks: RUN_BENCHMARKS=1 pytest tests/test_benchmark.py -v")
    print("  4. Read the documentation: README.md")


if __name__ == "__main__":
    main()
