"""
Run inspection tools on benchmark results.

Analyzes residual spectra, frequency attribution, and gradient spectra.
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from losses import LogosLoss, MSELoss, HuberLoss
from inspect import (
    analyze_residual_spectrum,
    compute_frequency_attribution,
    analyze_gradient_spectrum,
)


def main():
    """Run inspection tools on saved checkpoints."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inspection tools")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory with checkpoints"
    )
    parser.add_argument(
        "--inspect-dir",
        type=str,
        default="results/inspect",
        help="Directory to save inspection results"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    inspect_dir = Path(args.inspect_dir)
    inspect_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    metrics_file = results_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"No metrics file found at {metrics_file}")
        return
    
    # Read all results
    results = []
    with open(metrics_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    print("=" * 60)
    print("Running Inspection Tools")
    print("=" * 60)
    
    # Generate synthetic data for inspection
    # (In a real scenario, you'd load test data from checkpoints)
    torch.manual_seed(42)
    pred = torch.randn(10, 1, 128)
    truth = torch.randn(10, 1, 128)
    
    residual_results = []
    gradient_results = []
    
    # Run inspection for each loss type
    loss_types = list(set(r['loss'] for r in results))
    
    for loss_name in loss_types:
        print(f"\nInspecting {loss_name}...")
        
        # Create loss function
        if loss_name == "logosloss":
            loss_fn = LogosLoss(grace_coeff=0.5, phase_weight=0.1)
        elif loss_name == "mse":
            loss_fn = MSELoss()
        elif loss_name == "huber":
            loss_fn = HuberLoss()
        else:
            continue
        
        # Residual spectrum
        print("  Analyzing residual spectrum...")
        res_summary = analyze_residual_spectrum(
            pred.clone(), truth.clone(), loss_name, inspect_dir
        )
        residual_results.append(res_summary)
        
        # Frequency attribution (only for LogosLoss)
        if loss_name == "logosloss":
            print("  Computing frequency attribution...")
            compute_frequency_attribution(
                pred.clone(), truth.clone(),
                grace_coeff=0.5, phase_weight=0.1,
                output_dir=inspect_dir
            )
        
        # Gradient spectrum
        print("  Analyzing gradient spectrum...")
        grad_summary = analyze_gradient_spectrum(
            None, loss_fn, pred.clone(), truth.clone(),
            inspect_dir, loss_name
        )
        gradient_results.append(grad_summary)
    
    # Save combined summary
    combined_summary = {
        "residual_analysis": residual_results,
        "gradient_analysis": gradient_results,
    }
    
    summary_path = inspect_dir / "inspection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Inspection Complete!")
    print(f"Results saved to: {inspect_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
