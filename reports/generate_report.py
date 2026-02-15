"""
Auto-generate benchmark report from results.

Reads metrics and inspection results, generates a formatted report.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all results from metrics file."""
    metrics_file = results_dir / "metrics.jsonl"
    results = []
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    
    return results


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean and std for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def generate_report(results_dir: Path, output_path: Path) -> None:
    """Generate markdown report from results."""
    results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Extract metadata
    seeds = sorted(list(set(r['seed'] for r in results)))
    losses = sorted(list(set(r['loss'] for r in results)))
    tasks = sorted(list(set(r['task'] for r in results)))
    
    # Load reproducibility info
    repro_file = results_dir / "reproducibility.json"
    repro_info = {}
    if repro_file.exists():
        with open(repro_file, 'r') as f:
            repro_info = json.load(f)
    
    # Start building report
    lines = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"This report presents benchmark results comparing {len(losses)} loss functions ")
    lines.append(f"({', '.join(losses)}) across {len(tasks)} tasks ({', '.join(tasks)}) ")
    lines.append(f"with {len(seeds)} random seeds for reproducibility.")
    lines.append("")
    
    # Setup
    lines.append("## Setup")
    lines.append("")
    lines.append("### Configuration")
    lines.append(f"- **Seeds tested**: {seeds}")
    lines.append(f"- **Loss functions**: {losses}")
    lines.append(f"- **Tasks**: {tasks}")
    lines.append(f"- **Models**: {sorted(list(set(r['model'] for r in results)))}")
    lines.append("")
    
    lines.append("### Reproducibility")
    if repro_info:
        lines.append(f"- PyTorch version: {repro_info.get('torch_version', 'N/A')}")
        lines.append(f"- NumPy version: {repro_info.get('numpy_version', 'N/A')}")
        lines.append(f"- CUDA available: {repro_info.get('cuda_available', False)}")
        lines.append(f"- Random seeds: {seeds}")
    lines.append("")
    
    # Results by task
    lines.append("## Results")
    lines.append("")
    
    for task in tasks:
        lines.append(f"### Task: {task}")
        lines.append("")
        
        task_results = [r for r in results if r['task'] == task]
        
        # Group by loss
        for loss in losses:
            loss_results = [r for r in task_results if r['loss'] == loss]
            
            if not loss_results:
                continue
            
            lines.append(f"#### {loss}")
            lines.append("")
            
            # Extract metrics
            metric_keys = list(loss_results[0]['metrics'].keys())
            
            for metric in metric_keys:
                values = [r['metrics'][metric] for r in loss_results]
                stats = compute_statistics(values)
                lines.append(f"- **{metric}**: {stats['mean']:.4f} ± {stats['std']:.4f} "
                           f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
            
            # Training time
            times = [r['training_time'] for r in loss_results]
            time_stats = compute_statistics(times)
            lines.append(f"- **Training time**: {time_stats['mean']:.2f}s ± {time_stats['std']:.2f}s")
            lines.append("")
        
        lines.append("")
    
    # Compression results
    lines.append("## Compression Robustness")
    lines.append("")
    
    # Check if any results have compression data
    has_compression = any('compression' in r and r['compression'] for r in results)
    
    if has_compression:
        lines.append("### Quantization Tests")
        lines.append("")
        
        for task in tasks:
            lines.append(f"#### {task}")
            lines.append("")
            
            for loss in losses:
                loss_task_results = [r for r in results 
                                    if r['task'] == task and r['loss'] == loss]
                
                if not loss_task_results:
                    continue
                
                # Extract quantization results
                quant_results = []
                for r in loss_task_results:
                    if 'compression' in r:
                        quant_results.extend([c for c in r['compression'] 
                                            if c['type'] == 'quantization'])
                
                if quant_results:
                    lines.append(f"**{loss}**:")
                    for qr in quant_results:
                        bits = qr.get('bits', 'N/A')
                        drops = qr.get('drops', {})
                        lines.append(f"- {bits}-bit quantization: drops = {drops}")
                    lines.append("")
        
        lines.append("### Pruning Tests")
        lines.append("")
        
        for task in tasks:
            lines.append(f"#### {task}")
            lines.append("")
            
            for loss in losses:
                loss_task_results = [r for r in results 
                                    if r['task'] == task and r['loss'] == loss]
                
                if not loss_task_results:
                    continue
                
                # Extract pruning results
                prune_results = []
                for r in loss_task_results:
                    if 'compression' in r:
                        prune_results.extend([c for c in r['compression'] 
                                            if c['type'] == 'pruning'])
                
                if prune_results:
                    lines.append(f"**{loss}**:")
                    for pr in prune_results:
                        sparsity = pr.get('sparsity', 'N/A')
                        drops = pr.get('drops', {})
                        lines.append(f"- {sparsity*100:.0f}% pruning: drops = {drops}")
                    lines.append("")
    else:
        lines.append("No compression tests were run.")
        lines.append("")
    
    # Inspection findings
    inspect_dir = results_dir / "inspection"
    lines.append("## Inspection Findings")
    lines.append("")
    
    if inspect_dir.exists():
        summary_file = inspect_dir / "inspection_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                inspection = json.load(f)
            
            lines.append("### Residual Spectrum Analysis")
            lines.append("")
            if 'residual_analysis' in inspection:
                for res in inspection['residual_analysis']:
                    loss_name = res.get('loss_name', 'unknown')
                    ratio = res.get('residual_to_truth_ratio', 0)
                    lines.append(f"- **{loss_name}**: Residual/Truth ratio = {ratio:.4f}")
                lines.append("")
            
            lines.append("### Gradient Spectrum Analysis")
            lines.append("")
            if 'gradient_analysis' in inspection:
                for grad in inspection['gradient_analysis']:
                    loss_name = grad.get('loss_name', 'unknown')
                    hf_ratio = grad.get('high_freq_ratio', 0)
                    entropy = grad.get('gradient_entropy', 0)
                    lines.append(f"- **{loss_name}**: HF ratio = {hf_ratio:.4f}, entropy = {entropy:.4f}")
                lines.append("")
    else:
        lines.append("No inspection results found. Run `make inspect` to generate them.")
        lines.append("")
    
    # Conclusions
    lines.append("## Conclusions")
    lines.append("")
    lines.append("### Performance Summary")
    lines.append("")
    lines.append("- Results are reported across all seeds (no cherry-picking)")
    lines.append("- See full metrics in `results/metrics.jsonl` and `results/summary.csv`")
    lines.append("")
    
    lines.append("### Statistical Significance")
    lines.append("")
    lines.append(f"All results are averaged over {len(seeds)} independent runs with different ")
    lines.append("random seeds. Standard deviations are reported to show variability.")
    lines.append("")
    
    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("### Computational Constraints")
    lines.append("- Small models and datasets for fast iteration")
    lines.append("- Limited number of epochs")
    lines.append("- Synthetic data for controlled experiments")
    lines.append("")
    
    lines.append("### Scope")
    lines.append("- Results may not generalize to all domains")
    lines.append("- Further testing on real-world data recommended")
    lines.append("")
    
    # Artifacts
    lines.append("## Artifacts")
    lines.append("")
    lines.append("### Data Files")
    lines.append("- Metrics: `results/metrics.jsonl`")
    lines.append("- Summary: `results/summary.csv`")
    lines.append("- Checkpoints: `results/checkpoints/`")
    lines.append("- Inspection: `results/inspection/`")
    lines.append("")
    
    lines.append("### Reproducibility")
    lines.append("- Configuration: `bench/configs/suite.yaml`")
    lines.append("- Reproducibility info: `results/reproducibility.json`")
    lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report generated: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory with benchmark results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/report_latest.md",
        help="Output path for report"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_report(results_dir, output_path)


if __name__ == "__main__":
    main()
