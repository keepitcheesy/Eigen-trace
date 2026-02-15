"""
Main benchmark suite runner.

Executes the complete benchmark suite from YAML configuration.
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, List
import csv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from losses import LogosLoss, MSELoss, HuberLoss
from tasks import ImageDenoiseTask, TimeSeriesForecastTask
from models import UNet, MLP, LSTMModel
from bench.repro import set_seed, get_reproducible_config
from bench.measure import measure_time, measure_flops, estimate_energy


class BenchmarkRunner:
    """Main benchmark runner that executes the full suite."""
    
    def __init__(self, config_path: str):
        """
        Initialize benchmark runner.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.results_dir / self.config['output']['checkpoints_dir']
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.results_dir / self.config['output']['metrics_file']
        self.summary_file = self.results_dir / self.config['output']['summary_file']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def get_loss_function(self, loss_config: Dict[str, Any]) -> nn.Module:
        """Create loss function from config."""
        loss_name = loss_config['name']
        params = loss_config['params']
        
        if loss_name == "logosloss":
            return LogosLoss(**params)
        elif loss_name == "mse":
            return MSELoss(**params)
        elif loss_name == "huber":
            return HuberLoss(**params)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def get_task(self, task_config: Dict[str, Any], seed: int):
        """Create task from config."""
        task_name = task_config['name']
        params = task_config['params']
        params['seed'] = seed
        
        if task_name == "image_denoise":
            return ImageDenoiseTask(**params)
        elif task_name == "time_series":
            return TimeSeriesForecastTask(**params)
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def get_model(self, task_config: Dict[str, Any]) -> nn.Module:
        """Create model from config."""
        model_name = task_config['model']
        params = task_config.get('model_params', {})
        
        if model_name == "unet":
            return UNet(**params)
        elif model_name == "mlp":
            return MLP(**params)
        elif model_name == "lstm":
            return LSTMModel(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_model(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_data: tuple,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Train model and return metrics.
        
        Parameters
        ----------
        model : nn.Module
            Model to train
        loss_fn : nn.Module
            Loss function
        train_data : tuple
            (inputs, targets) tensors
        config : dict
            Training configuration
            
        Returns
        -------
        dict
            Training metrics and timing information
        """
        model = model.to(self.device)
        inputs, targets = train_data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate']
        )
        
        epochs = config['epochs']
        batch_size = config['batch_size']
        
        history = []
        
        with measure_time() as timer:
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Simple batching
                for i in range(0, len(inputs), batch_size):
                    batch_inputs = inputs[i:i+batch_size]
                    batch_targets = targets[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = loss_fn(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                history.append(avg_loss)
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        training_time = timer['elapsed']
        
        return {
            "training_time": training_time,
            "final_loss": history[-1],
            "loss_history": history,
        }
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_data: tuple,
        task: Any,
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        model.eval()
        inputs, targets = test_data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            outputs = model(inputs)
        
        # Compute task-specific metrics
        metrics = task.compute_metrics(outputs, targets)
        
        return metrics
    
    def apply_quantization(self, model: nn.Module, bits: int) -> nn.Module:
        """
        Simulate quantization by quantizing weights.
        
        NOTE: This is a simulation of uniform quantization and may not accurately
        represent actual hardware quantization behavior, especially for specialized
        accelerators or optimized 8-bit inference engines. For production use cases,
        consider using PyTorch's quantization APIs or framework-specific tools.
        """
        import copy
        quantized_model = copy.deepcopy(model)
        
        # Simple quantization simulation: quantize to N bits
        for param in quantized_model.parameters():
            if param.requires_grad:
                # Quantize to N bits
                min_val = param.data.min()
                max_val = param.data.max()
                scale = (max_val - min_val) / (2 ** bits - 1)
                quantized = torch.round((param.data - min_val) / scale)
                param.data = min_val + quantized * scale
        
        return quantized_model
    
    def apply_pruning(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply magnitude-based pruning."""
        import copy
        pruned_model = copy.deepcopy(model)
        
        for name, param in pruned_model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Magnitude pruning
                threshold = torch.quantile(torch.abs(param.data), sparsity)
                mask = torch.abs(param.data) > threshold
                param.data *= mask.float()
        
        return pruned_model
    
    def run_compression_tests(
        self,
        model: nn.Module,
        test_data: tuple,
        task: Any,
        baseline_metrics: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Run compression tests (quantization and pruning)."""
        compression_results = []
        
        # Quantization tests
        if self.config['compression']['quantization']['enabled']:
            for bits in self.config['compression']['quantization']['bits']:
                print(f"    Testing {bits}-bit quantization...")
                quantized = self.apply_quantization(model, bits)
                metrics = self.evaluate_model(quantized, test_data, task)
                
                compression_results.append({
                    "type": "quantization",
                    "bits": bits,
                    "metrics": metrics,
                    "drops": {k: baseline_metrics[k] - metrics[k] 
                             for k in baseline_metrics.keys()},
                })
        
        # Pruning tests
        if self.config['compression']['pruning']['enabled']:
            for sparsity in self.config['compression']['pruning']['sparsity']:
                print(f"    Testing {sparsity*100:.0f}% pruning...")
                pruned = self.apply_pruning(model, sparsity)
                metrics = self.evaluate_model(pruned, test_data, task)
                
                compression_results.append({
                    "type": "pruning",
                    "sparsity": sparsity,
                    "metrics": metrics,
                    "drops": {k: baseline_metrics[k] - metrics[k]
                             for k in baseline_metrics.keys()},
                })
        
        return compression_results
    
    def run_single_experiment(
        self,
        seed: int,
        loss_config: Dict[str, Any],
        task_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        print(f"\nRunning: seed={seed}, loss={loss_config['name']}, task={task_config['name']}")
        
        # Set seed
        set_seed(seed)
        
        # Create components
        loss_fn = self.get_loss_function(loss_config)
        task = self.get_task(task_config, seed)
        model = self.get_model(task_config)
        
        # Generate data
        train_data = task.generate_data()
        test_data = task.generate_data()  # New data for testing
        
        # Measure model size
        flops_info = measure_flops(model, train_data[0].shape, str(self.device))
        
        # Train model
        training_metrics = self.train_model(
            model, loss_fn, train_data, task_config['training']
        )
        
        # Evaluate
        eval_metrics = self.evaluate_model(model, test_data, task)
        
        # Energy estimate
        energy_info = estimate_energy(
            training_metrics['training_time'],
            flops_info['total_params'],
            str(self.device),
        )
        
        # Compression tests
        compression_results = self.run_compression_tests(
            model, test_data, task, eval_metrics
        )
        
        # Save checkpoint
        checkpoint_name = f"{task_config['name']}_{loss_config['name']}_seed{seed}.pt"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'seed': seed,
                'loss': loss_config,
                'task': task_config,
            },
        }, checkpoint_path)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "loss": loss_config['name'],
            "task": task_config['name'],
            "model": task_config['model'],
            "training_time": training_metrics['training_time'],
            "final_loss": training_metrics['final_loss'],
            "metrics": eval_metrics,
            "flops": flops_info,
            "energy": energy_info,
            "compression": compression_results,
            "checkpoint": str(checkpoint_path),
        }
        
        return result
    
    def save_metrics(self, result: Dict[str, Any]):
        """Save metrics to JSONL file."""
        with open(self.metrics_file, 'a') as f:
            json.dump(result, f)
            f.write('\n')
    
    def run_suite(self):
        """Run the complete benchmark suite."""
        print("=" * 60)
        print("Starting Benchmark Suite")
        print("=" * 60)
        
        # Save reproducibility info
        repro_config = get_reproducible_config()
        repro_file = self.results_dir / "reproducibility.json"
        with open(repro_file, 'w') as f:
            json.dump(repro_config, f, indent=2)
        
        # Clear previous metrics file
        if self.metrics_file.exists():
            self.metrics_file.unlink()
        
        all_results = []
        
        # Run all combinations
        for seed in self.config['seeds']:
            for loss_config in self.config['losses']:
                for task_config in self.config['tasks']:
                    result = self.run_single_experiment(seed, loss_config, task_config)
                    self.save_metrics(result)
                    all_results.append(result)
        
        # Generate summary CSV
        self.generate_summary(all_results)
        
        print("\n" + "=" * 60)
        print("Benchmark Suite Complete!")
        print(f"Results saved to: {self.results_dir}")
        print(f"Metrics: {self.metrics_file}")
        print(f"Summary: {self.summary_file}")
        print("=" * 60)
    
    def generate_summary(self, results: List[Dict[str, Any]]):
        """Generate summary CSV from results."""
        if not results:
            return
        
        # Extract flat data for CSV
        rows = []
        for r in results:
            row = {
                "seed": r['seed'],
                "loss": r['loss'],
                "task": r['task'],
                "model": r['model'],
                "training_time": r['training_time'],
                "final_loss": r['final_loss'],
                "total_params": r['flops']['total_params'],
                "estimated_flops": r['flops']['estimated_flops'],
                "estimated_joules": r['energy']['estimated_joules'],
            }
            # Add task-specific metrics
            for key, val in r['metrics'].items():
                row[f"metric_{key}"] = val
            rows.append(row)
        
        # Write CSV
        if rows:
            with open(self.summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument(
        "--config",
        type=str,
        default="bench/configs/suite.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.config)
    runner.run_suite()


if __name__ == "__main__":
    main()
