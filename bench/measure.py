"""
Performance measurement utilities.

Provides timing, FLOPs estimation, and energy proxy calculations.
"""

import time
import torch
from typing import Callable, Dict, Any, Tuple
from contextlib import contextmanager


@contextmanager
def measure_time():
    """
    Context manager to measure execution time.
    
    Yields
    ------
    dict
        Dictionary with 'elapsed' key containing elapsed time in seconds
        
    Example
    -------
    >>> with measure_time() as timer:
    ...     # some code
    ...     pass
    >>> print(f"Time: {timer['elapsed']:.4f}s")
    """
    times = {}
    start = time.perf_counter()
    yield times
    times['elapsed'] = time.perf_counter() - start


def measure_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Estimate FLOPs for a model.
    
    This is a simple estimation based on layer parameters.
    For more accurate measurements, use specialized tools like fvcore or ptflops.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to measure
    input_shape : tuple
        Shape of input tensor (including batch dimension)
    device : str, default="cpu"
        Device to run on
        
    Returns
    -------
    dict
        Dictionary with 'params', 'estimated_flops', and 'flops_per_param'
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Simple FLOPs estimation
    # This is a rough approximation: 2 * params (one multiply + one add per param)
    # Real FLOPs depend on architecture and operations
    estimated_flops = 2 * trainable_params
    
    # Try to get more accurate estimate for linear layers
    total_flops = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total_flops += 2 * module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv2d):
            # For conv: kernel_h * kernel_w * in_channels * out_channels * out_h * out_w
            # This is simplified - actual calculation needs input size
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            total_flops += 2 * kernel_ops * module.in_channels * module.out_channels
    
    if total_flops > 0:
        estimated_flops = total_flops
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "estimated_flops": estimated_flops,
        "flops_per_param": estimated_flops / max(trainable_params, 1),
    }


def estimate_energy(
    elapsed_time: float,
    num_params: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Estimate energy consumption (rough proxy).
    
    This is a very rough estimation based on typical power consumption
    and execution time. Real energy measurement requires specialized hardware.
    
    Parameters
    ----------
    elapsed_time : float
        Execution time in seconds
    num_params : int
        Number of model parameters
    device : str, default="cpu"
        Device type ("cpu" or "cuda")
        
    Returns
    -------
    dict
        Dictionary with 'estimated_watts' and 'estimated_joules'
    """
    # Rough power estimates (in watts)
    if device.startswith("cuda"):
        # GPU: varies widely, use conservative estimate
        base_power = 50.0  # Base GPU power
        power_per_1m_params = 5.0  # Additional power per 1M params
    else:
        # CPU
        base_power = 10.0  # Base CPU power
        power_per_1m_params = 2.0  # Additional power per 1M params
    
    estimated_watts = base_power + (num_params / 1e6) * power_per_1m_params
    estimated_joules = estimated_watts * elapsed_time
    
    return {
        "estimated_watts": estimated_watts,
        "estimated_joules": estimated_joules,
        "estimated_watt_hours": estimated_joules / 3600.0,
    }


def profile_forward_pass(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Profile model forward pass with warmup.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to profile
    input_tensor : torch.Tensor
        Input tensor
    num_warmup : int, default=10
        Number of warmup iterations
    num_iterations : int, default=100
        Number of timed iterations
        
    Returns
    -------
    dict
        Dictionary with timing statistics
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Time iterations
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
    
    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "std_time": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
    }
