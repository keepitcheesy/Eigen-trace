"""
Reproducibility utilities for deterministic benchmarking.

Provides functions to set seeds and ensure reproducible results.
"""

import torch
import numpy as np
import random
from typing import Dict, Any


def set_seed(seed: int, use_deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed value
    use_deterministic : bool, default=True
        If True, enable deterministic algorithms in PyTorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if use_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Some operations may not have deterministic implementations
        # Use torch.use_deterministic_algorithms(True) with caution


def get_reproducible_config() -> Dict[str, Any]:
    """
    Get current reproducibility configuration.
    
    Returns
    -------
    dict
        Dictionary with reproducibility settings
    """
    config = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    
    if torch.cuda.is_available():
        config["cuda_version"] = torch.version.cuda
        config["cudnn_version"] = torch.backends.cudnn.version()
    
    return config
