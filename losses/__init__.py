"""Loss functions for benchmarking."""

from .logosloss import LogosLoss
from .baselines import MSELoss, HuberLoss

__all__ = ["LogosLoss", "MSELoss", "HuberLoss"]
