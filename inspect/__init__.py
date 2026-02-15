"""Inspection tools for analyzing model behavior."""

from .residual_spectrum import analyze_residual_spectrum
from .frequency_attribution import compute_frequency_attribution
from .gradient_spectrum import analyze_gradient_spectrum

__all__ = [
    "analyze_residual_spectrum",
    "compute_frequency_attribution",
    "analyze_gradient_spectrum",
]
