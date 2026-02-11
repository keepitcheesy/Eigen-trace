"""
AVP (Adversarial Validation Pipeline) module for filtering LLM outputs.

This module provides functionality for:
- Processing JSONL input/output
- Computing instability scores using LogosLossV4
- Threshold gating
- Batch processing with FFT spectral+phase parity
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np

from .logosloss import LogosLossV4


def encode_text_to_tensor(text: str, max_length: int = 512) -> torch.Tensor:
    """
    Encode text to a tensor representation for instability scoring.
    Uses character encoding as a simple, deterministic, offline method.
    
    Args:
        text: Input text string
        max_length: Maximum sequence length
        
    Returns:
        Tensor of shape (1, max_length) with normalized character codes
    """
    # Convert text to character codes
    char_codes = [ord(c) for c in text[:max_length]]
    
    # Pad or truncate to max_length
    if len(char_codes) < max_length:
        char_codes.extend([0] * (max_length - len(char_codes)))
    
    # Normalize to [0, 1] range (assuming ASCII/Unicode range)
    tensor = torch.tensor(char_codes, dtype=torch.float32) / 1114111.0  # Max Unicode code point
    
    return tensor.unsqueeze(0)  # (1, max_length)


def compute_instability_score(
    pred_text: str,
    truth_text: str,
    loss_fn: LogosLossV4,
    max_length: int = 512,
) -> float:
    """
    Compute instability score between predicted and truth text using LogosLossV4.
    
    Args:
        pred_text: Predicted/generated text
        truth_text: Ground truth/reference text
        loss_fn: LogosLossV4 instance
        max_length: Maximum sequence length
        
    Returns:
        Instability score (float)
    """
    # Encode texts to tensors
    pred_tensor = encode_text_to_tensor(pred_text, max_length)
    truth_tensor = encode_text_to_tensor(truth_text, max_length)
    
    # Add channel dimension: (1, 1, max_length)
    pred_tensor = pred_tensor.unsqueeze(1)
    truth_tensor = truth_tensor.unsqueeze(1)
    
    # Compute loss
    with torch.no_grad():
        score = loss_fn(pred_tensor, truth_tensor).item()
    
    return score


class AVPProcessor:
    """
    Adversarial Validation Pipeline processor for filtering LLM outputs.
    
    Supports:
    - Structural analysis (deterministic, offline, default)
    - Optional belief streams (future extension)
    - Batch processing with FFT spectral+phase parity
    - Threshold gating
    """
    
    def __init__(
        self,
        threshold: float = 1.0,
        grace_coeff: float = 0.5,
        phase_weight: float = 0.1,
        max_length: int = 512,
        structural_only: bool = True,
        deterministic: bool = True,
    ):
        """
        Initialize AVP processor.
        
        Args:
            threshold: Instability score threshold for gating (higher = more permissive)
            grace_coeff: LogosLoss spectral weight coefficient
            phase_weight: LogosLoss phase weight coefficient
            max_length: Maximum sequence length for encoding
            structural_only: Use only structural analysis (default True)
            deterministic: Ensure deterministic behavior (default True)
        """
        self.threshold = threshold
        self.max_length = max_length
        self.structural_only = structural_only
        
        # Set random seed for determinism
        if deterministic:
            torch.manual_seed(42)
            np.random.seed(42)
        
        # Initialize LogosLoss
        self.loss_fn = LogosLossV4(
            grace_coeff=grace_coeff,
            phase_weight=phase_weight,
            reduction="mean",
        )
        
    def process_item(self, item: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Process a single JSONL item.
        
        Args:
            item: Dictionary with 'prediction' and 'truth' keys
            
        Returns:
            Tuple of (processed_item, pass_threshold)
        """
        pred_text = item.get("prediction", "")
        truth_text = item.get("truth", "")
        
        # Compute instability score
        score = compute_instability_score(
            pred_text,
            truth_text,
            self.loss_fn,
            self.max_length,
        )
        
        # Check threshold
        passed = score <= self.threshold
        
        # Add score to item
        result = item.copy()
        result["instability_score"] = float(score)
        result["passed_threshold"] = passed
        
        return result, passed
    
    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of JSONL items.
        
        Args:
            items: List of dictionaries with 'prediction' and 'truth' keys
            
        Returns:
            List of processed items with scores and pass/fail flags
        """
        results = []
        for item in items:
            result, _ = self.process_item(item)
            results.append(result)
        
        return results
    
    def get_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for processed results.
        
        Args:
            results: List of processed items
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                "total_items": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "mean_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
            }
        
        scores = [r["instability_score"] for r in results]
        passed = sum(1 for r in results if r["passed_threshold"])
        
        return {
            "total_items": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results),
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "threshold": self.threshold,
        }


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict[str, Any]], filepath: str) -> None:
    """Save items to JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
