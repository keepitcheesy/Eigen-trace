"""
Token-level heuristics for text quality assessment.

Provides heuristic features:
- repetition_score: Local n-gram repetition detection
- rolling_var_score: Rolling window variance of token features
- ttr_score: Type-token ratio (vocabulary diversity)
- heuristics_score: Weighted aggregate of all heuristics
"""

from typing import List, Dict, Any
from collections import defaultdict
import math


def compute_repetition_score(tokens: List[str], window_size: int = 5) -> float:
    """
    Compute local repetition score using sliding n-gram windows.
    
    For each sliding window of the last N tokens, compute how often
    that n-gram appears earlier in the sequence. Higher scores indicate
    more repetition.
    
    Args:
        tokens: List of tokens
        window_size: Size of sliding window (default: 5)
        
    Returns:
        Normalized repetition score [0, 1], higher = more repetition
    """
    if len(tokens) < window_size:
        return 0.0
    
    ngram_counts = defaultdict(int)
    total_ngrams = 0
    repetitions = 0
    
    for i in range(len(tokens) - window_size + 1):
        ngram = tuple(tokens[i:i + window_size])
        
        # If we've seen this n-gram before, it's a repetition
        if ngram_counts[ngram] > 0:
            repetitions += 1
        
        ngram_counts[ngram] += 1
        total_ngrams += 1
    
    if total_ngrams == 0:
        return 0.0
    
    return repetitions / total_ngrams


def compute_rolling_var_score(tokens: List[str], window_size: int = 32) -> float:
    """
    Compute rolling window variance of token lengths.
    
    Computes variance of token length across fixed windows and aggregates
    using the mean. Lower variance may indicate repetitive patterns.
    Higher variance indicates more diverse token usage.
    
    Args:
        tokens: List of tokens
        window_size: Size of rolling window (default: 32)
        
    Returns:
        Mean variance score, higher = more diverse
    """
    if len(tokens) < window_size:
        # For short sequences, compute variance of entire sequence
        if len(tokens) < 2:
            return 0.0
        lengths = [len(t) for t in tokens]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        return variance
    
    variances = []
    for i in range(len(tokens) - window_size + 1):
        window_tokens = tokens[i:i + window_size]
        lengths = [len(t) for t in window_tokens]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        variances.append(variance)
    
    if not variances:
        return 0.0
    
    return sum(variances) / len(variances)


def compute_ttr_score(tokens: List[str], window_size: int = 50) -> float:
    """
    Compute type-token ratio (TTR) within windows.
    
    TTR = unique_tokens / total_tokens. Lower TTR indicates more repetition.
    We compute TTR for each window and aggregate. The score is inverted
    so that higher values indicate more risk (more repetition).
    
    Args:
        tokens: List of tokens
        window_size: Size of window for TTR computation (default: 50)
        
    Returns:
        Inverted TTR score [0, 1], higher = more repetition (lower diversity)
    """
    if len(tokens) == 0:
        return 0.0
    
    if len(tokens) < window_size:
        # For short sequences, compute TTR for entire sequence
        unique_tokens = len(set(tokens))
        ttr = unique_tokens / len(tokens)
        # Invert: lower TTR = higher score (more repetition)
        return 1.0 - ttr
    
    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window_tokens = tokens[i:i + window_size]
        unique_tokens = len(set(window_tokens))
        ttr = unique_tokens / len(window_tokens)
        ttrs.append(ttr)
    
    if not ttrs:
        return 0.0
    
    mean_ttr = sum(ttrs) / len(ttrs)
    # Invert: lower TTR = higher score (more repetition)
    return 1.0 - mean_ttr


def compute_heuristics_score(
    repetition_score: float,
    rolling_var_score: float,
    ttr_score: float,
    repetition_weight: float = 0.4,
    variance_weight: float = 0.3,
    ttr_weight: float = 0.3,
) -> float:
    """
    Compute aggregate heuristics score as weighted sum.
    
    Args:
        repetition_score: Local repetition score
        rolling_var_score: Rolling variance score
        ttr_score: Type-token ratio score (inverted)
        repetition_weight: Weight for repetition (default: 0.4)
        variance_weight: Weight for variance (default: 0.3)
        ttr_weight: Weight for TTR (default: 0.3)
        
    Returns:
        Weighted heuristics score
    """
    # Normalize variance score to [0, 1] range using a reasonable max
    # Token length variance typically ranges 0-100 for normal text
    normalized_var = min(rolling_var_score / 100.0, 1.0)
    # Invert variance: lower variance = higher risk
    inverted_var = 1.0 - normalized_var
    
    score = (
        repetition_weight * repetition_score +
        variance_weight * inverted_var +
        ttr_weight * ttr_score
    )
    
    return score


def compute_all_heuristics(tokens: List[str]) -> Dict[str, float]:
    """
    Compute all heuristic features for a token sequence.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Dictionary with all heuristic scores
    """
    repetition = compute_repetition_score(tokens)
    rolling_var = compute_rolling_var_score(tokens)
    ttr = compute_ttr_score(tokens)
    heuristics = compute_heuristics_score(repetition, rolling_var, ttr)
    
    return {
        "repetition_score": float(repetition),
        "rolling_var_score": float(rolling_var),
        "ttr_score": float(ttr),
        "heuristics_score": float(heuristics),
    }
