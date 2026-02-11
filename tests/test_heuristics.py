"""
Unit tests for heuristics.
"""

import pytest
from logoslabs.heuristics import (
    compute_repetition_score,
    compute_rolling_var_score,
    compute_ttr_score,
    compute_heuristics_score,
    compute_all_heuristics,
)


class TestRepetitionScore:
    """Test repetition score computation."""
    
    def test_no_repetition(self):
        """Test tokens with no repetition."""
        tokens = ["a", "b", "c", "d", "e", "f"]
        score = compute_repetition_score(tokens, window_size=3)
        assert score == 0.0
    
    def test_full_repetition(self):
        """Test tokens with full repetition."""
        tokens = ["a", "b", "c"] * 3
        score = compute_repetition_score(tokens, window_size=3)
        assert score > 0.0
    
    def test_short_sequence(self):
        """Test sequence shorter than window."""
        tokens = ["a", "b"]
        score = compute_repetition_score(tokens, window_size=5)
        assert score == 0.0
    
    def test_empty_sequence(self):
        """Test empty sequence."""
        tokens = []
        score = compute_repetition_score(tokens, window_size=5)
        assert score == 0.0


class TestRollingVarScore:
    """Test rolling variance score computation."""
    
    def test_uniform_tokens(self):
        """Test tokens with uniform length."""
        tokens = ["aaaa"] * 50
        score = compute_rolling_var_score(tokens, window_size=32)
        assert score == 0.0
    
    def test_varied_tokens(self):
        """Test tokens with varied length."""
        tokens = ["a", "bb", "ccc", "dddd"] * 20
        score = compute_rolling_var_score(tokens, window_size=32)
        assert score > 0.0
    
    def test_short_sequence(self):
        """Test sequence shorter than window."""
        tokens = ["a", "bb", "ccc"]
        score = compute_rolling_var_score(tokens, window_size=50)
        assert score >= 0.0
    
    def test_empty_sequence(self):
        """Test empty sequence."""
        tokens = []
        score = compute_rolling_var_score(tokens, window_size=32)
        assert score == 0.0


class TestTTRScore:
    """Test type-token ratio score computation."""
    
    def test_high_diversity(self):
        """Test tokens with high diversity."""
        tokens = [str(i) for i in range(100)]
        score = compute_ttr_score(tokens, window_size=50)
        # High diversity = low TTR score (inverted)
        assert score < 0.5
    
    def test_low_diversity(self):
        """Test tokens with low diversity."""
        tokens = ["a"] * 100
        score = compute_ttr_score(tokens, window_size=50)
        # Low diversity = high TTR score (inverted)
        assert score > 0.5
    
    def test_short_sequence(self):
        """Test sequence shorter than window."""
        tokens = ["a", "b", "c"]
        score = compute_ttr_score(tokens, window_size=50)
        assert 0.0 <= score <= 1.0
    
    def test_empty_sequence(self):
        """Test empty sequence."""
        tokens = []
        score = compute_ttr_score(tokens, window_size=50)
        assert score == 0.0


class TestHeuristicsScore:
    """Test aggregate heuristics score computation."""
    
    def test_basic_computation(self):
        """Test basic weighted sum."""
        score = compute_heuristics_score(
            repetition_score=0.5,
            rolling_var_score=50.0,
            ttr_score=0.3,
        )
        assert 0.0 <= score <= 1.0
    
    def test_all_zeros(self):
        """Test with all zero inputs."""
        score = compute_heuristics_score(
            repetition_score=0.0,
            rolling_var_score=0.0,
            ttr_score=0.0,
        )
        # repetition=0, var normalized=0->inverted=1, ttr=0
        # score = 0.4*0 + 0.3*1 + 0.3*0 = 0.3
        assert score == 0.3


class TestComputeAllHeuristics:
    """Test compute_all_heuristics function."""
    
    def test_basic_tokens(self):
        """Test with basic token sequence."""
        tokens = ["hello", "world", "test", "example"]
        heuristics = compute_all_heuristics(tokens)
        
        assert "repetition_score" in heuristics
        assert "rolling_var_score" in heuristics
        assert "ttr_score" in heuristics
        assert "heuristics_score" in heuristics
        
        # All scores should be floats
        for key, value in heuristics.items():
            assert isinstance(value, float)
    
    def test_empty_tokens(self):
        """Test with empty token sequence."""
        tokens = []
        heuristics = compute_all_heuristics(tokens)
        
        assert "repetition_score" in heuristics
        assert "rolling_var_score" in heuristics
        assert "ttr_score" in heuristics
        assert "heuristics_score" in heuristics
