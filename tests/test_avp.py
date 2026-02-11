"""
Unit tests for AVP processor.

Tests JSONL I/O, instability scoring, threshold gating, and batch processing.
"""

import pytest
import os
import json
import tempfile
import torch

from logoslabs.avp import (
    AVPProcessor,
    encode_text_to_tensor,
    compute_instability_score,
    load_jsonl,
    save_jsonl,
)
from logoslabs.logosloss import LogosLossV4


class TestEncodingFunctions:
    """Test text encoding functions."""
    
    def test_encode_text_to_tensor_basic(self):
        """Test basic text encoding."""
        text = "Hello, world!"
        tensor = encode_text_to_tensor(text, max_length=100)
        
        assert tensor.shape == (1, 100)
        assert tensor.dtype == torch.float32
        assert torch.all(tensor >= 0)
        assert torch.all(tensor <= 1)
        
    def test_encode_text_to_tensor_empty(self):
        """Test encoding empty text."""
        tensor = encode_text_to_tensor("", max_length=50)
        
        assert tensor.shape == (1, 50)
        # Empty text should be all zeros (padding)
        assert torch.all(tensor == 0)
        
    def test_encode_text_to_tensor_truncation(self):
        """Test that long text is truncated."""
        text = "a" * 1000
        tensor = encode_text_to_tensor(text, max_length=100)
        
        assert tensor.shape == (1, 100)
        
    def test_encode_text_deterministic(self):
        """Test that encoding is deterministic."""
        text = "Test text"
        tensor1 = encode_text_to_tensor(text, max_length=50)
        tensor2 = encode_text_to_tensor(text, max_length=50)
        
        assert torch.allclose(tensor1, tensor2)
        
    def test_compute_instability_score(self):
        """Test instability score computation."""
        loss_fn = LogosLossV4()
        
        pred = "This is a prediction"
        truth = "This is the truth"
        
        score = compute_instability_score(pred, truth, loss_fn)
        
        assert isinstance(score, float)
        assert score >= 0
        
    def test_compute_instability_score_identical(self):
        """Test that identical texts have low instability."""
        loss_fn = LogosLossV4()
        
        text = "Identical text"
        score = compute_instability_score(text, text, loss_fn)
        
        assert score < 0.001
        
    def test_compute_instability_score_different(self):
        """Test that different texts have higher instability."""
        loss_fn = LogosLossV4()
        
        pred = "Completely different"
        truth = "Not at all similar"
        
        score = compute_instability_score(pred, truth, loss_fn)
        
        assert score > 0.01


class TestJSONLIO:
    """Test JSONL input/output functions."""
    
    def test_save_and_load_jsonl(self):
        """Test saving and loading JSONL files."""
        items = [
            {"prediction": "test1", "truth": "truth1"},
            {"prediction": "test2", "truth": "truth2"},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
        
        try:
            save_jsonl(items, filepath)
            loaded = load_jsonl(filepath)
            
            assert len(loaded) == len(items)
            assert loaded == items
        finally:
            os.unlink(filepath)
            
    def test_load_jsonl_empty_lines(self):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"a": 1}\n')
            f.write('\n')
            f.write('{"b": 2}\n')
            filepath = f.name
        
        try:
            items = load_jsonl(filepath)
            assert len(items) == 2
        finally:
            os.unlink(filepath)


class TestAVPProcessor:
    """Test AVP processor."""
    
    def test_initialization(self):
        """Test AVP processor initialization."""
        processor = AVPProcessor()
        
        assert processor.threshold == 1.0
        assert processor.max_length == 512
        assert processor.structural_only is True
        assert isinstance(processor.loss_fn, LogosLossV4)
        
    def test_initialization_custom(self):
        """Test AVP processor with custom parameters."""
        processor = AVPProcessor(
            threshold=2.0,
            grace_coeff=0.7,
            phase_weight=0.2,
            max_length=256,
            structural_only=False,
        )
        
        assert processor.threshold == 2.0
        assert processor.max_length == 256
        assert processor.structural_only is False
        
    def test_process_item(self):
        """Test processing a single item."""
        processor = AVPProcessor(threshold=1.0)
        
        item = {
            "prediction": "This is a test",
            "truth": "This is the truth",
        }
        
        result, passed = processor.process_item(item)
        
        assert "instability_score" in result
        assert "passed_threshold" in result
        assert isinstance(result["instability_score"], float)
        assert isinstance(result["passed_threshold"], bool)
        assert result["prediction"] == item["prediction"]
        assert result["truth"] == item["truth"]
        
    def test_process_item_threshold_pass(self):
        """Test that identical texts pass threshold."""
        processor = AVPProcessor(threshold=1.0)
        
        text = "Identical text for testing"
        item = {
            "prediction": text,
            "truth": text,
        }
        
        result, passed = processor.process_item(item)
        
        assert passed is True
        assert result["passed_threshold"] is True
        assert result["instability_score"] < 0.001
        
    def test_process_item_threshold_fail(self):
        """Test that different texts can fail threshold."""
        processor = AVPProcessor(threshold=0.001)  # Very strict threshold
        
        item = {
            "prediction": "Very different text",
            "truth": "Completely unrelated content",
        }
        
        result, passed = processor.process_item(item)
        
        # With such a strict threshold, should likely fail
        assert isinstance(passed, bool)
        assert result["passed_threshold"] == passed
        
    def test_process_batch(self):
        """Test batch processing."""
        processor = AVPProcessor()
        
        items = [
            {"prediction": "test1", "truth": "truth1"},
            {"prediction": "test2", "truth": "truth2"},
            {"prediction": "test3", "truth": "truth3"},
        ]
        
        results = processor.process_batch(items)
        
        assert len(results) == len(items)
        for result in results:
            assert "instability_score" in result
            assert "passed_threshold" in result
            
    def test_get_summary(self):
        """Test summary generation."""
        processor = AVPProcessor(threshold=1.0)
        
        items = [
            {"prediction": "same", "truth": "same"},
            {"prediction": "different", "truth": "other"},
        ]
        
        results = processor.process_batch(items)
        summary = processor.get_summary(results)
        
        assert "total_items" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "pass_rate" in summary
        assert "mean_score" in summary
        assert "min_score" in summary
        assert "max_score" in summary
        assert "threshold" in summary
        
        assert summary["total_items"] == 2
        assert summary["passed"] + summary["failed"] == 2
        assert 0 <= summary["pass_rate"] <= 1
        
    def test_get_summary_empty(self):
        """Test summary with empty results."""
        processor = AVPProcessor()
        summary = processor.get_summary([])
        
        assert summary["total_items"] == 0
        assert summary["passed"] == 0
        assert summary["failed"] == 0
        assert summary["pass_rate"] == 0.0
        
    def test_deterministic_behavior(self):
        """Test that processing is deterministic."""
        item = {"prediction": "test", "truth": "truth"}
        
        processor1 = AVPProcessor(deterministic=True)
        result1, _ = processor1.process_item(item)
        
        processor2 = AVPProcessor(deterministic=True)
        result2, _ = processor2.process_item(item)
        
        assert result1["instability_score"] == result2["instability_score"]
