"""
Unit tests for scripts/escalate.py.
"""

import pytest
import os
import json
import tempfile
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import escalate


class TestEscalate:
    """Test escalate.py helper script."""
    
    def test_escalate_default_behavior(self, capsys):
        """Test default behavior using passed_threshold field."""
        # Create test results file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1", "instability_score": 0.04, "passed_threshold": true}\n')
            f.write('{"id": "2", "instability_score": 0.06, "passed_threshold": false}\n')
            f.write('{"id": "3", "instability_score": 0.03, "passed_threshold": true}\n')
            results_file = f.name
        
        try:
            # Run escalate with default behavior
            sys.argv = ["escalate.py", results_file]
            result = escalate.main()
            
            assert result == 0
            
            # Check stdout contains only escalated items (passed_threshold=false)
            captured = capsys.readouterr()
            lines = [line for line in captured.out.strip().split('\n') if line]
            
            assert len(lines) == 1
            item = json.loads(lines[0])
            assert item["id"] == "2"
            assert item["passed_threshold"] == False
            
        finally:
            os.unlink(results_file)
    
    def test_escalate_custom_threshold(self, capsys):
        """Test custom threshold behavior."""
        # Create test results file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1", "instability_score": 0.04, "passed_threshold": true}\n')
            f.write('{"id": "2", "instability_score": 0.06, "passed_threshold": false}\n')
            f.write('{"id": "3", "instability_score": 0.03, "passed_threshold": true}\n')
            results_file = f.name
        
        try:
            # Run escalate with custom threshold
            sys.argv = ["escalate.py", results_file, "--threshold", "0.05"]
            result = escalate.main()
            
            assert result == 0
            
            # Check stdout contains only items with score > 0.05
            captured = capsys.readouterr()
            lines = [line for line in captured.out.strip().split('\n') if line]
            
            assert len(lines) == 1
            item = json.loads(lines[0])
            assert item["id"] == "2"
            assert item["instability_score"] == 0.06
            
        finally:
            os.unlink(results_file)
    
    def test_escalate_custom_threshold_multiple(self, capsys):
        """Test custom threshold with multiple escalated items."""
        # Create test results file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1", "instability_score": 0.04, "passed_threshold": true}\n')
            f.write('{"id": "2", "instability_score": 0.06, "passed_threshold": false}\n')
            f.write('{"id": "3", "instability_score": 0.03, "passed_threshold": true}\n')
            results_file = f.name
        
        try:
            # Run escalate with lower threshold
            sys.argv = ["escalate.py", results_file, "--threshold", "0.035"]
            result = escalate.main()
            
            assert result == 0
            
            # Check stdout contains items with score > 0.035
            captured = capsys.readouterr()
            lines = [line for line in captured.out.strip().split('\n') if line]
            
            assert len(lines) == 2
            ids = [json.loads(line)["id"] for line in lines]
            assert "1" in ids
            assert "2" in ids
            
        finally:
            os.unlink(results_file)
    
    def test_escalate_empty_result(self, capsys):
        """Test when no items are escalated."""
        # Create test results file with all passing items
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1", "instability_score": 0.04, "passed_threshold": true}\n')
            f.write('{"id": "2", "instability_score": 0.03, "passed_threshold": true}\n')
            results_file = f.name
        
        try:
            # Run escalate with default behavior
            sys.argv = ["escalate.py", results_file]
            result = escalate.main()
            
            assert result == 0
            
            # Check stdout is empty
            captured = capsys.readouterr()
            assert captured.out.strip() == ""
            
        finally:
            os.unlink(results_file)
    
    def test_escalate_file_not_found(self, capsys):
        """Test error handling for missing file."""
        sys.argv = ["escalate.py", "/nonexistent/file.jsonl"]
        result = escalate.main()
        
        assert result == 1
        captured = capsys.readouterr()
        assert "Error: File not found" in captured.err
    
    def test_escalate_missing_fields(self, capsys):
        """Test warning for missing fields."""
        # Create test results file with missing fields
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1"}\n')  # Missing both fields
            f.write('{"id": "2", "instability_score": 0.06, "passed_threshold": false}\n')
            results_file = f.name
        
        try:
            # Run escalate with default behavior
            sys.argv = ["escalate.py", results_file]
            result = escalate.main()
            
            assert result == 0
            
            # Check warning in stderr
            captured = capsys.readouterr()
            assert "Warning: Item missing 'passed_threshold'" in captured.err
            
            # Should still output item 2
            lines = [line for line in captured.out.strip().split('\n') if line]
            assert len(lines) == 1
            
        finally:
            os.unlink(results_file)
    
    def test_escalate_invalid_json(self, capsys):
        """Test handling of invalid JSON lines."""
        # Create test results file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "1", "instability_score": 0.04, "passed_threshold": true}\n')
            f.write('invalid json line\n')
            f.write('{"id": "2", "instability_score": 0.06, "passed_threshold": false}\n')
            results_file = f.name
        
        try:
            # Run escalate
            sys.argv = ["escalate.py", results_file]
            result = escalate.main()
            
            assert result == 0
            
            # Check warning in stderr
            captured = capsys.readouterr()
            assert "Warning: Skipping invalid JSON line" in captured.err
            
            # Should still output valid escalated items
            lines = [line for line in captured.out.strip().split('\n') if line]
            assert len(lines) == 1
            assert json.loads(lines[0])["id"] == "2"
            
        finally:
            os.unlink(results_file)
