"""
Unit tests for CLI.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch

from logoslabs.cli import main


class TestCLI:
    """Test command-line interface."""
    
    def test_cli_basic(self):
        """Test basic CLI functionality."""
        # Create test input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prediction": "test1", "truth": "truth1"}\n')
            f.write('{"prediction": "test2", "truth": "truth2"}\n')
            input_file = f.name
        
        # Create output file path
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name
        
        try:
            # Run CLI
            result = main([input_file, output_file])
            
            assert result == 0
            
            # Check output file exists and has content
            assert os.path.exists(output_file)
            
            with open(output_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2
                
                for line in lines:
                    item = json.loads(line)
                    assert "instability_score" in item
                    assert "passed_threshold" in item
                    
        finally:
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_cli_with_summary(self):
        """Test CLI with summary output."""
        # Create test input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prediction": "same", "truth": "same"}\n')
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name
        
        try:
            # Run CLI with summary
            result = main([input_file, output_file, "--summary"])
            
            assert result == 0
            
        finally:
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_cli_custom_threshold(self):
        """Test CLI with custom threshold."""
        # Create test input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"prediction": "test", "truth": "truth"}\n')
            input_file = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name
        
        try:
            # Run CLI with custom threshold
            result = main([
                input_file,
                output_file,
                "--threshold", "0.5",
                "--grace-coeff", "0.7",
                "--phase-weight", "0.2",
            ])
            
            assert result == 0
            
        finally:
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_cli_file_not_found(self):
        """Test CLI with non-existent input file."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            output_file = f.name
        
        try:
            result = main(["nonexistent.jsonl", output_file])
            assert result == 1
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
                
    def test_cli_help(self):
        """Test CLI help message."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        
        assert exc_info.value.code == 0
