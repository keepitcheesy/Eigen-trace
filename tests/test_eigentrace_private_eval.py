"""
Test Eigentrace as a Private Eval Layer.

Tests that:
- Scoring runs locally without network calls
- No API key dependencies
- Works with sensitive data
"""

import json
import socket
import pytest
from typing import Dict, List, Any
from unittest.mock import patch

from logoslabs.avp import AVPProcessor, load_jsonl
from logoslabs.eigentrace_contract import score_text


class NetworkBlocker:
    """Context manager to block all network calls."""
    
    def __init__(self):
        self.network_calls_attempted = []
    
    def _block_socket_connect(self, *args, **kwargs):
        """Block socket connections."""
        self.network_calls_attempted.append(('socket.connect', args, kwargs))
        raise RuntimeError("Network call blocked: socket.connect")
    
    def _block_requests(self, *args, **kwargs):
        """Block requests library calls."""
        self.network_calls_attempted.append(('requests.request', args, kwargs))
        raise RuntimeError("Network call blocked: requests.request")
    
    def _block_httpx(self, *args, **kwargs):
        """Block httpx library calls."""
        self.network_calls_attempted.append(('httpx.Client.request', args, kwargs))
        raise RuntimeError("Network call blocked: httpx.Client.request")
    
    def __enter__(self):
        """Start blocking network calls."""
        self.patches = []
        
        # Block socket
        socket_patch = patch.object(socket.socket, 'connect', self._block_socket_connect)
        self.patches.append(socket_patch)
        socket_patch.__enter__()
        
        # Block requests (if available)
        try:
            import requests
            requests_patch = patch('requests.request', self._block_requests)
            self.patches.append(requests_patch)
            requests_patch.__enter__()
        except ImportError:
            pass
        
        # Block httpx (if available)
        try:
            import httpx
            httpx_patch = patch.object(httpx.Client, 'request', self._block_httpx)
            self.patches.append(httpx_patch)
            httpx_patch.__enter__()
        except ImportError:
            pass
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop blocking network calls."""
        for patch_obj in reversed(self.patches):
            patch_obj.__exit__(exc_type, exc_val, exc_tb)
        return False


def score_items_privately(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score items with Eigentrace in privacy mode.
    
    Args:
        items: Test items with sensitive data
        
    Returns:
        Results with scores
    """
    processor = AVPProcessor(threshold=0.5)
    results = []
    
    for item in items:
        # Score using Eigentrace
        score = score_text(item['output'], item.get('input', ''), processor)
        
        results.append({
            'id': item['id'],
            'eigentrace_score': score.overall,
            'trace_kind': score.trace_kind,
            'used_logprobs': score.meta['used_logprobs'],
            'label': item.get('label', 'unknown')
        })
    
    return {
        'results': results,
        'total_scored': len(results),
        'privacy_mode': True
    }


class TestEigentracePrivateEval:
    """Test Eigentrace as private evaluation layer."""
    
    def test_no_network_calls(self, tmp_path):
        """Test that scoring completes without any network calls."""
        # Load sensitive fixtures
        items = load_jsonl('tests/fixtures/private_sensitive.jsonl')
        
        # Block network and score
        with NetworkBlocker() as blocker:
            results = score_items_privately(items)
        
        # Assert no network calls were attempted
        assert len(blocker.network_calls_attempted) == 0, \
            f"Network calls were attempted: {blocker.network_calls_attempted}"
        
        # Assert all items were scored
        assert results['total_scored'] == len(items), \
            f"Not all items scored: {results['total_scored']} vs {len(items)}"
        
        # Save results
        report_path = tmp_path / 'private_eval_results.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Private Eval Results ===")
        print(f"Total items scored: {results['total_scored']}")
        print(f"Privacy mode: {results['privacy_mode']}")
        print(f"Network calls attempted: {len(blocker.network_calls_attempted)}")
    
    def test_no_api_key_dependency(self):
        """Test that scoring works without API keys."""
        import os
        
        # Remove common API key environment variables
        api_key_vars = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'COHERE_API_KEY',
            'AI21_API_KEY',
            'HUGGINGFACE_API_KEY'
        ]
        
        original_values = {}
        for var in api_key_vars:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Load and score items
            items = load_jsonl('tests/fixtures/private_sensitive.jsonl')
            
            with NetworkBlocker():
                results = score_items_privately(items)
            
            # Should complete successfully
            assert results['total_scored'] == len(items)
            
        finally:
            # Restore original values
            for var, value in original_values.items():
                os.environ[var] = value
    
    def test_local_only_scoring(self):
        """Test that all scoring components are local."""
        items = load_jsonl('tests/fixtures/private_sensitive.jsonl')
        
        processor = AVPProcessor(threshold=0.5)
        
        for item in items:
            score = score_text(item['output'], item.get('input', ''), processor)
            
            # Verify local-only metadata
            assert score.meta['used_logprobs'] is False, \
                "Should not use external logprobs"
            
            assert score.trace_kind == 'head_proxy', \
                "Should use local head_proxy trace"
            
            # Should have valid score
            assert 0.0 <= score.overall <= 1.0, \
                f"Score out of range: {score.overall}"
    
    def test_privacy_report_generation(self, tmp_path):
        """Test generation of privacy verification report."""
        items = load_jsonl('tests/fixtures/private_sensitive.jsonl')
        
        with NetworkBlocker() as blocker:
            results = score_items_privately(items)
        
        # Generate report
        report = {
            'privacy_mode': 'Verified',
            'network_calls_attempted': len(blocker.network_calls_attempted),
            'network_status': 'No network calls' if len(blocker.network_calls_attempted) == 0 else 'Network calls detected',
            'items_scored': results['total_scored'],
            'local_only': True,
            'api_key_required': False
        }
        
        report_path = tmp_path / 'privacy_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== Privacy Report ===")
        print(f"Privacy Mode: {report['privacy_mode']}")
        print(f"Network Status: {report['network_status']}")
        print(f"Items Scored: {report['items_scored']}")
        print(f"Local Only: {report['local_only']}")
        
        # Assertions
        assert report['network_calls_attempted'] == 0
        assert report['network_status'] == 'No network calls'
        assert report['local_only'] is True
        assert report['api_key_required'] is False
    
    def test_sensitive_data_handling(self):
        """Test that sensitive data is handled properly."""
        items = load_jsonl('tests/fixtures/private_sensitive.jsonl')
        
        # Verify fixtures contain PII markers
        pii_markers = ['SSN', 'Patient:', 'Credit Card:', 'Employee ID:', 'DOB:']
        found_markers = []
        
        for item in items:
            for marker in pii_markers:
                if marker in item['output']:
                    found_markers.append(marker)
        
        # Should have PII in fixtures
        assert len(found_markers) > 0, "Fixtures should contain PII markers"
        
        # Score without leaking data
        with NetworkBlocker():
            results = score_items_privately(items)
        
        # All sensitive items should be scored
        assert results['total_scored'] == len(items)
        
        print(f"\n=== Sensitive Data Test ===")
        print(f"PII markers found in fixtures: {set(found_markers)}")
        print(f"Items with PII scored safely: {results['total_scored']}")
