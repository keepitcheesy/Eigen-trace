"""
Test Eigentrace as a Cognitive Linter for Code.

Tests that:
- Good code scores higher than bad code
- Anomaly detection flags specific issues
- Ghost imports detected
- Repetition patterns detected
"""

import json
import pytest
from typing import Dict, List, Any
import numpy as np

from logoslabs.avp import AVPProcessor, load_jsonl
from logoslabs.eigentrace_contract import score_text, AnomalyDetector


def score_code_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score code items with Eigentrace.
    
    Args:
        items: Code items to score
        
    Returns:
        Scored results
    """
    processor = AVPProcessor(threshold=0.5)
    results = []
    
    for item in items:
        # Score the code
        score = score_text(item['output'], item.get('input', ''), processor)
        
        results.append({
            'id': item['id'],
            'eigentrace_score': score.overall,
            'anomalies': score.anomalies,
            'num_anomalies': len(score.anomalies),
            'label': item.get('label', 'unknown'),
            'notes': item.get('notes', '')
        })
    
    return results


class TestEigentraceCodeLinter:
    """Test Eigentrace as cognitive linter for code."""
    
    def test_good_vs_bad_score_separation(self, tmp_path):
        """Test that good code scores significantly higher than bad code."""
        # Load fixtures
        good_items = load_jsonl('tests/fixtures/code_good.jsonl')
        bad_items = load_jsonl('tests/fixtures/code_bad.jsonl')
        
        # Score items
        good_results = score_code_items(good_items)
        bad_results = score_code_items(bad_items)
        
        # Calculate mean scores
        good_scores = [r['eigentrace_score'] for r in good_results]
        bad_scores = [r['eigentrace_score'] for r in bad_results]
        
        mean_good = np.mean(good_scores)
        mean_bad = np.mean(bad_scores)
        
        # Assert score separation
        margin = 0.15
        assert mean_good > mean_bad + margin, \
            f"Good code mean score ({mean_good:.3f}) not sufficiently higher than bad code ({mean_bad:.3f}), margin {margin}"
        
        # Save results
        results = {
            'good_scores': good_scores,
            'bad_scores': bad_scores,
            'mean_good': float(mean_good),
            'mean_bad': float(mean_bad),
            'separation': float(mean_good - mean_bad)
        }
        
        report_path = tmp_path / 'code_linter_scores.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Code Linter Score Separation ===")
        print(f"Mean good code score: {mean_good:.3f}")
        print(f"Mean bad code score: {mean_bad:.3f}")
        print(f"Separation: {mean_good - mean_bad:.3f}")
        print(f"Good scores: {[f'{s:.3f}' for s in good_scores]}")
        print(f"Bad scores: {[f'{s:.3f}' for s in bad_scores]}")
    
    def test_ghost_import_detection(self, tmp_path):
        """Test that ghost imports are detected."""
        bad_items = load_jsonl('tests/fixtures/code_bad.jsonl')
        
        # Find the ghost import item
        ghost_import_items = [
            item for item in bad_items 
            if 'ghost import' in item.get('notes', '').lower()
        ]
        
        assert len(ghost_import_items) > 0, "No ghost import items in fixtures"
        
        # Score and check for anomalies
        results = score_code_items(ghost_import_items)
        
        flagged_count = 0
        for result in results:
            # Check if any anomaly is related to imports
            for anomaly in result['anomalies']:
                if anomaly['kind'] == 'rare_token_burst' and 'import' in anomaly.get('description', '').lower():
                    flagged_count += 1
                    print(f"\n=== Ghost Import Detected ===")
                    print(f"Item: {result['id']}")
                    print(f"Anomaly: {anomaly}")
                    break
        
        # Should detect at least some ghost imports
        assert flagged_count > 0, "Ghost imports not detected by anomaly system"
        
        # Save results
        report_path = tmp_path / 'ghost_imports.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def test_repetition_detection(self, tmp_path):
        """Test that repetitive code patterns are detected."""
        bad_items = load_jsonl('tests/fixtures/code_bad.jsonl')
        
        # Find repetition items
        repetition_items = [
            item for item in bad_items 
            if 'repetit' in item.get('notes', '').lower()
        ]
        
        assert len(repetition_items) > 0, "No repetition items in fixtures"
        
        # Score and check for repetition anomalies
        results = score_code_items(repetition_items)
        
        flagged_count = 0
        for result in results:
            # Check for repetition or limit_cycle anomalies
            for anomaly in result['anomalies']:
                if anomaly['kind'] in ['repetition', 'limit_cycle']:
                    flagged_count += 1
                    print(f"\n=== Repetition Detected ===")
                    print(f"Item: {result['id']}")
                    print(f"Anomaly: {anomaly}")
                    break
        
        # Should detect at least some repetition
        assert flagged_count > 0, "Repetition patterns not detected by anomaly system"
        
        # Save results
        report_path = tmp_path / 'repetition_anomalies.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def test_anomaly_rate_difference(self):
        """Test that bad code has higher anomaly rate than good code."""
        good_items = load_jsonl('tests/fixtures/code_good.jsonl')
        bad_items = load_jsonl('tests/fixtures/code_bad.jsonl')
        
        good_results = score_code_items(good_items)
        bad_results = score_code_items(bad_items)
        
        # Count anomalies
        good_anomaly_count = sum(r['num_anomalies'] for r in good_results)
        bad_anomaly_count = sum(r['num_anomalies'] for r in bad_results)
        
        good_anomaly_rate = good_anomaly_count / len(good_results) if good_results else 0
        bad_anomaly_rate = bad_anomaly_count / len(bad_results) if bad_results else 0
        
        print(f"\n=== Anomaly Rates ===")
        print(f"Good code anomaly rate: {good_anomaly_rate:.2f} per item")
        print(f"Bad code anomaly rate: {bad_anomaly_rate:.2f} per item")
        
        # Bad code should have more anomalies (or at least not significantly less)
        assert bad_anomaly_rate >= good_anomaly_rate, \
            f"Bad code should have >= anomaly rate than good code"
    
    def test_undefined_variable_detection(self):
        """Test detection of undefined variables."""
        bad_items = load_jsonl('tests/fixtures/code_bad.jsonl')
        
        # Find items with undefined variables
        undefined_items = [
            item for item in bad_items 
            if 'undefined' in item.get('notes', '').lower()
        ]
        
        if len(undefined_items) > 0:
            results = score_code_items(undefined_items)
            
            flagged_count = 0
            for result in results:
                for anomaly in result['anomalies']:
                    if 'undefined' in anomaly.get('description', '').lower():
                        flagged_count += 1
                        print(f"\n=== Undefined Variable Detected ===")
                        print(f"Item: {result['id']}")
                        print(f"Anomaly: {anomaly}")
                        break
            
            # Should detect at least some undefined variables
            assert flagged_count > 0, "Undefined variables not detected"
    
    def test_flagged_items_report(self, tmp_path):
        """Generate report of flagged code items."""
        bad_items = load_jsonl('tests/fixtures/code_bad.jsonl')
        results = score_code_items(bad_items)
        
        # Collect flagged items
        flagged_items = []
        for result in results:
            if result['num_anomalies'] > 0:
                flagged_items.append({
                    'id': result['id'],
                    'score': result['eigentrace_score'],
                    'anomaly_count': result['num_anomalies'],
                    'anomaly_kinds': list(set(a['kind'] for a in result['anomalies'])),
                    'notes': result['notes']
                })
        
        report = {
            'total_bad_items': len(bad_items),
            'flagged_items': len(flagged_items),
            'flagged_rate': len(flagged_items) / len(bad_items) if bad_items else 0,
            'flagged_details': flagged_items
        }
        
        report_path = tmp_path / 'flagged_code_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== Flagged Code Report ===")
        print(f"Total bad items: {report['total_bad_items']}")
        print(f"Flagged items: {report['flagged_items']}")
        print(f"Flagged rate: {report['flagged_rate']:.2%}")
        for item in flagged_items:
            print(f"  - {item['id']}: {item['anomaly_kinds']}")
    
    def test_anomaly_detector_direct(self):
        """Test anomaly detector functions directly."""
        # Test repetition detection
        repetitive_text = "the same thing the same thing the same thing"
        repetition_anomalies = AnomalyDetector.detect_repetition(repetitive_text)
        assert len(repetition_anomalies) > 0, "Should detect repetition"
        
        # Test ghost import detection
        ghost_import_code = "import numpy.pandas as npd"
        ghost_anomalies = AnomalyDetector.detect_ghost_imports(ghost_import_code)
        assert len(ghost_anomalies) > 0, "Should detect ghost import"
        
        # Test undefined variable detection
        undefined_code = "result = undefined_variable * 2"
        undefined_anomalies = AnomalyDetector.detect_undefined_variables(undefined_code)
        assert len(undefined_anomalies) > 0, "Should detect undefined variable"
        
        print(f"\n=== Direct Anomaly Detector Tests ===")
        print(f"Repetition anomalies: {len(repetition_anomalies)}")
        print(f"Ghost import anomalies: {len(ghost_anomalies)}")
        print(f"Undefined variable anomalies: {len(undefined_anomalies)}")
