"""
Test Eigentrace as an Eval Cost Shaver.

Tests the "gate then escalate" pattern:
- Run Eigentrace on outputs
- If score < threshold, escalate to judge
- Otherwise, skip judge call
- Measure savings in judge calls
"""

import json
import pytest
from typing import Dict, List, Any

from logoslabs.avp import AVPProcessor, load_jsonl
from logoslabs.eigentrace_contract import score_text


# Configuration
JUDGE_COST_PER_1K_TOKENS_USD = 0.01
AVG_TOKENS_PER_JUDGE_CALL = 400
RECALL_BAD_THRESHOLD = 0.50  # Eigentrace catches structural issues, not semantic errors
JUDGE_CALLS_REDUCED_PCT_THRESHOLD = 0.50


class MockJudge:
    """Mock judge that returns labels from fixtures."""
    
    def __init__(self):
        self.call_count = 0
    
    def judge(self, item: Dict[str, Any]) -> str:
        """Return the label from the fixture."""
        self.call_count += 1
        return item.get('label', 'unknown')


def gated_evaluation(
    items: List[Dict[str, Any]],
    threshold: float,
    judge: MockJudge
) -> Dict[str, Any]:
    """
    Run evaluation with Eigentrace gating.
    
    Args:
        items: Test items with 'input', 'output', 'label'
        threshold: Eigentrace threshold (lower = stricter)
        judge: Mock judge for escalation
        
    Returns:
        Results with metrics
    """
    processor = AVPProcessor(threshold=threshold)
    results = []
    escalated_count = 0
    
    for item in items:
        # Score with Eigentrace
        score = score_text(item['output'], item.get('input', ''), processor)
        
        # Decide whether to escalate
        if score.overall < threshold:
            # Low quality score -> escalate to judge
            judge_result = judge.judge(item)
            escalated_count += 1
            decision = judge_result
        else:
            # High quality score -> trust it
            decision = 'good'
        
        results.append({
            'id': item['id'],
            'eigentrace_score': score.overall,
            'escalated': score.overall < threshold,
            'decision': decision,
            'true_label': item['label']
        })
    
    return {
        'results': results,
        'escalated_count': escalated_count,
        'total_count': len(items)
    }


def baseline_evaluation(
    items: List[Dict[str, Any]],
    judge: MockJudge
) -> Dict[str, Any]:
    """
    Run evaluation without gating (baseline).
    
    Args:
        items: Test items
        judge: Mock judge
        
    Returns:
        Results with metrics
    """
    results = []
    
    for item in items:
        decision = judge.judge(item)
        results.append({
            'id': item['id'],
            'decision': decision,
            'true_label': item['label']
        })
    
    return {
        'results': results,
        'judge_calls': len(items)
    }


def compute_metrics(gated_results: Dict[str, Any], baseline_calls: int) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    results = gated_results['results']
    
    # Count true positives (correctly identified bad items)
    bad_items = [r for r in results if r['true_label'] == 'bad']
    escalated_bad_items = [r for r in bad_items if r['escalated']]
    
    if len(bad_items) > 0:
        recall_bad = len(escalated_bad_items) / len(bad_items)
    else:
        recall_bad = 0.0
    
    # Calculate savings
    escalated_count = gated_results['escalated_count']
    total_count = gated_results['total_count']
    avoided_calls = baseline_calls - escalated_count
    
    if baseline_calls > 0:
        judge_calls_reduced_pct = avoided_calls / baseline_calls
    else:
        judge_calls_reduced_pct = 0.0
    
    # Estimate cost savings
    avoided_tokens = avoided_calls * AVG_TOKENS_PER_JUDGE_CALL
    estimated_savings_usd = (avoided_tokens / 1000) * JUDGE_COST_PER_1K_TOKENS_USD
    
    return {
        'recall_bad': recall_bad,
        'bad_items_total': len(bad_items),
        'bad_items_escalated': len(escalated_bad_items),
        'baseline_judge_calls': baseline_calls,
        'gated_judge_calls': escalated_count,
        'avoided_calls': avoided_calls,
        'judge_calls_reduced_pct': judge_calls_reduced_pct,
        'avoided_tokens': avoided_tokens,
        'estimated_savings_usd': estimated_savings_usd
    }


class TestEigentraceCostShaver:
    """Test Eigentrace as eval cost shaver."""
    
    def test_gating_reduces_judge_calls(self, tmp_path):
        """Test that gating reduces judge calls while maintaining recall."""
        # Load fixtures
        good_items = load_jsonl('tests/fixtures/prose_good.jsonl')
        bad_items = load_jsonl('tests/fixtures/prose_bad.jsonl')
        all_items = good_items + bad_items
        
        # Run baseline (no gating)
        baseline_judge = MockJudge()
        baseline_result = baseline_evaluation(all_items, baseline_judge)
        baseline_calls = baseline_judge.call_count
        
        # Run gated evaluation
        gated_judge = MockJudge()
        # Use threshold of 0.7 for gating (tune for good recall + savings)
        # Higher threshold catches more potential issues
        threshold = 0.7
        gated_result = gated_evaluation(all_items, threshold, gated_judge)
        
        # Compute metrics
        metrics = compute_metrics(gated_result, baseline_calls)
        
        # Assertions
        assert metrics['recall_bad'] >= RECALL_BAD_THRESHOLD, \
            f"Recall of bad items ({metrics['recall_bad']:.2f}) below threshold {RECALL_BAD_THRESHOLD}"
        
        assert metrics['judge_calls_reduced_pct'] >= JUDGE_CALLS_REDUCED_PCT_THRESHOLD, \
            f"Judge calls reduction ({metrics['judge_calls_reduced_pct']:.2%}) below threshold {JUDGE_CALLS_REDUCED_PCT_THRESHOLD}"
        
        # Save metrics for reporting
        report_path = tmp_path / 'cost_shaver_metrics.json'
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n=== Cost Shaver Metrics ===")
        print(f"Recall (bad items): {metrics['recall_bad']:.2%}")
        print(f"Baseline judge calls: {metrics['baseline_judge_calls']}")
        print(f"Gated judge calls: {metrics['gated_judge_calls']}")
        print(f"Avoided calls: {metrics['avoided_calls']}")
        print(f"Judge calls reduced: {metrics['judge_calls_reduced_pct']:.2%}")
        print(f"Estimated tokens saved: {metrics['avoided_tokens']}")
        print(f"Estimated cost savings: ${metrics['estimated_savings_usd']:.4f}")
    
    def test_savings_accounting_constants(self):
        """Test that savings constants are properly configured."""
        assert JUDGE_COST_PER_1K_TOKENS_USD > 0
        assert AVG_TOKENS_PER_JUDGE_CALL > 0
        assert RECALL_BAD_THRESHOLD > 0 and RECALL_BAD_THRESHOLD <= 1.0
        assert JUDGE_CALLS_REDUCED_PCT_THRESHOLD > 0 and JUDGE_CALLS_REDUCED_PCT_THRESHOLD <= 1.0
    
    def test_threshold_tuning(self):
        """Test different threshold values for tuning."""
        good_items = load_jsonl('tests/fixtures/prose_good.jsonl')
        bad_items = load_jsonl('tests/fixtures/prose_bad.jsonl')
        all_items = good_items + bad_items
        
        baseline_judge = MockJudge()
        baseline_result = baseline_evaluation(all_items, baseline_judge)
        baseline_calls = baseline_judge.call_count
        
        thresholds = [0.3, 0.4, 0.5, 0.6]
        results = []
        
        for threshold in thresholds:
            judge = MockJudge()
            gated_result = gated_evaluation(all_items, threshold, judge)
            metrics = compute_metrics(gated_result, baseline_calls)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        print("\n=== Threshold Tuning Results ===")
        for result in results:
            print(f"Threshold: {result['threshold']:.2f}, "
                  f"Recall: {result['recall_bad']:.2%}, "
                  f"Reduction: {result['judge_calls_reduced_pct']:.2%}")
        
        # At least one threshold should meet both criteria
        valid_configs = [r for r in results 
                        if r['recall_bad'] >= RECALL_BAD_THRESHOLD 
                        and r['judge_calls_reduced_pct'] >= JUDGE_CALLS_REDUCED_PCT_THRESHOLD]
        
        assert len(valid_configs) > 0, "No threshold configuration meets both recall and reduction targets"
