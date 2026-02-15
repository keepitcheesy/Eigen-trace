#!/usr/bin/env python3
"""
Run AVP test suite and collect metrics for reporting.

This script:
1. Runs pytest on AVP tests
2. Collects key metrics
3. Generates reports/avp_report.json
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


def run_pytest() -> dict:
    """Run pytest and capture results."""
    print("Running AVP test suite...")
    
    # Run pytest with JSON output
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            'tests/test_eigentrace_cost_shaver.py',
            'tests/test_eigentrace_private_eval.py',
            'tests/test_eigentrace_code_linter.py',
            'tests/test_report_generation.py',
            '-v',
            '--tb=short'
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return {
        'exit_code': result.returncode,
        'passed': result.returncode == 0,
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def collect_metrics() -> dict:
    """Collect metrics from test artifacts."""
    metrics = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test_suite': 'AVP Test Suite',
        'use_cases': {},
        'fixtures': {},
        'summary': {}
    }
    
    # Count fixture items
    fixtures = {
        'prose_good': 'tests/fixtures/prose_good.jsonl',
        'prose_bad': 'tests/fixtures/prose_bad.jsonl',
        'private_sensitive': 'tests/fixtures/private_sensitive.jsonl',
        'code_good': 'tests/fixtures/code_good.jsonl',
        'code_bad': 'tests/fixtures/code_bad.jsonl'
    }
    
    for name, path in fixtures.items():
        try:
            with open(path) as f:
                count = sum(1 for line in f if line.strip())
            metrics['fixtures'][name] = count
        except FileNotFoundError:
            metrics['fixtures'][name] = 0
    
    return metrics


def generate_report(pytest_result: dict, metrics: dict) -> None:
    """Generate JSON report."""
    # Combine results
    report = {
        **metrics,
        'pytest': {
            'passed': pytest_result['passed'],
            'exit_code': pytest_result['exit_code']
        },
        'use_cases': {
            'cost_shaver': {
                'name': 'Eval Cost Shaver',
                'description': 'Gate-then-escalate pattern to reduce judge API calls',
                'passed': pytest_result['passed']
            },
            'private_eval': {
                'name': 'Private Eval Layer',
                'description': 'Local-only evaluation with no network calls',
                'passed': pytest_result['passed']
            },
            'code_linter': {
                'name': 'Cognitive Linter for Code',
                'description': 'Detect code quality issues',
                'passed': pytest_result['passed']
            }
        },
        'summary': {
            'all_passed': pytest_result['passed'],
            'total_fixtures': sum(metrics['fixtures'].values())
        }
    }
    
    # Ensure reports directory exists
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Write JSON report
    report_path = reports_dir / 'avp_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ JSON report generated: {report_path}")
    
    return report


def main():
    """Main entry point."""
    print("=" * 60)
    print("AVP Test Suite Runner")
    print("=" * 60)
    
    # Run tests
    pytest_result = run_pytest()
    
    # Collect metrics
    metrics = collect_metrics()
    
    # Generate report
    report = generate_report(pytest_result, metrics)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Tests passed: {pytest_result['passed']}")
    print(f"Total fixtures: {sum(metrics['fixtures'].values())}")
    print(f"Report: reports/avp_report.json")
    
    # Exit with pytest exit code
    sys.exit(pytest_result['exit_code'])


if __name__ == '__main__':
    main()
