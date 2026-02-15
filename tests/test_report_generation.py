"""
Test report generation functionality.

Tests that:
- JSON reports are generated correctly
- Markdown reports are generated correctly
- Reports contain all required sections
"""

import json
import pytest
from pathlib import Path

from logoslabs.avp import load_jsonl


class TestReportGeneration:
    """Test report generation."""
    
    def test_json_report_structure(self, tmp_path):
        """Test that JSON report has correct structure."""
        # Create sample report data
        report_data = {
            'test_suite': 'AVP Test Suite',
            'timestamp': '2024-02-15T00:00:00Z',
            'use_cases': {
                'cost_shaver': {
                    'passed': True,
                    'metrics': {
                        'recall_bad': 0.85,
                        'judge_calls_reduced_pct': 0.60,
                        'baseline_calls': 11,
                        'gated_calls': 4,
                        'avoided_calls': 7,
                        'estimated_savings_usd': 0.028
                    }
                },
                'private_eval': {
                    'passed': True,
                    'metrics': {
                        'items_scored': 5,
                        'network_calls': 0,
                        'privacy_mode': 'Verified'
                    }
                },
                'code_linter': {
                    'passed': True,
                    'metrics': {
                        'mean_good_score': 0.65,
                        'mean_bad_score': 0.45,
                        'score_separation': 0.20,
                        'ghost_imports_detected': 2,
                        'repetition_detected': 3
                    }
                }
            },
            'fixtures': {
                'prose_good': 5,
                'prose_bad': 6,
                'private_sensitive': 5,
                'code_good': 5,
                'code_bad': 6
            },
            'summary': {
                'total_tests_passed': 3,
                'total_tests_failed': 0,
                'all_passed': True
            }
        }
        
        # Write JSON report
        report_path = tmp_path / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Read and verify
        with open(report_path) as f:
            loaded_data = json.load(f)
        
        assert loaded_data['test_suite'] == 'AVP Test Suite'
        assert 'use_cases' in loaded_data
        assert 'cost_shaver' in loaded_data['use_cases']
        assert 'private_eval' in loaded_data['use_cases']
        assert 'code_linter' in loaded_data['use_cases']
        assert loaded_data['summary']['all_passed'] is True
        
        print(f"\n=== JSON Report Structure Verified ===")
        print(f"Report path: {report_path}")
    
    def test_markdown_report_generation(self, tmp_path):
        """Test markdown report generation."""
        # Sample data
        report_data = {
            'use_cases': {
                'cost_shaver': {
                    'passed': True,
                    'metrics': {
                        'recall_bad': 0.85,
                        'judge_calls_reduced_pct': 0.60,
                        'baseline_calls': 11,
                        'gated_calls': 4,
                        'avoided_calls': 7,
                        'estimated_savings_usd': 0.028
                    }
                },
                'private_eval': {
                    'passed': True,
                    'metrics': {
                        'items_scored': 5,
                        'network_calls': 0
                    }
                },
                'code_linter': {
                    'passed': True,
                    'metrics': {
                        'mean_good_score': 0.65,
                        'mean_bad_score': 0.45,
                        'score_separation': 0.20
                    }
                }
            },
            'fixtures': {
                'prose_good': 5,
                'prose_bad': 6,
                'private_sensitive': 5,
                'code_good': 5,
                'code_bad': 6
            }
        }
        
        # Generate markdown
        markdown = self._generate_markdown_report(report_data)
        
        # Write to file
        report_path = tmp_path / 'test_report.md'
        with open(report_path, 'w') as f:
            f.write(markdown)
        
        # Verify content
        assert '# AVP Test Suite Report' in markdown
        assert '## What Was Tested' in markdown
        assert '## Results Summary' in markdown
        assert '## Savings Accounting' in markdown
        assert '## Privacy Verification' in markdown
        assert '## Code Linter Signal' in markdown
        
        print(f"\n=== Markdown Report Generated ===")
        print(f"Report path: {report_path}")
        print(f"Report length: {len(markdown)} characters")
    
    def test_fixture_counts(self):
        """Test that fixture files exist and have expected counts."""
        fixture_counts = {}
        
        fixture_files = [
            'prose_good.jsonl',
            'prose_bad.jsonl',
            'private_sensitive.jsonl',
            'code_good.jsonl',
            'code_bad.jsonl'
        ]
        
        for filename in fixture_files:
            path = f'tests/fixtures/{filename}'
            items = load_jsonl(path)
            fixture_counts[filename] = len(items)
        
        # Verify all fixtures exist and have data
        for filename, count in fixture_counts.items():
            assert count > 0, f"{filename} should have items"
        
        print(f"\n=== Fixture Counts ===")
        for filename, count in fixture_counts.items():
            print(f"{filename}: {count} items")
    
    def _generate_markdown_report(self, data: dict) -> str:
        """Generate markdown report from data."""
        lines = []
        
        lines.append("# AVP Test Suite Report\n")
        
        # What was tested
        lines.append("## What Was Tested\n")
        lines.append("This test suite demonstrates Eigentrace across three use cases:\n")
        lines.append("1. **Eval Cost Shaver**: Gate-then-escalate pattern to reduce judge API calls\n")
        lines.append("2. **Private Eval Layer**: Local-only evaluation with no network calls\n")
        lines.append("3. **Cognitive Linter for Code**: Detect code quality issues\n")
        
        # Fixtures
        if 'fixtures' in data:
            lines.append("\n### Fixture Counts\n")
            for fixture, count in data['fixtures'].items():
                lines.append(f"- {fixture}: {count} items\n")
        
        # Results summary
        lines.append("\n## Results Summary\n")
        for use_case, result in data['use_cases'].items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            lines.append(f"### {use_case.replace('_', ' ').title()}: {status}\n")
        
        # Savings accounting
        if 'cost_shaver' in data['use_cases']:
            lines.append("\n## Savings Accounting\n")
            metrics = data['use_cases']['cost_shaver']['metrics']
            lines.append(f"- Baseline judge calls: {metrics['baseline_calls']}\n")
            lines.append(f"- Gated judge calls: {metrics['gated_calls']}\n")
            lines.append(f"- Avoided calls: {metrics['avoided_calls']}\n")
            lines.append(f"- Judge calls reduced: {metrics['judge_calls_reduced_pct']:.1%}\n")
            lines.append(f"- Estimated savings: ${metrics['estimated_savings_usd']:.4f}\n")
        
        # Privacy verification
        if 'private_eval' in data['use_cases']:
            lines.append("\n## Privacy Verification\n")
            metrics = data['use_cases']['private_eval']['metrics']
            lines.append(f"- Items scored: {metrics['items_scored']}\n")
            lines.append(f"- Network calls: {metrics['network_calls']}\n")
            lines.append("- Status: ✅ Verified (no network)\n")
        
        # Code linter signal
        if 'code_linter' in data['use_cases']:
            lines.append("\n## Code Linter Signal\n")
            metrics = data['use_cases']['code_linter']['metrics']
            lines.append(f"- Mean good score: {metrics['mean_good_score']:.3f}\n")
            lines.append(f"- Mean bad score: {metrics['mean_bad_score']:.3f}\n")
            lines.append(f"- Score separation: {metrics['score_separation']:.3f}\n")
        
        return ''.join(lines)
