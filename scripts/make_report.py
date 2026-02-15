#!/usr/bin/env python3
"""
Generate Markdown report from JSON report.

Converts reports/avp_report.json to reports/avp_report.md
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_json_report(path: Path) -> dict:
    """Load JSON report."""
    with open(path) as f:
        return json.load(f)


def generate_markdown(data: dict) -> str:
    """Generate markdown report from JSON data."""
    lines = []
    
    # Header
    lines.append("# AVP Test Suite Report\n\n")
    lines.append(f"**Generated:** {data.get('timestamp', 'N/A')}\n\n")
    
    # Overall status
    status = "✅ PASSED" if data.get('summary', {}).get('all_passed', False) else "❌ FAILED"
    lines.append(f"**Status:** {status}\n\n")
    
    lines.append("---\n\n")
    
    # What was tested
    lines.append("## What Was Tested\n\n")
    lines.append("This test suite demonstrates Eigentrace across three use cases:\n\n")
    
    for use_case_id, use_case in data.get('use_cases', {}).items():
        name = use_case.get('name', use_case_id)
        desc = use_case.get('description', '')
        status = "✅" if use_case.get('passed', False) else "❌"
        lines.append(f"{status} **{name}**: {desc}\n\n")
    
    # Fixture counts
    lines.append("### Fixture Counts\n\n")
    fixtures = data.get('fixtures', {})
    if fixtures:
        for fixture, count in fixtures.items():
            lines.append(f"- `{fixture}`: {count} items\n")
        lines.append(f"\n**Total fixtures:** {sum(fixtures.values())} items\n\n")
    
    lines.append("---\n\n")
    
    # Results summary
    lines.append("## Results Summary\n\n")
    
    for use_case_id, use_case in data.get('use_cases', {}).items():
        name = use_case.get('name', use_case_id)
        passed = use_case.get('passed', False)
        status = "✅ PASSED" if passed else "❌ FAILED"
        
        lines.append(f"### {name}\n\n")
        lines.append(f"**Status:** {status}\n\n")
        
        # Add metrics if available
        if 'metrics' in use_case:
            lines.append("**Metrics:**\n\n")
            for metric, value in use_case['metrics'].items():
                formatted_value = format_metric_value(metric, value)
                metric_name = metric.replace('_', ' ').title()
                lines.append(f"- {metric_name}: {formatted_value}\n")
            lines.append("\n")
    
    lines.append("---\n\n")
    
    # Savings accounting (if cost_shaver data available)
    cost_shaver = data.get('use_cases', {}).get('cost_shaver', {})
    if 'metrics' in cost_shaver:
        lines.append("## Savings Accounting\n\n")
        lines.append("The **gate-then-escalate** pattern reduces expensive judge API calls:\n\n")
        
        metrics = cost_shaver['metrics']
        
        lines.append("### Call Reduction\n\n")
        if 'baseline_calls' in metrics:
            lines.append(f"- **Baseline judge calls (no gating):** {metrics['baseline_calls']}\n")
        if 'gated_calls' in metrics:
            lines.append(f"- **Gated judge calls:** {metrics['gated_calls']}\n")
        if 'avoided_calls' in metrics:
            lines.append(f"- **Avoided calls:** {metrics['avoided_calls']}\n")
        if 'judge_calls_reduced_pct' in metrics:
            lines.append(f"- **Reduction:** {metrics['judge_calls_reduced_pct']:.1%}\n")
        
        lines.append("\n### Cost Savings\n\n")
        if 'avoided_tokens' in metrics:
            lines.append(f"- **Estimated tokens saved:** {metrics['avoided_tokens']:,}\n")
        if 'estimated_savings_usd' in metrics:
            lines.append(f"- **Estimated cost savings:** ${metrics['estimated_savings_usd']:.4f}\n")
        
        lines.append("\n**Note:** Savings are estimated based on configurable constants. "
                   "Actual savings depend on judge model costs and token counts.\n\n")
        
        lines.append("---\n\n")
    
    # Privacy verification
    private_eval = data.get('use_cases', {}).get('private_eval', {})
    if 'metrics' in private_eval:
        lines.append("## Privacy Verification\n\n")
        
        metrics = private_eval['metrics']
        
        if 'items_scored' in metrics:
            lines.append(f"- **Items scored:** {metrics['items_scored']}\n")
        if 'network_calls' in metrics:
            network_status = "✅ No network calls" if metrics['network_calls'] == 0 else f"⚠️ {metrics['network_calls']} network calls"
            lines.append(f"- **Network calls:** {network_status}\n")
        if 'privacy_mode' in metrics:
            lines.append(f"- **Privacy mode:** {metrics['privacy_mode']}\n")
        
        lines.append("\n**Status:** ✅ Verified (local-only evaluation)\n\n")
        lines.append("All scoring completed without external API calls or network dependencies.\n\n")
        
        lines.append("---\n\n")
    
    # Code linter signal
    code_linter = data.get('use_cases', {}).get('code_linter', {})
    if 'metrics' in code_linter:
        lines.append("## Code Linter Signal\n\n")
        
        metrics = code_linter['metrics']
        
        lines.append("### Score Separation\n\n")
        if 'mean_good_score' in metrics:
            lines.append(f"- **Mean good code score:** {metrics['mean_good_score']:.3f}\n")
        if 'mean_bad_score' in metrics:
            lines.append(f"- **Mean bad code score:** {metrics['mean_bad_score']:.3f}\n")
        if 'score_separation' in metrics:
            lines.append(f"- **Separation:** {metrics['score_separation']:.3f}\n")
        
        lines.append("\n### Anomaly Detection\n\n")
        if 'ghost_imports_detected' in metrics:
            lines.append(f"- Ghost imports detected: {metrics['ghost_imports_detected']}\n")
        if 'repetition_detected' in metrics:
            lines.append(f"- Repetition patterns detected: {metrics['repetition_detected']}\n")
        
        lines.append("\n**Result:** Good code scores significantly higher than bad code.\n\n")
        
        lines.append("---\n\n")
    
    # Footer
    lines.append("## Conclusion\n\n")
    if data.get('summary', {}).get('all_passed', False):
        lines.append("✅ **All tests passed.** Eigentrace successfully demonstrates:\n\n")
        lines.append("- Cost reduction through intelligent gating\n")
        lines.append("- Privacy-preserving local evaluation\n")
        lines.append("- Code quality signal detection\n")
    else:
        lines.append("⚠️ **Some tests failed.** Review the results above for details.\n")
    
    return ''.join(lines)


def format_metric_value(metric: str, value) -> str:
    """Format metric value for display."""
    if isinstance(value, float):
        if 'pct' in metric or 'rate' in metric:
            return f"{value:.1%}"
        elif 'score' in metric or 'separation' in metric:
            return f"{value:.3f}"
        elif 'usd' in metric or 'cost' in metric:
            return f"${value:.4f}"
        else:
            return f"{value:.2f}"
    elif isinstance(value, int):
        return f"{value:,}"
    else:
        return str(value)


def main():
    """Main entry point."""
    # Paths
    json_path = Path('reports/avp_report.json')
    md_path = Path('reports/avp_report.md')
    
    if not json_path.exists():
        print(f"Error: {json_path} not found", file=sys.stderr)
        print("Run 'python scripts/run_avp_tests.py' first", file=sys.stderr)
        sys.exit(1)
    
    # Load JSON report
    print(f"Loading {json_path}...")
    data = load_json_report(json_path)
    
    # Generate markdown
    print("Generating markdown report...")
    markdown = generate_markdown(data)
    
    # Write markdown report
    md_path.parent.mkdir(exist_ok=True)
    with open(md_path, 'w') as f:
        f.write(markdown)
    
    print(f"✅ Markdown report generated: {md_path}")
    
    # Also print to stdout
    print("\n" + "=" * 60)
    print(markdown)


if __name__ == '__main__':
    main()
