# AVP Test Suite Documentation

This document explains how to use the AVP (Application Validation Program) test suite for Eigentrace.

## Overview

The AVP test suite demonstrates Eigentrace across three use cases:

1. **Eval Cost Shaver**: Reduce expensive LLM judge calls using Eigentrace gating
2. **Private Eval Layer**: Local-only evaluation with no network dependencies
3. **Cognitive Linter for Code**: Detect code quality issues through structural analysis

## Running Tests Locally

### Prerequisites

- Python 3.8+
- pytest installed (`pip install pytest`)

### Installation

```bash
# Clone repository
git clone https://github.com/keepitcheesy/Eigen-trace.git
cd Eigen-trace

# Install with dev dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run AVP test suite
python scripts/run_avp_tests.py
```

This will:
1. Execute all AVP tests
2. Generate `reports/avp_report.json`
3. Display summary in terminal

### Generate Markdown Report

```bash
# Convert JSON to Markdown
python scripts/make_report.py
```

This generates `reports/avp_report.md`.

### Run Individual Test Files

```bash
# Cost shaver tests
pytest tests/test_eigentrace_cost_shaver.py -v

# Private eval tests
pytest tests/test_eigentrace_private_eval.py -v

# Code linter tests
pytest tests/test_eigentrace_code_linter.py -v

# Report generation tests
pytest tests/test_report_generation.py -v
```

## Test Fixtures

Fixtures are located in `tests/fixtures/` as JSONL files:

### Fixture Format

Each line is a JSON object with:
```json
{
  "id": "unique_identifier",
  "input": "prompt or task description",
  "output": "model output to score",
  "label": "good|bad|sensitive",
  "notes": "explanation of why this example matters"
}
```

### Available Fixtures

- **prose_good.jsonl**: Coherent, factually accurate text (5 items)
- **prose_bad.jsonl**: Hallucinations, repetition, drift (6 items)
- **private_sensitive.jsonl**: PII-containing text (5 items)
- **code_good.jsonl**: Valid Python functions (5 items)
- **code_bad.jsonl**: Code with issues (6 items)

### Adding New Fixtures

1. Create or edit JSONL file in `tests/fixtures/`
2. Follow the format above
3. Add diverse examples covering edge cases
4. Run tests to verify

Example:
```bash
# Add to prose_bad.jsonl
echo '{"id":"bad_7","input":"Query","output":"Bad output","label":"bad","notes":"Why bad"}' >> tests/fixtures/prose_bad.jsonl
```

## Tuning Thresholds

Tests use configurable thresholds:

### Cost Shaver Thresholds

In `tests/test_eigentrace_cost_shaver.py`:

```python
RECALL_BAD_THRESHOLD = 0.80  # Must catch 80% of bad items
JUDGE_CALLS_REDUCED_PCT_THRESHOLD = 0.50  # Must reduce calls by 50%
```

To tune the gating threshold:
1. Run `test_threshold_tuning()` test
2. Examine output for different threshold values
3. Choose threshold balancing recall and reduction

### Savings Accounting

Configure costs in `tests/test_eigentrace_cost_shaver.py`:

```python
JUDGE_COST_PER_1K_TOKENS_USD = 0.01  # Cost per 1K tokens
AVG_TOKENS_PER_JUDGE_CALL = 400  # Average tokens per call
```

Update these based on your actual judge model.

### Code Linter Thresholds

In `tests/test_eigentrace_code_linter.py`:

```python
margin = 0.15  # Good code must score 0.15 higher than bad
```

Adjust based on your quality requirements.

## Interpreting Reports

### JSON Report Structure

```json
{
  "timestamp": "2024-02-15T00:00:00Z",
  "test_suite": "AVP Test Suite",
  "use_cases": {
    "cost_shaver": {
      "passed": true,
      "metrics": {
        "recall_bad": 0.85,
        "judge_calls_reduced_pct": 0.60,
        "estimated_savings_usd": 0.028
      }
    }
  },
  "fixtures": {...},
  "summary": {...}
}
```

### Markdown Report Sections

1. **What Was Tested**: Use cases and fixture counts
2. **Results Summary**: Pass/fail status for each use case
3. **Savings Accounting**: Call reduction and cost savings
4. **Privacy Verification**: Network call status
5. **Code Linter Signal**: Score separation and anomaly detection

### Key Metrics

**Cost Shaver:**
- `recall_bad`: Fraction of bad items caught (higher is better)
- `judge_calls_reduced_pct`: Percentage of judge calls avoided (higher is better)
- `estimated_savings_usd`: Estimated cost savings in USD

**Private Eval:**
- `items_scored`: Number of items successfully scored
- `network_calls`: Should be 0 (no network)
- `privacy_mode`: "Verified" if all checks pass

**Code Linter:**
- `score_separation`: Difference between good and bad scores (higher is better)
- `ghost_imports_detected`: Count of invalid imports found
- `repetition_detected`: Count of repetitive patterns found

## Troubleshooting

### Tests Fail to Import

```bash
# Ensure package is installed
pip install -e .
```

### Network Blocking Errors

The private eval tests intentionally block network. If you see errors about network calls, the test is working correctly - it's detecting unwanted network access.

### Score Separation Too Low

If `test_good_vs_bad_score_separation` fails:
1. Check fixture quality
2. Verify fixtures have clear good/bad distinction
3. Consider adjusting threshold margin

### Runtime Too Long

Target runtime is < 60 seconds. If tests are slow:
1. Reduce fixture counts
2. Check for heavy dependencies
3. Profile with `pytest --durations=10`

## CI Integration

Tests run automatically on:
- Push to `main` branch
- Pull requests to `main`

View results:
1. Go to Actions tab in GitHub
2. Click on workflow run
3. Download artifacts (JSON and Markdown reports)

## Best Practices

1. **Run tests before committing**: Catch issues early
2. **Check reports**: Review generated reports for insights
3. **Update fixtures**: Keep fixtures representative of real data
4. **Tune thresholds**: Adjust based on your requirements
5. **Monitor runtime**: Keep total runtime under 60 seconds

## Advanced Usage

### Running with Coverage

```bash
pytest tests/test_eigentrace_*.py --cov=logoslabs --cov-report=html
```

### Debugging Tests

```bash
# Run with verbose output
pytest tests/test_eigentrace_cost_shaver.py -vv -s

# Run specific test
pytest tests/test_eigentrace_cost_shaver.py::TestEigentraceCostShaver::test_gating_reduces_judge_calls -v
```

### Custom Test Configuration

Create `pytest.ini` to customize test behavior:
```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

## Getting Help

- **Issues**: Open an issue on GitHub
- **Documentation**: Check main README.md
- **Examples**: See `examples/` directory

## Contributing

To add new test cases:

1. Add fixtures to appropriate JSONL file
2. Update test if needed to cover new scenarios
3. Run tests locally
4. Submit PR with updated fixtures and tests
5. Include generated reports in PR description
