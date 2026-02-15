# AVP Test Suite for Eigentrace

This directory contains the Application Validation Program (AVP) test suite for Eigentrace, demonstrating its capabilities across three key use cases.

## Quick Start

```bash
# Run all AVP tests
python scripts/run_avp_tests.py

# Generate markdown report
python scripts/make_report.py

# View results
cat reports/avp_report.md
```

## Overview

The AVP test suite validates Eigentrace as:

1. **Eval Cost Shaver**: Reduces expensive LLM judge calls through intelligent gating
2. **Private Eval Layer**: Provides local-only evaluation without network dependencies
3. **Cognitive Linter for Code**: Detects structural code quality issues

## Test Results Summary

✅ **All 18 tests passing**
- Runtime: ~2 seconds (well under 60s requirement)
- No external web calls
- Deterministic, reproducible results

### Key Metrics

**Cost Savings:**
- 63.6% reduction in judge API calls
- ~2,800 tokens saved per batch
- Estimated $0.028 savings per batch

**Privacy:**
- 100% local evaluation
- Zero network calls
- No API key dependencies

**Code Quality:**
- Clear score separation between good/bad code
- Anomaly detection: ghost imports, repetition, undefined variables
- 50% recall on structurally problematic outputs

## Structure

```
/tests/
  /fixtures/          # JSONL test data (27 items)
    prose_good.jsonl
    prose_bad.jsonl
    private_sensitive.jsonl
    code_good.jsonl
    code_bad.jsonl
  test_eigentrace_cost_shaver.py     # Cost reduction tests
  test_eigentrace_private_eval.py    # Privacy verification tests
  test_eigentrace_code_linter.py     # Code quality tests
  test_report_generation.py          # Reporting tests

/scripts/
  run_avp_tests.py    # Test runner and metric collector
  make_report.py      # Markdown report generator

/reports/             # Generated reports (gitignored)
  avp_report.json
  avp_report.md

/.github/workflows/
  avp_tests.yml       # CI automation

/docs/
  TEST_SUITE.md       # Usage documentation
  SAVINGS.md          # Savings accounting details
  SAMPLE_REPORT.md    # Example report output
```

## Running Tests

### All Tests

```bash
python scripts/run_avp_tests.py
```

### Individual Test Suites

```bash
# Cost shaver tests
pytest tests/test_eigentrace_cost_shaver.py -v

# Private eval tests
pytest tests/test_eigentrace_private_eval.py -v

# Code linter tests
pytest tests/test_eigentrace_code_linter.py -v
```

### With Coverage

```bash
pytest tests/test_eigentrace_*.py --cov=logoslabs --cov-report=html
```

## Reports

### JSON Report

Contains raw metrics and test results:
```bash
cat reports/avp_report.json
```

### Markdown Report

Human-readable summary with all key metrics:
```bash
cat reports/avp_report.md
```

See [docs/SAMPLE_REPORT.md](docs/SAMPLE_REPORT.md) for an example.

## CI Integration

Tests run automatically on:
- Push to `main` branch
- Pull requests to `main`

View workflow: `.github/workflows/avp_tests.yml`

## Documentation

- **[TEST_SUITE.md](docs/TEST_SUITE.md)** - Complete usage guide
  - How to run tests locally
  - Adding new fixtures
  - Tuning thresholds
  - Interpreting results

- **[SAVINGS.md](docs/SAVINGS.md)** - Savings accounting explained
  - What is saved (calls, tokens, $)
  - What is not claimed
  - Gate-then-escalate pattern
  - Real-world examples

- **[SAMPLE_REPORT.md](docs/SAMPLE_REPORT.md)** - Example report output

## Key Features

### 1. Deterministic Fixtures

All test data is in JSONL format with clear labels:
```json
{
  "id": "unique_id",
  "input": "task description",
  "output": "model output to score",
  "label": "good|bad|sensitive",
  "notes": "why this example matters"
}
```

### 2. Stable Scoring Contract

Eigentrace scoring provides:
- `scores.overall` - Quality score (0..1)
- `confidence_trace` - Temporal confidence values
- `trace_kind` - Scoring method ("head_proxy")
- `anomalies` - Detected issues with severity
- `meta.used_logprobs` - bool (false for local)

### 3. Anomaly Detection

Minimal heuristics detect:
- Repetitive patterns (loops, excessive word reuse)
- Ghost imports (invalid module names)
- Undefined variables
- Structural collapse

### 4. Configurable Thresholds

Tune for your use case:
```python
JUDGE_COST_PER_1K_TOKENS_USD = 0.01  # Your judge model cost
AVG_TOKENS_PER_JUDGE_CALL = 400       # Your average
RECALL_BAD_THRESHOLD = 0.50           # Quality bar
JUDGE_CALLS_REDUCED_PCT_THRESHOLD = 0.50  # Savings target
```

## Constraints

- **Runtime:** < 60 seconds ✅ (actual: ~2s)
- **Dependencies:** Minimal (torch, numpy, pytest)
- **Network:** Zero external calls ✅
- **Deterministic:** Reproducible results ✅

## Caveats

Eigentrace detects **structural** issues, not semantic correctness:

✅ **Detects:**
- Repetitive loops
- Entropy bursts
- Structural collapse
- Ghost imports (code)
- Undefined variables (code)

❌ **Does not detect:**
- Subtle factual errors
- Semantic incorrectness
- Bias or appropriateness
- Content-level quality

This is by design - Eigentrace is a pre-filter, not a replacement for content-aware judges.

## Contributing

To add test cases:
1. Add items to appropriate fixture file
2. Run tests: `python scripts/run_avp_tests.py`
3. Verify all tests pass
4. Include generated report in PR

## Questions?

- See [TEST_SUITE.md](docs/TEST_SUITE.md) for detailed usage
- See [SAVINGS.md](docs/SAVINGS.md) for economics
- Open an issue on GitHub

## License

MIT - See [LICENSE](../LICENSE) file
