# Savings Accounting Documentation

This document explains how Eigentrace saves costs through intelligent gating and what is (and is not) claimed.

## The Gate-Then-Escalate Pattern

### Traditional Approach

Without Eigentrace:
```
Every output → LLM Judge → $$$ per call
```

- **Cost**: Every item requires expensive judge call
- **Latency**: Network round-trip for every evaluation
- **Privacy**: Data sent to external API

### Eigentrace Approach

With Eigentrace gating:
```
Every output → Eigentrace (local, free) → Low score? → LLM Judge
                                        → High score? → Trust it (skip judge)
```

- **Cost**: Only questionable items escalated to judge
- **Latency**: Most items evaluated locally in milliseconds
- **Privacy**: Only escalated items sent externally (optional)

## What Is Saved

### 1. Judge API Calls

**Baseline (no gating):**
- N outputs → N judge calls

**With gating:**
- N outputs → M judge calls (where M < N)
- Savings: N - M calls avoided

**Example from tests:**
- 11 total items
- 4 escalated to judge
- 7 calls avoided (64% reduction)

### 2. Tokens

Each judge call consumes tokens:
- Prompt tokens (context + output to judge)
- Completion tokens (judge's response)

**Estimated tokens saved:**
```
avoided_calls × avg_tokens_per_call
```

**Example:**
- 7 calls avoided
- 400 tokens/call average
- 2,800 tokens saved

### 3. Cost (USD)

Based on judge model pricing:

```
saved_tokens ÷ 1000 × cost_per_1k_tokens
```

**Example with GPT-4:**
- 2,800 tokens saved
- $0.01 per 1K tokens
- $0.028 saved

**Note**: This is per batch. Multiply by number of batches for total savings.

### 4. Latency

Eigentrace runs in milliseconds locally:
- No network latency
- No API queue wait time
- Parallel processing possible

Judge calls add latency:
- Network round-trip: 100-500ms
- API processing: 500-2000ms
- Total: ~1-2s per call

**Latency saved:**
```
avoided_calls × avg_latency_per_judge_call
```

### 5. Carbon Footprint

LLM inference has environmental cost:
- Energy for GPU computation
- Data center cooling
- Network transmission

Local evaluation:
- Minimal CPU usage
- No network transmission
- Deterministic (no repeated calls)

## What Is NOT Claimed

### 1. Perfect Recall

Eigentrace gates based on structural signals, not content correctness:
- **What we claim**: 80%+ recall of bad items (configurable)
- **What we don't claim**: 100% catch rate
- **Trade-off**: Some bad items may pass gating

Tune threshold based on your risk tolerance:
- Lower threshold → More conservative → More escalations
- Higher threshold → More permissive → Fewer escalations

### 2. Replacement of Human Review

Eigentrace is a **pre-filter**, not a complete solution:
- **What we claim**: Reduces load on expensive judges
- **What we don't claim**: Eliminates need for quality control

Use case determines final review:
- Low-stakes: Eigentrace alone may suffice
- High-stakes: Eigentrace + judge + human review

### 3. Content-Level Judgments

Eigentrace measures structural coherence, not semantic correctness:
- **Detects**: Repetition, entropy bursts, structural collapse
- **Doesn't detect**: Subtle factual errors, bias, appropriateness

**Example:**
- ✅ Detects: "Paris is the capital the capital the capital..."
- ❌ May miss: "Paris is the capital of Germany" (structurally fine, factually wrong)

### 4. Specific Dollar Amounts

Savings estimates depend on:
- Your judge model choice
- Actual token counts
- Your pricing tier
- Batch size and frequency

**Test constants are configurable:**
```python
JUDGE_COST_PER_1K_TOKENS_USD = 0.01  # Update for your model
AVG_TOKENS_PER_JUDGE_CALL = 400      # Update for your prompts
```

Actual savings may vary significantly.

## Accounting Constants

### Default Values

```python
# Judge model costs (default: GPT-3.5-turbo ballpark)
JUDGE_COST_PER_1K_TOKENS_USD = 0.01

# Average tokens per judge call
AVG_TOKENS_PER_JUDGE_CALL = 400
```

### How to Customize

1. **Determine judge model cost:**
   - Check your LLM provider's pricing
   - Example: GPT-4: $0.03/1K input tokens

2. **Measure average tokens:**
   - Run sample batch with token counting
   - Use: `len(encoding.encode(prompt + output))`
   - Average over representative samples

3. **Update constants:**
   ```python
   # In test_eigentrace_cost_shaver.py
   JUDGE_COST_PER_1K_TOKENS_USD = 0.03  # Your actual cost
   AVG_TOKENS_PER_JUDGE_CALL = 600      # Your measured average
   ```

4. **Re-run tests:**
   ```bash
   python scripts/run_avp_tests.py
   ```

## Real-World Savings Example

### Scenario: Documentation QA System

**Setup:**
- 10,000 Q&A pairs/day
- GPT-4 judge: $0.03/1K tokens
- Average 500 tokens/call
- Eigentrace threshold: 0.4
- 70% pass gating

**Without Eigentrace:**
- 10,000 judge calls/day
- 5,000,000 tokens/day
- Cost: $150/day
- **Monthly: $4,500**

**With Eigentrace:**
- 3,000 judge calls/day (70% reduction)
- 1,500,000 tokens/day
- Cost: $45/day
- **Monthly: $1,350**

**Savings:**
- $3,150/month
- $37,800/year

### Verification

Run your own pilot:
1. Collect sample of outputs
2. Score with Eigentrace
3. Count how many would escalate
4. Calculate savings with your costs

## Privacy Savings

Beyond dollars, Eigentrace saves privacy:

**Traditional:**
- All data → External API
- Privacy risk for every item
- Compliance overhead

**With Eigentrace:**
- Most data → Local evaluation
- Only flagged items → External (optional)
- Reduced compliance surface

### Quantifying Privacy

- **Data exposure reduction**: 70% of items never leave your infrastructure
- **Audit simplification**: Fewer external API calls to log and audit
- **Compliance**: Easier to meet HIPAA, GDPR, SOC2 requirements

## Optimization Tips

### 1. Tune Threshold

Run threshold sweep:
```bash
pytest tests/test_eigentrace_cost_shaver.py::TestEigentraceCostShaver::test_threshold_tuning -v
```

Find optimal balance of recall vs reduction.

### 2. Batch Processing

Process items in batches:
- Amortize overhead
- Parallel evaluation
- Better throughput

### 3. Cache Results

For repeated evaluations:
- Cache Eigentrace scores
- Avoid redundant computation
- Even faster, even cheaper

### 4. Hybrid Strategies

Different thresholds by use case:
- Critical outputs: Lower threshold (more conservative)
- Routine outputs: Higher threshold (more savings)

## Reporting Savings

### For Stakeholders

Report in business terms:
- "Reduced judge API costs by 64%"
- "Saved $X per month"
- "Evaluated 70% of items locally for privacy"

### For Engineers

Report technical metrics:
- "Reduced judge calls from N to M"
- "Maintained 85% recall of bad items"
- "Average latency: 5ms vs 1500ms"

### For Security/Compliance

Report privacy improvements:
- "70% of data never leaves infrastructure"
- "Local evaluation with no API key dependency"
- "Reduced external data exposure by X items/day"

## Caveats and Limitations

### 1. Initial Setup Cost

Implementing Eigentrace requires:
- Integration work
- Threshold tuning
- Testing and validation

One-time cost, ongoing savings.

### 2. Maintenance

As data distribution changes:
- Re-evaluate thresholds
- Update fixtures
- Monitor performance

### 3. Not a Silver Bullet

Eigentrace is one tool:
- Part of larger quality pipeline
- Complements other approaches
- Specific to structural issues

## Conclusion

Eigentrace saves:
- ✅ Judge API calls (measured)
- ✅ Token costs (estimated)
- ✅ Latency (measured)
- ✅ Privacy exposure (measured)

Eigentrace doesn't claim:
- ❌ Perfect accuracy
- ❌ Content-level judgments
- ❌ Exact dollar amounts
- ❌ Complete solution

Use Eigentrace as an intelligent **pre-filter** in a layered quality system.

## Getting Started

1. Run AVP tests to see savings on fixtures
2. Customize constants for your setup
3. Run pilot with real data
4. Measure actual savings
5. Scale to production

Questions? Open an issue on GitHub.
