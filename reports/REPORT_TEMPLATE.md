# Benchmark Report Template

## Executive Summary

[Provide a 2-3 sentence summary of the benchmark results]

## Setup

### Configuration
- **Seeds tested**: [list seeds]
- **Loss functions**: [list loss functions]
- **Tasks**: [list tasks]
- **Models**: [list models]

### Reproducibility
- PyTorch version: [version]
- NumPy version: [version]
- CUDA available: [yes/no]
- Random seeds: [list]

## Results

### Overall Performance

[Table comparing losses across tasks]

| Loss | Task | Metric 1 | Metric 2 | Training Time |
|------|------|----------|----------|---------------|
| ... | ... | ... | ... | ... |

### Per-Task Analysis

#### Task 1: [Task Name]

**Metrics**: [metric definitions]

**Results**:
- LogosLoss: [results with std]
- MSE: [results with std]
- Huber: [results with std]

**Key Findings**:
- [Finding 1]
- [Finding 2]

#### Task 2: [Task Name]

**Metrics**: [metric definitions]

**Results**:
- LogosLoss: [results with std]
- MSE: [results with std]
- Huber: [results with std]

**Key Findings**:
- [Finding 1]
- [Finding 2]

## Compression Robustness

### Quantization Tests

[Results for 8-bit quantization]

| Loss | Task | Baseline | Quantized | Drop |
|------|------|----------|-----------|------|
| ... | ... | ... | ... | ... |

### Pruning Tests

[Results for 20% and 40% pruning]

| Loss | Task | Baseline | 20% Pruned | 40% Pruned |
|------|------|----------|------------|------------|
| ... | ... | ... | ... | ... |

**Key Findings**:
- [Finding about compression robustness]

## Inspection Findings

### Residual Spectrum Analysis

[Summary of residual spectrum comparisons]

**Key Observations**:
- [Observation 1]
- [Observation 2]

### Frequency Attribution (LogosLoss)

[Summary of frequency penalty attribution]

**Key Observations**:
- [Which frequencies contribute most]
- [Spectral vs phase contributions]

### Gradient Spectrum Analysis

[Summary of gradient frequency content]

**Key Observations**:
- [How different losses shape gradients]
- [High-frequency vs low-frequency emphasis]

## Conclusions

### Performance Summary
- [Main conclusion 1]
- [Main conclusion 2]

### When to Use Each Loss
- **LogosLoss**: [recommended use cases]
- **MSE**: [recommended use cases]
- **Huber**: [recommended use cases]

### Statistical Significance
[Note about variance across seeds and statistical significance]

## Limitations

### Experimental Limitations
- [Limitation 1]
- [Limitation 2]

### Computational Constraints
- Number of epochs: [value and rationale]
- Model sizes: [small for fast iteration]

### Scope
- [What was not tested]
- [Future work suggestions]

## Artifacts

### Data Files
- Metrics: `results/metrics.jsonl`
- Summary: `results/summary.csv`
- Checkpoints: `results/checkpoints/`
- Inspection: `results/inspection/`

### Reproducibility
- Configuration: `bench/configs/suite.yaml`
- Reproducibility info: `results/reproducibility.json`

## Appendix

### Complete Results

[Include full result tables if needed]

### Figures

[Reference to plots in results/inspection/]
