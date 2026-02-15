# Benchmark Harness for LogosLoss (Eigentrace)

This benchmark harness provides a comprehensive evaluation framework for comparing LogosLoss against baseline loss functions across multiple tasks.

## Quick Start

```bash
# Install dependencies
make install

# Run complete benchmark suite
make all

# Or run steps individually:
make suite      # Run benchmarks
make inspect    # Run inspection tools
make report     # Generate report
```

## Structure

```
bench/
├── configs/
│   └── suite.yaml          # Benchmark configuration
├── run_suite.py            # Main benchmark runner
├── repro.py                # Reproducibility utilities
└── measure.py              # Performance measurement

losses/
├── logosloss.py            # LogosLoss implementation
└── baselines.py            # MSE and Huber losses

tasks/
├── image_denoise.py        # Image denoising task
└── time_series.py          # Time series forecasting

models/
├── unet.py                 # Small U-Net
├── mlp.py                  # Small MLP
└── lstm.py                 # Small LSTM

inspect/
├── residual_spectrum.py    # Residual analysis
├── frequency_attribution.py # Per-frequency penalties
├── gradient_spectrum.py    # Gradient analysis
└── run_inspection.py       # Inspection runner

reports/
├── REPORT_TEMPLATE.md      # Report template
├── generate_report.py      # Auto-report generation
└── report_latest.md        # Generated report (gitignored)

results/                    # All results (gitignored)
├── metrics.jsonl           # Per-run metrics
├── summary.csv             # Summary table
├── checkpoints/            # Model checkpoints
└── inspect/                # Inspection outputs
```

## Configuration

Edit `bench/configs/suite.yaml` to configure:

- **Seeds**: Random seeds for reproducibility (default: [42, 123, 456])
- **Losses**: Loss functions to compare (logosloss, mse, huber)
- **Tasks**: Tasks to evaluate (image_denoise, time_series)
- **Models**: Model architectures per task
- **Training**: Epochs, batch size, learning rate
- **Compression**: Quantization and pruning settings

## Loss Functions

### LogosLoss
Multi-component loss combining:
- **Material loss**: Time-domain MSE
- **Spectral loss**: Log-magnitude difference, energy-weighted
- **Spectral noise**: Wrap-safe phase error

Parameters:
- `grace_coeff` (default: 0.5): Spectral weight
- `phase_weight` (default: 0.1): Phase weight
- `freq_power` (default: 1.0): Frequency emphasis
- `mercy_power` (default: 1.0): Phase weighting
- `presence_power` (default: 1.0): Spectral energy weighting

### Baseline Losses
- **MSE**: Standard L2 loss
- **Huber**: Robust loss with L1/L2 transition

## Tasks

### Image Denoising
- Synthetic images with various patterns
- Noise types: Gaussian, salt-pepper, structured
- Metrics: PSNR, SSIM
- Model: Small U-Net (16 base channels)

### Time Series Forecasting
- Synthetic signals with controllable spectrum
- Multiple frequency components
- Metrics: MAE, MSE, spectral error
- Model: Small LSTM (32 hidden, 2 layers)

## Compression Tests

### Quantization
Simulates 8-bit weight quantization to test robustness.

### Pruning
Tests magnitude-based pruning at 20% and 40% sparsity.

For each compression method, the harness logs:
- Post-compression metrics
- Metric drops from baseline
- Comparison across losses

## Inspection Tools

### Residual Spectrum Analysis
Analyzes frequency content of prediction errors.
- Plots residual magnitude spectrum
- Computes relative spectral error
- Compares residual power across losses

### Frequency Attribution (LogosLoss only)
Shows which frequencies contribute most to the loss.
- Per-frequency spectral penalty
- Per-frequency phase penalty
- Combined attribution

### Gradient Spectrum Analysis
Analyzes gradient frequency content during training.
- Gradient magnitude spectrum
- High-frequency vs low-frequency ratio
- Gradient entropy (diversity measure)

All inspection results include:
- Plots saved to `results/inspect/`
- JSON summaries with numeric statistics

## Output Files

### results/metrics.jsonl
One JSON object per experiment run with:
- Timestamp, seed, loss, task, model
- Training time, final loss, loss history
- Task-specific metrics (PSNR, SSIM, MAE, etc.)
- FLOPs and energy estimates
- Compression test results
- Checkpoint path

### results/summary.csv
Flattened table for easy analysis:
- All runs in rows
- Columns: seed, loss, task, metrics, timing

### results/checkpoints/
Model checkpoints named:
```
{task}_{loss}_seed{seed}.pt
```

Each checkpoint contains:
- `model_state_dict`: Trained weights
- `config`: Full experiment configuration

### results/inspect/
Inspection outputs:
- `residual_spectrum_{loss}.png`
- `frequency_attribution.png` (LogosLoss only)
- `gradient_spectrum_{loss}.png`
- JSON summaries with statistics

### reports/report_latest.md
Auto-generated markdown report with:
- Executive summary
- Setup and reproducibility info
- Results by task and loss
- Compression robustness analysis
- Inspection findings
- Conclusions and limitations

## Reproducibility

The harness ensures reproducibility through:

1. **Deterministic seeding**: `bench.repro.set_seed()`
2. **Config snapshots**: Saved with each checkpoint
3. **Environment recording**: `results/reproducibility.json`
4. **No cherry-picking**: All seeds reported

To reproduce results:
```bash
make suite  # Uses seeds from config
```

## Customization

### Add a New Loss
1. Implement in `losses/` following the API
2. Add to `losses/__init__.py`
3. Update `bench/run_suite.py` in `get_loss_function()`
4. Add to `bench/configs/suite.yaml`

### Add a New Task
1. Implement in `tasks/` with:
   - `generate_data()` → (inputs, targets)
   - `compute_metrics(pred, truth)` → dict
2. Add to `tasks/__init__.py`
3. Update `bench/run_suite.py` in `get_task()`
4. Add to `bench/configs/suite.yaml`

### Add a New Model
1. Implement in `models/` as `nn.Module`
2. Add to `models/__init__.py`
3. Update `bench/run_suite.py` in `get_model()`
4. Assign to tasks in config

## Performance Tips

### Reduce Compute
- Decrease epochs in config
- Use smaller models (reduce base channels, hidden size)
- Reduce num_samples per task
- Use fewer seeds

### Scale Up
- Increase epochs for convergence
- Use larger models for capacity
- Add more seeds for statistical power
- Add real-world tasks and data

## Guardrails

The harness follows strict guidelines:

1. **No cherry-picking**: Report all seeds
2. **Clear underperformance**: State if a loss underperforms
3. **Small compute**: Keep runs fast for iteration
4. **Reproducible**: Deterministic seeds and config logging

## Dependencies

See `requirements.txt`:
- torch >= 2.0.0
- numpy >= 1.20.0
- pyyaml >= 6.0
- matplotlib >= 3.5.0
- pytest >= 7.0.0 (for tests)

## Testing

```bash
make test
```

Tests cover:
- Loss function correctness
- Task data generation
- Model forward passes
- Benchmark pipeline components

## License

MIT License - see main repository LICENSE file.
