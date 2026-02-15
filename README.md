# EigenTrace

EigenTrace measures whether the model’s output is internally stable — not whether it agrees with another model. It's a deterministic structural auditor for LLM outputs.
It replaces LLM-as-a-Judge for first-pass filtering.
It runs in milliseconds, costs nothing per token, and works offline.
It measures entropy bursts and phase instability instead of asking another model for opinions.

It detects:
Confident nonsense (smooth but semantically wrong)
High-entropy hallucination bursts
Structural collapse
Overfit jitter
Degenerate loops

It does not judge content.
It measures coherence stability.
total = MSE(time) 
      + α · spectral_error 
      + β · phase_error
Time = “does it resemble normal language flow?”
Spectral = “does it explode in entropy?”
Phase = “does structure stay coherent?”

1. Convert text into a numeric waveform. 
2. Compare waveform to stable structural priors.
3. Penalize noise spikes (entropy bursts).
4. Penalize phase drift (structural inconsistency).
5. Output a confidence score.

LLM-as-a-Judge:
Requires second model
Doubles inference cost
Adds network latency
Adds carbon cost

EigenTrace:
Pure Python
NumPy FFT
O(n log n)
Milliseconds
Deterministic
Offline

Privacy:
LLM peer review means:
User data → external API → second model → stored somewhere

EigenTrace:
User data → local math → done

Useful For:
Healthcare
Finance
Defense
Enterprise compliance
Censorship-averse users

## Features

- **JSONL Input/Output**: Process text data in standard JSONL format
- **Structural Analysis**: Deterministic, offline text analysis (default mode)
- **Instability Scoring**: Batch FFT spectral+phase parity using LogosLossV4
- **Threshold Gating**: Filter outputs based on configurable instability thresholds
- **Summary Statistics**: Generate comprehensive summary reports
- **CLI Interface**: Easy-to-use command-line interface
- **Framework Integrations**: Easy integration with LangChain, LlamaIndex, and ZenML

## Installation

### From Source

```bash
git clone https://github.com/keepitcheesy/Eigen-trace.git
cd Eigen-trace
pip install -e .
```

### With Framework Integrations

```bash
# Install with all integrations
pip install -e ".[all]"

# Or install specific integrations
pip install -e ".[langchain]"  # LangChain integration
pip install -e ".[llamaindex]"  # LlamaIndex integration
pip install -e ".[zenml]"       # ZenML integration
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0

Optional dependencies for integrations:
- LangChain >= 0.0.200
- LlamaIndex >= 0.9.0
- ZenML >= 0.40.0

## Usage

## Debugging & Example Files

This repo includes a small example input file and debug helpers:

- `outputs.jsonl` — sample input data (uses `prediction` and `truth`)
- `debug_signal.py` — prints basic stats on the encoded signal to help explain zero/near‑zero scores
- `scripts/escalate.py` — optional helper to filter results for escalation to an external judge

### Example input format

```jsonl
{"id":"1","prediction":"Photosynthesis is how plants convert light into energy.","truth":"Photosynthesis converts light energy into chemical energy in plants."}
{"id":"2","prediction":"Photosynthesis is when plants teleport nutrients across space.","truth":"Photosynthesis converts light energy into chemical energy in plants."}
```

### Debug signal stats
```bash
python debug_signal.py
```
If the standard deviation is near zero, the signal is flat and instability scores will be near zero.

### Run evaluation
```bash
python -m logoslabs.cli outputs.jsonl results.jsonl --threshold 0.05 --summary
```

The CLI writes a `results.jsonl` file containing the original fields plus `instability_score` and `passed_threshold`:

```jsonl
{"id":"1","prediction":"...","truth":"...","instability_score":0.0466,"passed_threshold":true}
{"id":"2","prediction":"...","truth":"...","instability_score":0.0548,"passed_threshold":false}
```

### Optional GPT-4 Escalation

For items that fail the threshold or require additional review, use `scripts/escalate.py` to generate a JSONL queue for your external judge pipeline (e.g., GPT-4, human review):

**Default behavior** (uses `passed_threshold` field):
```bash
python scripts/escalate.py results.jsonl > escalate_queue.jsonl
```

**Custom threshold** (escalates if `instability_score > threshold`):
```bash
python scripts/escalate.py results.jsonl --threshold 0.048 > escalate_queue.jsonl
```

This outputs JSONL lines to stdout for only the items that should be escalated. The escalation step is **completely optional**—EigenTrace provides offline filtering, and you can integrate escalation into your existing judge pipeline as needed.
### Command-Line Interface

Basic usage:

```bash
logoslabs input.jsonl output.jsonl
```

With options:

```bash
logoslabs input.jsonl output.jsonl \
  --threshold 1.0 \
  --grace-coeff 0.5 \
  --phase-weight 0.1 \
  --summary
```

### Input Format

Input JSONL file should contain objects with `prediction` and `truth` fields:

```jsonl
{"prediction": "This is a predicted text", "truth": "This is the ground truth"}
{"prediction": "Another prediction", "truth": "Another truth"}
```

### Output Format

Output JSONL includes the original fields plus scoring information:

```jsonl
{"prediction": "...", "truth": "...", "instability_score": 0.234, "passed_threshold": true}
```

### CLI Options

- `input`: Input JSONL file path (required)
- `output`: Output JSONL file path (required)
- `--threshold`: Instability score threshold (default: 1.0)
- `--grace-coeff`: LogosLoss spectral weight coefficient (default: 0.5)
- `--phase-weight`: LogosLoss phase weight coefficient (default: 0.1)
- `--max-length`: Maximum sequence length for encoding (default: 512)
- `--summary`: Print summary statistics to stdout
- `--no-deterministic`: Disable deterministic behavior
- `--enable-belief-streams`: Enable belief streams (experimental, not yet implemented)
- `--heuristics`: Enable token-level heuristic feature computation (opt-in)
- `--tokenizer`: Tokenizer to use for heuristics - `whitespace` (default), `tiktoken`, `sentencepiece`

### Python API

```python
from logoslabs import LogosLossV4
from logoslabs.avp import AVPProcessor

# Initialize processor
processor = AVPProcessor(
    threshold=1.0,
    grace_coeff=0.5,
    phase_weight=0.1,
)

# Process items
items = [
    {"prediction": "test", "truth": "truth"},
]
results = processor.process_batch(items)

# Get summary
summary = processor.get_summary(results)
print(summary)
```

### LogosLossV4 Direct Usage

```python
import torch
from logoslabs import LogosLossV4

# Initialize loss function
loss_fn = LogosLossV4(
    grace_coeff=0.5,
    phase_weight=0.1,
)

# Compute loss
pred = torch.randn(4, 2, 128)  # (batch, channels, sequence)
truth = torch.randn(4, 2, 128)
loss = loss_fn(pred, truth)
```

## Token-Level Heuristics (Opt-In)

EigenTrace now supports optional token-level heuristic features that can provide additional signals about text quality. These features are **opt-in** and disabled by default to maintain backward compatibility.

### Heuristic Features

When `--heuristics` is enabled, the following additional fields are computed and added to the output:

1. **`repetition_score`**: Detects local n-gram repetition patterns using sliding windows (size 5). Measures how often the same sequence of tokens appears multiple times. Higher values indicate more repetition.

2. **`rolling_var_score`**: Computes the variance of token lengths across rolling windows (size 32). Measures diversity in token characteristics. The raw variance is reported (typically 0-100 for normal text).

3. **`ttr_score`**: Type-Token Ratio within windows (size 50). Measures vocabulary diversity - the ratio of unique tokens to total tokens. The score is inverted so that higher values indicate more repetition (lower diversity).

4. **`heuristics_score`**: A weighted aggregate of all heuristics (40% repetition + 30% variance + 30% TTR). Provides a single score combining all heuristic signals.

**Important**: The `passed_threshold` field remains based **solely on LogosLossV4** instability score. Heuristics are emitted as additional information but do not affect the default pass/fail gating.

### Tokenizer Support

Heuristics require tokenization. Three tokenizers are supported:

- **`whitespace`** (default): No dependencies, splits on whitespace
- **`tiktoken`**: Requires `pip install tiktoken` - uses GPT-style tokenization
- **`sentencepiece`**: Requires `pip install sentencepiece` - uses SentencePiece tokenization

If you select `tiktoken` or `sentencepiece` without installing the package, you'll receive a clear error message with installation instructions.

### Usage Examples

**Basic usage with heuristics (whitespace tokenizer):**
```bash
logoslabs input.jsonl output.jsonl --heuristics
```

**With custom tokenizer:**
```bash
# Using tiktoken (requires: pip install tiktoken)
logoslabs input.jsonl output.jsonl --heuristics --tokenizer tiktoken

# Using sentencepiece (requires: pip install sentencepiece)
logoslabs input.jsonl output.jsonl --heuristics --tokenizer sentencepiece
```

**With all options:**
```bash
logoslabs input.jsonl output.jsonl \
  --threshold 0.5 \
  --heuristics \
  --tokenizer whitespace \
  --summary
```

### Example Output with Heuristics

Without `--heuristics` (default):
```jsonl
{"id":"1","prediction":"...","truth":"...","instability_score":0.0466,"passed_threshold":true}
```

With `--heuristics`:
```jsonl
{"id":"1","prediction":"...","truth":"...","instability_score":0.0466,"passed_threshold":true,"repetition_score":0.0,"rolling_var_score":12.0,"ttr_score":0.0,"heuristics_score":0.264}
```

### Python API with Heuristics

```python
from logoslabs.avp import AVPProcessor
from logoslabs.tokenizers import get_tokenizer

# Initialize with heuristics enabled
tokenizer = get_tokenizer("whitespace")
processor = AVPProcessor(
    threshold=1.0,
    enable_heuristics=True,
    tokenizer=tokenizer,
)

# Process items - output will include heuristic fields
items = [
    {"prediction": "test prediction text", "truth": "ground truth"},
]
results = processor.process_batch(items)

# Results include: repetition_score, rolling_var_score, ttr_score, heuristics_score
print(results[0])
```

## Framework Integrations

LogosLabs integrates seamlessly with popular ML and LLM frameworks.

### LangChain Integration

```python
from langchain.llms import OpenAI
from logoslabs.integrations.langchain import LogosLabsChain

# Create chain with built-in validation
chain = LogosLabsChain.from_llm(
    llm=OpenAI(),
    threshold=0.5,
)

result = chain.run("Generate some text")
print(f"Validated: {result.get('validated', False)}")
```

### LlamaIndex Integration

```python
from llama_index import VectorStoreIndex
from logoslabs.integrations.llamaindex import LogosLabsPostprocessor

# Add quality filtering to query engine
query_engine = index.as_query_engine(
    node_postprocessors=[
        LogosLabsPostprocessor(threshold=0.5, filter_failures=True)
    ]
)

response = query_engine.query("What is...?")
```

### ZenML Integration

```python
from zenml import pipeline
from logoslabs.integrations.zenml import logoslabs_filter_step

@pipeline
def validation_pipeline():
    data = load_data()
    filtered, summary = logoslabs_filter_step(data, threshold=0.5)
    return filtered

validation_pipeline().run()
```

**See [INTEGRATIONS.md](INTEGRATIONS.md) for detailed integration guides and examples.**

## Benchmark Harness

A comprehensive benchmark harness is provided to evaluate LogosLoss performance against baseline losses across multiple tasks.

### Quick Start

```bash
# Install dependencies
make install

# Run full benchmark suite
make all
```

This will:
1. Train models with LogosLoss, MSE, and Huber losses
2. Evaluate on image denoising and time series tasks
3. Test compression robustness (quantization, pruning)
4. Run inspection tools (spectral analysis, gradient analysis)
5. Generate a comprehensive report

### Benchmark Features

- **Multiple Loss Functions**: LogosLoss, MSE, Huber
- **Tasks**: Image denoising, time series forecasting
- **Models**: U-Net, LSTM, MLP
- **Compression Tests**: Quantization (8-bit), Pruning (20%, 40%)
- **Inspection Tools**:
  - Residual spectrum analysis
  - Per-frequency penalty attribution
  - Gradient spectrum comparison
- **Reproducibility**: Deterministic seeding, config snapshots
- **Artifacts**: Metrics (JSONL), summaries (CSV), checkpoints, plots

See [bench/README.md](bench/README.md) for detailed documentation.

### Configuration

Edit `bench/configs/suite.yaml` to customize:
- Random seeds
- Loss function parameters
- Task configurations
- Training hyperparameters
- Compression settings

### Output Structure

```
results/
├── metrics.jsonl           # Per-run detailed metrics
├── summary.csv             # Summary table
├── checkpoints/            # Trained model weights
├── inspection/             # Analysis plots and summaries
└── reproducibility.json    # Environment info

reports/
└── report_latest.md        # Auto-generated report
```

## Development

### Running Tests

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run all tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=logoslabs --cov-report=html
```

Run benchmark tests (requires `RUN_BENCHMARKS=1`):

```bash
RUN_BENCHMARKS=1 pytest tests/test_benchmark.py -v
```

### Project Structure

```
Eigen-trace/
├── src/
│   └── logoslabs/
│       ├── __init__.py       # Package initialization
│       ├── logosloss.py      # LogosLossV4 implementation
│       ├── avp.py            # AVP processor
│       └── cli.py            # Command-line interface
├── tests/
│   ├── test_logosloss.py     # LogosLoss tests
│   ├── test_avp.py           # AVP processor tests
│   ├── test_cli.py           # CLI tests
│   └── test_benchmark.py     # Benchmark tests
├── pyproject.toml            # Package configuration
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## LogosLossV4 Algorithm

LogosLossV4 combines three components:

1. **Time-Domain (Material)**: MSE between predicted and truth signals
2. **Spectral Magnitude**: Log-magnitude difference weighted by truth energy and frequency
3. **Phase**: Wrap-safe phase error (1 - cos(angle)) weighted by dominant bins

Formula:
```
total = material + (grace_coeff × spectral) + (phase_weight × phase)
```

### Parameters

- `grace_coeff` (default: 0.5): Weight for spectral component
- `phase_weight` (default: 0.1): Weight for phase component
- `eps` (default: 1e-8): Numerical stability epsilon
- `freq_power` (default: 1.0): Frequency weighting power
- `mercy_power` (default: 1.0): Phase error weighting power (focuses on dominant bins)
- `presence_power` (default: 1.0): Spectral energy weighting power
- `reduction` (default: "mean"): Output reduction ("mean" or "none")

## Design Philosophy

- **Deterministic**: Same inputs always produce same outputs (default)
- **Offline**: No external API calls or network dependencies
- **Structural-Only**: Default mode uses only structural text analysis
- **Minimal Dependencies**: Only requires PyTorch and NumPy

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use LogosLabs in your research, please cite:

```bibtex
@software{logoslabs2026,
  title={LogosLabs: LogosLoss-based AVP for LLM Output Filtering},
  author={keepitcheesy},
  year={2026},
  url={https://github.com/keepitcheesy/Eigen-trace}
}
```
