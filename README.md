# EigenTrace

Adversarial Validation Pipeline (AVP) for pre-filtering LLM outputs. It provides deterministic, offline structural analysis of text using FFT spectral and phase parity metrics.

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
- `scripts/escalate.py` — optional helper to filter results for escalation to GPT-4 or an external judge

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
