# LogosLabs Integration Quick Start

This guide shows how to quickly get started with LogosLabs integrations for popular ML/LLM frameworks.

## TL;DR

```bash
# Install with integrations
pip install logoslabs[all]

# Use with LangChain
from logoslabs.integrations.langchain import LogosLabsChain
chain = LogosLabsChain.from_llm(llm, threshold=0.5)

# Use with LlamaIndex
from logoslabs.integrations.llamaindex import LogosLabsPostprocessor
query_engine = index.as_query_engine(
    node_postprocessors=[LogosLabsPostprocessor(threshold=0.5)]
)

# Use with ZenML
from logoslabs.integrations.zenml import logoslabs_filter_step
filtered, summary = logoslabs_filter_step(data, threshold=0.5)
```

## Why Use LogosLabs with These Frameworks?

### LangChain

**Problem**: LLMs sometimes generate low-quality, inconsistent, or hallucinated outputs.

**Solution**: LogosLabs provides automatic validation and filtering of LLM outputs based on structural quality metrics.

**Benefits**:
- Catch low-quality generations before they reach production
- Reduce hallucinations by filtering unreliable outputs
- Improve overall system reliability
- Track quality metrics over time

### LlamaIndex

**Problem**: RAG systems retrieve and present nodes that may be low-quality or irrelevant.

**Solution**: LogosLabs postprocessors filter retrieved nodes based on quality scores.

**Benefits**:
- Improve RAG response quality
- Reduce noise in retrieved results
- Rank nodes by structural quality
- Enrich node metadata with quality scores

### ZenML

**Problem**: ML pipelines need quality gates for data filtering and validation.

**Solution**: LogosLabs provides reusable pipeline steps for data quality filtering.

**Benefits**:
- Standardize quality filtering across pipelines
- Track quality metrics in ML experiments
- Ensure training data quality
- Reproducible validation logic

## Installation Options

### All Integrations

```bash
pip install logoslabs[all]
```

### Individual Integrations

```bash
# LangChain only
pip install logoslabs[langchain]

# LlamaIndex only  
pip install logoslabs[llamaindex]

# ZenML only
pip install logoslabs[zenml]
```

### Core Package Only

```bash
pip install logoslabs
```

## Quick Examples

### LangChain: Validate LLM Outputs

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from logoslabs.integrations.langchain import LogosLabsOutputParser

# Create LLM and prompt
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer this question: {question}"
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Create validator
validator = LogosLabsOutputParser(
    threshold=0.5,
    raise_on_fail=False
)

# Generate and validate
output = chain.run(question="What is AI?")
result = validator.parse(output)

if result['passed_threshold']:
    print("✓ High-quality output")
    print(output)
else:
    print("✗ Low-quality output, regenerating...")
    # Regenerate or use fallback
```

### LangChain: Automated Validation Chain

```python
from logoslabs.integrations.langchain import LogosLabsChain

# Create self-validating chain
chain = LogosLabsChain.from_llm(
    llm=OpenAI(),
    threshold=0.5
)

# Outputs are automatically validated
result = chain.run("Explain quantum computing")

# Check validation results
print(f"Validated: {result.get('validated', False)}")
print(f"Score: {result.get('instability_score', 'N/A')}")
```

### LlamaIndex: Filter Retrieved Nodes

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from logoslabs.integrations.llamaindex import LogosLabsPostprocessor

# Load documents
documents = SimpleDirectoryReader('docs/').load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine with quality filtering
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[
        LogosLabsPostprocessor(
            threshold=0.5,
            filter_failures=True  # Remove low-quality nodes
        )
    ]
)

# Query - only high-quality nodes will be used
response = query_engine.query("What is machine learning?")

# Check node quality
for node in response.source_nodes:
    score = node.node.metadata.get('instability_score', 'N/A')
    print(f"Node quality score: {score}")
```

### ZenML: Pipeline with Quality Filtering

```python
from zenml import pipeline, step
from logoslabs.integrations.zenml import LogosLabsFilterStep

@step
def load_data():
    return [
        {"prediction": "Good quality text", "truth": "Reference"},
        {"prediction": "Poor quality", "truth": "Reference"},
    ]

@step
def process_data(data):
    print(f"Processing {len(data)} items")
    return data

# Create filter step
filter_step = LogosLabsFilterStep(threshold=0.5)

@pipeline
def ml_pipeline():
    raw_data = load_data()
    filtered_data, summary = filter_step(raw_data)
    result = process_data(filtered_data)
    return result

# Run pipeline
ml_pipeline().run()
```

## Configuration

### Threshold Selection

The threshold parameter controls filtering sensitivity:

- **threshold=0.1**: Very strict (only very high-quality outputs pass)
- **threshold=0.5**: Moderate (balanced filtering)
- **threshold=1.0**: Permissive (default, most outputs pass)
- **threshold=2.0**: Very permissive (almost everything passes)

Start with the default (1.0) and adjust based on your use case.

### Advanced Parameters

```python
# Fine-tune LogosLoss behavior
processor = AVPProcessor(
    threshold=0.5,
    grace_coeff=0.1,      # Spectral component weight
    phase_weight=0.01,     # Phase component weight
    max_length=512,       # Maximum text length
    deterministic=True,   # Reproducible results
)
```

## Common Patterns

### Pattern 1: Retry with Fallback

```python
from logoslabs.integrations.langchain import LogosLabsOutputParser

parser = LogosLabsOutputParser(threshold=0.5)

max_retries = 3
for attempt in range(max_retries):
    output = llm.generate(prompt)
    result = parser.parse(output)
    
    if result['passed_threshold']:
        break
else:
    # All retries failed, use fallback
    output = get_fallback_response()
```

### Pattern 2: Quality-Based Ranking

```python
from logoslabs.avp import AVPProcessor

processor = AVPProcessor(threshold=1.0)  # Don't filter

# Score all outputs
outputs = [{"prediction": text, "truth": ref} for text in candidates]
scored = processor.process_batch(outputs)

# Sort by quality
sorted_outputs = sorted(
    scored,
    key=lambda x: x['instability_score']
)

# Use best output
best_output = sorted_outputs[0]
```

### Pattern 3: Pipeline Quality Gate

```python
from logoslabs.integrations.zenml import logoslabs_filter_step

@pipeline
def training_pipeline():
    # Extract raw data
    raw_data = extract_data()
    
    # Quality gate
    clean_data, summary = logoslabs_filter_step(
        raw_data,
        threshold=0.7
    )
    
    # Only train on high-quality data
    model = train_model(clean_data)
    
    return model
```

## Best Practices

1. **Start Conservative**: Begin with default threshold (1.0) and adjust based on results
2. **Monitor Metrics**: Track pass rates and adjust thresholds accordingly
3. **Use Reference Text**: Provide meaningful reference text when available
4. **Batch Processing**: Process multiple items at once for better performance
5. **Log Quality Scores**: Keep track of quality metrics for debugging and improvement

## Troubleshooting

### Everything is being filtered out

- **Increase** the threshold value
- Check that reference text is meaningful
- Verify input data quality

### Nothing is being filtered

- **Decrease** the threshold value
- Ensure you're providing appropriate reference text
- Check if inputs are too similar

### Import errors

Make sure you've installed the integration:

```bash
pip install logoslabs[langchain]  # or llamaindex, zenml, all
```

## Next Steps

- Read the full [INTEGRATIONS.md](INTEGRATIONS.md) guide
- Check out [examples/](examples/) directory for complete working examples
- Review the main [README.md](README.md) for core functionality
- Experiment with different threshold values for your use case

## Support

- **Issues**: https://github.com/keepitcheesy/Eigen-trace/issues
- **Documentation**: https://github.com/keepitcheesy/Eigen-trace
- **Examples**: See `examples/` directory in the repository

## License

MIT License - see [LICENSE](LICENSE) file for details.
