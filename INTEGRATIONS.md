# LogosLabs Framework Integrations

This guide explains how to integrate LogosLabs with popular ML and LLM frameworks.

## Table of Contents

- [LangChain Integration](#langchain-integration)
- [LlamaIndex Integration](#llamaindex-integration)
- [ZenML Integration](#zenml-integration)
- [Installation](#installation)

## LangChain Integration

LogosLabs provides seamless integration with LangChain for validating and filtering LLM outputs.

### Installation

```bash
pip install logoslabs[langchain]
# or
pip install logoslabs langchain
```

### Components

#### LogosLabsOutputParser

A LangChain output parser that validates LLM outputs using LogosLoss instability scoring.

```python
from langchain.llms import OpenAI
from logoslabs.integrations.langchain import LogosLabsOutputParser

# Create parser
parser = LogosLabsOutputParser(
    threshold=0.5,
    reference_text="Expected output pattern",
    raise_on_fail=False
)

# Parse LLM output
result = parser.parse(llm_output)
print(f"Score: {result['instability_score']}")
print(f"Valid: {result['passed_threshold']}")
```

#### LogosLabsChain

A custom LangChain that automatically validates outputs.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from logoslabs.integrations.langchain import LogosLabsChain

# Create validated chain
chain = LogosLabsChain.from_llm(
    llm=OpenAI(),
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic}"
    ),
    threshold=0.8,
    reference_text="Provide a clear explanation"
)

# Run with automatic validation
result = chain.run(topic="machine learning")
print(f"Output: {result}")
```

### Use Cases

- **Quality Filtering**: Remove low-quality LLM outputs before downstream processing
- **Response Validation**: Ensure generated text meets quality thresholds
- **Batch Processing**: Validate multiple outputs efficiently
- **Prompt Engineering**: Iterate on prompts using quality metrics

### Example

See `examples/example_langchain.py` for complete working examples.

## LlamaIndex Integration

LogosLabs integrates with LlamaIndex to filter and validate retrieved nodes.

### Installation

```bash
pip install logoslabs[llamaindex]
# or
pip install logoslabs llama-index
```

### Components

#### LogosLabsPostprocessor

A LlamaIndex postprocessor that validates and filters retrieved nodes.

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from logoslabs.integrations.llamaindex import LogosLabsPostprocessor

# Load documents
documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine with validation
query_engine = index.as_query_engine(
    node_postprocessors=[
        LogosLabsPostprocessor(
            threshold=0.5,
            filter_failures=True  # Remove low-quality nodes
        )
    ]
)

# Query with automatic filtering
response = query_engine.query("What is machine learning?")
```

#### LogosLabsNodeFilter

Standalone node filter for custom pipelines.

```python
from logoslabs.integrations.llamaindex import LogosLabsNodeFilter

# Create filter
filter = LogosLabsNodeFilter(threshold=0.8)

# Filter nodes
filtered_nodes = filter.filter_nodes(
    nodes=retrieved_nodes,
    reference_text="What is AI?"
)
```

### Use Cases

- **RAG Quality**: Filter low-quality retrieved documents
- **Node Ranking**: Rank nodes by quality scores
- **Response Filtering**: Ensure only high-quality nodes are used
- **Metadata Enrichment**: Add quality scores to node metadata

### Metadata

After postprocessing, each node contains quality metadata:

```python
for node in response.source_nodes:
    score = node.node.metadata['instability_score']
    passed = node.node.metadata['passed_threshold']
    print(f"Node quality: {score:.4f}, Valid: {passed}")
```

### Example

See `examples/example_llamaindex.py` for complete working examples.

## ZenML Integration

LogosLabs provides ZenML steps and materializers for ML pipeline integration.

### Installation

```bash
pip install logoslabs[zenml]
# or
pip install logoslabs zenml
```

### Components

#### logoslabs_filter_step

A ready-to-use ZenML step for filtering data.

```python
from zenml import pipeline
from logoslabs.integrations.zenml import logoslabs_filter_step

@pipeline
def validation_pipeline():
    data = load_data()
    
    # Filter with LogosLabs
    filtered_data, summary = logoslabs_filter_step(
        items=data,
        threshold=0.5
    )
    
    return filtered_data, summary

validation_pipeline().run()
```

#### LogosLabsFilterStep

Reusable step class with custom configuration.

```python
from zenml import pipeline
from logoslabs.integrations.zenml import LogosLabsFilterStep

# Create configured step
filter_step = LogosLabsFilterStep(
    threshold=0.8,
    grace_coeff=0.6,
    phase_weight=0.15
)

@pipeline
def my_pipeline():
    data = load_data()
    filtered, summary = filter_step(data)
    return filtered

my_pipeline().run()
```

#### logoslabs_score_step

Score items without filtering (for downstream decisions).

```python
from logoslabs.integrations.zenml import logoslabs_score_step

@pipeline
def scoring_pipeline():
    data = load_data()
    
    # Add scores without filtering
    scored_data, summary = logoslabs_score_step(
        items=data,
        threshold=0.5
    )
    
    # Downstream steps can use scores
    return scored_data

scoring_pipeline().run()
```

#### LogosLabsScoreMaterializer

Custom materializer for artifact storage.

```python
from logoslabs.integrations.zenml import LogosLabsScoreMaterializer

# Materializer automatically handles serialization
# Results are stored in ZenML artifact store
```

### Use Cases

- **Data Cleaning**: Filter low-quality training data
- **Pipeline Validation**: Add quality checks to ML pipelines
- **Experiment Tracking**: Track quality metrics across runs
- **Artifact Storage**: Save validation results with custom materializers

### Example

See `examples/example_zenml.py` for complete working examples.

## Installation

### Install All Integrations

```bash
pip install logoslabs[all]
```

### Install Specific Integrations

```bash
# LangChain only
pip install logoslabs[langchain]

# LlamaIndex only
pip install logoslabs[llamaindex]

# ZenML only
pip install logoslabs[zenml]

# Development dependencies
pip install logoslabs[dev]
```

## Common Patterns

### Pattern 1: Quality Filtering in Production

```python
# Use threshold gating to filter outputs
from logoslabs.integrations.langchain import LogosLabsOutputParser

parser = LogosLabsOutputParser(threshold=0.5)
result = parser.parse(llm_output)

if result['passed_threshold']:
    # Use the output
    process(result['text'])
else:
    # Regenerate or use fallback
    fallback_response()
```

### Pattern 2: Batch Validation

```python
# Process multiple outputs efficiently
from logoslabs.avp import AVPProcessor

processor = AVPProcessor(threshold=0.8)
items = [
    {"prediction": output1, "truth": reference1},
    {"prediction": output2, "truth": reference2},
]
results = processor.process_batch(items)
```

### Pattern 3: Pipeline Integration

```python
# Integrate into existing pipelines
from logoslabs.integrations.zenml import LogosLabsFilterStep

@pipeline
def ml_pipeline():
    raw_data = extract()
    validated_data, _ = LogosLabsFilterStep(threshold=0.7)(raw_data)
    model = train(validated_data)
    return model
```

## Best Practices

1. **Threshold Selection**: Start with default threshold (1.0) and adjust based on your use case
2. **Reference Text**: Provide meaningful reference text when available
3. **Batch Processing**: Use batch operations for better performance
4. **Metadata**: Leverage quality scores in downstream decisions
5. **Monitoring**: Track quality metrics over time

## Troubleshooting

### ImportError: Framework not installed

If you see import errors, make sure the framework is installed:

```bash
pip install langchain  # or llama-index, or zenml
```

### Low Pass Rates

If too many outputs are being filtered:
- Increase threshold value
- Adjust grace_coeff and phase_weight
- Review reference text quality

### High Pass Rates (Everything Passes)

If nothing is being filtered:
- Decrease threshold value
- Ensure reference text is meaningful
- Check input data quality

## Support

For issues or questions:
- GitHub Issues: https://github.com/keepitcheesy/Eigen-trace/issues
- Documentation: https://github.com/keepitcheesy/Eigen-trace
- Examples: See `examples/` directory

## License

MIT License - see LICENSE file for details.
