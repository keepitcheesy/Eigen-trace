#!/usr/bin/env python3
"""
Example: Using LogosLabs with LlamaIndex

This example demonstrates how to integrate LogosLabs validation
into LlamaIndex query engines for filtering retrieved nodes.

Requirements:
    pip install logoslabs llama-index
"""

from typing import List
from logoslabs.integrations.llamaindex import LogosLabsPostprocessor, LogosLabsNodeFilter


def example_postprocessor():
    """Example using LogosLabsPostprocessor."""
    print("=" * 60)
    print("Example 1: LogosLabsPostprocessor")
    print("=" * 60)
    
    # Note: This is a conceptual example
    # In real usage, you would integrate with an actual LlamaIndex query engine
    
    print("""
    Usage with LlamaIndex:
    
    from llama_index import VectorStoreIndex, SimpleDirectoryReader
    from logoslabs.integrations.llamaindex import LogosLabsPostprocessor
    
    # Load documents and create index
    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # Create query engine with LogosLabs postprocessor
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
    print(response)
    """)
    
    print("\n✓ Postprocessor example complete\n")


def example_node_filter():
    """Example using LogosLabsNodeFilter."""
    print("=" * 60)
    print("Example 2: LogosLabsNodeFilter")
    print("=" * 60)
    
    print("""
    Usage as standalone filter:
    
    from logoslabs.integrations.llamaindex import LogosLabsNodeFilter
    
    # Create filter
    filter = LogosLabsNodeFilter(threshold=0.8)
    
    # Filter nodes manually
    filtered_nodes = filter.filter_nodes(
        nodes=retrieved_nodes,
        reference_text="What is artificial intelligence?"
    )
    
    # Only high-quality nodes remain
    for node in filtered_nodes:
        score = node.node.metadata.get('instability_score', 'N/A')
        print(f"Node score: {score}")
    """)
    
    print("\n✓ Node filter example complete\n")


def example_custom_pipeline():
    """Example of custom pipeline with LogosLabs."""
    print("=" * 60)
    print("Example 3: Custom Pipeline Integration")
    print("=" * 60)
    
    print("""
    Integration in a custom retrieval pipeline:
    
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from logoslabs.integrations.llamaindex import LogosLabsPostprocessor
    
    # Combine multiple postprocessors
    query_engine = index.as_query_engine(
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7),
            LogosLabsPostprocessor(
                threshold=0.5,
                grace_coeff=0.5,
                phase_weight=0.1,
                filter_failures=True
            )
        ],
        similarity_top_k=10
    )
    
    # First filter by similarity, then by quality
    response = query_engine.query("Explain neural networks")
    
    # Access metadata
    for node in response.source_nodes:
        print(f"Similarity: {node.score}")
        print(f"Quality: {node.node.metadata['instability_score']}")
    """)
    
    print("\n✓ Custom pipeline example complete\n")


def example_with_metadata():
    """Example showing metadata enrichment."""
    print("=" * 60)
    print("Example 4: Metadata Enrichment")
    print("=" * 60)
    
    print("""
    LogosLabs adds quality metadata to nodes:
    
    # After postprocessing, each node has:
    node.metadata['instability_score']  # Quality score
    node.metadata['passed_threshold']   # Pass/fail flag
    
    # Use this for ranking, filtering, or display:
    for node in response.source_nodes:
        score = node.node.metadata['instability_score']
        passed = node.node.metadata['passed_threshold']
        
        print(f"Node quality: {score:.4f}")
        print(f"Meets threshold: {passed}")
        
        if passed:
            print(f"Content: {node.node.get_content()}")
    """)
    
    print("\n✓ Metadata enrichment example complete\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LogosLabs + LlamaIndex Integration Examples")
    print("=" * 60 + "\n")
    
    try:
        example_postprocessor()
        example_node_filter()
        example_custom_pipeline()
        example_with_metadata()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNote: These are conceptual examples.")
        print("For real usage, you need to:")
        print("  1. pip install logoslabs llama-index")
        print("  2. Set up your document index")
        print("  3. Integrate the postprocessor in your query engine")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure to install required dependencies:")
        print("  pip install logoslabs llama-index")


if __name__ == "__main__":
    main()
