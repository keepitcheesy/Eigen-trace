#!/usr/bin/env python3
"""
Example: Using LogosLabs with ZenML

This example demonstrates how to integrate LogosLabs validation
into ZenML pipelines for filtering LLM outputs.

Requirements:
    pip install logoslabs zenml
"""

from typing import List, Dict, Any


def example_basic_step():
    """Example using basic filter step."""
    print("=" * 60)
    print("Example 1: Basic Filter Step")
    print("=" * 60)
    
    print("""
    Usage with ZenML pipelines:
    
    from zenml import pipeline
    from logoslabs.integrations.zenml import logoslabs_filter_step
    
    @pipeline
    def validation_pipeline():
        # Load data
        data = load_llm_outputs()
        
        # Filter with LogosLabs
        filtered_data, summary = logoslabs_filter_step(
            items=data,
            threshold=0.5,
            grace_coeff=0.5,
            phase_weight=0.1
        )
        
        # Continue pipeline with filtered data
        process_data(filtered_data)
        
        return filtered_data, summary
    
    # Run pipeline
    pipeline_instance = validation_pipeline()
    pipeline_instance.run()
    """)
    
    print("\n✓ Basic step example complete\n")


def example_reusable_step():
    """Example using reusable step class."""
    print("=" * 60)
    print("Example 2: Reusable Step Class")
    print("=" * 60)
    
    print("""
    Create reusable steps with custom configuration:
    
    from zenml import pipeline
    from logoslabs.integrations.zenml import LogosLabsFilterStep
    
    # Create configured step
    quality_filter = LogosLabsFilterStep(
        threshold=0.8,
        grace_coeff=0.6,
        phase_weight=0.15
    )
    
    @pipeline
    def my_pipeline():
        raw_data = load_data()
        
        # Use the configured step
        filtered, summary = quality_filter(raw_data)
        
        # Log summary
        log_summary(summary)
        
        return filtered
    
    my_pipeline().run()
    """)
    
    print("\n✓ Reusable step example complete\n")


def example_scoring_only():
    """Example of scoring without filtering."""
    print("=" * 60)
    print("Example 3: Scoring Without Filtering")
    print("=" * 60)
    
    print("""
    Add scores without filtering (for downstream decisions):
    
    from zenml import pipeline
    from logoslabs.integrations.zenml import logoslabs_score_step
    
    @pipeline
    def scoring_pipeline():
        data = load_data()
        
        # Score all items (don't filter)
        scored_data, summary = logoslabs_score_step(
            items=data,
            threshold=0.5
        )
        
        # Downstream steps can use scores
        analyze_scores(scored_data)
        
        # Custom filtering logic
        high_quality = [
            item for item in scored_data
            if item['instability_score'] < 0.3
        ]
        
        return high_quality
    
    scoring_pipeline().run()
    """)
    
    print("\n✓ Scoring example complete\n")


def example_full_pipeline():
    """Example of a complete ML pipeline with LogosLabs."""
    print("=" * 60)
    print("Example 4: Complete ML Pipeline")
    print("=" * 60)
    
    print("""
    Full pipeline with data loading, filtering, and processing:
    
    from zenml import pipeline, step
    from logoslabs.integrations.zenml import LogosLabsFilterStep
    
    @step
    def load_llm_outputs() -> List[Dict[str, Any]]:
        # Load outputs from LLM
        return [
            {"prediction": "Generated text 1", "truth": "Reference 1"},
            {"prediction": "Generated text 2", "truth": "Reference 2"},
            # ... more items
        ]
    
    @step
    def train_model(data: List[Dict[str, Any]]) -> None:
        # Train on filtered, high-quality data
        print(f"Training on {len(data)} high-quality examples")
    
    @step
    def log_summary(summary: Dict[str, Any]) -> None:
        print(f"Pass rate: {summary['pass_rate']:.1%}")
        print(f"Mean score: {summary['mean_score']:.4f}")
    
    # Create filter
    filter_step = LogosLabsFilterStep(threshold=0.7)
    
    @pipeline
    def ml_pipeline():
        # Load data
        raw_data = load_llm_outputs()
        
        # Filter with LogosLabs
        filtered_data, summary = filter_step(raw_data)
        
        # Log results
        log_summary(summary)
        
        # Train model on clean data
        train_model(filtered_data)
    
    # Execute pipeline
    ml_pipeline().run()
    """)
    
    print("\n✓ Full pipeline example complete\n")


def example_custom_materializer():
    """Example using custom materializer."""
    print("=" * 60)
    print("Example 5: Custom Materializer")
    print("=" * 60)
    
    print("""
    Use custom materializer for artifact storage:
    
    from zenml import pipeline
    from logoslabs.integrations.zenml import (
        logoslabs_score_step,
        LogosLabsScoreMaterializer
    )
    
    @pipeline
    def pipeline_with_custom_materializer():
        data = load_data()
        
        # Results will be saved with custom materializer
        scored_data, summary = logoslabs_score_step(data)
        
        return scored_data
    
    # The materializer handles serialization/deserialization
    # Results are stored in ZenML artifact store
    pipeline_instance = pipeline_with_custom_materializer()
    pipeline_instance.run()
    
    # Later, access the artifacts
    # run = pipeline_instance.get_run()
    # scored_data = run.steps['logoslabs_score_step'].outputs['scored_items']
    """)
    
    print("\n✓ Custom materializer example complete\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LogosLabs + ZenML Integration Examples")
    print("=" * 60 + "\n")
    
    try:
        example_basic_step()
        example_reusable_step()
        example_scoring_only()
        example_full_pipeline()
        example_custom_materializer()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNote: These are conceptual examples.")
        print("For real usage, you need to:")
        print("  1. pip install logoslabs zenml")
        print("  2. Initialize ZenML: zenml init")
        print("  3. Define your pipeline with LogosLabs steps")
        print("  4. Run: python your_pipeline.py")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure to install required dependencies:")
        print("  pip install logoslabs zenml")


if __name__ == "__main__":
    main()
