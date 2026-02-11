#!/usr/bin/env python3
"""
Example: Using LogosLabs with LangChain

This example demonstrates how to integrate LogosLabs validation
into LangChain workflows for filtering and validating LLM outputs.

Requirements:
    pip install logoslabs langchain
"""

from langchain.llms import FakeListLLM
from langchain.prompts import PromptTemplate
from logoslabs.integrations.langchain import LogosLabsOutputParser, LogosLabsChain


def example_output_parser():
    """Example using LogosLabsOutputParser."""
    print("=" * 60)
    print("Example 1: LogosLabsOutputParser")
    print("=" * 60)
    
    # Create parser with threshold
    parser = LogosLabsOutputParser(
        threshold=0.5,
        reference_text="This is a high-quality response",
        raise_on_fail=False,
    )
    
    # Parse some outputs
    outputs = [
        "This is a high-quality response",
        "This is a different response",
        "Something completely unrelated",
    ]
    
    for output in outputs:
        result = parser.parse(output)
        print(f"\nOutput: {output[:50]}...")
        print(f"  Score: {result['instability_score']:.4f}")
        print(f"  Passed: {result['passed_threshold']}")
    
    print("\n✓ Output parser example complete\n")


def example_chain():
    """Example using LogosLabsChain."""
    print("=" * 60)
    print("Example 2: LogosLabsChain with Validation")
    print("=" * 60)
    
    # Create a fake LLM for testing
    responses = [
        "This is response one about AI",
        "This is response two about AI",
        "This is response three about AI",
    ]
    llm = FakeListLLM(responses=responses)
    
    # Create prompt
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a brief explanation about {topic}.",
    )
    
    # Create chain with validation
    chain = LogosLabsChain.from_llm(
        llm=llm,
        prompt=prompt,
        threshold=1.0,
        reference_text="Explain artificial intelligence",
    )
    
    # Run chain
    result = chain.run(topic="artificial intelligence")
    
    print(f"\nGenerated text: {result}")
    print(f"Validation score: {chain.output_key}")
    
    print("\n✓ Chain example complete\n")


def example_batch_validation():
    """Example batch validation with LangChain."""
    print("=" * 60)
    print("Example 3: Batch Validation")
    print("=" * 60)
    
    parser = LogosLabsOutputParser(threshold=0.8)
    
    # Simulate multiple LLM outputs
    outputs = [
        "The weather is sunny today",
        "Machine learning is fascinating",
        "Python is a programming language",
        "Data science involves statistics",
    ]
    
    results = []
    for output in outputs:
        result = parser.parse(output)
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r['passed_threshold'])
    total = len(results)
    
    print(f"\nProcessed {total} outputs")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    print("\nDetailed results:")
    for i, (output, result) in enumerate(zip(outputs, results), 1):
        print(f"  {i}. Score: {result['instability_score']:.4f} - "
              f"{'✓ PASS' if result['passed_threshold'] else '✗ FAIL'}")
    
    print("\n✓ Batch validation example complete\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LogosLabs + LangChain Integration Examples")
    print("=" * 60 + "\n")
    
    try:
        example_output_parser()
        example_chain()
        example_batch_validation()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure to install required dependencies:")
        print("  pip install logoslabs langchain")


if __name__ == "__main__":
    main()
