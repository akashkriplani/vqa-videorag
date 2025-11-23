#!/usr/bin/env python3
"""
demo_answer_generation.py
Quick demo of answer generation with GPT-4o-mini.

Usage:
    python demo_answer_generation.py

Requirements:
    - Set OPENAI_API_KEY in .env file or environment
    - Run retrieval first to generate search results
"""

import os
import json
from answer_generation import AnswerGenerator, format_answer_output


def demo_single_query():
    """Demo answer generation for a single query"""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Set it in .env file or export OPENAI_API_KEY=your_key")
        return

    # Load search results
    results_file = "multimodal_search_results_hybrid.json"

    if not os.path.exists(results_file):
        print(f"❌ Error: {results_file} not found")
        print("   Run query_faiss.py first to generate search results")
        return

    print(f"Loading search results from {results_file}...")
    with open(results_file, 'r') as f:
        search_data = json.load(f)

    query = search_data['query']
    segments = search_data['results']

    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    print(f"Retrieved {len(segments)} segments")
    print(f"Top 3 segments will be used for answer generation")

    # Initialize generator
    print("\nInitializing GPT-4o-mini answer generator...")
    generator = AnswerGenerator(model_name="gpt-4o-mini")

    # Generate answer
    print("\nGenerating answer (this may take 2-5 seconds)...")
    result = generator.generate_answer(
        query=query,
        segment_contexts=segments,
        max_tokens=250,
        temperature=0.3,
        top_k_evidence=3
    )

    # Display result
    print("\n" + format_answer_output(result))

    # Save result
    output_file = "demo_answer_with_generation.json"
    with open(output_file, 'w') as f:
        json.dump({
            'query': query,
            'search_results': segments[:3],
            'generated_answer': result
        }, f, indent=2)

    print(f"\n✅ Complete! Full result saved to: {output_file}")
    print(f"\nCost Summary:")
    print(f"  - This query cost: ${result['cost_estimate']:.6f}")
    print(f"  - Estimated cost for 100 queries: ${result['cost_estimate'] * 100:.4f}")
    print(f"  - Estimated cost for 1000 queries: ${result['cost_estimate'] * 1000:.2f}")


def demo_batch_queries():
    """Demo batch answer generation"""

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        return

    # Sample queries for testing
    test_queries = [
        "How to do a mouth cancer check at home?",
        "How to assess the supraclavicular lymph nodes?",
        "What are the steps for CPR?"
    ]

    print(f"\n{'='*80}")
    print(f"BATCH DEMO: {len(test_queries)} queries")
    print(f"{'='*80}")

    # Note: In real scenario, you'd run retrieval for each query first
    print("\nNote: This demo requires pre-generated search results for each query")
    print("For full batch processing, integrate with query_faiss.py")

    generator = AnswerGenerator(model_name="gpt-4o-mini")

    print("\nEstimated batch costs:")
    print(f"  - 10 queries: ~$0.003-0.005")
    print(f"  - 100 queries: ~$0.03-0.05")
    print(f"  - 1000 queries: ~$0.30-0.50")
    print("\nGPT-4o-mini is 60x cheaper than GPT-4o!")


if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("MEDICAL VIDEORAG - ANSWER GENERATION DEMO")
    print("=" * 80)

    # Check for .env file
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        demo_batch_queries()
    else:
        demo_single_query()
