"""
test_hybrid_search.py

Test script to demonstrate hybrid search (BM25 + dense embeddings) improvements.

Usage:
    python test_hybrid_search.py --query "How to do a mouth cancer check at home?"
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Test hybrid search vs pure dense search")
    parser.add_argument("--query", type=str, default="How to do a mouth cancer check at home?",
                       help="Test query")
    parser.add_argument("--text_index", type=str, default="faiss_db/textual_train.index",
                       help="Path to textual FAISS index")
    parser.add_argument("--visual_index", type=str, default="faiss_db/visual_train.index",
                       help="Path to visual FAISS index")
    parser.add_argument("--json_dir", type=str, default="feature_extraction/textual/train",
                       help="Directory with JSON feature files")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of results to show")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Dense weight for hybrid search (0.7 = 70% dense, 30% BM25)")

    args = parser.parse_args()

    print("="*100)
    print("HYBRID SEARCH TEST")
    print("="*100)
    print(f"\nQuery: {args.query}")
    print(f"Text Index: {args.text_index}")
    print(f"JSON Dir: {args.json_dir}")
    print(f"Alpha (dense weight): {args.alpha}")
    print("\n" + "="*100)

    # Test 1: Pure Dense Search (baseline)
    print("\n" + "="*100)
    print("TEST 1: PURE DENSE SEARCH (Baseline)")
    print("="*100)

    from query_faiss import main as query_main

    # Temporarily modify sys.argv to pass arguments
    original_argv = sys.argv
    sys.argv = [
        'query_faiss.py',
        '--query', args.query,
        '--text_index', args.text_index,
        '--visual_index', args.visual_index,
        '--json_dir', args.json_dir,
        '--final_k', str(args.top_k),
        '--mode', 'segment',
        '--hierarchical'
    ]

    try:
        print("\nRunning pure dense search...")
        query_main()
    except Exception as e:
        print(f"Error in dense search: {e}")

    print("\n\n" + "="*100)
    print("TEST 2: HYBRID SEARCH (BM25 + Dense)")
    print("="*100)

    # Test 2: Hybrid Search
    sys.argv = [
        'query_faiss.py',
        '--query', args.query,
        '--text_index', args.text_index,
        '--visual_index', args.visual_index,
        '--json_dir', args.json_dir,
        '--final_k', str(args.top_k),
        '--mode', 'segment',
        '--hierarchical',
        '--hybrid',  # Enable hybrid search
        '--alpha', str(args.alpha),
        '--fusion', 'linear',
        '--expand_query',
        '--analyze_fusion'  # Show detailed analysis
    ]

    try:
        print("\nRunning hybrid search with BM25 + dense embeddings...")
        query_main()
    except Exception as e:
        print(f"Error in hybrid search: {e}")

    # Restore original argv
    sys.argv = original_argv

    print("\n\n" + "="*100)
    print("COMPARISON COMPLETE")
    print("="*100)
    print("\nCheck the 'multimodal_search_results.json' file for detailed results.")
    print("\nKey Improvements to Look For:")
    print("1. Expected video ID ranking higher in results")
    print("2. Better match for medical terminology (e.g., 'oral cancer' for 'mouth cancer')")
    print("3. Fusion analysis showing BM25 contribution for keyword-rich queries")
    print("\nTo experiment with different fusion weights:")
    print(f"  python {sys.argv[0]} --alpha 0.5  # 50% dense, 50% BM25")
    print(f"  python {sys.argv[0]} --alpha 0.8  # 80% dense, 20% BM25")

if __name__ == "__main__":
    main()
