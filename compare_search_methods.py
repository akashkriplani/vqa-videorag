"""
compare_search_methods.py

Compare pure dense search vs hybrid search to analyze improvements.

Usage:
    python compare_search_methods.py --query "How to do a mouth cancer check at home?" --expected_video "_6csIJAWj_s"
"""

import argparse
import subprocess
import json
import os

def run_search(query, text_index, visual_index, json_dir, top_k, hybrid=False, alpha=0.7):
    """Run search and return results"""
    cmd = [
        'python', 'query_faiss.py',
        '--query', query,
        '--text_index', text_index,
        '--visual_index', visual_index,
        '--json_dir', json_dir,
        '--final_k', str(top_k),
        '--mode', 'segment',
        '--hierarchical'
    ]

    if hybrid:
        cmd.extend(['--hybrid', '--alpha', str(alpha), '--fusion', 'linear', '--expand_query'])

    # Run command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Load results from JSON file
    if os.path.exists('multimodal_search_results.json'):
        with open('multimodal_search_results.json', 'r') as f:
            return json.load(f)

    return None

def analyze_results(results, expected_video_id):
    """Analyze search results and find position of expected video"""
    if not results or 'results' not in results:
        return None, []

    video_positions = {}
    top_results = []

    for rank, result in enumerate(results['results'], 1):
        video_id = result.get('video_id', 'unknown')
        score = result.get('combined_score', 0.0)

        top_results.append({
            'rank': rank,
            'video_id': video_id,
            'score': score,
            'segment_id': result.get('segment_id', 'unknown')
        })

        if video_id not in video_positions:
            video_positions[video_id] = rank

    expected_rank = video_positions.get(expected_video_id, None)

    return expected_rank, top_results

def main():
    parser = argparse.ArgumentParser(description="Compare pure dense vs hybrid search")
    parser.add_argument("--query", type=str, default="How to do a mouth cancer check at home?",
                       help="Test query")
    parser.add_argument("--expected_video", type=str, default="_6csIJAWj_s",
                       help="Expected video ID for this query")
    parser.add_argument("--text_index", type=str, default="faiss_db/textual_train.index",
                       help="Path to textual FAISS index")
    parser.add_argument("--visual_index", type=str, default="faiss_db/visual_train.index",
                       help="Path to visual FAISS index")
    parser.add_argument("--json_dir", type=str, default="feature_extraction/textual",
                       help="Directory with JSON feature files (searches recursively through train/test/val)")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of results to analyze")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Dense weight for hybrid search")

    args = parser.parse_args()

    print("="*100)
    print("SEARCH METHOD COMPARISON")
    print("="*100)
    print(f"\nQuery: {args.query}")
    print(f"Expected Video: {args.expected_video}")
    print(f"Top-K: {args.top_k}")
    print("\n" + "="*100)

    # Run pure dense search
    print("\n[1/2] Running PURE DENSE search...")
    dense_results = run_search(
        args.query, args.text_index, args.visual_index,
        args.json_dir, args.top_k, hybrid=False
    )

    dense_rank, dense_top = analyze_results(dense_results, args.expected_video)

    # Run hybrid search
    print("\n[2/2] Running HYBRID search (BM25 + Dense)...")
    hybrid_results = run_search(
        args.query, args.text_index, args.visual_index,
        args.json_dir, args.top_k, hybrid=True, alpha=args.alpha
    )

    hybrid_rank, hybrid_top = analyze_results(hybrid_results, args.expected_video)

    # Display comparison
    print("\n" + "="*100)
    print("COMPARISON RESULTS")
    print("="*100)

    print(f"\n📍 Expected Video: {args.expected_video}")
    print(f"\n🔍 Pure Dense Search:")
    if dense_rank:
        print(f"   Rank: #{dense_rank} / {args.top_k}")
        print(f"   ✅ Found in top-{args.top_k}")
    else:
        print(f"   ❌ Not found in top-{args.top_k}")

    print(f"\n🔬 Hybrid Search (BM25 + Dense, α={args.alpha}):")
    if hybrid_rank:
        print(f"   Rank: #{hybrid_rank} / {args.top_k}")
        print(f"   ✅ Found in top-{args.top_k}")
    else:
        print(f"   ❌ Not found in top-{args.top_k}")

    # Calculate improvement
    if dense_rank and hybrid_rank:
        improvement = dense_rank - hybrid_rank
        if improvement > 0:
            print(f"\n📈 IMPROVEMENT: Moved up {improvement} positions!")
        elif improvement < 0:
            print(f"\n📉 REGRESSION: Moved down {abs(improvement)} positions")
        else:
            print(f"\n➡️  NO CHANGE: Same ranking")
    elif hybrid_rank and not dense_rank:
        print(f"\n🎯 SUCCESS: Expected video now appears in top-{args.top_k}!")
    elif dense_rank and not hybrid_rank:
        print(f"\n⚠️  WARNING: Expected video dropped out of top-{args.top_k}")

    # Display top results side by side
    print(f"\n" + "="*100)
    print(f"TOP-{min(5, args.top_k)} RESULTS COMPARISON")
    print("="*100)

    print(f"\n{'RANK':<6} {'DENSE SEARCH':<40} {'HYBRID SEARCH':<40}")
    print("-" * 100)

    for i in range(min(5, args.top_k)):
        rank = i + 1

        dense_video = dense_top[i]['video_id'] if i < len(dense_top) else 'N/A'
        dense_score = f"{dense_top[i]['score']:.4f}" if i < len(dense_top) else 'N/A'

        hybrid_video = hybrid_top[i]['video_id'] if i < len(hybrid_top) else 'N/A'
        hybrid_score = f"{hybrid_top[i]['score']:.4f}" if i < len(hybrid_top) else 'N/A'

        # Highlight expected video
        if dense_video == args.expected_video:
            dense_display = f"🎯 {dense_video} ({dense_score})"
        else:
            dense_display = f"   {dense_video} ({dense_score})"

        if hybrid_video == args.expected_video:
            hybrid_display = f"🎯 {hybrid_video} ({hybrid_score})"
        else:
            hybrid_display = f"   {hybrid_video} ({hybrid_score})"

        print(f"{rank:<6} {dense_display:<40} {hybrid_display:<40}")

    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    if hybrid_rank and (not dense_rank or hybrid_rank < dense_rank):
        print("\n✅ Hybrid search shows improvement!")
        print("\nNext steps:")
        print("1. Try different alpha values (0.5-0.9) to find optimal balance")
        print("2. Consider RRF fusion: --fusion rrf")
        print("3. Expand medical terminology dictionary in hybrid_search.py")
    else:
        print("\n⚠️  Hybrid search needs tuning")
        print("\nNext steps:")
        print("1. Reduce alpha (give more weight to BM25): --alpha 0.5")
        print("2. Check query expansion - add more medical synonyms")
        print("3. Consider RRF fusion instead of linear: --fusion rrf")

    print("\n" + "="*100)

if __name__ == "__main__":
    main()
