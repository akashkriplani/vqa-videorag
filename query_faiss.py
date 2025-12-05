"""
query_faiss.py

Load FAISS indices and metadata from `faiss_db/` and run retrieval for a user query.

Features:
- Embed queries with Bio_ClinicalBERT (text) using CLS token pooling to match training embeddings
- Embed queries with BiomedCLIP text encoder (open_clip) to search visual FAISS indexes for cross-modal retrieval
- Support for sliding window, chunk overlapping, and hybrid embedding strategies
- CLI: choose index paths, top_k, and mode (text / visual / both)

Usage examples:
python query_faiss.py --query "laparoscopic surgery" --text_index faiss_db/textual_train.index --visual_index faiss_db/visual_train.index --top_k 5 --mode both

"""

import os
import json
import argparse
import numpy as np
import torch

# Import from search module
from search import (
    FaissIndex,
    EmbeddingModels,
    aggregate_results_by_segment,
    aggregate_results_by_video,
    print_segment_results,
    print_video_contexts,
    extract_metadata,
    HybridSearchEngine,
    load_segments_from_json_dir
)

def pretty_print_results(results, query=None):
    """Format search results with rich metadata context"""
    if not results:
        print("No results found.")
        return []

    out = []
    print("\nTop Results:")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        metadata = extract_metadata(r)
        modal = r.get("modal", "unknown")
        sim = float(r.get("sim", 0.0))
        source = os.path.basename(r.get("source_index", "unknown"))

        entry = {
            "rank": i,
            "similarity": f"{sim:.3f}",
            "modality": modal,
            "video_id": metadata["video_id"],
            "timestamp": metadata["timestamp"],
            "source": source,
            "processing_type": metadata.get("processing_type", "N/A")
        }

        # Add text snippet for textual results
        if modal == "text" and metadata["text"]:
            entry["transcript"] = metadata["text"][:200]
        if metadata["entities"]:
            entry["entities"] = metadata["entities"]

        # Add window information for debugging/analysis
        if "window_size" in metadata:
            entry["window_info"] = {
                "size": metadata["window_size"],
                "range": metadata.get("window_range", "N/A")
            }

        out.append(entry)

        # Print human-friendly format
        print(f"\n{i}. Video: {metadata['video_id']} ({modal}, sim={sim:.3f})")
        print(f"   Time: {metadata['timestamp']}")
        print(f"   Source: {source}")
        print(f"   Processing: {metadata.get('processing_type', 'N/A')}")

        if modal == "text" and metadata["text"]:
            # Show snippet with context
            snippet = metadata["text"][:250] + "..." if len(metadata["text"]) > 250 else metadata["text"]
            print(f"   Transcript: {snippet}")

        if metadata["entities"]:
            print(f"   Medical Entities: {metadata['entities']}")

        if "window_size" in metadata:
            print(f"   Window Info: {metadata.get('window_range', 'N/A')} (size: {metadata['window_size']})")

        print("-" * 100)

    # Save structured output to JSON
    output_file = "search_results.json"
    print(f"\nStructured results saved to: {output_file}")
    with open(output_file, "w") as f:
        json.dump({
            "query": query,
            "total_results": len(out),
            "results": out
        }, f, indent=2)

    return out


def main():
    parser = argparse.ArgumentParser(description="Query FAISS textual/visual indices using user query embeddings")
    parser.add_argument("--query", type=str, required=False, help="Text query to search (if omitted you'll be prompted)")
    parser.add_argument("--text_index", type=str, default=None, help="Path to textual FAISS index (optional; auto-discover in faiss_db/ if omitted)")
    parser.add_argument("--visual_index", type=str, default=None, help="Path to visual FAISS index (optional; auto-discover in faiss_db/ if omitted)")
    parser.add_argument("--final_k", type=int, default=10, help="FINAL number of combined results to return (default: 10)")
    parser.add_argument("--local_k", type=int, default=50, help="Number of results to fetch from each index before merging (default: 50)")
    parser.add_argument("--device", type=str, choices=["cpu","cuda","mps"], default=None, help="Device to run models on; if omitted auto-detects")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for query embedding (default: 512)")
    parser.add_argument("--mode", type=str, choices=["segment", "video"], default="segment", help="Result aggregation mode: 'segment' for segment-level multimodal linking (default), 'video' for video-level aggregation")
    parser.add_argument("--top_videos", type=int, default=5, help="Number of top videos to return in video aggregation mode (default: 5)")
    parser.add_argument("--text_weight", type=float, default=0.6, help="Weight for textual similarity in video aggregation (default: 0.6)")
    parser.add_argument("--visual_weight", type=float, default=0.4, help="Weight for visual similarity in video aggregation (default: 0.4)")
    parser.add_argument("--filter_sampling", type=str, default=None, choices=["uniform", "adaptive", "quality_based"], help="Filter results by sampling strategy (optional)")
    parser.add_argument("--filter_aggregation", type=str, default=None, choices=["mean", "max"], help="Filter results by aggregation method (optional)")
    parser.add_argument("--hierarchical", default=True, action="store_true", help="Enable hierarchical search with fine-grained timestamp refinement")
    parser.add_argument("--json_dir", type=str, default="feature_extraction/", help="Directory containing JSON feature files (will search recursively through train/test/val)")

    # Hybrid search arguments
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid search (BM25 + dense embeddings)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for dense retrieval in hybrid search (0-1, default: 0.7)")
    parser.add_argument("--fusion", type=str, choices=["linear", "rrf"], default="linear", help="Fusion strategy for hybrid search: 'linear' or 'rrf' (Reciprocal Rank Fusion)")
    parser.add_argument("--expand_query", action="store_true", default=True, help="Expand query with medical synonyms in BM25 search")
    parser.add_argument("--analyze_fusion", action="store_true", help="Show detailed fusion analysis (BM25 vs dense contribution)")
    parser.add_argument("--output", type=str, default=None, help="Output file path for search results (default: auto-generated based on search type)")

    # Answer generation arguments
    parser.add_argument("--generate_answer", action="store_true", help="Generate answer using LLM (GPT-4o-mini)")
    parser.add_argument("--answer_model", type=str, default="gpt-4o-mini", help="LLM model for answer generation (default: gpt-4o-mini)")
    parser.add_argument("--answer_max_tokens", type=int, default=250, help="Maximum tokens for answer (default: 250)")
    parser.add_argument("--answer_temperature", type=float, default=0.3, help="Temperature for answer generation (default: 0.3)")

    # Adaptive context selection arguments
    parser.add_argument("--enable_curation", action="store_true", default=False, help="Enable adaptive context selection with factual grounding (default: False)")
    parser.add_argument("--enable_attribution", action="store_true", default=False, help="Enable self-reflection attribution mapping (default: False)")
    parser.add_argument("--quality_threshold", type=float, default=0.3, help="Minimum quality score for context filtering (default: 0.3)")
    parser.add_argument("--nli_top_k", type=int, default=15, help="Number of top candidates for NLI factuality scoring (default: 15)")
    parser.add_argument("--token_budget", type=int, default=600, help="Maximum tokens for curated context (default: 600)")

    args = parser.parse_args()

    # Auto-generate output filename based on search type if not specified
    if args.output is None:
        if args.hybrid:
            args.output = "multimodal_search_results_hybrid.json"
        else:
            args.output = "multimodal_search_results_dense.json"

    # If query not provided, prompt the user
    if not args.query:
        try:
            args.query = input("Enter search query: ")
        except Exception:
            raise ValueError("No query provided")

    # Auto-discover indexes in faiss_db/ if explicit paths are not provided
    def discover_indexes(prefix):
        import glob
        paths = sorted(glob.glob(os.path.join("faiss_db", f"{prefix}_*.index")))
        return paths

    text_index_paths = [args.text_index] if args.text_index else discover_indexes("textual")
    visual_index_paths = [args.visual_index] if args.visual_index else discover_indexes("visual")

    if not text_index_paths and not visual_index_paths:
        raise ValueError("No FAISS indexes found in faiss_db/. Place textual_*.index or visual_*.index or pass explicit paths.")

    device = None
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")

    print(f"Loading embedding models on device: {device or 'auto'}")
    models = EmbeddingModels(device=device)

    # Search each index for a larger local pool, normalize distances to similarities, and merge both modalities
    def search_index_pool(index_paths, query_vec, local_k=50):
        pool = []
        for p in index_paths:
            try:
                idx = FaissIndex(p)
                idx.load()
                res = idx.search(query_vec, top_k=local_k)
                for r in res:
                    pool.append({
                        "raw_score": r.get("score"),
                        "meta": r.get("meta"),
                        "source_index": p
                    })
            except Exception as e:
                print(f"Warning: failed to search index {p}: {e}")
        return pool

    # Generate query embeddings
    print(f"\nQuery: '{args.query}'")
    print(f"Generating embeddings with max_length={args.max_length}...")

    q_vec_text = models.embed_text_bio(args.query, max_length=args.max_length) if text_index_paths else None
    q_vec_clip = models.embed_text_clip(args.query) if visual_index_paths else None

    print(f"Searching indices with local_k={args.local_k} per index...")
    text_pool = search_index_pool(text_index_paths, q_vec_text, local_k=args.local_k) if q_vec_text is not None else []
    visual_pool = search_index_pool(visual_index_paths, q_vec_clip, local_k=args.local_k) if q_vec_clip is not None else []

    print(f"Retrieved {len(text_pool)} text results and {len(visual_pool)} visual results")

    # Apply hybrid search if enabled
    if args.hybrid and text_pool:
        print(f"\n{'='*80}")
        print("HYBRID SEARCH ENABLED (BM25 + Dense Embeddings)")
        print(f"{'='*80}")

        try:
            # Load segments for BM25 index
            segments_data = load_segments_from_json_dir(args.json_dir)

            # Initialize hybrid search engine
            hybrid_engine = HybridSearchEngine(
                segments_data=segments_data,
                alpha=args.alpha
            )

            # Perform hybrid search on text results
            hybrid_text_results = hybrid_engine.hybrid_search(
                query=args.query,
                dense_results=text_pool,
                top_k=args.local_k,
                fusion=args.fusion,
                expand_query=args.expand_query
            )

            # Show fusion analysis if requested
            if args.analyze_fusion:
                hybrid_engine.analyze_fusion_contribution(hybrid_text_results, top_k=args.final_k)

            # Replace text_pool with hybrid results
            text_pool = []
            for hybrid_result in hybrid_text_results:
                meta = hybrid_result.get('metadata', {})
                # Convert hybrid score to raw_score format
                combined_score = hybrid_result.get('combined_score', 0.0)
                # Convert back to distance-like metric (lower is better)
                raw_score = -np.log(combined_score + 1e-10)

                text_pool.append({
                    'raw_score': raw_score,
                    'meta': meta,
                    'source_index': 'hybrid',
                    'hybrid_info': {
                        'dense_score': hybrid_result.get('dense_score', 0.0),
                        'bm25_score': hybrid_result.get('bm25_score', 0.0),
                        'fusion_method': hybrid_result.get('fusion_method', 'unknown')
                    }
                })

            print(f"✅ Hybrid search complete: {len(text_pool)} results")

        except Exception as e:
            print(f"⚠️  Hybrid search failed: {e}")
            print("Falling back to dense-only search...")

    # Normalize raw distances to similarity in (0,1] using sim = 1/(1+dist)
    combined = []
    for item in text_pool:
        dist = item.get("raw_score", float('inf'))
        sim = 1.0 / (1.0 + dist) if np.isfinite(dist) else 0.0
        combined.append({
            "sim": sim,
            "meta": item.get("meta"),
            "source_index": item.get("source_index"),
            "modal": "text",
            "raw_score": dist,
            "hybrid_info": item.get("hybrid_info")  # Preserve hybrid search info
        })

    for item in visual_pool:
        dist = item.get("raw_score", float('inf'))
        sim = 1.0 / (1.0 + dist) if np.isfinite(dist) else 0.0

        # Optional: Filter by hyperparameters (useful for comparing experiments)
        meta = item.get("meta", {}) or {}
        if args.filter_sampling and meta.get("sampling_strategy") != args.filter_sampling:
            continue
        if args.filter_aggregation and meta.get("aggregation_method") != args.filter_aggregation:
            continue

        combined.append({
            "sim": sim,
            "meta": item.get("meta"),
            "source_index": item.get("source_index"),
            "modal": "visual",
            "raw_score": dist
        })

    # Choose result aggregation mode
    if args.mode == "video":
        # Video-level aggregation: group by video_id for comprehensive context
        print(f"\nAggregating results by video (top {args.top_videos} videos)")
        print(f"Using weights: text={args.text_weight}, visual={args.visual_weight}")

        video_contexts = aggregate_results_by_video(
            text_pool,
            visual_pool,
            top_k=args.top_videos,
            text_weight=args.text_weight,
            visual_weight=args.visual_weight
        )

        print_video_contexts(video_contexts, query=args.query, output_file=args.output)

    else:
        # Segment-level aggregation: precise multimodal linking via segment_id
        print(f"\nAggregating results by segment (top {args.final_k} segments)")
        print(f"Using weights: text={args.text_weight}, visual={args.visual_weight}")

        segment_contexts = aggregate_results_by_segment(
            text_pool,
            visual_pool,
            top_k=args.final_k,
            text_weight=args.text_weight,
            visual_weight=args.visual_weight,
            enable_hierarchical=args.hierarchical,
            json_dir=args.json_dir if args.hierarchical else None
        )

        print_segment_results(segment_contexts, query=args.query, output_file=args.output)

        # Generate answer if requested
        if args.generate_answer:
            print(f"\n{'='*80}")
            print("GENERATING ANSWER (GPT-4o-mini)")
            print(f"{'='*80}")

            try:
                from generation import AnswerGenerator, format_answer_output

                # Configure curation settings if enabled
                curation_config = None
                if args.enable_curation:
                    curation_config = {
                        'quality_threshold': args.quality_threshold,
                        'token_budget': args.token_budget,
                        'use_nli': True
                    }
                    print(f"\n🔧 Adaptive Context Selection: ENABLED")
                    print(f"   - Quality threshold: {args.quality_threshold}")
                    print(f"   - Token budget: {args.token_budget}")
                    print(f"   - NLI scoring: Enabled (top-15 candidates)")
                else:
                    print(f"\n🔧 Adaptive Context Selection: DISABLED (use --enable_curation to enable)")

                if args.enable_attribution:
                    print(f"🔧 Self-Reflection Attribution: ENABLED")
                else:
                    print(f"🔧 Self-Reflection Attribution: DISABLED (use --enable_attribution to enable)")

                generator = AnswerGenerator(
                    model_name=args.answer_model,
                    enable_curation=args.enable_curation,
                    enable_attribution=args.enable_attribution,
                    curation_config=curation_config
                )

                answer_result = generator.generate_answer(
                    query=args.query,
                    segment_contexts=segment_contexts,
                    max_tokens=args.answer_max_tokens,
                    temperature=args.answer_temperature,
                    top_k_evidence=3
                )

                # Display formatted answer
                print("\n" + format_answer_output(answer_result))

                # Save answer with search results
                answer_output_file = args.output.replace('.json', '_with_answer.json')
                import json
                with open(answer_output_file, 'w') as f:
                    json.dump({
                        'query': args.query,
                        'search_results': segment_contexts,
                        'generated_answer': answer_result
                    }, f, indent=2)

                print(f"\n✅ Answer saved to: {answer_output_file}")

            except Exception as e:
                print(f"\n❌ Answer generation failed: {e}")
                print("   Make sure OPENAI_API_KEY is set in environment or .env file")


if __name__ == "__main__":
    main()