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
from transformers import AutoTokenizer, AutoModel
import open_clip
import faiss


class FaissIndex:
    def __init__(self, index_path):
        self.index_path = index_path
        self.index = None
        self.metadata = []
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(index_path + ".meta.json"):
            raise FileNotFoundError(f"Metadata JSON file not found for index: {index_path}. Expected: {index_path}.meta.json")

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".meta.json", "r") as f:
            self.metadata = json.load(f)

    def search(self, query_vec, top_k=5):
        """
        Search FAISS index with normalized query vector.

        Args:
            query_vec: 1D numpy array (should already be normalized)
            top_k: Number of results to return

        Returns:
            List of dicts with score and metadata
        """
        q = query_vec.astype(np.float32).reshape(1, -1)
        D, I = self.index.search(q, top_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                results.append({"score": float(dist), "meta": None})
            else:
                results.append({"score": float(dist), "meta": self.metadata[idx]})
        return results


class EmbeddingModels:
    def __init__(self, device=None, biomedclip_model_id="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        # Load Bio Clinical BERT for textual embeddings
        self.bio_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bio_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
        self.bio_model.eval()

        # Load BiomedCLIP (open_clip) for cross-modal text->visual matching
        self.clip_model, _, _ = open_clip.create_model_and_transforms(biomedclip_model_id, pretrained=True)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

    def embed_text_bio(self, text, max_length=512):
        """
        Generate text embedding using Bio_ClinicalBERT with CLS token pooling.
        This matches the embedding generation strategy in multimodal_pipeline_with_sliding_window.py

        Args:
            text: Input text query
            max_length: Maximum sequence length (default 512 to match training)
        """
        inputs = self.bio_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        # move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.bio_model(**inputs)
            # Use CLS token pooling (position 0) to match training embeddings
            emb = outputs.last_hidden_state[:, 0, :].squeeze()

        # Normalize embedding to match FAISS index normalization
        vec = emb.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_text_clip(self, text):
        """
        Generate text embedding using BiomedCLIP for cross-modal retrieval.
        """
        tokens = open_clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens).squeeze(0)

        # Normalize embedding to match FAISS index normalization
        vec = emb.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def format_timestamp(ts):
    """Format timestamp tuple (start, end) to readable string"""
    if not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
        return "unknown time"
    start, end = ts
    return f"{start:.2f}s - {end:.2f}s"


def extract_metadata(result):
    """Extract useful metadata fields from a search result"""
    meta = result.get("meta", {}) or {}

    # Common fields across text and visual results
    video_id = meta.get("video_id", "unknown")
    timestamp = format_timestamp(meta.get("timestamp"))

    # Text-specific fields
    text = meta.get("text", "")
    entities = meta.get("entities", [])
    entity_text = ", ".join(ent[0] for ent in entities) if entities else ""

    # Sliding window specific metadata
    window_info = meta.get("window_info", {})
    processing_type = meta.get("processing_type", "unknown")

    # Handle both chunk_id (hybrid) and window_id (sliding window)
    chunk_id = meta.get("chunk_id")
    window_id = meta.get("window_id")
    mini_window_id = meta.get("mini_window_id")

    result_dict = {
        "video_id": video_id,
        "timestamp": timestamp,
        "text": text,
        "entities": entity_text,
        "processing_type": processing_type
    }

    # Add window information if available
    if window_info:
        result_dict["window_size"] = window_info.get("window_size")
        result_dict["window_range"] = f"tokens {window_info.get('start_token', 'N/A')}-{window_info.get('end_token', 'N/A')}"

    if chunk_id is not None:
        result_dict["chunk_id"] = chunk_id
    if window_id is not None:
        result_dict["window_id"] = window_id
    if mini_window_id is not None:
        result_dict["mini_window_id"] = mini_window_id

    return result_dict


def aggregate_results_by_segment(text_results, visual_results, top_k=10, text_weight=0.6, visual_weight=0.4):
    """
    Aggregate multimodal search results by segment_id for precise multimodal linking.

    This leverages the segment_id created during embedding generation to precisely
    match text and visual embeddings from the same video segment.

    Args:
        text_results: List of dicts from textual index search
        visual_results: List of dicts from visual index search
        top_k: Number of top segments to return
        text_weight: Weight for textual similarity (default: 0.6)
        visual_weight: Weight for visual similarity (default: 0.4)

    Returns:
        List of segment contexts sorted by relevance, each containing:
        - video_id
        - segment_id (unique identifier linking text and visual)
        - combined_score (weighted average when both modalities available, otherwise single modality)
        - text_evidence (text content, entities, similarity)
        - visual_evidence (frame info, similarity)
        - timestamp
    """
    from collections import defaultdict

    # First, group by segment_id for precise multimodal linking
    segment_contexts = defaultdict(lambda: {
        "video_id": None,
        "segment_id": None,
        "timestamp": None,
        "text_evidence": None,
        "visual_evidence": None,
        "text_score": 0.0,
        "visual_score": 0.0,
        "combined_score": 0.0,
        "has_both_modalities": False
    })

    # Process textual results
    for result in text_results:
        meta = result.get("meta", {}) or {}
        segment_id = meta.get("segment_id", f"{meta.get('video_id', 'unknown')}_seg_unknown")
        video_id = meta.get("video_id", "unknown")

        # Normalize distance to similarity using exponential decay for better score distribution
        dist = result.get("raw_score", float('inf'))
        similarity = np.exp(-dist) if np.isfinite(dist) else 0.0

        # Only update if this is a better text match for this segment
        if similarity > segment_contexts[segment_id]["text_score"]:
            segment_contexts[segment_id].update({
                "video_id": video_id,
                "segment_id": segment_id,
                "timestamp": meta.get("timestamp", "unknown"),
                "text_score": similarity,
                "text_evidence": {
                    "text": meta.get("text", ""),
                    "entities": meta.get("entities", []),
                    "similarity": similarity,
                    "raw_distance": dist,
                    "window_info": meta.get("window_info", {})
                }
            })

    # Process visual results
    for result in visual_results:
        meta = result.get("meta", {}) or {}
        segment_id = meta.get("segment_id", f"{meta.get('video_id', 'unknown')}_seg_unknown")
        video_id = meta.get("video_id", "unknown")

        dist = result.get("raw_score", float('inf'))
        similarity = np.exp(-dist) if np.isfinite(dist) else 0.0

        # Only update if this is a better visual match for this segment
        if similarity > segment_contexts[segment_id]["visual_score"]:
            if segment_contexts[segment_id]["video_id"] is None:
                segment_contexts[segment_id]["video_id"] = video_id
                segment_contexts[segment_id]["segment_id"] = segment_id
                segment_contexts[segment_id]["timestamp"] = meta.get("timestamp", "unknown")

            segment_contexts[segment_id].update({
                "visual_score": similarity,
                "visual_evidence": {
                    "frame_info": meta.get("frame_path", "in-memory"),
                    "num_frames": meta.get("num_frames_averaged", 1),
                    "similarity": similarity,
                    "raw_distance": dist
                }
            })

    # Calculate combined scores with proper weighting
    for segment_id, ctx in segment_contexts.items():
        has_text = ctx["text_score"] > 0
        has_visual = ctx["visual_score"] > 0

        if has_text and has_visual:
            # Both modalities available: weighted combination
            ctx["combined_score"] = (text_weight * ctx["text_score"] +
                                    visual_weight * ctx["visual_score"])
            ctx["has_both_modalities"] = True
        elif has_text:
            # Only text available
            ctx["combined_score"] = ctx["text_score"]
        elif has_visual:
            # Only visual available
            ctx["combined_score"] = ctx["visual_score"]
        else:
            ctx["combined_score"] = 0.0

    # Sort by combined score and return top-k
    sorted_segments = sorted(
        [ctx for ctx in segment_contexts.values() if ctx["combined_score"] > 0],
        key=lambda x: (x["has_both_modalities"], x["combined_score"]),  # Prioritize segments with both modalities
        reverse=True
    )

    return sorted_segments[:top_k]


def aggregate_results_by_video(text_results, visual_results, top_k=5, text_weight=0.6, visual_weight=0.4):
    """
    Aggregate multimodal search results by video_id for video-level context.

    This groups all segments from the same video and provides comprehensive evidence.

    Args:
        text_results: List of dicts from textual index search
        visual_results: List of dicts from visual index search
        top_k: Number of top videos to return
        text_weight: Weight for textual similarity
        visual_weight: Weight for visual similarity

    Returns:
        List of video contexts sorted by relevance with all matching segments
    """
    from collections import defaultdict

    video_contexts = defaultdict(lambda: {
        "video_id": None,
        "segments": [],
        "best_combined_score": 0.0,
        "num_text_matches": 0,
        "num_visual_matches": 0,
        "num_multimodal_matches": 0
    })

    # First get segment-level aggregation
    segment_results = aggregate_results_by_segment(
        text_results, visual_results,
        top_k=len(text_results) + len(visual_results),  # Get all segments first
        text_weight=text_weight,
        visual_weight=visual_weight
    )

    # Group segments by video
    for segment in segment_results:
        video_id = segment["video_id"]

        if video_contexts[video_id]["video_id"] is None:
            video_contexts[video_id]["video_id"] = video_id

        video_contexts[video_id]["segments"].append(segment)
        video_contexts[video_id]["best_combined_score"] = max(
            video_contexts[video_id]["best_combined_score"],
            segment["combined_score"]
        )

        if segment["has_both_modalities"]:
            video_contexts[video_id]["num_multimodal_matches"] += 1
        if segment["text_evidence"]:
            video_contexts[video_id]["num_text_matches"] += 1
        if segment["visual_evidence"]:
            video_contexts[video_id]["num_visual_matches"] += 1

    # Sort segments within each video
    for ctx in video_contexts.values():
        ctx["segments"] = sorted(
            ctx["segments"],
            key=lambda x: x["combined_score"],
            reverse=True
        )

    # Sort videos by best segment score and multimodal coverage
    sorted_videos = sorted(
        video_contexts.values(),
        key=lambda x: (x["num_multimodal_matches"], x["best_combined_score"]),
        reverse=True
    )

    return sorted_videos[:top_k]


def print_segment_results(segment_contexts, query=None):
    """
    Pretty print segment-level multimodal search results.
    """
    if not segment_contexts:
        print("No segment contexts found.")
        return

    print("\n" + "=" * 100)
    print("MULTIMODAL SEGMENT-LEVEL SEARCH RESULTS")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Found {len(segment_contexts)} relevant segments")
    print("=" * 100)

    for i, ctx in enumerate(segment_contexts, 1):
        video_id = ctx["video_id"]
        segment_id = ctx["segment_id"]
        combined_score = ctx["combined_score"]
        timestamp = ctx["timestamp"]
        has_both = ctx["has_both_modalities"]

        modality_indicator = "🔗 [TEXT+VISUAL]" if has_both else ("📝 [TEXT]" if ctx["text_evidence"] else "🖼️ [VISUAL]")

        print(f"\n{i}. {modality_indicator} Video: {video_id} | Segment: {segment_id}")
        print(f"   📊 Combined Score: {combined_score:.4f} (text: {ctx['text_score']:.4f}, visual: {ctx['visual_score']:.4f})")
        print(f"   ⏱️  Timestamp: {format_timestamp(timestamp) if isinstance(timestamp, (list, tuple)) else timestamp}")

        # Show textual evidence
        if ctx["text_evidence"]:
            text_ev = ctx["text_evidence"]
            print(f"\n   📝 Text Evidence (similarity: {text_ev['similarity']:.4f}):")
            snippet = text_ev['text'][:200] + "..." if len(text_ev['text']) > 200 else text_ev['text']
            print(f"      {snippet}")
            if text_ev.get('entities'):
                entities_str = ", ".join([ent[0] if isinstance(ent, (list, tuple)) else str(ent)
                                         for ent in text_ev['entities'][:5]])
                print(f"      Medical Entities: {entities_str}")

        # Show visual evidence
        if ctx["visual_evidence"]:
            vis_ev = ctx["visual_evidence"]
            print(f"\n   🖼️  Visual Evidence (similarity: {vis_ev['similarity']:.4f}):")
            print(f"      Frames averaged: {vis_ev['num_frames']}")

        print("-" * 100)

    # Save to JSON
    output_file = "multimodal_search_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "query": query,
            "total_segments": len(segment_contexts),
            "aggregation_mode": "segment",
            "results": segment_contexts
        }, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_file}")


def print_video_contexts(video_contexts, query=None):
    """
    Pretty print video-aggregated search results showing all matching segments.
    """
    if not video_contexts:
        print("No video contexts found.")
        return

    print("\n" + "=" * 100)
    print("MULTIMODAL VIDEO-LEVEL SEARCH RESULTS")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Found {len(video_contexts)} relevant videos")
    print("=" * 100)

    for i, ctx in enumerate(video_contexts, 1):
        video_id = ctx["video_id"]
        best_score = ctx["best_combined_score"]
        num_segments = len(ctx["segments"])
        num_multimodal = ctx["num_multimodal_matches"]

        print(f"\n{i}. 🎥 Video: {video_id}")
        print(f"   📊 Best Score: {best_score:.4f} | Segments: {num_segments} (multimodal: {num_multimodal})")

        # Show top 3 segments from this video
        print(f"\n   🔝 Top Segments:")
        for j, seg in enumerate(ctx["segments"][:3], 1):
            timestamp = format_timestamp(seg["timestamp"]) if isinstance(seg["timestamp"], (list, tuple)) else seg["timestamp"]
            modality_str = "TEXT+VISUAL" if seg["has_both_modalities"] else ("TEXT" if seg["text_evidence"] else "VISUAL")

            print(f"\n      {j}. [{timestamp}] {modality_str} (score: {seg['combined_score']:.4f})")

            if seg["text_evidence"]:
                snippet = seg["text_evidence"]["text"][:120] + "..." if len(seg["text_evidence"]["text"]) > 120 else seg["text_evidence"]["text"]
                print(f"         📝 {snippet}")

            if seg["visual_evidence"]:
                print(f"         🖼️  {seg['visual_evidence']['num_frames']} frames averaged")

        if num_segments > 3:
            print(f"\n      ... and {num_segments - 3} more segments")

        print("-" * 100)

    # Save to JSON
    output_file = "multimodal_search_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "query": query,
            "total_videos": len(video_contexts),
            "aggregation_mode": "video",
            "results": video_contexts
        }, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_file}")


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
    args = parser.parse_args()

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
            "raw_score": dist
        })

    for item in visual_pool:
        dist = item.get("raw_score", float('inf'))
        sim = 1.0 / (1.0 + dist) if np.isfinite(dist) else 0.0
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

        print_video_contexts(video_contexts, query=args.query)

    else:
        # Segment-level aggregation: precise multimodal linking via segment_id
        print(f"\nAggregating results by segment (top {args.final_k} segments)")
        print(f"Using weights: text={args.text_weight}, visual={args.visual_weight}")

        segment_contexts = aggregate_results_by_segment(
            text_pool,
            visual_pool,
            top_k=args.final_k,
            text_weight=args.text_weight,
            visual_weight=args.visual_weight
        )

        print_segment_results(segment_contexts, query=args.query)


if __name__ == "__main__":
    main()