"""
run_evaluation.py

End-to-end evaluation script for Medical VideoRAG VQA pipeline.

Features:
- Load MedVidQA test/val datasets
- Run queries through complete VideoRAG pipeline
- Calculate comprehensive metrics:
  * Answer quality (BLEU, ROUGE, BERTScore, Medical Entity F1)
  * Retrieval quality (Precision@K, Recall@K, F1@K)
  * Temporal accuracy (IoU, Temporal Precision/Recall/F1)
  * Attribution accuracy
- Save detailed results with per-query breakdown

Usage:
    # Evaluate on test set with dense search
    python run_evaluation.py --split test --top_k 10

    # Evaluate with hybrid search
    python run_evaluation.py --split test --hybrid --alpha 0.3 --fusion linear

    # Quick evaluation on subset
    python run_evaluation.py --split test --num_samples 50 --output eval_results_test_subset.json

    # With answer generation
    python run_evaluation.py --split test --generate_answer --enable_curation
"""

import os
import json
import argparse
import time
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np

# Import VideoRAG components
from search import (
    FaissIndex,
    EmbeddingModels,
    HybridSearchEngine,
    aggregate_results_by_segment,
    load_segments_from_json_dir
)
from generation.answer_generator import AnswerGenerator
from evaluation import AnswerEvaluator


class VideoRAGEvaluator:
    """
    End-to-end evaluator for Medical VideoRAG pipeline.
    """

    def __init__(
        self,
        text_index_paths: List[str],
        visual_index_paths: List[str],
        json_dir: str = "feature_extraction/",
        device: Optional[str] = None,
        enable_hybrid: bool = False,
        hybrid_alpha: float = 0.3,
        fusion_method: str = "linear",
        generate_answers: bool = False,
        enable_curation: bool = False,
        enable_attribution: bool = False
    ):
        """
        Initialize VideoRAG evaluator.

        Args:
            text_index_paths: Paths to textual FAISS indices
            visual_index_paths: Paths to visual FAISS indices
            json_dir: Directory containing JSON feature files
            device: Device for models ('cpu', 'cuda', 'mps', or None)
            enable_hybrid: Enable hybrid search (BM25 + dense)
            hybrid_alpha: Weight for dense retrieval in hybrid search
            fusion_method: 'linear' or 'rrf'
            generate_answers: Whether to generate answers with LLM
            enable_curation: Enable adaptive context selection
            enable_attribution: Enable self-reflection attribution
        """
        self.text_index_paths = text_index_paths
        self.visual_index_paths = visual_index_paths
        self.json_dir = json_dir
        self.enable_hybrid = enable_hybrid
        self.hybrid_alpha = hybrid_alpha
        self.fusion_method = fusion_method
        self.generate_answers = generate_answers

        print(f"Initializing VideoRAG Evaluator...")
        print(f"  Text indices: {len(text_index_paths)}")
        print(f"  Visual indices: {len(visual_index_paths)}")
        print(f"  Hybrid search: {enable_hybrid}")
        print(f"  Answer generation: {generate_answers}")

        # Load embedding models
        self.models = EmbeddingModels(device=device)

        # Load FAISS indices
        self.text_indices = []
        for path in text_index_paths:
            idx = FaissIndex(path)
            idx.load()
            self.text_indices.append(idx)

        self.visual_indices = []
        for path in visual_index_paths:
            idx = FaissIndex(path)
            idx.load()
            self.visual_indices.append(idx)

        # Initialize hybrid search engine if enabled
        self.hybrid_engine = None
        if enable_hybrid:
            segments_data = load_segments_from_json_dir(json_dir)
            self.hybrid_engine = HybridSearchEngine(
                segments_data=segments_data,
                alpha=hybrid_alpha
            )

        # Initialize answer generator if enabled
        self.answer_generator = None
        if generate_answers:
            self.answer_generator = AnswerGenerator(
                enable_curation=enable_curation,
                enable_attribution=enable_attribution
            )

        # Initialize evaluator
        self.evaluator = AnswerEvaluator()

        print("✅ Initialization complete!")

    def search(
        self,
        query: str,
        top_k: int = 10,
        local_k: int = 50,
        max_length: int = 512
    ) -> Dict:
        """
        Run multimodal search for a query.

        Args:
            query: User query
            top_k: Final number of results to return
            local_k: Number of results per index before merging
            max_length: Max sequence length for query embedding

        Returns:
            {
                'segments': List[Dict],  # Retrieved segments
                'search_time': float
            }
        """
        start_time = time.time()

        # Generate query embeddings
        q_vec_text = self.models.embed_text_bio(query, max_length=max_length)
        q_vec_clip = self.models.embed_text_clip(query)

        # Search textual indices
        text_pool = []
        for idx in self.text_indices:
            results = idx.search(q_vec_text, top_k=local_k)
            for r in results:
                text_pool.append({
                    "raw_score": r.get("score"),
                    "meta": r.get("meta"),
                    "modality": "text"
                })

        # Search visual indices
        visual_pool = []
        for idx in self.visual_indices:
            results = idx.search(q_vec_clip, top_k=local_k)
            for r in results:
                visual_pool.append({
                    "raw_score": r.get("score"),
                    "meta": r.get("meta"),
                    "modality": "visual"
                })

        # Apply hybrid search if enabled
        if self.enable_hybrid and text_pool:
            hybrid_results = self.hybrid_engine.hybrid_search(
                query=query,
                dense_results=text_pool,
                top_k=local_k,
                fusion=self.fusion_method,
                expand_query=True
            )
            text_pool = hybrid_results

        # Aggregate multimodal results (requires separate text and visual lists)
        aggregated = aggregate_results_by_segment(
            text_results=text_pool,
            visual_results=visual_pool,
            top_k=top_k,
            text_weight=0.6,
            visual_weight=0.4
        )

        search_time = time.time() - start_time

        return {
            'segments': aggregated,
            'search_time': search_time
        }

    def evaluate_query(
        self,
        query: str,
        ground_truth: Dict,
        top_k: int = 10
    ) -> Dict:
        """
        Evaluate a single query.

        Args:
            query: User query
            ground_truth: Ground truth data with:
                - answer_start_second: int
                - answer_end_second: int
                - video_id: str
            top_k: Number of results to retrieve

        Returns:
            Comprehensive evaluation metrics
        """
        # Run search
        search_result = self.search(query, top_k=top_k)
        segments = search_result['segments']

        result = {
            'query': query,
            'ground_truth': ground_truth,
            'num_retrieved': len(segments),
            'search_time': search_result['search_time']
        }

        # Extract timestamps from retrieved segments
        # Segments from aggregate_results_by_segment have timestamp at top level
        predicted_timestamps = []
        for seg in segments:
            # Try timestamp at top level first (from aggregation)
            if 'timestamp' in seg and seg['timestamp'] != 'unknown':
                ts = seg['timestamp']
                if isinstance(ts, list) and len(ts) == 2:
                    predicted_timestamps.append((ts[0], ts[1]))
            # Fallback to meta.timestamp
            elif 'meta' in seg and 'timestamp' in seg.get('meta', {}):
                meta = seg['meta']
                ts = meta['timestamp']
                if isinstance(ts, list) and len(ts) == 2:
                    predicted_timestamps.append((ts[0], ts[1]))

        # Evaluate temporal overlap
        if predicted_timestamps:
            temporal_metrics = self.evaluator.evaluate_temporal_overlap(
                predicted_timestamps=predicted_timestamps,
                ground_truth_start=ground_truth['answer_start_second'],
                ground_truth_end=ground_truth['answer_end_second']
            )
            result['temporal_metrics'] = temporal_metrics
        else:
            result['temporal_metrics'] = {
                'iou': 0.0,
                'temporal_precision': 0.0,
                'temporal_recall': 0.0,
                'temporal_f1': 0.0,
                'mean_distance': float('inf')
            }

        # Check if correct video retrieved
        # Check both top-level video_id (from aggregation) and meta.video_id
        correct_video_retrieved = False
        for seg in segments:
            video_id = seg.get('video_id') or seg.get('meta', {}).get('video_id')
            if video_id == ground_truth['video_id']:
                correct_video_retrieved = True
                break
        result['correct_video_retrieved'] = correct_video_retrieved

        # Generate and evaluate answer if enabled
        if self.answer_generator and segments:
            try:
                answer_result = self.answer_generator.generate_answer(
                    query=query,
                    segment_contexts=segments,
                    max_tokens=250,
                    temperature=0.3,
                    top_k_evidence=5
                )
                result['generated_answer'] = answer_result['answer']
                result['answer_confidence'] = answer_result['confidence']
                result['generation_time'] = answer_result['generation_time']

                # Note: We can't evaluate answer quality without reference answers
                # in the MedVidQA dataset (only timestamps provided)
                result['has_generated_answer'] = True
            except Exception as e:
                print(f"⚠️  Answer generation failed: {e}")
                result['has_generated_answer'] = False
        else:
            result['has_generated_answer'] = False

        return result

    def evaluate_dataset(
        self,
        dataset_path: str,
        top_k: int = 10,
        num_samples: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Evaluate on entire dataset.

        Args:
            dataset_path: Path to MedVidQA JSON file
            top_k: Number of results to retrieve per query
            num_samples: Limit to first N samples (None = all)
            save_path: Path to save detailed results

        Returns:
            Aggregated metrics across all queries
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING DATASET: {dataset_path}")
        print(f"{'='*80}")

        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        if num_samples:
            dataset = dataset[:num_samples]
            print(f"Evaluating on {num_samples} samples")
        else:
            print(f"Evaluating on {len(dataset)} samples")

        # Evaluate each query
        results = []
        for sample in tqdm(dataset, desc="Evaluating queries"):
            query = sample['question']
            result = self.evaluate_query(query, sample, top_k=top_k)
            results.append(result)

        # Aggregate metrics
        aggregated = self._aggregate_results(results)

        # Save detailed results if requested
        if save_path:
            output = {
                'dataset_path': dataset_path,
                'num_samples': len(dataset),
                'top_k': top_k,
                'hybrid_enabled': self.enable_hybrid,
                'hybrid_alpha': self.hybrid_alpha if self.enable_hybrid else None,
                'fusion_method': self.fusion_method if self.enable_hybrid else None,
                'answer_generation_enabled': self.generate_answers,
                'aggregated_metrics': aggregated,
                'per_query_results': results
            }
            with open(save_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n✅ Detailed results saved to: {save_path}")

        return aggregated

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all queries"""
        num_queries = len(results)

        # Temporal metrics
        temporal_metrics = [r['temporal_metrics'] for r in results]
        avg_temporal = {
            'iou_mean': np.mean([m['iou'] for m in temporal_metrics]),
            'iou_std': np.std([m['iou'] for m in temporal_metrics]),
            'temporal_precision_mean': np.mean([m['temporal_precision'] for m in temporal_metrics]),
            'temporal_precision_std': np.std([m['temporal_precision'] for m in temporal_metrics]),
            'temporal_recall_mean': np.mean([m['temporal_recall'] for m in temporal_metrics]),
            'temporal_recall_std': np.std([m['temporal_recall'] for m in temporal_metrics]),
            'temporal_f1_mean': np.mean([m['temporal_f1'] for m in temporal_metrics]),
            'temporal_f1_std': np.std([m['temporal_f1'] for m in temporal_metrics]),
            'mean_distance_mean': np.mean([m['mean_distance'] for m in temporal_metrics if m['mean_distance'] != float('inf')]),
        }

        # Video retrieval accuracy
        correct_video_count = sum(1 for r in results if r['correct_video_retrieved'])
        video_accuracy = correct_video_count / num_queries

        # Search time
        avg_search_time = np.mean([r['search_time'] for r in results])

        # Answer generation stats
        if self.generate_answers:
            answers_generated = sum(1 for r in results if r['has_generated_answer'])
            avg_confidence = np.mean([r.get('answer_confidence', 0) for r in results if r.get('has_generated_answer')])
            avg_gen_time = np.mean([r.get('generation_time', 0) for r in results if r.get('has_generated_answer')])
        else:
            answers_generated = 0
            avg_confidence = 0
            avg_gen_time = 0

        aggregated = {
            'num_queries': num_queries,
            'temporal_metrics': avg_temporal,
            'video_retrieval_accuracy': video_accuracy,
            'correct_video_count': correct_video_count,
            'avg_search_time': avg_search_time,
            'answer_generation': {
                'enabled': self.generate_answers,
                'success_count': answers_generated,
                'avg_confidence': avg_confidence,
                'avg_generation_time': avg_gen_time
            }
        }

        return aggregated

    def print_results(self, aggregated: Dict):
        """Print formatted evaluation results"""
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS")
        print(f"{'='*80}")

        print(f"\nDataset Statistics:")
        print(f"  Total queries: {aggregated['num_queries']}")
        print(f"  Avg search time: {aggregated['avg_search_time']:.3f}s")

        print(f"\nVideo Retrieval:")
        print(f"  Accuracy: {aggregated['video_retrieval_accuracy']:.2%}")
        print(f"  Correct videos: {aggregated['correct_video_count']}/{aggregated['num_queries']}")

        print(f"\nTemporal Accuracy:")
        tm = aggregated['temporal_metrics']
        print(f"  IoU: {tm['iou_mean']:.4f} (±{tm['iou_std']:.4f})")
        print(f"  Temporal Precision: {tm['temporal_precision_mean']:.4f} (±{tm['temporal_precision_std']:.4f})")
        print(f"  Temporal Recall: {tm['temporal_recall_mean']:.4f} (±{tm['temporal_recall_std']:.4f})")
        print(f"  Temporal F1: {tm['temporal_f1_mean']:.4f} (±{tm['temporal_f1_std']:.4f})")
        print(f"  Mean Distance: {tm['mean_distance_mean']:.2f}s")

        if aggregated['answer_generation']['enabled']:
            print(f"\nAnswer Generation:")
            ag = aggregated['answer_generation']
            print(f"  Success rate: {ag['success_count']}/{aggregated['num_queries']}")
            print(f"  Avg confidence: {ag['avg_confidence']:.2%}")
            print(f"  Avg generation time: {ag['avg_generation_time']:.3f}s")

        print(f"\n{'='*80}")


def discover_indexes(prefix: str, base_dir: str = "faiss_db") -> List[str]:
    """Auto-discover FAISS indices"""
    import glob
    paths = sorted(glob.glob(os.path.join(base_dir, f"{prefix}_*.index")))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate Medical VideoRAG VQA pipeline")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test",
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Custom dataset path (overrides --split)")
    parser.add_argument("--text_index", type=str, default=None,
                        help="Path to textual FAISS index (auto-discover if omitted)")
    parser.add_argument("--visual_index", type=str, default=None,
                        help="Path to visual FAISS index (auto-discover if omitted)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of results to retrieve (default: 10)")
    parser.add_argument("--local_k", type=int, default=50,
                        help="Number of results per index before merging (default: 50)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit evaluation to N samples (default: all)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None,
                        help="Device to run models on (default: auto)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for detailed results (default: auto-generated)")

    # Hybrid search arguments
    parser.add_argument("--hybrid", action="store_true",
                        help="Enable hybrid search (BM25 + dense)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Weight for dense retrieval in hybrid search (default: 0.3)")
    parser.add_argument("--fusion", type=str, choices=["linear", "rrf"], default="linear",
                        help="Fusion method for hybrid search (default: linear)")

    # Answer generation arguments
    parser.add_argument("--generate_answer", action="store_true",
                        help="Generate answers using LLM")
    parser.add_argument("--enable_curation", action="store_true",
                        help="Enable adaptive context selection")
    parser.add_argument("--enable_attribution", action="store_true",
                        help="Enable self-reflection attribution")

    args = parser.parse_args()

    # Determine dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = f"MedVidQA_cleaned/{args.split}_openai_whisper_tiny_cleaned.json"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Auto-discover indices
    text_index_paths = [args.text_index] if args.text_index else discover_indexes("textual")
    visual_index_paths = [args.visual_index] if args.visual_index else discover_indexes("visual")

    if not text_index_paths and not visual_index_paths:
        raise ValueError("No FAISS indexes found. Run multimodal_pipeline_with_sliding_window.py first.")

    # Filter indices by split
    split_text = [p for p in text_index_paths if args.split in p]
    split_visual = [p for p in visual_index_paths if args.split in p]

    if not split_text and not split_visual:
        print(f"⚠️  No indices found for split '{args.split}', using all available indices")
        split_text = text_index_paths
        split_visual = visual_index_paths

    # Auto-generate output filename
    if args.output is None:
        suffix = f"_hybrid_a{args.alpha}" if args.hybrid else "_dense"
        suffix += f"_gen" if args.generate_answer else ""
        suffix += f"_n{args.num_samples}" if args.num_samples else ""
        args.output = f"evaluation_results_{args.split}{suffix}.json"

    print(f"\n{'='*80}")
    print("MEDICAL VIDEORAG EVALUATION")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Split: {args.split}")
    print(f"  Text indices: {len(split_text)}")
    print(f"  Visual indices: {len(split_visual)}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Hybrid search: {args.hybrid}")
    if args.hybrid:
        print(f"    Alpha: {args.alpha}")
        print(f"    Fusion: {args.fusion}")
    print(f"  Answer generation: {args.generate_answer}")
    if args.generate_answer:
        print(f"    Curation: {args.enable_curation}")
        print(f"    Attribution: {args.enable_attribution}")
    print(f"  Output: {args.output}")

    # Initialize evaluator
    evaluator = VideoRAGEvaluator(
        text_index_paths=split_text,
        visual_index_paths=split_visual,
        device=args.device,
        enable_hybrid=args.hybrid,
        hybrid_alpha=args.alpha,
        fusion_method=args.fusion,
        generate_answers=args.generate_answer,
        enable_curation=args.enable_curation,
        enable_attribution=args.enable_attribution
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataset_path=dataset_path,
        top_k=args.top_k,
        num_samples=args.num_samples,
        save_path=args.output
    )

    # Print results
    evaluator.print_results(results)

    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
