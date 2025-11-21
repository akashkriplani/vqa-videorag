"""
run_tuned_pipeline.py
Run the multimodal pipeline with your optimized hyperparameters.

This script demonstrates how to use the results from hyperparameter tuning
to process videos with your best-performing configuration.
"""

import json
import os
from multimodal_pipeline_with_sliding_window import demo_pipeline, process_split

def load_best_config(results_path, metric='text_coverage', higher_is_better=True):
    """
    Load the best configuration from tuning results.

    Args:
        results_path: Path to tuning results JSON
        metric: Metric to optimize for
        higher_is_better: Whether higher values are better

    Returns:
        Best configuration dict
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Filter out failed configs
    valid_results = [r for r in results if 'error' not in r and r.get(metric) is not None]

    if not valid_results:
        print("No valid results found! Using default configuration.")
        return None

    # Find best
    if higher_is_better:
        best = max(valid_results, key=lambda x: x[metric])
    else:
        best = min(valid_results, key=lambda x: x[metric])

    print(f"\n{'='*80}")
    print(f"Loading best configuration (optimizing {metric}):")
    print(f"{'='*80}")
    for k, v in best['config'].items():
        print(f"  {k}: {v}")
    print(f"\nMetrics:")
    print(f"  text_coverage: {best.get('text_coverage', 'N/A')}")
    print(f"  text_embeddings_count: {best.get('text_embeddings_count', 'N/A')}")
    print(f"  visual_embeddings_count: {best.get('visual_embeddings_count', 'N/A')}")
    print(f"  processing_time: {best.get('processing_time', 'N/A')}")

    return best['config']


def run_demo_with_tuned_params(config=None):
    """
    Run demo pipeline with tuned or default parameters.

    Args:
        config: Config dict from tuning results (optional)
    """
    # Default configuration (baseline)
    default_config = {
        'window_size': 256,
        'stride': 192,
        'min_coverage_contribution': 0.15,
        'deduplication_mode': 'coverage',
        'frames_per_segment': 2,
        'sampling_strategy': 'uniform',
        'quality_filter': False,
        'aggregation_method': 'mean'
    }

    # Use provided config or default
    params = config if config else default_config

    print(f"\n{'='*80}")
    print("Running DEMO PIPELINE with configuration:")
    print(f"{'='*80}")
    for k, v in params.items():
        print(f"  {k}: {v}")

    demo_pipeline(
        video_path="videos_train/_6csIJAWj_s.mp4",
        text_feat_dir="feature_extraction_tuned/textual/demo",
        visual_feat_dir="feature_extraction_tuned/visual/demo",
        faiss_text_path="faiss_db_tuned/textual_demo.index",
        faiss_visual_path="faiss_db_tuned/visual_demo.index",
        **params
    )


def run_batch_with_tuned_params(config=None, splits=None):
    """
    Run batch processing with tuned or default parameters.

    Args:
        config: Config dict from tuning results (optional)
        splits: List of (split_name, video_dir) tuples to process
    """
    # Default configuration
    default_config = {
        'window_size': 256,
        'stride': 192,
        'min_coverage_contribution': 0.15,
        'deduplication_mode': 'coverage',
        'frames_per_segment': 2,
        'sampling_strategy': 'uniform',
        'quality_filter': False,
        'aggregation_method': 'mean'
    }

    # Use provided config or default
    params = config if config else default_config

    # Default splits
    if splits is None:
        splits = [
            ("train", "videos_train"),
            ("val", "videos_val"),
            ("test", "videos_test")
        ]

    print(f"\n{'='*80}")
    print("Running BATCH PROCESSING with configuration:")
    print(f"{'='*80}")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"\nSplits to process: {[s[0] for s in splits]}")

    for split, video_dir in splits:
        print(f"\n{'='*80}")
        print(f"Processing split: {split}")
        print(f"{'='*80}")

        process_split(
            split=split,
            video_dir=video_dir,
            text_feat_dir=f"feature_extraction_tuned/textual/{split}",
            visual_feat_dir=f"feature_extraction_tuned/visual/{split}",
            faiss_text_path=f"faiss_db_tuned/textual_{split}.index",
            faiss_visual_path=f"faiss_db_tuned/visual_{split}.index",
            **params
        )


def compare_configurations():
    """
    Compare different configurations side-by-side.
    """
    configs = {
        'Baseline': {
            'window_size': 256,
            'stride': 192,
            'min_coverage_contribution': 0.15,
            'deduplication_mode': 'coverage',
            'frames_per_segment': 2,
            'sampling_strategy': 'uniform',
            'quality_filter': False,
            'aggregation_method': 'mean'
        },
        'Fast': {
            'window_size': 128,
            'stride': 128,
            'min_coverage_contribution': 0.20,
            'deduplication_mode': 'aggressive',
            'frames_per_segment': 1,
            'sampling_strategy': 'uniform',
            'quality_filter': False,
            'aggregation_method': 'mean'
        },
        'Quality': {
            'window_size': 384,
            'stride': 96,
            'min_coverage_contribution': 0.10,
            'deduplication_mode': 'coverage',
            'frames_per_segment': 4,
            'sampling_strategy': 'quality_based',
            'quality_filter': True,
            'aggregation_method': 'mean'
        }
    }

    print(f"\n{'='*80}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*80}")

    for name, config in configs.items():
        print(f"\n{name}:")
        run_demo_with_tuned_params(config)


def main():
    """
    Main execution with multiple options.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run pipeline with tuned hyperparameters')
    parser.add_argument('--mode', choices=['demo', 'batch', 'compare'], default='demo',
                       help='Execution mode: demo (single video), batch (all splits), or compare (multiple configs)')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to hyperparameter tuning results JSON')
    parser.add_argument('--metric', type=str, default='text_coverage',
                       help='Metric to optimize for (text_coverage, processing_time, etc.)')
    parser.add_argument('--split', type=str, default=None,
                       help='Process only this split (train/val/test) in batch mode')

    args = parser.parse_args()

    # Load best config if results provided
    config = None
    if args.results and os.path.exists(args.results):
        config = load_best_config(args.results, metric=args.metric)
    else:
        print("\nNo tuning results provided. Using default configuration.")
        print("To use tuned parameters, run: python hyperparameter_tuning.py first")

    # Execute based on mode
    if args.mode == 'demo':
        run_demo_with_tuned_params(config)

    elif args.mode == 'batch':
        splits = [
            ("train", "videos_train"),
            ("val", "videos_val"),
            ("test", "videos_test")
        ]

        # Filter to specific split if requested
        if args.split:
            splits = [(s, d) for s, d in splits if s == args.split]
            if not splits:
                print(f"Error: Invalid split '{args.split}'. Choose from: train, val, test")
                return

        run_batch_with_tuned_params(config, splits)

    elif args.mode == 'compare':
        compare_configurations()

    print(f"\n{'='*80}")
    print("PIPELINE EXECUTION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
