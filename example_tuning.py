"""
example_tuning.py
Simple example demonstrating hyperparameter tuning usage.

This shows the most common workflows for finding optimal parameters.
"""

from hyperparameter_tuning import (
    HyperparameterEvaluator,
    quick_test_configs,
    random_search,
    grid_search
)
import os

# Example 1: Quick Test (Recommended First Step)
def example_quick_test():
    """
    Test 4 preset configurations to understand the landscape.
    Takes ~10-20 minutes depending on video length.
    """
    print("="*80)
    print("EXAMPLE 1: Quick Test with Preset Configurations")
    print("="*80)
    
    video_path = "videos_train/_6csIJAWj_s.mp4"  # Change to your video
    evaluator = HyperparameterEvaluator(video_path)
    
    # Test 4 preset configs: Baseline, Fast, Quality, Balanced
    quick_test_configs(evaluator)
    
    # Save results
    output_dir = "hyperparameter_tuning_results"
    os.makedirs(output_dir, exist_ok=True)
    evaluator.save_results(f"{output_dir}/quick_test_results.json")
    
    # Find best config
    best = evaluator.get_best_config(metric='text_coverage', higher_is_better=True)
    
    # Generate plots
    evaluator.plot_results(output_dir=f"{output_dir}/plots")
    
    print("\n✅ Quick test complete!")
    print(f"   Results: {output_dir}/quick_test_results.json")
    print(f"   Plots: {output_dir}/plots/")
    
    return best


# Example 2: Random Search
def example_random_search(n_iterations=10):
    """
    Test N random configurations.
    Good balance between coverage and speed.
    """
    print("="*80)
    print(f"EXAMPLE 2: Random Search ({n_iterations} configurations)")
    print("="*80)
    
    video_path = "videos_train/_6csIJAWj_s.mp4"
    evaluator = HyperparameterEvaluator(video_path)
    
    # Test N random configurations
    random_search(evaluator, n_iterations=n_iterations)
    
    # Save and analyze
    output_dir = "hyperparameter_tuning_results"
    os.makedirs(output_dir, exist_ok=True)
    evaluator.save_results(f"{output_dir}/random_search_results.json")
    
    # Find best configs for different metrics
    print("\nBest for coverage:")
    best_coverage = evaluator.get_best_config(metric='text_coverage', higher_is_better=True)
    
    print("\nBest for speed:")
    best_speed = evaluator.get_best_config(metric='processing_time', higher_is_better=False)
    
    evaluator.plot_results(output_dir=f"{output_dir}/plots")
    
    return best_coverage, best_speed


# Example 3: Limited Grid Search
def example_grid_search():
    """
    Exhaustively test a limited grid of parameters.
    Warning: Can be slow if grid is too large!
    """
    print("="*80)
    print("EXAMPLE 3: Limited Grid Search")
    print("="*80)
    
    video_path = "videos_train/_6csIJAWj_s.mp4"
    evaluator = HyperparameterEvaluator(video_path)
    
    # Define a limited grid (only 8 combinations = 2×2×2×1×1×1×1×1)
    limited_grid = {
        'window_size': [256, 384],
        'stride': [128, 192],
        'min_coverage_contribution': [0.15],
        'deduplication_mode': ['coverage'],
        'frames_per_segment': [2, 3],
        'sampling_strategy': ['uniform'],
        'quality_filter': [False],
        'aggregation_method': ['mean']
    }
    
    # Run grid search
    grid_search(evaluator, param_subset=limited_grid)
    
    # Save and analyze
    output_dir = "hyperparameter_tuning_results"
    os.makedirs(output_dir, exist_ok=True)
    evaluator.save_results(f"{output_dir}/grid_search_results.json")
    
    best = evaluator.get_best_config(metric='text_coverage', higher_is_better=True)
    evaluator.plot_results(output_dir=f"{output_dir}/plots")
    
    return best


# Example 4: Custom Configuration Testing
def example_custom_config():
    """
    Test a specific configuration you designed.
    """
    print("="*80)
    print("EXAMPLE 4: Test Custom Configuration")
    print("="*80)
    
    video_path = "videos_train/_6csIJAWj_s.mp4"
    evaluator = HyperparameterEvaluator(video_path)
    
    # Your custom configuration
    my_config = {
        'window_size': 300,
        'stride': 150,
        'min_coverage_contribution': 0.12,
        'deduplication_mode': 'coverage',
        'frames_per_segment': 3,
        'sampling_strategy': 'adaptive',
        'quality_filter': False,
        'aggregation_method': 'mean'
    }
    
    print("\nTesting your custom configuration:")
    metrics = evaluator.evaluate_config(my_config)
    
    print("\n📊 Results:")
    print(f"   Text Coverage: {metrics['text_coverage']:.2f}%")
    print(f"   Text Embeddings: {metrics['text_embeddings_count']}")
    print(f"   Visual Embeddings: {metrics['visual_embeddings_count']}")
    print(f"   Processing Time: {metrics['processing_time']:.2f}s")
    
    return metrics


# Example 5: With Ground Truth Evaluation
def example_with_ground_truth():
    """
    Evaluate configurations with search relevance metrics.
    Requires ground truth queries with known relevant segments.
    """
    print("="*80)
    print("EXAMPLE 5: Evaluation with Ground Truth")
    print("="*80)
    
    video_path = "videos_train/_6csIJAWj_s.mp4"
    
    # Define ground truth queries
    # You need to manually label which segments are relevant for each query
    ground_truth = [
        {
            'query': 'How to check for mouth cancer at home?',
            'relevant_segments': ['_6csIJAWj_s_seg_0', '_6csIJAWj_s_seg_1']
        },
        {
            'query': 'What are the warning signs?',
            'relevant_segments': ['_6csIJAWj_s_seg_2']
        }
    ]
    
    evaluator = HyperparameterEvaluator(video_path, ground_truth_queries=ground_truth)
    
    # Now search_relevance metric will be computed
    quick_test_configs(evaluator)
    
    # Find best by search relevance
    best = evaluator.get_best_config(metric='search_relevance', higher_is_better=True)
    
    print(f"\n✅ Best configuration for search relevance found!")
    print(f"   Search Relevance: {best['search_relevance']:.4f}")
    
    return best


# Main workflow
def main():
    """
    Choose which example to run.
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING EXAMPLES")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Quick Test (4 preset configs) - RECOMMENDED FIRST")
    print("  2. Random Search (10 random configs)")
    print("  3. Grid Search (limited grid)")
    print("  4. Custom Configuration Test")
    print("  5. With Ground Truth Evaluation")
    print("\n")
    
    choice = input("Enter example number (1-5) or 'all' to run example 1: ").strip()
    
    if choice == '1' or choice == '' or choice == 'all':
        best = example_quick_test()
        
    elif choice == '2':
        best_coverage, best_speed = example_random_search(n_iterations=10)
        
    elif choice == '3':
        best = example_grid_search()
        
    elif choice == '4':
        metrics = example_custom_config()
        
    elif choice == '5':
        best = example_with_ground_truth()
        
    else:
        print(f"Invalid choice: {choice}")
        print("Running default example 1...")
        best = example_quick_test()
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the results JSON in: hyperparameter_tuning_results/")
    print("2. View the plots in: hyperparameter_tuning_results/plots/")
    print("3. Use the best config with:")
    print("   python run_tuned_pipeline.py --mode demo --results hyperparameter_tuning_results/quick_test_results.json")


if __name__ == "__main__":
    main()
