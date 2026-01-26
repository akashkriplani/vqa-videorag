#!/usr/bin/env python3
"""
Quick validation script to verify curation and attribution fixes.
Runs a small sample to check if confidence and metrics improve.
"""

import json
import numpy as np
from pathlib import Path

def analyze_results(results_path):
    """Analyze evaluation results and print key metrics."""
    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"\n{'='*80}")
    print(f"ANALYSIS: {results_path}")
    print(f"{'='*80}\n")

    # Extract metrics
    confidences = []
    support_breakdowns = []
    curation_stats = []
    conflicts_counts = []

    for result in results:
        gen = result.get('generated_answer', {})

        # Confidence
        conf = gen.get('confidence', 0.0)
        confidences.append(conf)

        # Attribution breakdown
        attr_map = gen.get('attribution_map', {})
        if attr_map:
            support_breakdowns.append(attr_map.get('support_breakdown', {}))

        # Curation stats
        curation = gen.get('curation_stats', {})
        if curation:
            curation_stats.append(curation)

        # Conflicts
        conflicts = gen.get('conflicts_detected', [])
        conflicts_counts.append(len(conflicts) if conflicts else 0)

    # Print confidence statistics
    print("CONFIDENCE STATISTICS")
    print(f"  Mean: {np.mean(confidences):.2%}")
    print(f"  Median: {np.median(confidences):.2%}")
    print(f"  Std: {np.std(confidences):.2%}")
    print(f"  Min: {np.min(confidences):.2%}")
    print(f"  Max: {np.max(confidences):.2%}")

    # Print support breakdown aggregates
    if support_breakdowns:
        print("\nSUPPORT BREAKDOWN (Average per answer)")
        all_support = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'UNSUPPORTED': [], 'CONFLICTED': []}
        for breakdown in support_breakdowns:
            for level in all_support.keys():
                all_support[level].append(breakdown.get(level, 0))

        for level, counts in all_support.items():
            print(f"  {level:12s}: {np.mean(counts):.2f} claims (avg)")

    # Print curation statistics
    if curation_stats:
        print("\nCURATION STATISTICS (Average)")
        keys = ['input_segments', 'after_quality_filter', 'after_contradiction_detection',
                'final_selected', 'conflicts_found']
        for key in keys:
            values = [s.get(key, 0) for s in curation_stats]
            print(f"  {key:30s}: {np.mean(values):.2f}")

    # Print conflict counts
    print("\nCONFLICT DETECTION")
    print(f"  Average conflicts per query: {np.mean(conflicts_counts):.2f}")
    print(f"  Queries with conflicts: {sum(1 for c in conflicts_counts if c > 0)}/{len(conflicts_counts)}")
    print(f"  Max conflicts: {max(conflicts_counts)}")

    print(f"\n{'='*80}\n")

def compare_results(old_path, new_path):
    """Compare old vs new results."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: OLD vs NEW")
    print(f"{'='*80}\n")

    with open(old_path, 'r') as f:
        old_results = json.load(f)
    with open(new_path, 'r') as f:
        new_results = json.load(f)

    old_conf = [r.get('generated_answer', {}).get('confidence', 0.0) for r in old_results]
    new_conf = [r.get('generated_answer', {}).get('confidence', 0.0) for r in new_results]

    old_segments = [r.get('generated_answer', {}).get('curation_stats', {}).get('final_selected', 0)
                    for r in old_results]
    new_segments = [r.get('generated_answer', {}).get('curation_stats', {}).get('final_selected', 0)
                    for r in new_results]

    print(f"CONFIDENCE")
    print(f"  Old: {np.mean(old_conf):.2%}")
    print(f"  New: {np.mean(new_conf):.2%}")
    print(f"  Change: {(np.mean(new_conf) - np.mean(old_conf)):.2%} ({'+' if np.mean(new_conf) > np.mean(old_conf) else ''}{((np.mean(new_conf) - np.mean(old_conf)) / np.mean(old_conf) * 100):.1f}%)")

    print(f"\nSEGMENTS SELECTED")
    print(f"  Old: {np.mean(old_segments):.2f}")
    print(f"  New: {np.mean(new_segments):.2f}")
    print(f"  Change: {np.mean(new_segments) - np.mean(old_segments):.2f}")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Analyze current results (with issues)
    current_results = Path("evaluation_results/evaluation_results_test_20260126_012949.json")

    if current_results.exists():
        print("\n🔍 Analyzing CURRENT results (with aggressive curation/attribution)...")
        analyze_results(current_results)
    else:
        print(f"⚠️  Current results not found: {current_results}")

    print("\n" + "="*80)
    print("✅ FIXES APPLIED TO:")
    print("   - generation/context_curator.py")
    print("   - generation/attribution.py")
    print("\n📋 CHANGES MADE:")
    print("   1. Stricter contradiction detection (fewer false positives)")
    print("   2. Relaxed attribution thresholds (medical domain adjusted)")
    print("   3. Increased max segments (5-7 vs 1-3)")
    print("   4. Flag conflicts instead of removing segments")
    print("   5. Less punitive confidence weights")
    print("\n🚀 NEXT STEPS:")
    print("   Run: python ./run_full_evaluation.py --split test --enable-curation --enable-attribution")
    print("   Expected: Confidence 31% → 70-80%")
    print("   Expected: Final segments 1-2 → 5-7")
    print("   Expected: Improved temporal metrics and IoU")
    print("="*80 + "\n")
