"""
evaluation.py

Automated evaluation metrics for Medical VideoRAG VQA.

Metrics:
- BLEU: N-gram precision (1-4 grams)
- ROUGE-L: Longest common subsequence
- Precision@K: % relevant segments in top-K
- Recall@K: % of relevant segments retrieved in top-K
- F1@K: Harmonic mean of Precision@K and Recall@K
- Attribution Accuracy: Correctness of claim-to-evidence mapping

Usage:
    from evaluation import AnswerEvaluator

    evaluator = AnswerEvaluator()
    metrics = evaluator.evaluate_answer(generated, reference)
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
import warnings

warnings.filterwarnings('ignore')


class AnswerEvaluator:
    """
    Comprehensive evaluator for answer generation and retrieval quality.
    """

    def __init__(self):
        """Initialize evaluator with required libraries"""

        # Initialize BLEU scorer
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.sentence_bleu = sentence_bleu
            self.smoothing = SmoothingFunction().method1
            self.bleu_available = True
        except ImportError:
            print("⚠️  NLTK not available. BLEU scoring disabled.")
            self.bleu_available = False

        # Initialize ROUGE scorer
        try:
            from rouge_score import rouge_scorer
            self.rouge = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self.rouge_available = True
        except ImportError:
            print("⚠️  rouge-score not available. ROUGE scoring disabled.")
            self.rouge_available = False

    def evaluate_answer(
        self,
        generated: str,
        reference: str,
        return_all: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate generated answer against reference.

        Args:
            generated: Generated answer text
            reference: Ground truth reference answer
            return_all: Return all metrics (default: True)

        Returns:
            {
                'bleu_1': float,  # BLEU-1 score
                'bleu_2': float,  # BLEU-2 score
                'bleu_3': float,  # BLEU-3 score
                'bleu_4': float,  # BLEU-4 score
                'rouge_1': float, # ROUGE-1 F1
                'rouge_2': float, # ROUGE-2 F1
                'rouge_l': float, # ROUGE-L F1
                'average_score': float  # Average of all metrics
            }
        """
        metrics = {}

        # BLEU scores
        if self.bleu_available:
            bleu_scores = self._compute_bleu(generated, reference)
            metrics.update(bleu_scores)

        # ROUGE scores
        if self.rouge_available:
            rouge_scores = self._compute_rouge(generated, reference)
            metrics.update(rouge_scores)

        # Calculate average
        if metrics:
            metrics['average_score'] = np.mean(list(metrics.values()))
        else:
            metrics['average_score'] = 0.0

        return metrics

    def evaluate_retrieval(
        self,
        retrieved_segments: List[Dict],
        relevant_segment_ids: Set[str],
        k_values: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval quality using Precision@K, Recall@K, F1@K.

        Args:
            retrieved_segments: List of retrieved segments (ordered by relevance)
            relevant_segment_ids: Set of ground truth relevant segment IDs
            k_values: List of K values to evaluate (default: [1, 3, 5, 10])

        Returns:
            {
                'k=1': {'precision': float, 'recall': float, 'f1': float},
                'k=3': {'precision': float, 'recall': float, 'f1': float},
                ...
            }
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        results = {}

        for k in k_values:
            metrics = self._compute_retrieval_metrics(
                retrieved_segments[:k],
                relevant_segment_ids,
                k
            )
            results[f'k={k}'] = metrics

        return results

    def evaluate_attribution(
        self,
        attribution_map: List[Dict],
        ground_truth_support: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Evaluate attribution accuracy.

        Args:
            attribution_map: Generated attribution from SelfReflectionAttribution
            ground_truth_support: Dict mapping claims → correct evidence IDs

        Returns:
            {
                'attribution_accuracy': float,  # % correct mappings
                'unsupported_rate': float,      # % claims marked unsupported
                'precision': float,             # Precision of supported claims
                'recall': float                 # Recall of supported claims
            }
        """
        if not ground_truth_support:
            return {
                'attribution_accuracy': 0.0,
                'unsupported_rate': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

        correct_mappings = 0
        unsupported_count = 0
        predicted_supported = 0
        total_claims = len(attribution_map)

        # Create claim → evidence mapping from attribution
        predicted_mappings = {}
        for attr in attribution_map:
            claim = attr['claim']
            evidence_id = attr.get('evidence_id')
            support_level = attr.get('support_level', 'UNSUPPORTED')

            if support_level == 'UNSUPPORTED':
                unsupported_count += 1
            else:
                predicted_supported += 1
                predicted_mappings[claim] = evidence_id

        # Count correct mappings
        for claim, true_evidence in ground_truth_support.items():
            predicted_evidence = predicted_mappings.get(claim)
            if predicted_evidence == true_evidence:
                correct_mappings += 1

        # Calculate metrics
        attribution_accuracy = correct_mappings / total_claims if total_claims > 0 else 0.0
        unsupported_rate = unsupported_count / total_claims if total_claims > 0 else 0.0

        # Precision: of claims we said are supported, how many are correct?
        precision = correct_mappings / predicted_supported if predicted_supported > 0 else 0.0

        # Recall: of all ground truth claims, how many did we find?
        recall = correct_mappings / len(ground_truth_support) if ground_truth_support else 0.0

        return {
            'attribution_accuracy': attribution_accuracy,
            'unsupported_rate': unsupported_rate,
            'precision': precision,
            'recall': recall
        }

    def evaluate_batch(
        self,
        results: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """
        Evaluate a batch of results.

        Args:
            results: List of result dicts from generate_answer()
            ground_truth: List of ground truth dicts with 'answer' and 'relevant_segments'

        Returns:
            Aggregated metrics across all examples
        """
        all_answer_metrics = []
        all_retrieval_metrics = []
        all_attribution_metrics = []

        for result, gt in zip(results, ground_truth):
            # Answer quality
            if 'answer' in gt:
                answer_metrics = self.evaluate_answer(
                    result['answer'],
                    gt['answer']
                )
                all_answer_metrics.append(answer_metrics)

            # Retrieval quality
            if 'relevant_segments' in gt:
                retrieved = result.get('evidence_segments', [])
                retrieval_metrics = self.evaluate_retrieval(
                    retrieved,
                    set(gt['relevant_segments'])
                )
                all_retrieval_metrics.append(retrieval_metrics)

            # Attribution quality
            if 'attribution_support' in gt and result.get('attribution_map'):
                attr_metrics = self.evaluate_attribution(
                    result['attribution_map']['attribution_map'],
                    gt['attribution_support']
                )
                all_attribution_metrics.append(attr_metrics)

        # Aggregate results
        aggregated = {
            'num_examples': len(results),
            'answer_quality': self._aggregate_metrics(all_answer_metrics),
            'retrieval_quality': self._aggregate_retrieval_metrics(all_retrieval_metrics),
            'attribution_quality': self._aggregate_metrics(all_attribution_metrics)
        }

        return aggregated

    # Helper methods

    def _compute_bleu(self, generated: str, reference: str) -> Dict[str, float]:
        """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4"""
        try:
            gen_tokens = generated.split()
            ref_tokens = reference.split()

            bleu_1 = self.sentence_bleu(
                [ref_tokens], gen_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=self.smoothing
            )
            bleu_2 = self.sentence_bleu(
                [ref_tokens], gen_tokens,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=self.smoothing
            )
            bleu_3 = self.sentence_bleu(
                [ref_tokens], gen_tokens,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=self.smoothing
            )
            bleu_4 = self.sentence_bleu(
                [ref_tokens], gen_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothing
            )

            return {
                'bleu_1': bleu_1,
                'bleu_2': bleu_2,
                'bleu_3': bleu_3,
                'bleu_4': bleu_4
            }
        except Exception as e:
            print(f"⚠️  BLEU computation failed: {e}")
            return {
                'bleu_1': 0.0,
                'bleu_2': 0.0,
                'bleu_3': 0.0,
                'bleu_4': 0.0
            }

    def _compute_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L"""
        try:
            scores = self.rouge.score(reference, generated)

            return {
                'rouge_1': scores['rouge1'].fmeasure,
                'rouge_2': scores['rouge2'].fmeasure,
                'rouge_l': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"⚠️  ROUGE computation failed: {e}")
            return {
                'rouge_1': 0.0,
                'rouge_2': 0.0,
                'rouge_l': 0.0
            }

    def _compute_retrieval_metrics(
        self,
        retrieved_segments: List[Dict],
        relevant_segment_ids: Set[str],
        k: int
    ) -> Dict[str, float]:
        """Compute Precision@K, Recall@K, F1@K"""

        # Extract segment IDs from retrieved segments
        retrieved_ids = set()
        for seg in retrieved_segments:
            seg_id = seg.get('segment_id') or seg.get('evidence_id')
            if seg_id:
                retrieved_ids.add(seg_id)

        # Calculate metrics
        relevant_in_topk = retrieved_ids & relevant_segment_ids

        precision = len(relevant_in_topk) / k if k > 0 else 0.0
        recall = len(relevant_in_topk) / len(relevant_segment_ids) if relevant_segment_ids else 0.0
        f1 = (2 * precision * recall) / (precision + recall + 1e-10)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across multiple examples"""
        if not metrics_list:
            return {}

        aggregated = {}

        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        # Average each metric
        for key in all_keys:
            values = [m.get(key, 0.0) for m in metrics_list]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

        return aggregated

    def _aggregate_retrieval_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate retrieval metrics across multiple examples"""
        if not metrics_list:
            return {}

        aggregated = {}

        # Get all k values
        all_k_values = set()
        for metrics in metrics_list:
            all_k_values.update(metrics.keys())

        # Aggregate for each k
        for k_str in all_k_values:
            k_metrics = []
            for metrics in metrics_list:
                if k_str in metrics:
                    k_metrics.append(metrics[k_str])

            if k_metrics:
                aggregated[k_str] = {
                    'precision_mean': np.mean([m['precision'] for m in k_metrics]),
                    'precision_std': np.std([m['precision'] for m in k_metrics]),
                    'recall_mean': np.mean([m['recall'] for m in k_metrics]),
                    'recall_std': np.std([m['recall'] for m in k_metrics]),
                    'f1_mean': np.mean([m['f1'] for m in k_metrics]),
                    'f1_std': np.std([m['f1'] for m in k_metrics])
                }

        return aggregated

    def format_evaluation_results(self, metrics: Dict) -> str:
        """Format evaluation results for display"""
        output = []
        output.append("=" * 80)
        output.append("EVALUATION RESULTS")
        output.append("=" * 80)

        # Answer quality metrics
        if 'answer_quality' in metrics:
            output.append("\nAnswer Quality:")
            output.append("-" * 80)
            answer_metrics = metrics['answer_quality']

            # BLEU scores
            if any('bleu' in k for k in answer_metrics.keys()):
                output.append("\nBLEU Scores:")
                for i in range(1, 5):
                    key = f'bleu_{i}_mean'
                    if key in answer_metrics:
                        output.append(f"  BLEU-{i}: {answer_metrics[key]:.4f} (±{answer_metrics.get(f'bleu_{i}_std', 0):.4f})")

            # ROUGE scores
            if any('rouge' in k for k in answer_metrics.keys()):
                output.append("\nROUGE Scores:")
                for rouge_type in ['rouge_1', 'rouge_2', 'rouge_l']:
                    key = f'{rouge_type}_mean'
                    if key in answer_metrics:
                        output.append(f"  {rouge_type.upper()}: {answer_metrics[key]:.4f} (±{answer_metrics.get(f'{rouge_type}_std', 0):.4f})")

        # Retrieval quality metrics
        if 'retrieval_quality' in metrics:
            output.append("\n" + "-" * 80)
            output.append("Retrieval Quality:")
            output.append("-" * 80)
            retrieval_metrics = metrics['retrieval_quality']

            for k_str in sorted(retrieval_metrics.keys()):
                k_metrics = retrieval_metrics[k_str]
                output.append(f"\n{k_str.upper()}:")
                output.append(f"  Precision: {k_metrics['precision_mean']:.4f} (±{k_metrics['precision_std']:.4f})")
                output.append(f"  Recall:    {k_metrics['recall_mean']:.4f} (±{k_metrics['recall_std']:.4f})")
                output.append(f"  F1:        {k_metrics['f1_mean']:.4f} (±{k_metrics['f1_std']:.4f})")

        # Attribution quality metrics
        if 'attribution_quality' in metrics:
            output.append("\n" + "-" * 80)
            output.append("Attribution Quality:")
            output.append("-" * 80)
            attr_metrics = metrics['attribution_quality']

            for key, value in attr_metrics.items():
                if '_mean' in key:
                    metric_name = key.replace('_mean', '').replace('_', ' ').title()
                    std_key = key.replace('_mean', '_std')
                    std = attr_metrics.get(std_key, 0)
                    output.append(f"  {metric_name}: {value:.4f} (±{std:.4f})")

        output.append("\n" + "=" * 80)

        return "\n".join(output)


# Example usage
if __name__ == "__main__":
    print("Evaluation Demo")
    print("=" * 80)

    # Initialize evaluator
    evaluator = AnswerEvaluator()

    # Sample answer evaluation
    generated = "CPR involves chest compressions at 100-120 per minute and rescue breaths."
    reference = "CPR consists of chest compressions performed at a rate of 100-120 compressions per minute, combined with rescue breathing."

    print("\n1. Answer Quality Evaluation")
    print("-" * 80)
    answer_metrics = evaluator.evaluate_answer(generated, reference)

    for metric, value in answer_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Sample retrieval evaluation
    print("\n2. Retrieval Quality Evaluation")
    print("-" * 80)

    retrieved = [
        {'segment_id': 'seg_1'},
        {'segment_id': 'seg_2'},
        {'segment_id': 'seg_3'},
        {'segment_id': 'seg_4'},
        {'segment_id': 'seg_5'}
    ]
    relevant = {'seg_1', 'seg_3', 'seg_7'}

    retrieval_metrics = evaluator.evaluate_retrieval(retrieved, relevant)

    for k, metrics in retrieval_metrics.items():
        print(f"\n{k}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Sample attribution evaluation
    print("\n3. Attribution Quality Evaluation")
    print("-" * 80)

    attribution_map = [
        {'claim': 'CPR involves compressions', 'support_level': 'HIGH', 'evidence_id': 'seg_1'},
        {'claim': 'Rate is 100-120 per minute', 'support_level': 'HIGH', 'evidence_id': 'seg_1'},
        {'claim': 'Includes rescue breaths', 'support_level': 'MEDIUM', 'evidence_id': 'seg_2'}
    ]

    ground_truth = {
        'CPR involves compressions': 'seg_1',
        'Rate is 100-120 per minute': 'seg_1',
        'Includes rescue breaths': 'seg_3'  # Wrong evidence
    }

    attr_metrics = evaluator.evaluate_attribution(attribution_map, ground_truth)

    for metric, value in attr_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
