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

        # Initialize BERTScore (for semantic similarity)
        try:
            from bert_score import score as bert_score
            self.bert_score = bert_score
            self.bertscore_available = True
        except ImportError:
            print("⚠️  BERTScore not available. Semantic scoring disabled.")
            self.bertscore_available = False

        # Initialize SciSpacy for medical entity extraction
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_md")
            self.scispacy_available = True
        except Exception:
            print("⚠️  SciSpacy not available. Medical entity F1 disabled.")
            self.nlp = None
            self.scispacy_available = False

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
                'bert_score_f1': float,  # BERTScore F1
                'medical_entity_f1': float,  # Medical entity overlap F1
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

        # BERTScore (semantic similarity)
        if self.bertscore_available:
            bert_scores = self._compute_bertscore(generated, reference)
            metrics.update(bert_scores)

        # Medical entity F1
        if self.scispacy_available:
            entity_f1 = self._compute_medical_entity_f1(generated, reference)
            metrics['medical_entity_f1'] = entity_f1

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
    ) -> Dict:
        """
        Evaluate retrieval quality using Precision@K, Recall@K, F1@K, mAP, and nDCG@K.

        Args:
            retrieved_segments: List of retrieved segments (ordered by relevance)
            relevant_segment_ids: Set of ground truth relevant segment IDs
            k_values: List of K values to evaluate (default: [5, 10])

        Returns:
            {
                'precision@5': float,
                'recall@5': float,
                'f1@5': float,
                'precision@10': float,
                'recall@10': float,
                'f1@10': float,
                'mAP': float,
                'nDCG@5': float,
                'nDCG@10': float
            }
        """
        if k_values is None:
            k_values = [5, 10]

        results = {}

        # Precision@K, Recall@K, F1@K for each k
        for k in k_values:
            metrics = self._compute_retrieval_metrics(retrieved_segments, relevant_segment_ids, k)
            results[f'precision@{k}'] = metrics['precision']
            results[f'recall@{k}'] = metrics['recall']
            results[f'f1@{k}'] = metrics['f1']

        # Mean Average Precision
        results['mAP'] = self.compute_average_precision(
            retrieved_segments,
            relevant_segment_ids
        )

        # nDCG@K for each k
        for k in k_values:
            results[f'nDCG@{k}'] = self.compute_ndcg(
                retrieved_segments,
                relevant_segment_ids,
                k=k
            )

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

    def evaluate_temporal_overlap(
        self,
        predicted_timestamps: List[Tuple[float, float]],
        ground_truth_start: float,
        ground_truth_end: float
    ) -> Dict[str, float]:
        """
        Evaluate temporal overlap between predicted and ground truth timestamps.

        Args:
            predicted_timestamps: List of (start, end) tuples from retrieved segments
            ground_truth_start: Ground truth answer start time (seconds)
            ground_truth_end: Ground truth answer end time (seconds)

        Returns:
            {
                'iou': float,           # Intersection over Union (0-1)
                'temporal_precision': float,  # % of predicted time that overlaps
                'temporal_recall': float,     # % of ground truth time covered
                'temporal_f1': float,   # Harmonic mean of precision/recall
                'mean_distance': float  # Average distance from ground truth
            }
        """
        if not predicted_timestamps:
            return {
                'iou': 0.0,
                'temporal_precision': 0.0,
                'temporal_recall': 0.0,
                'temporal_f1': 0.0,
                'mean_distance': float('inf')
            }

        gt_interval = (ground_truth_start, ground_truth_end)
        gt_duration = ground_truth_end - ground_truth_start

        # Calculate union of all predicted intervals
        predicted_union = self._merge_intervals(predicted_timestamps)
        pred_duration = sum(end - start for start, end in predicted_union)

        # Calculate intersection with ground truth
        intersection_duration = 0.0
        for pred_start, pred_end in predicted_union:
            overlap_start = max(pred_start, ground_truth_start)
            overlap_end = min(pred_end, ground_truth_end)
            if overlap_start < overlap_end:
                intersection_duration += (overlap_end - overlap_start)

        # IoU (Intersection over Union)
        union_duration = pred_duration + gt_duration - intersection_duration
        iou = intersection_duration / union_duration if union_duration > 0 else 0.0

        # Temporal Precision: how much of predicted time overlaps with GT
        temporal_precision = intersection_duration / pred_duration if pred_duration > 0 else 0.0

        # Temporal Recall: how much of GT time is covered by predictions
        temporal_recall = intersection_duration / gt_duration if gt_duration > 0 else 0.0

        # Temporal F1
        if temporal_precision + temporal_recall > 0:
            temporal_f1 = 2 * (temporal_precision * temporal_recall) / (temporal_precision + temporal_recall)
        else:
            temporal_f1 = 0.0

        # Mean distance from ground truth center
        gt_center = (ground_truth_start + ground_truth_end) / 2
        distances = []
        for start, end in predicted_timestamps:
            pred_center = (start + end) / 2
            distances.append(abs(pred_center - gt_center))
        mean_distance = np.mean(distances) if distances else float('inf')

        return {
            'iou': iou,
            'temporal_precision': temporal_precision,
            'temporal_recall': temporal_recall,
            'temporal_f1': temporal_f1,
            'mean_distance': mean_distance
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

    def _compute_bertscore(self, generated: str, reference: str) -> Dict[str, float]:
        """Compute BERTScore for semantic similarity"""
        try:
            P, R, F1 = self.bert_score(
                [generated],
                [reference],
                lang='en',
                model_type='microsoft/deberta-xlarge-mnli',  # High-quality model
                verbose=False
            )
            return {
                'bert_score_precision': float(P[0]),
                'bert_score_recall': float(R[0]),
                'bert_score_f1': float(F1[0])
            }
        except Exception as e:
            print(f"⚠️  BERTScore computation failed: {e}")
            return {
                'bert_score_precision': 0.0,
                'bert_score_recall': 0.0,
                'bert_score_f1': 0.0
            }

    def _compute_medical_entity_f1(self, generated: str, reference: str) -> float:
        """Compute F1 score for medical entity overlap"""
        try:
            gen_doc = self.nlp(generated)
            ref_doc = self.nlp(reference)

            # Extract named entities
            gen_entities = set([ent.text.lower() for ent in gen_doc.ents])
            ref_entities = set([ent.text.lower() for ent in ref_doc.ents])

            if not ref_entities:
                return 1.0 if not gen_entities else 0.0

            # Calculate F1
            true_positives = len(gen_entities & ref_entities)
            precision = true_positives / len(gen_entities) if gen_entities else 0.0
            recall = true_positives / len(ref_entities) if ref_entities else 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            return f1
        except Exception as e:
            print(f"⚠️  Medical entity F1 computation failed: {e}")
            return 0.0

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping time intervals"""
        if not intervals:
            return []

        # Sort by start time
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]

        for current in sorted_intervals[1:]:
            last = merged[-1]
            # If intervals overlap, merge them
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

        return merged

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string (MM:SS or HH:MM:SS) to seconds"""
        try:
            parts = timestamp_str.strip().split(':')
            if len(parts) == 2:  # MM:SS
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            else:
                return float(timestamp_str)
        except Exception:
            return 0.0

    def compute_average_precision(
        self,
        retrieved_segments: List[Dict],
        relevant_segment_ids: Set[str]
    ) -> float:
        """
        Compute Average Precision for ranked retrieval results.

        AP = sum(P@k * rel(k)) / number_of_relevant_items

        Args:
            retrieved_segments: List of retrieved segments (ordered by relevance)
            relevant_segment_ids: Set of ground truth relevant segment IDs

        Returns:
            Average Precision score (0.0 to 1.0)
        """
        if not relevant_segment_ids:
            return 0.0

        num_relevant = len(relevant_segment_ids)
        precision_at_k = []
        num_relevant_seen = 0

        for k, seg in enumerate(retrieved_segments, 1):
            seg_id = seg.get('segment_id') or seg.get('evidence_id') or seg.get('meta', {}).get('segment_id')

            if seg_id in relevant_segment_ids:
                num_relevant_seen += 1
                precision_at_k.append(num_relevant_seen / k)

        if not precision_at_k:
            return 0.0

        return sum(precision_at_k) / num_relevant

    def compute_ndcg(
        self,
        retrieved_segments: List[Dict],
        relevant_segment_ids: Set[str],
        k: int = 10
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K.

        Accounts for ranking quality - relevant items ranked higher get more weight.

        Args:
            retrieved_segments: List of retrieved segments (ordered by relevance)
            relevant_segment_ids: Set of ground truth relevant segment IDs
            k: Cutoff position (default: 10)

        Returns:
            nDCG@K score (0.0 to 1.0)
        """
        import math

        # DCG calculation
        dcg = 0.0
        for i, seg in enumerate(retrieved_segments[:k], 1):
            seg_id = seg.get('segment_id') or seg.get('evidence_id') or seg.get('meta', {}).get('segment_id')
            relevance = 1.0 if seg_id in relevant_segment_ids else 0.0
            dcg += relevance / math.log2(i + 1)

        # IDCG (ideal DCG - all relevant items at top)
        num_relevant = min(len(relevant_segment_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

        return dcg / idcg if idcg > 0 else 0.0

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
