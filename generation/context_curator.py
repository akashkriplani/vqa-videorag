"""
generation/context_curator.py

Adaptive context selection with factual grounding for Medical VideoRAG.

Features:
- Quality filtering (score thresholds, completeness checks)
- Rule-based relevance scoring (semantic similarity, entity overlap)
- NLI-based factuality scoring (entailment/contradiction detection)
- Contradiction detection and resolution
- Query-type aware adaptive selection
- Diversity sampling and token budgeting
"""

import re
import os
import torch
import spacy
import warnings
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Suppress warnings
warnings.filterwarnings('ignore')


class ContextSelector:
    """
    Adaptive context selector with factual grounding.

    Pipeline:
    1. Quality filtering (remove low-quality segments)
    2. Rule-based relevance scoring (semantic + entity + question type)
    3. NLI factuality scoring (entailment/contradiction detection)
    4. Contradiction detection and resolution
    5. Adaptive selection (query-type aware, diversity, token budget)
    """

    def __init__(
        self,
        quality_threshold: float = 0.3,
        nli_model_path: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        token_budget: int = 600,
        use_nli: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize context selector.

        Args:
            quality_threshold: Minimum combined_score for filtering (default: 0.3)
            nli_model_path: HuggingFace model for NLI (default: BiomedNLP-PubMedBERT)
            token_budget: Maximum tokens for context (default: 600)
            use_nli: Whether to use NLI model for factuality scoring (default: True)
            device: Device for models ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.quality_threshold = quality_threshold
        self.token_budget = token_budget
        self.use_nli = use_nli

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"ContextSelector initialized on device: {self.device}")

        # Load NLI model if enabled
        self.nli_model = None
        self.nli_tokenizer = None

        if self.use_nli:
            try:
                print(f"Loading NLI model: {nli_model_path}")
                self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_path)
                self.nli_model.to(self.device)
                self.nli_model.eval()
                print("✅ NLI model loaded for factuality scoring")

            except Exception as e:
                print(f"⚠️  NLI model loading failed: {e}")
                print("   Factuality scoring will be limited")
                self.use_nli = False

        # Load medical entity recognizer (scispacy)
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_sm")
            print("✅ SciSpacy loaded for entity extraction")
        except Exception as e:
            print(f"⚠️  SciSpacy not available: {e}")
            print("   Entity overlap scoring will be limited")

        # Selection strategies by query type
        self.selection_strategies = {
            'procedural': {
                'max_segments': 5,
                'prioritize': 'temporal_order',
                'diversity_threshold': 0.85
            },
            'diagnostic': {
                'max_segments': 6,
                'prioritize': 'diversity',
                'diversity_threshold': 0.70
            },
            'factoid': {
                'max_segments': 2,
                'prioritize': 'top_score',
                'diversity_threshold': 0.95
            },
            'general': {
                'max_segments': 3,
                'prioritize': 'balanced',
                'diversity_threshold': 0.80
            }
        }

    def curate_context(
        self,
        query: str,
        segments: List[Dict],
        query_type: str = 'general',
        enable_hierarchical: bool = True
    ) -> Dict:
        """
        Main pipeline: curate context with factual grounding.

        Args:
            query: User query
            segments: Retrieved segments from query_faiss
            query_type: Question type ('procedural', 'diagnostic', 'factoid', 'general')
            enable_hierarchical: Whether to enable advanced processing

        Returns:
            {
                'selected_segments': List[Dict],
                'conflicts_detected': List[Dict],
                'stats': Dict
            }
        """
        print(f"\n{'='*80}")
        print(f"ADAPTIVE CONTEXT SELECTION")
        print(f"{'='*80}")
        print(f"Query type: {query_type}")
        print(f"Input segments: {len(segments)}")

        stats = {
            'input_segments': len(segments),
            'after_quality_filter': 0,
            'after_relevance_scoring': 0,
            'after_nli_scoring': 0,
            'after_contradiction_detection': 0,
            'final_selected': 0,
            'conflicts_found': 0
        }

        # Step 1: Quality filtering
        filtered_segments = self.filter_quality(segments)
        stats['after_quality_filter'] = len(filtered_segments)
        print(f"After quality filtering: {len(filtered_segments)} segments")

        if not filtered_segments:
            return {
                'selected_segments': [],
                'conflicts_detected': [],
                'stats': stats
            }

        # Step 2: Rule-based relevance scoring
        scored_segments = self.score_relevance(query, filtered_segments, query_type)
        stats['after_relevance_scoring'] = len(scored_segments)
        print(f"After relevance scoring: {len(scored_segments)} segments")

        # Step 3: NLI factuality scoring
        if self.use_nli and len(scored_segments) > 0:
            top_candidates = scored_segments[:min(15, len(scored_segments))]
            nli_scored = self.score_nli_factuality(query, top_candidates)
            scored_segments = nli_scored + scored_segments[len(nli_scored):]
            stats['after_nli_scoring'] = len(scored_segments)
            print(f"After NLI scoring: {len(scored_segments)} segments")

        # Step 4: Contradiction detection
        conflicts = []
        if enable_hierarchical and len(scored_segments) > 1:
            cleaned_segments, conflicts = self.detect_and_resolve_contradictions(scored_segments)
            stats['after_contradiction_detection'] = len(cleaned_segments)
            stats['conflicts_found'] = len(conflicts)
            print(f"After contradiction detection: {len(cleaned_segments)} segments, {len(conflicts)} conflicts")
        else:
            cleaned_segments = scored_segments

        # Step 5: Adaptive selection
        selected_segments = self.adaptive_select(
            query=query,
            segments=cleaned_segments,
            query_type=query_type
        )
        stats['final_selected'] = len(selected_segments)
        print(f"Final selected: {len(selected_segments)} segments")
        print(f"{'='*80}\n")

        return {
            'selected_segments': selected_segments,
            'conflicts_detected': conflicts,
            'stats': stats
        }

    def filter_quality(self, segments: List[Dict]) -> List[Dict]:
        """
        Filter out low-quality segments.

        Criteria:
        - Score threshold (combined_score > quality_threshold)
        - Text completeness (min 50 characters)
        - Valid timestamp
        """
        filtered = []

        for seg in segments:
            # Check score threshold
            combined_score = seg.get('combined_score', 0.0)
            if combined_score < self.quality_threshold:
                continue

            # Check text completeness
            text_evidence = seg.get('text_evidence', {})
            if text_evidence:
                text = text_evidence.get('text', '')
                if len(text) < 50:
                    continue

            # Check valid timestamp
            timestamp = seg.get('timestamp')
            if not timestamp or not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                continue

            filtered.append(seg)

        return filtered

    def score_relevance(
        self,
        query: str,
        segments: List[Dict],
        query_type: str
    ) -> List[Dict]:
        """
        Score segments using rule-based relevance metrics.

        Components:
        - Semantic similarity (from existing embeddings)
        - Entity overlap (medical term matching)
        - Question type alignment

        Formula:
            relevance = 0.4 * semantic_sim + 0.3 * entity_overlap +
                       0.2 * question_match + 0.1 * retrieval_confidence
        """
        query_entities = self._extract_entities(query)

        scored = []
        for seg in segments:
            text_score = seg.get('text_score', 0.0)
            visual_score = seg.get('visual_score', 0.0)
            combined_score = seg.get('combined_score', 0.0)

            # Get segment text
            text_evidence = seg.get('text_evidence', {})
            seg_text = text_evidence.get('text', '') if text_evidence else ''

            # Calculate entity overlap
            seg_entities = self._extract_entities(seg_text)
            entity_overlap = self._calculate_entity_overlap(query_entities, seg_entities)

            # Calculate question type alignment
            question_match = self._score_question_alignment(query_type, seg)

            # Compute final relevance score
            relevance_score = (
                0.4 * text_score +
                0.3 * entity_overlap +
                0.2 * question_match +
                0.1 * combined_score
            )

            seg['relevance_score'] = relevance_score
            scored.append(seg)

        # Sort by relevance score
        scored.sort(key=lambda x: x['relevance_score'], reverse=True)

        return scored

    def score_nli_factuality(
        self,
        query: str,
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Score factual alignment using NLI model.

        Returns segments with NLI scores added.
        """
        if not self.use_nli or not self.nli_model:
            return segments

        scored = []

        for seg in segments:
            text_evidence = seg.get('text_evidence', {})
            seg_text = text_evidence.get('text', '') if text_evidence else ''

            if not seg_text:
                scored.append(seg)
                continue

            # Compute NLI scores
            nli_scores = self._compute_nli_scores(query, seg_text)

            # Add NLI scores to segment
            seg['nli_entailment'] = nli_scores.get('entailment', 0.0)
            seg['nli_neutral'] = nli_scores.get('neutral', 0.0)
            seg['nli_contradiction'] = nli_scores.get('contradiction', 0.0)

            # Update final score
            relevance_score = seg.get('relevance_score', 0.0)
            final_score = (
                0.7 * relevance_score +
                0.3 * seg['nli_entailment']
            )
            seg['final_score'] = final_score

            scored.append(seg)

        # Sort by final score
        scored.sort(key=lambda x: x.get('final_score', x.get('relevance_score', 0.0)), reverse=True)

        return scored

    def detect_and_resolve_contradictions(
        self,
        segments: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect and resolve contradictions across segments.

        Returns:
            (cleaned_segments, conflicts_detected)
        """
        conflicts = []
        segments_to_remove = set()

        # Check all pairs for contradictions
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                conflict_type = self._check_contradiction(segments[i], segments[j])

                if conflict_type:
                    # Record conflict
                    conflict = {
                        'type': conflict_type,
                        'segments': [
                            segments[i].get('segment_id'),
                            segments[j].get('segment_id')
                        ],
                        'description': self._describe_conflict(conflict_type, segments[i], segments[j]),
                        'resolution': 'Removed lower-scored segment'
                    }
                    conflicts.append(conflict)

                    # Remove lower-scored segment
                    score_i = segments[i].get('final_score', segments[i].get('relevance_score', 0.0))
                    score_j = segments[j].get('final_score', segments[j].get('relevance_score', 0.0))

                    if score_i < score_j:
                        segments_to_remove.add(i)
                    else:
                        segments_to_remove.add(j)

        # Remove conflicting segments
        cleaned_segments = [seg for i, seg in enumerate(segments) if i not in segments_to_remove]

        return cleaned_segments, conflicts

    def adaptive_select(
        self,
        query: str,
        segments: List[Dict],
        query_type: str
    ) -> List[Dict]:
        """
        Adaptive selection based on query type, diversity, and token budget.

        Strategies:
        - procedural: Sequential, more segments
        - diagnostic: Diverse perspectives
        - factoid: Single best answer
        - general: Balanced
        """
        strategy = self.selection_strategies.get(query_type, self.selection_strategies['general'])

        max_segments = strategy['max_segments']
        diversity_threshold = strategy['diversity_threshold']
        prioritize = strategy['prioritize']

        # Apply strategy
        if prioritize == 'top_score':
            selected = segments[:max_segments]

        elif prioritize == 'diversity':
            selected = self._diversity_sample(segments, max_segments, diversity_threshold)

        elif prioritize == 'temporal_order':
            sorted_by_time = sorted(segments[:max_segments * 2], key=lambda x: x.get('timestamp', [0])[0])
            selected = sorted_by_time[:max_segments]

        else:  # balanced
            selected = segments[:max_segments]

        # Apply token budget
        selected = self._fit_within_token_budget(selected, self.token_budget)

        return selected

    def _diversity_sample(
        self,
        segments: List[Dict],
        target_count: int,
        diversity_threshold: float
    ) -> List[Dict]:
        """
        Select diverse segments avoiding semantic duplicates.
        """
        if not segments:
            return []

        selected = [segments[0]]

        for seg in segments[1:]:
            if len(selected) >= target_count:
                break

            # Check diversity with already selected
            is_diverse = True
            seg_text = seg.get('text_evidence', {}).get('text', '')

            for selected_seg in selected:
                selected_text = selected_seg.get('text_evidence', {}).get('text', '')
                similarity = self._calculate_text_similarity(seg_text, selected_text)

                if similarity > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(seg)

        return selected

    def _fit_within_token_budget(
        self,
        segments: List[Dict],
        max_tokens: int
    ) -> List[Dict]:
        """
        Select segments that fit within token budget.
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        except:
            # Fallback: estimate 4 chars per token
            def estimate_tokens(text):
                return len(text) // 4

        selected = []
        cumulative_tokens = 0

        for seg in segments:
            text_evidence = seg.get('text_evidence', {})
            text = text_evidence.get('text', '') if text_evidence else ''

            if not text:
                continue

            # Count tokens
            if 'tokenizer' in locals():
                seg_tokens = len(tokenizer.encode(text))
            else:
                seg_tokens = estimate_tokens(text)

            if cumulative_tokens + seg_tokens <= max_tokens:
                selected.append(seg)
                cumulative_tokens += seg_tokens
            else:
                break

        return selected

    # Helper methods

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract medical entities from text"""
        if not self.nlp:
            # Fallback: extract noun phrases
            words = re.findall(r'\b[A-Za-z]{4,}\b', text)
            return set(words[:20])

        doc = self.nlp(text)
        entities = set()

        # Extract named entities
        for ent in doc.ents:
            entities.add(ent.text.lower())

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            entities.add(chunk.text.lower())

        return entities

    def _calculate_entity_overlap(
        self,
        query_entities: Set[str],
        segment_entities: Set[str]
    ) -> float:
        """Calculate Jaccard similarity of entities"""
        if not query_entities or not segment_entities:
            return 0.0

        intersection = len(query_entities & segment_entities)
        union = len(query_entities | segment_entities)

        return intersection / union if union > 0 else 0.0

    def _score_question_alignment(self, query_type: str, segment: Dict) -> float:
        """
        Score how well segment aligns with question type.
        """
        text_evidence = segment.get('text_evidence', {})
        if not text_evidence:
            return 0.0

        text = text_evidence.get('text', '').lower()

        if query_type == 'procedural':
            procedural_keywords = ['step', 'first', 'then', 'next', 'finally', 'procedure', 'method']
            score = sum(1 for kw in procedural_keywords if kw in text)
            return min(1.0, score / 3.0)

        elif query_type == 'diagnostic':
            diagnostic_keywords = ['symptom', 'sign', 'diagnosis', 'assess', 'evaluate', 'test']
            score = sum(1 for kw in diagnostic_keywords if kw in text)
            return min(1.0, score / 3.0)

        elif query_type == 'factoid':
            factoid_keywords = ['is', 'are', 'definition', 'means', 'called', 'refers']
            score = sum(1 for kw in factoid_keywords if kw in text)
            return min(1.0, score / 3.0)

        else:  # general
            return 0.5

    def _compute_nli_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute NLI scores (entailment, neutral, contradiction).
        """
        if not self.nli_model or not self.nli_tokenizer:
            return {'entailment': 0.5, 'neutral': 0.5, 'contradiction': 0.0}

        try:
            # Tokenize
            inputs = self.nli_tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]

            # Map to labels (assuming standard NLI format)
            return {
                'entailment': float(probs[0]),
                'neutral': float(probs[1]),
                'contradiction': float(probs[2])
            }

        except Exception as e:
            return {'entailment': 0.5, 'neutral': 0.5, 'contradiction': 0.0}

    def _check_contradiction(self, seg_i: Dict, seg_j: Dict) -> Optional[str]:
        """
        Check if two segments contradict each other.

        Returns conflict type or None.
        """
        text_i = seg_i.get('text_evidence', {}).get('text', '')
        text_j = seg_j.get('text_evidence', {}).get('text', '')

        if not text_i or not text_j:
            return None

        # Check NLI contradiction
        if 'nli_contradiction' in seg_i and 'nli_contradiction' in seg_j:
            if seg_i['nli_contradiction'] > 0.6 or seg_j['nli_contradiction'] > 0.6:
                return 'nli_contradiction'

        # Check negation patterns
        if self._has_negation_conflict(text_i, text_j):
            return 'negation'

        # Check numeric conflicts
        if self._has_numeric_conflict(text_i, text_j):
            return 'numeric'

        return None

    def _has_negation_conflict(self, text_i: str, text_j: str) -> bool:
        """Check for negation conflicts"""
        negation_words = {'not', 'never', 'avoid', "don't", 'do not', 'no', 'cannot', "can't"}

        words_i = set(re.findall(r'\b[a-z]+\b', text_i.lower()))
        words_j = set(re.findall(r'\b[a-z]+\b', text_j.lower()))

        has_neg_i = bool(negation_words & words_i)
        has_neg_j = bool(negation_words & words_j)

        # XOR: one has negation, other doesn't, but similar words
        if has_neg_i != has_neg_j:
            common_words = (words_i - negation_words) & (words_j - negation_words)
            if len(common_words) > 5:
                return True

        return False

    def _has_numeric_conflict(self, text_i: str, text_j: str) -> bool:
        """Check for conflicting numeric values"""
        pattern = r'(\d+\.?\d*)\s*([a-z/%]+)?'

        nums_i = re.findall(pattern, text_i.lower())
        nums_j = re.findall(pattern, text_j.lower())

        if not nums_i or not nums_j:
            return False

        # Check for same units with different values
        for val_i, unit_i in nums_i:
            for val_j, unit_j in nums_j:
                if unit_i and unit_j and unit_i == unit_j:
                    try:
                        if abs(float(val_i) - float(val_j)) / max(float(val_i), float(val_j)) > 0.2:
                            return True
                    except:
                        pass

        return False

    def _describe_conflict(self, conflict_type: str, seg_i: Dict, seg_j: Dict) -> str:
        """Generate human-readable conflict description"""
        if conflict_type == 'negation':
            return "Segments contain contradictory statements (negation detected)"
        elif conflict_type == 'numeric':
            return "Segments contain conflicting numeric values"
        elif conflict_type == 'nli_contradiction':
            return "NLI model detected logical contradiction"
        else:
            return "Unknown conflict type"

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard on words)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
