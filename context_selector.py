"""
context_selector.py

Adaptive context selection with factual grounding for Medical VideoRAG.

Features:
- Quality filtering (score thresholds, completeness checks)
- Rule-based relevance scoring (semantic similarity, entity overlap)
- NLI-based factuality scoring (entailment/contradiction detection)
- Contradiction detection and resolution
- Query-type aware adaptive selection
- Diversity sampling and token budgeting

Usage:
    from context_selector import ContextSelector

    selector = ContextSelector(config)
    curated = selector.curate_context(query, segments, query_type)
"""

import numpy as np
import torch
import re
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
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
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                print(f"Loading NLI model: {nli_model_path}...")
                self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_path)
                self.nli_model.to(self.device)
                self.nli_model.eval()
                print("✅ NLI model loaded successfully")

            except Exception as e:
                print(f"⚠️  Failed to load NLI model: {e}")
                print("   Falling back to rule-based scoring only")
                self.use_nli = False

        # Load medical entity recognizer (scispacy)
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_sm")
        except Exception as e:
            print(f"⚠️  SciSpacy not available: {e}")
            print("   Entity overlap scoring will be limited")

        # Selection strategies by query type
        self.selection_strategies = {
            'procedural': {
                'max_segments': 5,
                'prioritize': 'temporal_order',
                'diversity_threshold': 0.85  # Low diversity (prefer sequential)
            },
            'diagnostic': {
                'max_segments': 6,
                'prioritize': 'diversity',
                'diversity_threshold': 0.70  # High diversity
            },
            'factoid': {
                'max_segments': 2,
                'prioritize': 'top_score',
                'diversity_threshold': 0.95  # No diversity needed
            },
            'general': {
                'max_segments': 3,
                'prioritize': 'balanced',
                'diversity_threshold': 0.80  # Medium diversity
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
                'selected_segments': List[Dict],  # Curated segments
                'conflicts_detected': List[Dict],  # Contradictions found
                'stats': Dict                       # Processing statistics
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

        # Step 3: NLI factuality scoring (on top-15 candidates)
        if self.use_nli and len(scored_segments) > 0:
            top_candidates = scored_segments[:min(15, len(scored_segments))]
            nli_scored = self.score_nli_factuality(query, top_candidates)
            # Merge back with remaining segments
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
                if len(text.strip()) < 50:
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
        # Extract query entities
        query_entities = self._extract_entities(query)

        scored = []
        for seg in segments:
            # Get existing retrieval scores
            text_score = seg.get('text_score', 0.0)
            visual_score = seg.get('visual_score', 0.0)
            combined_score = seg.get('combined_score', 0.0)

            # Semantic similarity (use existing text_score as proxy)
            semantic_sim = text_score

            # Entity overlap
            text_evidence = seg.get('text_evidence', {})
            if text_evidence:
                seg_text = text_evidence.get('text', '')
                seg_entities = self._extract_entities(seg_text)
                entity_overlap = self._calculate_entity_overlap(query_entities, seg_entities)
            else:
                entity_overlap = 0.0

            # Question type alignment
            question_match = self._score_question_alignment(query_type, seg)

            # Calculate relevance score
            relevance_score = (
                0.4 * semantic_sim +
                0.3 * entity_overlap +
                0.2 * question_match +
                0.1 * combined_score  # Original retrieval confidence
            )

            # Add relevance score to segment
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

        Returns segments with NLI scores added:
        - nli_entailment: Segment supports query
        - nli_neutral: Segment unrelated to query
        - nli_contradiction: Segment conflicts with query
        """
        if not self.use_nli or not self.nli_model:
            return segments

        scored = []

        for seg in segments:
            text_evidence = seg.get('text_evidence', {})
            if not text_evidence:
                scored.append(seg)
                continue

            seg_text = text_evidence.get('text', '')
            if not seg_text:
                scored.append(seg)
                continue

            # Score NLI alignment
            nli_scores = self._compute_nli_scores(query, seg_text)

            # Add NLI scores to segment
            seg['nli_entailment'] = nli_scores['entailment']
            seg['nli_neutral'] = nli_scores['neutral']
            seg['nli_contradiction'] = nli_scores['contradiction']

            # Update relevance score with NLI
            relevance_score = seg.get('relevance_score', 0.0)

            # Combine rule-based and NLI scores
            final_score = (
                0.6 * relevance_score +
                0.4 * nli_scores['entailment']
            )

            # Heavy penalty for contradictions
            if nli_scores['contradiction'] > 0.5:
                final_score *= 0.5

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
            if i in segments_to_remove:
                continue

            for j in range(i + 1, len(segments)):
                if j in segments_to_remove:
                    continue

                seg_i = segments[i]
                seg_j = segments[j]

                # Check for contradictions
                conflict_type = self._check_contradiction(seg_i, seg_j)

                if conflict_type:
                    # Found a conflict
                    score_i = seg_i.get('final_score', seg_i.get('relevance_score', 0.0))
                    score_j = seg_j.get('final_score', seg_j.get('relevance_score', 0.0))

                    # Keep higher scoring segment
                    if score_i >= score_j:
                        segments_to_remove.add(j)
                        kept_seg = i
                        removed_seg = j
                    else:
                        segments_to_remove.add(i)
                        kept_seg = j
                        removed_seg = i

                    conflicts.append({
                        'segments': [
                            segments[i].get('segment_id', f'seg_{i}'),
                            segments[j].get('segment_id', f'seg_{j}')
                        ],
                        'type': conflict_type,
                        'description': self._describe_conflict(conflict_type, segments[i], segments[j]),
                        'resolution': f"Kept segment {kept_seg} (score: {segments[kept_seg].get('final_score', 0.0):.4f}), removed segment {removed_seg} (score: {segments[removed_seg].get('final_score', 0.0):.4f})"
                    })

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
            # Simply take top segments
            selected = segments[:max_segments]

        elif prioritize == 'diversity':
            # Diversity sampling
            selected = self._diversity_sample(segments, max_segments, diversity_threshold)

        elif prioritize == 'temporal_order':
            # Sort by timestamp for procedural questions
            temporal_sorted = sorted(segments, key=lambda x: x.get('timestamp', [0, 0])[0])
            selected = temporal_sorted[:max_segments]

        else:  # balanced
            # Mix of top scores and diversity
            selected = self._diversity_sample(segments, max_segments, diversity_threshold)

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

        Uses text similarity to ensure selected segments cover different aspects.
        """
        if not segments:
            return []

        selected = [segments[0]]  # Always take top segment

        for seg in segments[1:]:
            if len(selected) >= target_count:
                break

            # Check diversity with already selected segments
            is_diverse = True
            seg_text = seg.get('text_evidence', {}).get('text', '')

            for selected_seg in selected:
                selected_text = selected_seg.get('text_evidence', {}).get('text', '')

                # Simple text overlap similarity
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
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except:
            # Fallback: estimate tokens as words * 1.3
            encoding = None

        selected = []
        cumulative_tokens = 0

        for seg in segments:
            text_evidence = seg.get('text_evidence', {})
            if not text_evidence:
                continue

            seg_text = text_evidence.get('text', '')

            # Estimate tokens
            if encoding:
                seg_tokens = len(encoding.encode(seg_text))
            else:
                seg_tokens = int(len(seg_text.split()) * 1.3)

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
            # Fallback: simple keyword extraction
            medical_keywords = set(re.findall(r'\b[a-z]{4,}\b', text.lower()))
            return medical_keywords

        doc = self.nlp(text)
        entities = set()

        # Extract named entities
        for ent in doc.ents:
            entities.add(ent.text.lower())

        # Extract noun chunks (medical terms often multi-word)
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

        Heuristics:
        - procedural: Look for step indicators, action verbs
        - diagnostic: Look for symptoms, assessment terms
        - factoid: Look for definitions, facts
        """
        text_evidence = segment.get('text_evidence', {})
        if not text_evidence:
            return 0.5  # Neutral score

        text = text_evidence.get('text', '').lower()

        if query_type == 'procedural':
            # Look for procedural indicators
            indicators = ['step', 'first', 'next', 'then', 'procedure', 'perform', 'how to']
            score = sum(1 for ind in indicators if ind in text) / len(indicators)
            return min(1.0, score * 2)

        elif query_type == 'diagnostic':
            # Look for diagnostic terms
            indicators = ['symptom', 'diagnose', 'assess', 'evaluate', 'identify', 'sign']
            score = sum(1 for ind in indicators if ind in text) / len(indicators)
            return min(1.0, score * 2)

        elif query_type == 'factoid':
            # Look for definitional content
            indicators = ['is', 'are', 'defined as', 'means', 'refers to', 'definition']
            score = sum(1 for ind in indicators if ind in text) / len(indicators)
            return min(1.0, score * 2)

        else:  # general
            return 0.5  # Neutral score

    def _compute_nli_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Compute NLI scores (entailment, neutral, contradiction).

        Args:
            premise: Query (what we want to verify)
            hypothesis: Segment text (evidence)

        Returns:
            {'entailment': float, 'neutral': float, 'contradiction': float}
        """
        if not self.nli_model or not self.nli_tokenizer:
            return {'entailment': 0.5, 'neutral': 0.5, 'contradiction': 0.0}

        try:
            # Tokenize
            inputs = self.nli_tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]

            # Map to labels (model-dependent, but typically: contradiction, neutral, entailment)
            # For BiomedNLP-PubMedBERT, check model config
            scores = {
                'contradiction': probs[0].item(),
                'neutral': probs[1].item(),
                'entailment': probs[2].item()
            }

            return scores

        except Exception as e:
            print(f"⚠️  NLI scoring failed: {e}")
            return {'entailment': 0.5, 'neutral': 0.5, 'contradiction': 0.0}

    def _check_contradiction(self, seg_i: Dict, seg_j: Dict) -> Optional[str]:
        """
        Check if two segments contradict each other.

        Returns conflict type or None:
        - 'negation': Opposite statements
        - 'numeric': Conflicting numbers
        - 'nli_contradiction': NLI detected contradiction
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

        # Extract key medical actions/terms (simplified)
        words_i = set(re.findall(r'\b[a-z]+\b', text_i.lower()))
        words_j = set(re.findall(r'\b[a-z]+\b', text_j.lower()))

        # Check if one has negation and they share similar content
        has_neg_i = bool(negation_words & words_i)
        has_neg_j = bool(negation_words & words_j)

        # XOR: one has negation, other doesn't, but similar words
        if has_neg_i != has_neg_j:
            common_words = (words_i - negation_words) & (words_j - negation_words)
            if len(common_words) > 5:  # Significant overlap
                return True

        return False

    def _has_numeric_conflict(self, text_i: str, text_j: str) -> bool:
        """Check for conflicting numeric values"""
        # Extract numbers with context
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
                        diff = abs(float(val_i) - float(val_j))
                        avg = (float(val_i) + float(val_j)) / 2
                        # Significant difference (>20%)
                        if avg > 0 and (diff / avg) > 0.2:
                            return True
                    except:
                        pass

        return False

    def _describe_conflict(self, conflict_type: str, seg_i: Dict, seg_j: Dict) -> str:
        """Generate human-readable conflict description"""
        if conflict_type == 'negation':
            return "Conflicting advice: one segment affirms, another negates"
        elif conflict_type == 'numeric':
            return "Conflicting numerical values for same measurement"
        elif conflict_type == 'nli_contradiction':
            return "Logical contradiction detected by NLI model"
        else:
            return f"Conflict of type: {conflict_type}"

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard on words)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# Example usage
if __name__ == "__main__":
    import json

    print("ContextSelector Demo")
    print("=" * 80)

    # Load sample segments
    try:
        with open("multimodal_search_results_hybrid.json", "r") as f:
            data = json.load(f)
            query = data['query']
            segments = data['results']

        print(f"Query: {query}")
        print(f"Input segments: {len(segments)}")

        # Initialize selector
        selector = ContextSelector(
            quality_threshold=0.3,
            token_budget=600,
            use_nli=True
        )

        # Curate context
        result = selector.curate_context(
            query=query,
            segments=segments,
            query_type='general'
        )

        print(f"\n{'='*80}")
        print("CURATION RESULTS")
        print(f"{'='*80}")
        print(f"Selected segments: {len(result['selected_segments'])}")
        print(f"Conflicts detected: {len(result['conflicts_detected'])}")
        print(f"\nStatistics:")
        for key, value in result['stats'].items():
            print(f"  {key}: {value}")

        if result['conflicts_detected']:
            print(f"\n{'='*80}")
            print("CONFLICTS DETECTED")
            print(f"{'='*80}")
            for i, conflict in enumerate(result['conflicts_detected'], 1):
                print(f"\n{i}. {conflict['description']}")
                print(f"   Segments: {conflict['segments']}")
                print(f"   Resolution: {conflict['resolution']}")

    except FileNotFoundError:
        print("⚠️  Sample file not found. Run query_faiss.py first to generate search results.")
