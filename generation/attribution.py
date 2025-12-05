"""
generation/attribution.py

Self-reflection attribution for Medical VideoRAG answers.

Features:
- Sentence-level claim extraction
- Evidence mapping (link claims to supporting segments)
- Confidence annotation (HIGH/MEDIUM/LOW/UNSUPPORTED)
- Conflict highlighting for contradictory evidence
- Attribution accuracy tracking
"""

import numpy as np
import re
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class SelfReflectionAttribution:
    """
    Self-reflection attribution system for answer generation.

    Maps answer sentences to supporting evidence with confidence levels.
    Highlights unsupported claims and conflicts.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize attribution system.

        Args:
            device: Device for models ('cpu', 'cuda', 'mps', or None for auto)
        """
        import torch

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

        print(f"SelfReflectionAttribution initialized on device: {self.device}")

        # Load SciSpacy for medical text processing
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_sm")
            print("✅ SciSpacy loaded for claim extraction")
        except Exception as e:
            print(f"⚠️  SciSpacy not available: {e}")
            print("   Using fallback sentence segmentation")

        # Load sentence transformer for semantic similarity
        self.sentence_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentence_model.to(self.device)
            print("✅ Sentence transformer loaded for similarity matching")
        except Exception as e:
            print(f"⚠️  Sentence transformer not available: {e}")
            print("   Using fallback text similarity")

    def generate_attribution(
        self,
        answer: str,
        evidence_segments: List[Dict],
        conflicts: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate complete attribution map for an answer.

        Args:
            answer: Generated answer text
            evidence_segments: List of evidence segments used for generation
            conflicts: Optional list of detected conflicts

        Returns:
            {
                'attribution_map': List[Dict],
                'overall_confidence': float,
                'unsupported_claims': int,
                'conflicted_claims': int,
                'claim_count': int,
                'support_breakdown': Dict
            }
        """
        print(f"\n{'='*80}")
        print("GENERATING ATTRIBUTION MAP")
        print(f"{'='*80}")

        # Extract claims from answer
        claims = self.extract_claims(answer)
        print(f"Extracted {len(claims)} claims from answer")

        # Map each claim to evidence
        attribution_map = []
        for i, claim in enumerate(claims, 1):
            print(f"  Mapping claim {i}/{len(claims)}...")
            attribution = self.map_claim_to_evidence(claim, evidence_segments)

            # Check for conflicts
            if conflicts:
                attribution = self._annotate_conflicts(attribution, conflicts)

            attribution_map.append(attribution)

        # Calculate overall statistics
        stats = self._calculate_attribution_stats(attribution_map)

        print(f"\nAttribution complete:")
        print(f"  Overall confidence: {stats['overall_confidence']:.2%}")
        print(f"  Unsupported claims: {stats['unsupported_claims']}/{stats['claim_count']}")
        print(f"  Conflicted claims: {stats['conflicted_claims']}/{stats['claim_count']}")
        print(f"{'='*80}\n")

        return {
            'attribution_map': attribution_map,
            **stats
        }

    def extract_claims(self, answer_text: str) -> List[str]:
        """
        Extract individual claims (sentences) from answer.

        Uses SciSpacy if available, otherwise falls back to simple segmentation.
        """
        if not answer_text:
            return []

        if self.nlp:
            doc = self.nlp(answer_text)
            claims = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback: simple sentence splitting
            text = answer_text
            # Protect abbreviations
            text = re.sub(r'\b([A-Z][a-z]*\.)', r'\1<PROT>', text)
            text = re.sub(r'\b([A-Z]{2,}\.)', r'\1<PROT>', text)

            # Split on sentence boundaries
            sentences = re.split(r'[.!?]+\s+', text)

            # Restore protected periods
            claims = [s.replace('<PROT>', '') for s in sentences if s.strip()]

        return claims

    def map_claim_to_evidence(
        self,
        claim: str,
        evidence_segments: List[Dict]
    ) -> Dict:
        """
        Find the best supporting evidence segment for a claim.

        Returns:
            {
                'claim': str,
                'support_level': str,
                'evidence_id': Optional[str],
                'video_id': Optional[str],
                'timestamp': Optional[list],
                'formatted_time': Optional[str],
                'exact_quote': Optional[str],
                'similarity_score': float,
                'conflict_warning': Optional[str]
            }
        """
        if not evidence_segments:
            return {
                'claim': claim,
                'support_level': 'UNSUPPORTED',
                'evidence_id': None,
                'video_id': None,
                'timestamp': None,
                'formatted_time': None,
                'exact_quote': None,
                'similarity_score': 0.0,
                'conflict_warning': None
            }

        # Find best matching segment
        best_match = None
        best_similarity = 0.0
        best_quote = None

        for seg in evidence_segments:
            text_evidence = seg.get('text_evidence', {})
            if not text_evidence:
                continue

            seg_text = text_evidence.get('text', '')
            if not seg_text:
                continue

            # Calculate similarity
            similarity = self._calculate_semantic_similarity(claim, seg_text)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = seg
                best_quote = self._extract_quote(claim, seg_text)

        # Assess support level
        support_level = self._assess_support_level(claim, best_match, best_similarity)

        # Format result
        if best_match:
            timestamp = best_match.get('timestamp')
            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                formatted_time = self._format_timestamp(timestamp)
            else:
                formatted_time = "unknown"

            return {
                'claim': claim,
                'support_level': support_level,
                'evidence_id': best_match.get('segment_id'),
                'video_id': best_match.get('video_id'),
                'timestamp': timestamp,
                'formatted_time': formatted_time,
                'exact_quote': best_quote,
                'similarity_score': best_similarity,
                'conflict_warning': None
            }
        else:
            return {
                'claim': claim,
                'support_level': 'UNSUPPORTED',
                'evidence_id': None,
                'video_id': None,
                'timestamp': None,
                'formatted_time': None,
                'exact_quote': None,
                'similarity_score': 0.0,
                'conflict_warning': None
            }

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        """
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(similarity)
            except:
                pass

        # Fallback: Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _extract_quote(self, claim: str, segment_text: str, context_words: int = 10) -> Optional[str]:
        """
        Extract relevant quote from segment that supports claim.
        """
        # Split segment into sentences
        if self.nlp:
            doc = self.nlp(segment_text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            sentences = re.split(r'[.!?]+\s+', segment_text)

        if not sentences:
            return None

        # Find most similar sentence
        best_sentence = None
        best_similarity = 0.0

        for sent in sentences:
            if not sent.strip():
                continue

            similarity = self._calculate_semantic_similarity(claim, sent)
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence = sent

        # Return truncated quote if too long
        if best_sentence and len(best_sentence) > 150:
            return best_sentence[:147] + "..."

        return best_sentence

    def _assess_support_level(
        self,
        claim: str,
        segment: Optional[Dict],
        similarity: float
    ) -> str:
        """
        Assess support level for a claim.

        Levels:
        - HIGH: Strong semantic similarity (>0.7)
        - MEDIUM: Moderate similarity (0.5-0.7)
        - LOW: Weak similarity (0.3-0.5)
        - UNSUPPORTED: No good match (<0.3)
        """
        if not segment or similarity < 0.3:
            return 'UNSUPPORTED'

        # Check for exact phrase match
        text_evidence = segment.get('text_evidence', {})
        if text_evidence:
            seg_text = text_evidence.get('text', '').lower()
            if claim.lower() in seg_text or seg_text in claim.lower():
                return 'HIGH'

        # Use similarity thresholds
        if similarity >= 0.7:
            return 'HIGH'
        elif similarity >= 0.5:
            return 'MEDIUM'
        elif similarity >= 0.3:
            return 'LOW'
        else:
            return 'UNSUPPORTED'

    def _annotate_conflicts(
        self,
        attribution: Dict,
        conflicts: List[Dict]
    ) -> Dict:
        """
        Annotate attribution with conflict warnings.
        """
        evidence_id = attribution.get('evidence_id')
        if not evidence_id:
            return attribution

        # Check if this evidence is involved in conflicts
        for conflict in conflicts:
            if evidence_id in conflict.get('segments', []):
                attribution['conflict_warning'] = conflict.get('description', 'Evidence conflicts with other segments')
                attribution['support_level'] = 'CONFLICTED'
                break

        return attribution

    def _calculate_attribution_stats(self, attribution_map: List[Dict]) -> Dict:
        """
        Calculate overall statistics for attribution.
        """
        if not attribution_map:
            return {
                'overall_confidence': 0.0,
                'unsupported_claims': 0,
                'conflicted_claims': 0,
                'claim_count': 0
            }

        # Count support levels
        support_counts = {
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'UNSUPPORTED': 0,
            'CONFLICTED': 0
        }

        for attr in attribution_map:
            level = attr.get('support_level', 'UNSUPPORTED')
            support_counts[level] = support_counts.get(level, 0) + 1

        # Calculate overall confidence
        total_claims = len(attribution_map)
        confidence = (
            support_counts['HIGH'] * 1.0 +
            support_counts['MEDIUM'] * 0.7 +
            support_counts['LOW'] * 0.4 +
            support_counts['UNSUPPORTED'] * 0.0 +
            support_counts['CONFLICTED'] * 0.3
        ) / total_claims if total_claims > 0 else 0.0

        return {
            'overall_confidence': confidence,
            'unsupported_claims': support_counts['UNSUPPORTED'],
            'conflicted_claims': support_counts['CONFLICTED'],
            'claim_count': total_claims,
            'support_breakdown': support_counts
        }

    def _format_timestamp(self, timestamp: List[float]) -> str:
        """Format timestamp as MM:SS-MM:SS"""
        start, end = timestamp
        start_mm = int(start // 60)
        start_ss = int(start % 60)
        end_mm = int(end // 60)
        end_ss = int(end % 60)
        return f"{start_mm:02d}:{start_ss:02d}-{end_mm:02d}:{end_ss:02d}"

    def format_attribution_output(self, attribution_result: Dict) -> str:
        """
        Format attribution map for display.
        """
        output = []
        output.append("=" * 80)
        output.append("ATTRIBUTION MAP")
        output.append("=" * 80)
        output.append(f"Overall Confidence: {attribution_result['overall_confidence']:.2%}")
        output.append(f"Total Claims: {attribution_result['claim_count']}")
        output.append(f"Unsupported: {attribution_result['unsupported_claims']}")
        output.append(f"Conflicted: {attribution_result['conflicted_claims']}")

        if 'support_breakdown' in attribution_result:
            breakdown = attribution_result['support_breakdown']
            output.append(f"\nSupport Breakdown:")
            output.append(f"  HIGH: {breakdown.get('HIGH', 0)}")
            output.append(f"  MEDIUM: {breakdown.get('MEDIUM', 0)}")
            output.append(f"  LOW: {breakdown.get('LOW', 0)}")
            output.append(f"  UNSUPPORTED: {breakdown.get('UNSUPPORTED', 0)}")
            output.append(f"  CONFLICTED: {breakdown.get('CONFLICTED', 0)}")

        output.append("\n" + "=" * 80)
        output.append("CLAIMS AND EVIDENCE")
        output.append("=" * 80)

        for i, attr in enumerate(attribution_result['attribution_map'], 1):
            level = attr['support_level']

            if level == 'HIGH':
                indicator = "✅"
            elif level == 'MEDIUM':
                indicator = "🟡"
            elif level == 'LOW':
                indicator = "🟠"
            elif level == 'CONFLICTED':
                indicator = "⚠️"
            else:
                indicator = "❌"

            output.append(f"\n{i}. {indicator} [{level}] {attr['claim']}")

            if attr.get('evidence_id'):
                output.append(f"   📍 Evidence: {attr['video_id']} @ {attr['formatted_time']}")
                output.append(f"   🎯 Similarity: {attr['similarity_score']:.3f}")

                if attr.get('exact_quote'):
                    output.append(f"   💬 Quote: \"{attr['exact_quote']}\"")

                if attr.get('conflict_warning'):
                    output.append(f"   ⚠️  CONFLICT: {attr['conflict_warning']}")
            else:
                output.append(f"   ⚠️  No supporting evidence found")

            output.append("-" * 80)

        output.append("=" * 80)

        return "\n".join(output)
