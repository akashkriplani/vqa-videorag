"""
generation/answer_generator.py

LLM-based answer generation for Medical VideoRAG VQA.

Features:
- Cost-optimized prompting (GPT-4o-mini)
- Timestamp-aware medical responses
- Evidence-based answer generation
- Confidence scoring
- Integrated context curation and attribution
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _convert_to_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    return obj


class AnswerGenerator:
    """
    Generate concise medical answers using GPT-4o-mini.

    Cost optimization:
    - GPT-4o-mini: $0.150/1M input tokens, $0.600/1M output tokens
    - Target: 150-200 word answers (~200 tokens output)
    - Average cost per query: ~$0.0003
    """

    def __init__(
        self,
        model_name="gpt-4o-mini",
        api_key=None,
        enable_curation=True,
        enable_attribution=True,
        curation_config=None
    ):
        """
        Initialize answer generator.

        Args:
            model_name: OpenAI model name (default: gpt-4o-mini)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            enable_curation: Enable adaptive context selection (default: True)
            enable_attribution: Enable self-reflection attribution (default: True)
            curation_config: Optional config dict for ContextSelector
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.enable_curation = enable_curation
        self.enable_attribution = enable_attribution

        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize context selector
        self.context_selector = None
        if self.enable_curation:
            try:
                from generation.context_curator import ContextSelector
                config = curation_config or {}
                self.context_selector = ContextSelector(**config)
                print("✅ Context curation enabled")
            except Exception as e:
                print(f"⚠️  Context curation unavailable: {e}")
                self.enable_curation = False

        # Initialize attribution system
        self.attributor = None
        if self.enable_attribution:
            try:
                from generation.attribution import SelfReflectionAttribution
                self.attributor = SelfReflectionAttribution()
                print("✅ Self-reflection attribution enabled")
            except Exception as e:
                print(f"⚠️  Attribution unavailable: {e}")
                self.enable_attribution = False

        # Initialize prompt manager
        self.prompt_manager = None
        try:
            from generation.prompts import PromptManager
            self.prompt_manager = PromptManager()
            print("✅ Prompt manager loaded")
        except Exception as e:
            print(f"⚠️  Prompt manager unavailable: {e}")

    def generate_answer(
        self,
        query: str,
        segment_contexts: List[Dict],
        max_tokens: int = 250,
        temperature: float = 0.3,
        top_k_evidence: int = 3,
        query_type: Optional[str] = None
    ) -> Dict:
        """
        Generate answer from retrieved multimodal segments.

        Args:
            query: User question
            segment_contexts: Retrieved segments from query_faiss
            max_tokens: Maximum answer length (default: 250 for ~200 words)
            temperature: Sampling temperature (0.3 for medical accuracy)
            top_k_evidence: Number of segments to include (default: 3)
            query_type: Optional query type override ('procedural', 'diagnostic', 'factoid', 'general')

        Returns:
            {
                'answer': str,
                'confidence': float,
                'evidence_segments': List[Dict],
                'model_used': str,
                'generation_time': float,
                'cost_estimate': float,
                'token_usage': Dict,
                'attribution_map': Optional[Dict],
                'curation_stats': Optional[Dict],
                'conflicts_detected': Optional[List[Dict]],
                'query_type': str
            }
        """
        start_time = time.time()

        # Step 1: Classify query type
        if query_type is None and self.prompt_manager:
            query_type = self.prompt_manager.classify_question(query)
        elif query_type is None:
            query_type = 'general'

        print(f"\n{'='*80}")
        print(f"ANSWER GENERATION PIPELINE")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Query type: {query_type}")
        print(f"Input segments: {len(segment_contexts)}")

        # Step 2: Adaptive context selection
        curated_segments = segment_contexts
        curation_stats = None
        conflicts_detected = None

        if self.enable_curation and self.context_selector:
            print(f"\n🔍 Applying adaptive context selection...")
            curation_result = self.context_selector.curate_context(
                query=query,
                segments=segment_contexts,
                query_type=query_type
            )

            curated_segments = curation_result['selected_segments']
            curation_stats = curation_result['stats']
            conflicts_detected = curation_result['conflicts_detected']

            print(f"✅ Context curated: {len(curated_segments)} segments selected")

            if conflicts_detected:
                print(f"⚠️  {len(conflicts_detected)} conflicts detected and resolved")
        else:
            # Fallback: simple top-k truncation
            curated_segments = segment_contexts[:top_k_evidence]
            print(f"Using top-{top_k_evidence} segments (curation disabled)")

        # Assess retrieval quality before generation
        quality_check = self._assess_retrieval_quality(curated_segments, query)

        if not quality_check['sufficient']:
            print(f"\n⚠️  WARNING: Low retrieval quality detected!")
            print(f"   Max similarity: {quality_check['max_similarity']:.3f}")
            print(f"   Recommendation: {quality_check['message']}")

            # Return a refusal message instead of hallucinating
            return _convert_to_serializable({
                'answer': quality_check['message'],
                'confidence': 0.0,
                'evidence_segments': self._extract_evidence(curated_segments),
                'model_used': self.model_name,
                'generation_time': float(time.time() - start_time),
                'cost_estimate': 0.0,
                'token_usage': {'refused': True},
                'attribution_map': None,
                'curation_stats': curation_stats,
                'conflicts_detected': conflicts_detected,
                'query_type': query_type,
                'quality_check': quality_check
            })

        # Format prompt with evidence
        prompt = self._format_medical_prompt(query, curated_segments)

        # Step 3: Generate answer
        print(f"\n💬 Generating answer with {self.model_name}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical AI assistant specializing in medical video education. "
                            "Provide accurate, concise answers (150-200 words) citing evidence with video IDs and timestamps.\n\n"
                            "CITATION FORMAT: Use [Video 1: video_id, Time: MM:SS-MM:SS] or just [Video 1] when referencing evidence.\n\n"
                            "CRITICAL: You must ONLY use information from the provided video evidence. "
                            "If the evidence does not contain relevant information to answer the query, "
                            "respond with: 'The provided video evidence does not contain information about [topic]. "
                            "The available videos cover different medical topics.'\n\n"
                            "DO NOT use your general medical knowledge if the evidence is irrelevant."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer_text = response.choices[0].message.content
            token_usage = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            # Calculate cost (GPT-4o-mini pricing)
            cost_estimate = (
                token_usage['input_tokens'] * 0.150 / 1_000_000 +
                token_usage['output_tokens'] * 0.600 / 1_000_000
            )

            # Post-process: inject timestamps for evidence citations
            answer_text = self._inject_timestamps(answer_text, curated_segments)

            print(f"✅ Answer generated ({token_usage['output_tokens']} tokens, ${cost_estimate:.6f})")

        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            return _convert_to_serializable({
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'evidence_segments': [],
                'model_used': self.model_name,
                'generation_time': float(time.time() - start_time),
                'cost_estimate': 0.0,
                'token_usage': {'error': str(e)},
                'attribution_map': None,
                'curation_stats': curation_stats,
                'conflicts_detected': conflicts_detected,
                'query_type': query_type
            })

        # Step 4: Self-reflection attribution
        attribution_result = None
        if self.enable_attribution and self.attributor:
            print(f"\n🔍 Generating attribution map...")
            attribution_result = self.attributor.generate_attribution(
                answer=answer_text,
                evidence_segments=curated_segments,
                conflicts=conflicts_detected
            )
            print(f"✅ Attribution complete: {attribution_result['overall_confidence']:.2%} confidence")

        # Extract evidence segments with timestamps
        evidence_segments = self._extract_evidence(curated_segments)

        # Estimate confidence
        if attribution_result:
            confidence = attribution_result['overall_confidence']
        else:
            confidence = self._estimate_confidence(curated_segments)

        generation_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {generation_time:.2f}s")
        print(f"Final confidence: {confidence:.2%}")
        print(f"{'='*80}\n")

        return _convert_to_serializable({
            'answer': answer_text,
            'confidence': float(confidence),
            'evidence_segments': evidence_segments,
            'model_used': self.model_name,
            'generation_time': float(generation_time),
            'cost_estimate': float(cost_estimate),
            'token_usage': token_usage,
            'attribution_map': attribution_result,
            'curation_stats': curation_stats,
            'conflicts_detected': conflicts_detected,
            'query_type': query_type
        })

    def _format_medical_prompt(self, query: str, segments: List[Dict]) -> str:
        """
        Format prompt for medical VQA with timestamp-aware context.

        Uses PromptManager if available, otherwise default template.
        """
        if self.prompt_manager:
            return self.prompt_manager.format_prompt(query, segments)

        # Fallback: default template
        evidence_parts = []
        for i, seg in enumerate(segments, 1):
            video_id = seg.get('video_id', 'unknown')
            timestamp = seg.get('timestamp', [0, 0])

            # Get precise timestamp if available
            text_ev = seg.get('text_evidence', {})
            if text_ev and text_ev.get('precise_timestamp'):
                precise_ts = text_ev['precise_timestamp']
                if isinstance(precise_ts, (list, tuple)) and len(precise_ts) == 2:
                    ts_str = f"{self._format_time(precise_ts[0])}-{self._format_time(precise_ts[1])}"
                else:
                    ts_str = f"{self._format_time(timestamp[0])}-{self._format_time(timestamp[1])}"
            elif isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                ts_str = f"{self._format_time(timestamp[0])}-{self._format_time(timestamp[1])}"
            else:
                ts_str = "unknown"

            text = text_ev.get('text', '') if text_ev else ''

            if len(text) > 400:
                text = text[:400] + "..."

            evidence_parts.append(
                f"[Video {i}: {video_id}, Time: {ts_str}]\n"
                f"Content: {text}\n"
            )

        evidence_context = "\n".join(evidence_parts)

        return f"""Question: {query}

Retrieved Video Evidence:
{evidence_context}

Instructions:
1. Answer the medical question accurately and concisely (150-200 words)
2. Cite evidence using format: [Video 1: video_id, Time: MM:SS-MM:SS] or just [Video 1]
3. When referencing specific information, cite the video number (e.g., "As shown in [Video 1]")
4. Use ONLY information from the provided evidence transcripts
5. If the evidence doesn't fully answer the question, acknowledge what's covered and what's not
6. Use clear medical terminology appropriate for patient education

Answer:"""

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _assess_retrieval_quality(
        self,
        segments: List[Dict],
        query: str,
        min_similarity_threshold: float = 0.45,
        min_avg_similarity: float = 0.35
    ) -> Dict:
        """
        Assess whether retrieved segments are relevant enough to answer the query.

        Prevents hallucination by detecting when retrieval has failed.

        Args:
            segments: Retrieved segments
            query: User query
            min_similarity_threshold: Minimum required top similarity (default: 0.45 - lowered for better recall)
            min_avg_similarity: Minimum required average similarity (default: 0.35)

        Returns:
            {
                'sufficient': bool,
                'max_similarity': float,
                'avg_similarity': float,
                'message': str,
                'segment_count': int
            }
        """
        if not segments:
            return {
                'sufficient': False,
                'max_similarity': 0.0,
                'avg_similarity': 0.0,
                'message': 'No relevant video evidence found in the database for this query. The video collection may not cover this topic.',
                'segment_count': 0
            }

        # Extract similarity scores (check original score before curation adjustments)
        similarities = []
        for seg in segments:
            # Prioritize original_combined_score (pre-curation) > combined_score > final_score > text_score
            score = (
                seg.get('original_combined_score') or
                seg.get('combined_score') or
                seg.get('final_score') or
                seg.get('text_score') or
                0.0
            )
            similarities.append(score)

        max_sim = max(similarities) if similarities else 0.0
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

        # Quality checks
        sufficient = max_sim >= min_similarity_threshold and avg_sim >= min_avg_similarity

        if not sufficient:
            if max_sim < 0.35:
                reason = f"The retrieved segments have very low relevance (max: {max_sim:.2f}). This query may be outside the scope of the video database."
            elif max_sim < min_similarity_threshold:
                reason = f"The retrieved segments have moderate relevance (max: {max_sim:.2f}) but may not provide sufficient detail to answer accurately."
            else:
                reason = f"The average quality of retrieved segments is low (avg: {avg_sim:.2f}). Multiple segments needed but quality is insufficient."

            message = (
                f"Unable to provide a reliable answer based on available video evidence. "
                f"{reason}\n\n"
                f"Suggestion: Try queries about physical therapy, injury treatment, or medical procedures "
                f"that are within the scope of the video collection."
            )
        else:
            message = "Sufficient evidence available"

        return {
            'sufficient': bool(sufficient),
            'max_similarity': float(max_sim),
            'avg_similarity': float(avg_sim),
            'message': message,
            'segment_count': int(len(segments))
        }

    def _assess_retrieval_quality(
        self,
        segments: List[Dict],
        query: str,
        min_similarity_threshold: float = 0.45,
        min_avg_similarity: float = 0.35
    ) -> Dict:
        """
        Assess whether retrieved segments are relevant enough to answer the query.

        Prevents hallucination by detecting when retrieval has failed.

        Args:
            segments: Retrieved segments
            query: User query
            min_similarity_threshold: Minimum required top similarity (default: 0.45 - lowered for better recall)
            min_avg_similarity: Minimum required average similarity (default: 0.35)

        Returns:
            {
                'sufficient': bool,
                'max_similarity': float,
                'avg_similarity': float,
                'message': str,
                'segment_count': int
            }
        """
        if not segments:
            return {
                'sufficient': False,
                'max_similarity': 0.0,
                'avg_similarity': 0.0,
                'message': 'No relevant video evidence found in the database for this query. The video collection may not cover this topic.',
                'segment_count': 0
            }

        # Extract similarity scores (check original score before curation adjustments)
        similarities = []
        for seg in segments:
            # Prioritize original_combined_score (pre-curation) > combined_score > final_score > text_score
            score = (
                seg.get('original_combined_score') or
                seg.get('combined_score') or
                seg.get('final_score') or
                seg.get('text_score') or
                0.0
            )
            similarities.append(score)

        max_sim = max(similarities) if similarities else 0.0
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

        # Quality checks
        sufficient = max_sim >= min_similarity_threshold and avg_sim >= min_avg_similarity

        if not sufficient:
            if max_sim < 0.35:
                reason = f"The retrieved segments have very low relevance (max: {max_sim:.2f}). This query may be outside the scope of the video database."
            elif max_sim < min_similarity_threshold:
                reason = f"The retrieved segments have moderate relevance (max: {max_sim:.2f}) but may not provide sufficient detail to answer accurately."
            else:
                reason = f"The average quality of retrieved segments is low (avg: {avg_sim:.2f}). Multiple segments needed but quality is insufficient."

            message = (
                f"Unable to provide a reliable answer based on available video evidence. "
                f"{reason}\n\n"
                f"Suggestion: Try queries about physical therapy, injury treatment, or medical procedures "
                f"that are within the scope of the video collection."
            )
        else:
            message = "Sufficient evidence available"

        return {
            'sufficient': bool(sufficient),
            'max_similarity': float(max_sim),
            'avg_similarity': float(avg_sim),
            'message': message,
            'segment_count': int(len(segments))
        }

    def _inject_timestamps(self, answer_text: str, segments: List[Dict]) -> str:
        """
        Post-process answer to inject timestamps for evidence citations.

        Converts:
        - [Evidence 1] -> [Evidence 1, 02:21-02:23]
        - Evidence 1 -> Evidence 1 (02:21-02:23)
        """
        import re

        # Build evidence number to timestamp mapping
        evidence_map = {}
        for i, seg in enumerate(segments, 1):
            timestamp = seg.get('timestamp', [0, 0])

            # Get precise timestamp if available
            text_ev = seg.get('text_evidence', {})
            if text_ev and text_ev.get('precise_timestamp'):
                precise_ts = text_ev['precise_timestamp']
                if isinstance(precise_ts, (list, tuple)) and len(precise_ts) == 2:
                    timestamp = precise_ts

            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                ts_str = f"{self._format_time(timestamp[0])}-{self._format_time(timestamp[1])}"
                evidence_map[i] = ts_str

        # Pattern 1: [Evidence N] -> [Evidence N, HH:MM-HH:MM]
        def replace_bracketed(match):
            num = int(match.group(1))
            if num in evidence_map:
                return f"[Evidence {num}, {evidence_map[num]}]"
            return match.group(0)

        answer_text = re.sub(r'\[Evidence\s+(\d+)\]', replace_bracketed, answer_text)

        # Pattern 2: Evidence N (not already in brackets) -> Evidence N (HH:MM-HH:MM)
        def replace_unbracketed(match):
            # Check if already followed by timestamp
            num = int(match.group(1))
            if num in evidence_map:
                # Don't replace if already has timestamp
                if re.match(r'\s*\(?\d{2}:\d{2}', match.group(2) or ''):
                    return match.group(0)
                return f"Evidence {num} ({evidence_map[num]})"
            return match.group(0)

        # Only replace if not already in brackets
        answer_text = re.sub(r'(?<!\[)Evidence\s+(\d+)(?!\])(\s|[,.])?', replace_unbracketed, answer_text)

        return answer_text

    def _extract_evidence(self, segments: List[Dict]) -> List[Dict]:
        """Extract evidence metadata with timestamps"""
        evidence = []

        for seg in segments:
            video_id = seg.get('video_id', 'unknown')
            segment_id = seg.get('segment_id', 'unknown')
            timestamp = seg.get('timestamp', [0, 0])
            combined_score = seg.get('combined_score', 0.0)

            # Get precise timestamp if available
            text_ev = seg.get('text_evidence', {})
            if text_ev and text_ev.get('precise_timestamp'):
                precise_ts = text_ev['precise_timestamp']
            else:
                precise_ts = timestamp

            evidence.append({
                'video_id': video_id,
                'segment_id': segment_id,
                'timestamp': timestamp,
                'precise_timestamp': precise_ts,
                'relevance_score': combined_score,
                'formatted_time': (
                    f"{self._format_time(precise_ts[0])}-{self._format_time(precise_ts[1])}"
                    if isinstance(precise_ts, (list, tuple)) and len(precise_ts) == 2
                    else "unknown"
                )
            })

        return evidence

    def _estimate_confidence(self, segments: List[Dict]) -> float:
        """
        Estimate answer confidence based on retrieval scores.

        Heuristic:
        - High confidence: top segment score > 0.7
        - Medium confidence: top segment score > 0.5
        - Low confidence: top segment score <= 0.5
        """
        if not segments:
            return 0.0

        top_score = segments[0].get('combined_score', 0.0)

        if top_score > 0.7:
            return min(0.9, top_score)
        elif top_score > 0.5:
            return min(0.7, top_score)
        else:
            return min(0.5, top_score)

    def batch_generate(
        self,
        queries: List[str],
        segment_contexts_list: List[List[Dict]],
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate answers for multiple queries (for evaluation).

        Args:
            queries: List of questions
            segment_contexts_list: List of segment contexts (one per query)
            save_path: Optional path to save results JSON

        Returns:
            List of answer dictionaries
        """
        assert len(queries) == len(segment_contexts_list), "Queries and contexts must match"

        results = []
        total_cost = 0.0

        print(f"Generating answers for {len(queries)} queries...")

        for i, (query, contexts) in enumerate(zip(queries, segment_contexts_list), 1):
            print(f"[{i}/{len(queries)}] Processing: {query[:50]}...")

            result = self.generate_answer(query, contexts)
            results.append({
                'query': query,
                **result
            })

            total_cost += result['cost_estimate']

            if i % 10 == 0:
                print(f"Progress: {i}/{len(queries)} | Cost so far: ${total_cost:.4f}")

        print(f"\n✅ Batch complete!")
        print(f"   Total queries: {len(queries)}")
        print(f"   Total cost: ${total_cost:.4f}")
        print(f"   Average cost per query: ${total_cost/len(queries):.6f}")

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   Results saved to: {save_path}")

        return results


def format_answer_output(result: Dict, show_attribution: bool = True, show_curation: bool = True) -> str:
    """
    Format answer result for display.

    Args:
        result: Output from generate_answer()
        show_attribution: Whether to show attribution map (default: True)
        show_curation: Whether to show curation statistics (default: True)

    Returns:
        Formatted string for console/UI display
    """
    output = []
    output.append("=" * 80)
    output.append("ANSWER")
    output.append("=" * 80)
    output.append(result['answer'])
    output.append("\n" + "-" * 80)
    output.append(f"Confidence: {result['confidence']:.2%}")
    output.append(f"Query Type: {result.get('query_type', 'general')}")
    output.append(f"Model: {result['model_used']}")
    output.append(f"Generation time: {result['generation_time']:.2f}s")
    output.append(f"Cost: ${result['cost_estimate']:.6f}")

    if result.get('token_usage'):
        tokens = result['token_usage']
        output.append(
            f"Tokens: {tokens.get('input_tokens', 0)} in, "
            f"{tokens.get('output_tokens', 0)} out, "
            f"{tokens.get('total_tokens', 0)} total"
        )

    # Show curation statistics
    if show_curation and result.get('curation_stats'):
        output.append("\n" + "-" * 80)
        output.append("CONTEXT CURATION STATISTICS")
        output.append("-" * 80)
        stats = result['curation_stats']
        output.append(f"Input segments: {stats.get('input_segments', 0)}")
        output.append(f"After quality filter: {stats.get('after_quality_filter', 0)}")
        output.append(f"After relevance scoring: {stats.get('after_relevance_scoring', 0)}")
        output.append(f"After NLI scoring: {stats.get('after_nli_scoring', 0)}")
        output.append(f"Final selected: {stats.get('final_selected', 0)}")

        if stats.get('conflicts_found', 0) > 0:
            output.append(f"⚠️  Conflicts detected: {stats['conflicts_found']}")

    # Show conflicts
    if result.get('conflicts_detected'):
        output.append("\n" + "-" * 80)
        output.append("⚠️  CONFLICTS DETECTED")
        output.append("-" * 80)
        for i, conflict in enumerate(result['conflicts_detected'], 1):
            output.append(f"{i}. {conflict['description']}")
            output.append(f"   Resolution: {conflict['resolution']}")

    # Show evidence sources
    output.append("\n" + "-" * 80)
    output.append("EVIDENCE SOURCES")
    output.append("-" * 80)

    for i, ev in enumerate(result.get('evidence_segments', []), 1):
        output.append(
            f"{i}. Video: {ev['video_id']} | Time: {ev['formatted_time']} | "
            f"Score: {ev['relevance_score']:.4f}"
        )

    # Show attribution map
    if show_attribution and result.get('attribution_map'):
        output.append("\n" + "-" * 80)
        output.append("ATTRIBUTION MAP")
        output.append("-" * 80)

        attr = result['attribution_map']
        output.append(f"Overall Confidence: {attr['overall_confidence']:.2%}")
        output.append(f"Unsupported Claims: {attr['unsupported_claims']}/{attr['claim_count']}")
        output.append(f"Conflicted Claims: {attr['conflicted_claims']}/{attr['claim_count']}")

        output.append("\nClaim-by-Claim Attribution:")
        for i, claim_attr in enumerate(attr['attribution_map'], 1):
            level = claim_attr['support_level']
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

            output.append(f"{i}. {indicator} [{level}] {claim_attr['claim']}")
            if claim_attr.get('evidence_id'):
                output.append(f"   📍 {claim_attr['video_id']} @ {claim_attr['formatted_time']}")

    output.append("\n" + "=" * 80)

    return "\n".join(output)
