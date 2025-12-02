"""
answer_generation.py
Answer generation for Medical VideoRAG VQA using GPT-4o-mini.

Features:
- Cost-optimized prompting (150-200 word answers)
- Timestamp-aware medical responses
- Evidence-based answer generation
- Confidence scoring
- Adaptive context selection with factual grounding (NEW)
- Self-reflection attribution (NEW)
"""

import os
import time
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import new modules (optional dependencies)
try:
    from context_selector import ContextSelector
    CONTEXT_SELECTION_AVAILABLE = True
except ImportError:
    CONTEXT_SELECTION_AVAILABLE = False
    print("⚠️  context_selector not available. Install dependencies to enable adaptive context selection.")

try:
    from attribution import SelfReflectionAttribution
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False
    print("⚠️  attribution not available. Install dependencies to enable self-reflection attribution.")

try:
    from prompts import PromptManager
    PROMPT_MANAGER_AVAILABLE = True
except ImportError:
    PROMPT_MANAGER_AVAILABLE = False


class AnswerGenerator:
    """
    Generate concise medical answers using GPT-4o-mini.

    Cost optimization:
    - GPT-4o-mini: $0.150/1M input tokens, $0.600/1M output tokens
    - Target: 150-200 word answers (~200 tokens output)
    - Average cost per query: ~$0.0003 (with retrieval context)
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
        self.enable_curation = enable_curation and CONTEXT_SELECTION_AVAILABLE
        self.enable_attribution = enable_attribution and ATTRIBUTION_AVAILABLE

        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize context selector
        self.context_selector = None
        if self.enable_curation:
            try:
                config = curation_config or {}
                self.context_selector = ContextSelector(
                    quality_threshold=config.get('quality_threshold', 0.3),
                    token_budget=config.get('token_budget', 600),
                    use_nli=config.get('use_nli', True),
                    device=config.get('device', None)
                )
                print("✅ Adaptive context selection enabled")
            except Exception as e:
                print(f"⚠️  Failed to initialize context selector: {e}")
                self.enable_curation = False

        # Initialize attribution system
        self.attributor = None
        if self.enable_attribution:
            try:
                self.attributor = SelfReflectionAttribution()
                print("✅ Self-reflection attribution enabled")
            except Exception as e:
                print(f"⚠️  Failed to initialize attributor: {e}")
                self.enable_attribution = False

        # Initialize prompt manager
        self.prompt_manager = None
        if PROMPT_MANAGER_AVAILABLE:
            self.prompt_manager = PromptManager()

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
            segment_contexts: Retrieved segments from query_faiss (with text_evidence, visual_evidence)
            max_tokens: Maximum answer length (default: 250 for ~200 words)
            temperature: Sampling temperature (0.3 for medical accuracy)
            top_k_evidence: Number of segments to include in context (default: 3)
            query_type: Optional query type override ('procedural', 'diagnostic', 'factoid', 'general')

        Returns:
            {
                'answer': str,                    # Generated answer (150-200 words)
                'confidence': float,              # Confidence score (0-1)
                'evidence_segments': List[Dict],  # Evidence with timestamps
                'model_used': str,
                'generation_time': float,
                'cost_estimate': float,           # Estimated API cost in USD
                'token_usage': Dict,              # Input/output token counts
                'attribution_map': Optional[Dict], # Self-reflection attribution (if enabled)
                'curation_stats': Optional[Dict],  # Context curation statistics (if enabled)
                'conflicts_detected': Optional[List[Dict]]  # Conflicts in evidence (if enabled)
            }
        """
        start_time = time.time()

        # Step 1: Classify query type (if not provided)
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

        # Step 2: Adaptive context selection (if enabled)
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
                            "Provide accurate, concise answers (150-200 words) with timestamp references."
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

            print(f"✅ Answer generated ({token_usage['output_tokens']} tokens, ${cost_estimate:.6f})")

        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'evidence_segments': [],
                'model_used': self.model_name,
                'generation_time': time.time() - start_time,
                'cost_estimate': 0.0,
                'token_usage': {'error': str(e)},
                'attribution_map': None,
                'curation_stats': curation_stats,
                'conflicts_detected': conflicts_detected
            }

        # Step 4: Self-reflection attribution (if enabled)
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

        # Estimate confidence based on segment scores (or use attribution confidence)
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

        return {
            'answer': answer_text,
            'confidence': confidence,
            'evidence_segments': evidence_segments,
            'model_used': self.model_name,
            'generation_time': generation_time,
            'cost_estimate': cost_estimate,
            'token_usage': token_usage,
            'attribution_map': attribution_result,
            'curation_stats': curation_stats,
            'conflicts_detected': conflicts_detected,
            'query_type': query_type
        }

    def _format_medical_prompt(self, query: str, segments: List[Dict]) -> str:
        """
        Format prompt for medical VQA with timestamp-aware context.

        Optimized for:
        - Concise answers (150-200 words)
        - Timestamp citations
        - Medical accuracy
        - Cost efficiency (minimal tokens)
        """
        # Build evidence context
        evidence_parts = []
        for i, seg in enumerate(segments, 1):
            video_id = seg.get('video_id', 'unknown')
            timestamp = seg.get('timestamp', [0, 0])

            # Format timestamp
            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                ts_str = f"{self._format_time(timestamp[0])}-{self._format_time(timestamp[1])}"
            else:
                ts_str = "unknown"

            # Get text evidence
            text_ev = seg.get('text_evidence', {})
            text = text_ev.get('text', '') if text_ev else ''

            # Truncate long text for cost optimization
            if len(text) > 400:
                text = text[:400] + "..."

            evidence_parts.append(
                f"[Segment {i}] Video: {video_id} | Time: {ts_str}\n"
                f"Content: {text}\n"
            )

        evidence_context = "\n".join(evidence_parts)

        # Construct optimized prompt
        prompt = f"""Question: {query}

Retrieved Video Evidence:
{evidence_context}

Instructions:
1. Answer the medical question accurately and concisely (150-200 words)
2. Reference specific timestamps using format [MM:SS-MM:SS]
3. Use evidence from the video transcripts provided
4. Structure answer as: direct answer, key steps/points with timestamps, brief summary
5. Use clear medical terminology appropriate for patient education

Answer:"""

        return prompt

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _extract_evidence(self, segments: List[Dict]) -> List[Dict]:
        """Extract evidence metadata with timestamps"""
        evidence = []

        for seg in segments:
            video_id = seg.get('video_id', 'unknown')
            segment_id = seg.get('segment_id', 'unknown')
            timestamp = seg.get('timestamp', [0, 0])
            combined_score = seg.get('combined_score', 0.0)

            # Get precise timestamp if available (from hierarchical search)
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

        # Normalize score to 0-1 confidence
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
        import json

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

            # Progress update
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(queries)} | Total cost so far: ${total_cost:.4f}")

        print(f"\n✅ Batch complete!")
        print(f"   Total queries: {len(queries)}")
        print(f"   Total cost: ${total_cost:.4f}")
        print(f"   Average cost per query: ${total_cost/len(queries):.6f}")

        # Save if path provided
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

            output.append(f"\n{i}. {indicator} [{level}] {claim_attr['claim']}")

            if claim_attr.get('evidence_id'):
                output.append(f"   Evidence: {claim_attr['video_id']} @ {claim_attr['formatted_time']}")
                if claim_attr.get('exact_quote'):
                    quote = claim_attr['exact_quote']
                    if len(quote) > 100:
                        quote = quote[:97] + "..."
                    output.append(f"   Quote: \"{quote}\"")
            else:
                output.append(f"   ⚠️  No supporting evidence")

    output.append("\n" + "=" * 80)

    return "\n".join(output)


# Example usage
if __name__ == "__main__":
    import json

    # Load sample search results
    with open("multimodal_search_results_hybrid.json", "r") as f:
        search_results = json.load(f)

    query = search_results['query']
    segments = search_results['results']

    # Initialize generator
    generator = AnswerGenerator(model_name="gpt-4o-mini")

    # Generate answer
    print(f"\nQuery: {query}\n")
    result = generator.generate_answer(query, segments, top_k_evidence=3)

    # Display result
    print(format_answer_output(result))

    # Save result
    with open("answer_generation_demo.json", "w") as f:
        json.dump({
            'query': query,
            **result
        }, f, indent=2)

    print(f"\n✅ Demo complete! Result saved to answer_generation_demo.json")
