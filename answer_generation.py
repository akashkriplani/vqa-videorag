"""
answer_generation.py
Answer generation for Medical VideoRAG VQA using GPT-4o-mini.

Features:
- Cost-optimized prompting (150-200 word answers)
- Timestamp-aware medical responses
- Evidence-based answer generation
- Confidence scoring
"""

import os
import time
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AnswerGenerator:
    """
    Generate concise medical answers using GPT-4o-mini.

    Cost optimization:
    - GPT-4o-mini: $0.150/1M input tokens, $0.600/1M output tokens
    - Target: 150-200 word answers (~200 tokens output)
    - Average cost per query: ~$0.0003 (with retrieval context)
    """

    def __init__(self, model_name="gpt-4o-mini", api_key=None):
        """
        Initialize answer generator.

        Args:
            model_name: OpenAI model name (default: gpt-4o-mini)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        if not self.client.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

    def generate_answer(
        self,
        query: str,
        segment_contexts: List[Dict],
        max_tokens: int = 250,
        temperature: float = 0.3,
        top_k_evidence: int = 3
    ) -> Dict:
        """
        Generate answer from retrieved multimodal segments.

        Args:
            query: User question
            segment_contexts: Retrieved segments from query_faiss (with text_evidence, visual_evidence)
            max_tokens: Maximum answer length (default: 250 for ~200 words)
            temperature: Sampling temperature (0.3 for medical accuracy)
            top_k_evidence: Number of segments to include in context (default: 3)

        Returns:
            {
                'answer': str,                    # Generated answer (150-200 words)
                'confidence': float,              # Confidence score (0-1)
                'evidence_segments': List[Dict],  # Evidence with timestamps
                'model_used': str,
                'generation_time': float,
                'cost_estimate': float,           # Estimated API cost in USD
                'token_usage': Dict               # Input/output token counts
            }
        """
        start_time = time.time()

        # Limit evidence to top-k for cost optimization
        top_segments = segment_contexts[:top_k_evidence]

        # Format prompt with evidence
        prompt = self._format_medical_prompt(query, top_segments)

        # Generate answer
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

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'evidence_segments': [],
                'model_used': self.model_name,
                'generation_time': time.time() - start_time,
                'cost_estimate': 0.0,
                'token_usage': {'error': str(e)}
            }

        # Extract evidence segments with timestamps
        evidence_segments = self._extract_evidence(top_segments)

        # Estimate confidence based on segment scores
        confidence = self._estimate_confidence(top_segments)

        return {
            'answer': answer_text,
            'confidence': confidence,
            'evidence_segments': evidence_segments,
            'model_used': self.model_name,
            'generation_time': time.time() - start_time,
            'cost_estimate': cost_estimate,
            'token_usage': token_usage
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


def format_answer_output(result: Dict) -> str:
    """
    Format answer result for display.

    Args:
        result: Output from generate_answer()

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

    output.append("\n" + "-" * 80)
    output.append("EVIDENCE SOURCES")
    output.append("-" * 80)

    for i, ev in enumerate(result.get('evidence_segments', []), 1):
        output.append(
            f"{i}. Video: {ev['video_id']} | Time: {ev['formatted_time']} | "
            f"Score: {ev['relevance_score']:.4f}"
        )

    output.append("=" * 80)

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
