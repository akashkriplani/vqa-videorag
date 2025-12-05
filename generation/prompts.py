"""
generation/prompts.py

Prompt templates for Medical VideoRAG VQA.

Features:
- Question type classification (procedural, diagnostic, factoid, general)
- Type-specific prompt templates
- Cost-optimized formatting
- Timestamp citation support
"""

from typing import List, Dict


class PromptManager:
    """
    Manage prompt templates for different medical question types.
    """

    def __init__(self):
        self.system_prompt = (
            "You are a medical AI assistant specializing in medical video education. "
            "Provide accurate, concise answers (150-200 words) with timestamp references."
        )

        # Question type templates
        self.templates = {
            'factoid': self._factoid_template,
            'procedural': self._procedural_template,
            'diagnostic': self._diagnostic_template,
            'general': self._general_template
        }

    def format_prompt(
        self,
        query: str,
        segments: List[Dict],
        question_type: str = 'general',
        max_segments: int = 3
    ) -> str:
        """
        Format prompt based on question type.

        Args:
            query: User question
            segments: Retrieved video segments
            question_type: 'factoid', 'procedural', 'diagnostic', or 'general'
            max_segments: Maximum segments to include

        Returns:
            Formatted prompt string
        """
        # Auto-detect question type if general
        if question_type == 'general':
            question_type = self.classify_question(query)

        template_func = self.templates.get(question_type, self._general_template)
        return template_func(query, segments[:max_segments])

    def classify_question(self, query: str) -> str:
        """
        Classify question type using keyword heuristics.

        Returns:
            'factoid', 'procedural', 'diagnostic', or 'general'
        """
        query_lower = query.lower()

        # Procedural questions
        if any(kw in query_lower for kw in ['how to', 'steps', 'procedure', 'perform', 'examine']):
            return 'procedural'

        # Diagnostic questions
        if any(kw in query_lower for kw in ['diagnose', 'identify', 'assess', 'evaluate', 'symptoms']):
            return 'diagnostic'

        # Factoid questions
        if any(kw in query_lower for kw in ['what is', 'what are', 'when', 'where', 'which']):
            return 'factoid'

        # Default to general
        return 'general'

    def _format_evidence(self, segments: List[Dict]) -> str:
        """Format evidence context from segments"""
        evidence_parts = []

        for i, seg in enumerate(segments, 1):
            video_id = seg.get('video_id', 'unknown')
            timestamp = seg.get('timestamp', [0, 0])

            # Get precise timestamp if available
            text_ev = seg.get('text_evidence', {})
            if text_ev and text_ev.get('precise_timestamp'):
                precise_ts = text_ev['precise_timestamp']
                if isinstance(precise_ts, (list, tuple)) and len(precise_ts) == 2:
                    timestamp = precise_ts

            # Format timestamp
            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                start_mm = int(timestamp[0] // 60)
                start_ss = int(timestamp[0] % 60)
                end_mm = int(timestamp[1] // 60)
                end_ss = int(timestamp[1] % 60)
                ts_str = f"{start_mm:02d}:{start_ss:02d}-{end_mm:02d}:{end_ss:02d}"
            else:
                ts_str = "unknown"

            # Get text evidence
            text = text_ev.get('text', '') if text_ev else ''

            # Truncate for cost optimization
            if len(text) > 350:
                text = text[:350] + "..."

            evidence_parts.append(
                f"[Video {i}: {video_id}, Time: {ts_str}]\n"
                f"Content: {text}"
            )

        return "\n\n".join(evidence_parts)

    def _factoid_template(self, query: str, segments: List[Dict]) -> str:
        """Template for factoid questions (What is...?)"""
        evidence = self._format_evidence(segments)

        return f"""Question: {query}

Video Evidence:
{evidence}

Instructions:
1. Provide a direct, concise answer (100-150 words)
2. Include key definition or fact from the evidence
3. Cite using format: [Video 1: video_id, Time: MM:SS-MM:SS] or just [Video 1]
4. Use clear medical terminology
5. Only use information from the provided evidence

Answer:"""

    def _procedural_template(self, query: str, segments: List[Dict]) -> str:
        """Template for procedural questions (How to...?)"""
        evidence = self._format_evidence(segments)

        return f"""Question: {query}

Video Evidence:
{evidence}

Instructions:
1. Provide step-by-step procedural answer (150-200 words)
2. Cite using format: [Video 1: video_id, Time: MM:SS-MM:SS] or just [Video 1]
3. Reference specific evidence when describing steps
4. Include important safety notes or considerations from the evidence
5. Only use information present in the provided evidence

Answer:"""

    def _diagnostic_template(self, query: str, segments: List[Dict]) -> str:
        """Template for diagnostic questions"""
        evidence = self._format_evidence(segments)

        return f"""Question: {query}

Video Evidence:
{evidence}

Instructions:
1. Provide diagnostic guidance (150-200 words)
2. List key assessment criteria, citing evidence using format: [Video 1: video_id, Time: MM:SS-MM:SS] or [Video 1]
3. Include what to look for and warning signs from the evidence
4. Only use information from the provided evidence
5. Use clear medical terminology

Answer:"""

    def _general_template(self, query: str, segments: List[Dict]) -> str:
        """General template for all question types"""
        evidence = self._format_evidence(segments)

        return f"""Question: {query}

Video Evidence:
{evidence}

Instructions:
1. Answer the medical question accurately (150-200 words)
2. Use ONLY evidence from video segments provided
3. Cite using format: [Video 1: video_id, Time: MM:SS-MM:SS] or just [Video 1]
4. Structure: direct answer → supporting details → brief summary
5. Use medical terminology appropriate for patient education
6. If evidence doesn't fully answer the question, acknowledge what's covered

Answer:"""

    def get_system_prompt(self) -> str:
        """Get system prompt for chat models"""
        return self.system_prompt


class CostOptimizedPrompts:
    """
    Ultra-minimal prompts for extreme cost optimization.
    Use when processing large batches.
    """

    @staticmethod
    def minimal_prompt(query: str, text_snippet: str, timestamp: str) -> str:
        """Minimal prompt (~50% token reduction)"""
        return f"""Q: {query}
Evidence [{timestamp}]: {text_snippet[:200]}

Answer in 150 words with timestamps:"""

    @staticmethod
    def batch_prompt(queries_with_context: List[tuple]) -> str:
        """Batch multiple queries in one API call (experimental)"""
        prompts = []
        for i, (query, context, ts) in enumerate(queries_with_context, 1):
            prompts.append(f"{i}. Q: {query}\nContext [{ts}]: {context[:150]}")

        return f"""Answer these medical questions (150 words each with timestamps):

{chr(10).join(prompts)}

Answers:"""
