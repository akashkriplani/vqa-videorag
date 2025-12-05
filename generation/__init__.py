"""
generation/

Answer generation and context curation module for Medical VideoRAG VQA.

Components:
- answer_generator: LLM-based answer generation (GPT-4o-mini)
- context_curator: Adaptive context selection with NLI factuality scoring
- attribution: Self-reflection attribution system
- prompts: Prompt templates and management

Usage:
    from generation import AnswerGenerator, ContextSelector, SelfReflectionAttribution, PromptManager

    # Initialize components
    generator = AnswerGenerator()
    curator = ContextSelector()
    attributor = SelfReflectionAttribution()
    prompt_manager = PromptManager()

    # Curate context
    curated = curator.curate_context(query, segments, query_type='procedural')

    # Generate answer
    result = generator.generate_answer(query, curated['selected_segments'])
"""

# Answer generator
from generation.answer_generator import (
    AnswerGenerator,
    format_answer_output
)

# Context curator
from generation.context_curator import (
    ContextSelector
)

# Attribution system
from generation.attribution import (
    SelfReflectionAttribution
)

# Prompt manager
from generation.prompts import (
    PromptManager,
    CostOptimizedPrompts
)

__all__ = [
    # Answer generation
    'AnswerGenerator',
    'format_answer_output',

    # Context curation
    'ContextSelector',

    # Attribution
    'SelfReflectionAttribution',

    # Prompts
    'PromptManager',
    'CostOptimizedPrompts'
]

__version__ = '1.0.0'
