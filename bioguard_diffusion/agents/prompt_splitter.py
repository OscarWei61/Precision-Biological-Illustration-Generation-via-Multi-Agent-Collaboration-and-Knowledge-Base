"""
Prompt Splitter — splits a structured biological prompt into two parts
optimized for FLUX's dual-encoder architecture:

  prompt   → CLIP encoder (≤77 tokens): visual keywords, style, core subject
  prompt_2 → T5 encoder (unlimited):    full natural-language detail,
             biological structure, labels, spatial relationships

Two separate LLM calls ensure each encoder gets a prompt tuned for its
strengths rather than a truncated version of one long prompt.
"""

import anthropic

try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_anthropic as _wrap_anthropic
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator
    def _wrap_anthropic(client): return client

MODEL = "claude-haiku-4-5-20251001"

_CLIP_SYSTEM = """\
You are an expert at writing image generation prompts for the FLUX model's CLIP text encoder.

CLIP encoder rules:
- HARD LIMIT: 77 tokens maximum (approximately 55–65 words)
- Style: comma-separated keywords and short noun phrases
- NO full sentences, NO conjunctions, NO explanations
- Include: biological subject, key visible structures (2–4 max), diagram style, quality keywords
- Examples of good CLIP prompts:
    "scientific cross-section diagram, mitochondria, cristae, matrix, white background, biology textbook, no text"
    "neuron diagram, axon, dendrites, myelin sheath, educational illustration, clean linework, no labels"

Respond with ONLY the CLIP prompt text. No explanation, no quotes, no labels."""

_T5_SYSTEM = """\
You are an expert at writing image generation prompts for the FLUX model's T5 text encoder.

T5 encoder rules:
- No token limit — use as much detail as needed
- Write in clear natural language with full sentences
- Include ALL biological structures and their spatial relationships
- Describe layering, cross-sections, color coding where relevant
- Mention biological functions to reinforce structural accuracy
- Include diagram style directives (educational, scientific, textbook, white background, no text, no labels)

Respond with ONLY the T5 prompt text. No explanation, no quotes, no labels."""

_CLIP_USER = """\
Original biological illustration prompt:
{prompt}

Generate the CLIP encoder prompt (≤77 tokens, keyword style)."""

_T5_USER = """\
Original biological illustration prompt:
{prompt}

Generate the T5 encoder prompt (detailed natural language, include every structural and biological detail)."""


class PromptSplitter:
    """
    Calls claude-haiku-4-5 twice to produce CLIP and T5 sub-prompts
    from a structured SpatialArchitect prompt.

    api_key: Anthropic API key. Reads ANTHROPIC_API_KEY env var if not provided.
    """

    def __init__(self, api_key: str = None):
        self._client = _wrap_anthropic(
        )
        print(f"PromptSplitter: {MODEL} ready.")

    @traceable(name="prompt_splitter_llm_call", run_type="llm")
    def _call(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    @traceable(name="prompt_splitter_split", run_type="chain")
    def split(self, prompt: str) -> dict:
        """
        Returns:
          clip_prompt  str — short keyword prompt for CLIP encoder (≤77 tokens)
          t5_prompt    str — detailed natural-language prompt for T5 encoder
        """
        clip_prompt = self._call(_CLIP_SYSTEM, _CLIP_USER.format(prompt=prompt))
        t5_prompt   = self._call(_T5_SYSTEM,   _T5_USER.format(prompt=prompt))
        return {"clip_prompt": clip_prompt, "t5_prompt": t5_prompt}
