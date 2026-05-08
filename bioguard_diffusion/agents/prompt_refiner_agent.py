"""
Prompt Refiner Agent — surgically rewrites CLIP+T5 prompts based on verifier feedback.

Hard token limits:
  CLIP : 55 words  (~77 CLIP tokens, safe margin)
  T5   : 150 words (enough structural detail, prevents JSON truncation)

If LLM output exceeds these limits it is word-truncated — never falls back to the
original prompt.
"""

import json
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

CLIP_MAX_WORDS = 55   # ≤77 CLIP tokens
T5_MAX_WORDS   = 150  # keeps JSON well under 2048 output tokens


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate to max_words at a word boundary. Never raises."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


_SYSTEM = f"""\
You are an expert at refining biological image generation prompts for the FLUX diffusion model.
FLUX uses two text encoders:
  - CLIP prompt : keyword style, HARD LIMIT {CLIP_MAX_WORDS} words maximum
  - T5 prompt   : natural language, HARD LIMIT {T5_MAX_WORDS} words maximum

Your task: given the current CLIP+T5 prompts and verifier feedback, produce REFINED versions that:
1. PRESERVE all keywords/phrases for structures already correctly depicted
2. EXPLICITLY add every missing structure by name — in BOTH prompts
3. CLIP: insert missing structure names as keywords near the front
4. T5: add one concise sentence per missing structure
5. Do NOT rewrite working parts — surgical additions only
6. Keep style directives (scientific diagram, white background, no text, no labels)

CRITICAL RULES:
- CLIP output MUST be ≤{CLIP_MAX_WORDS} words — cut low-priority style words if needed
- T5 output MUST be ≤{T5_MAX_WORDS} words — be concise, no padding
- Every item in "missing_structures" MUST appear by name in BOTH prompts
- Every item in "violations" MUST be addressed with emphasis
- Items in "present_structures" → do NOT remove or alter

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "refined_clip_prompt": "<≤{CLIP_MAX_WORDS} word keyword prompt>",
  "refined_t5_prompt": "<≤{T5_MAX_WORDS} word natural language prompt>",
  "changes_made": ["<change 1>", "<change 2>"]
}}
"""

_USER_TEMPLATE = """\
Current CLIP prompt ({clip_words} words):
{clip_prompt}

Current T5 prompt ({t5_words} words — truncated to {T5_MAX_WORDS} for context):
{t5_prompt_truncated}

Already correctly depicted (DO NOT remove):
{present_structures}

MISSING — add explicitly by name to BOTH prompts:
{missing_structures}

CLIP violations (add emphasis):
{violations}

Verifier suggestions:
{improvement_suggestions}
"""


class PromptRefinerAgent:
    def __init__(self, api_key: str = None):
        self._client = _wrap_anthropic(anthropic.Anthropic(api_key=api_key))
        print(f"PromptRefinerAgent: {MODEL} ready. "
              f"Limits: CLIP≤{CLIP_MAX_WORDS}w  T5≤{T5_MAX_WORDS}w")

    @traceable(name="prompt_refiner_refine", run_type="llm")
    def refine(
        self,
        clip_prompt: str,
        t5_prompt: str,
        present_structures: list[str] | None = None,
        missing_structures: list[str] | None = None,
        violations: list[str] | None = None,
        improvement_suggestions: str = "",
    ) -> dict:
        """
        Refines CLIP and T5 prompts based on verifier feedback.
        Output is always word-truncated to CLIP_MAX_WORDS / T5_MAX_WORDS.
        Never falls back to original — always returns something refined or truncated.

        Returns:
          refined_clip_prompt  str   (≤CLIP_MAX_WORDS words)
          refined_t5_prompt    str   (≤T5_MAX_WORDS words)
          changes_made         list[str]
        """
        present_structures = present_structures or []
        missing_structures = missing_structures or []
        violations         = violations         or []

        # Always truncate inputs to their limits before sending
        clip_in = _truncate_words(clip_prompt, CLIP_MAX_WORDS)
        t5_in   = _truncate_words(t5_prompt,   T5_MAX_WORDS)

        # Nothing to fix — return truncated originals
        if not missing_structures and not violations and not improvement_suggestions:
            return {
                "refined_clip_prompt": clip_in,
                "refined_t5_prompt":   t5_in,
                "changes_made":        ["no changes needed"],
            }

        def _fmt(lst): return "\n".join(f"  - {x}" for x in lst) if lst else "  (none)"

        user_msg = _USER_TEMPLATE.format(
            clip_words          = len(clip_in.split()),
            clip_prompt         = clip_in,
            t5_words            = len(t5_prompt.split()),
            T5_MAX_WORDS        = T5_MAX_WORDS,
            t5_prompt_truncated = t5_in,
            present_structures  = _fmt(present_structures),
            missing_structures  = _fmt(missing_structures),
            violations          = _fmt(violations),
            improvement_suggestions = improvement_suggestions or "(none)",
        )

        refined_clip = clip_in
        refined_t5   = t5_in
        changes      = []

        try:
            resp = self._client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw  = resp.content[0].text.strip()
            raw  = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            refined_clip = data.get("refined_clip_prompt", clip_in).strip()
            refined_t5   = data.get("refined_t5_prompt",   t5_in).strip()
            changes      = data.get("changes_made", [])
        except json.JSONDecodeError as e:
            changes = [f"JSON parse error ({e}) — using truncated original prompts"]
        except Exception as e:
            changes = [f"refiner error ({e}) — using truncated original prompts"]

        # Always enforce word limits on output regardless of LLM compliance
        refined_clip = _truncate_words(refined_clip, CLIP_MAX_WORDS)
        refined_t5   = _truncate_words(refined_t5,   T5_MAX_WORDS)

        return {
            "refined_clip_prompt": refined_clip,
            "refined_t5_prompt":   refined_t5,
            "changes_made":        changes,
        }
