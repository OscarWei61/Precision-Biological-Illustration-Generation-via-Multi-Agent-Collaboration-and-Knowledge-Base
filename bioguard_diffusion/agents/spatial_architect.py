"""
Spatial Architect Agent — converts biological constraints into a
structured Stable Diffusion prompt.

Uses a template-based approach (no LLM API key required).
The prompt is assembled in layers:
  1. Subject declaration  (what to draw)
  2. Structural details   (from RAG constraints)
  3. Style/quality suffix (educational diagram aesthetics)
"""

import re

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

# SD ViT-B/32 tokenizer limit is 77 tokens; we keep prompts under ~200 chars
# to stay safely within that bound after tokenization.
MAX_CONSTRAINTS = 5
STYLE_SUFFIX = (
    "scientific educational illustration, detailed diagram, "
    "white background, accurate biology textbook style, no text, no labels"
)
NEGATIVE_PROMPT = (
    "blurry, cartoon, deformed, extra limbs, wrong anatomy, "
    "missing structures, low quality, watermark, text, labels, annotations, "
    "words, letters, numbers, abstract art"
)

# Maps common query verbs/phrases to a cleaner subject form
_STRIP_PREFIXES = re.compile(
    r"^(draw|create|generate|make|show|illustrate)\s+(a\s+)?"
    r"(detailed\s+)?(diagram\s+of\s+|cross[- ]section\s+of\s+|"
    r"illustration\s+of\s+)?",
    re.IGNORECASE,
)


def _extract_subject(query: str) -> str:
    """Strip instruction verbs and return the biological subject."""
    subject = _STRIP_PREFIXES.sub("", query.strip())
    # Also trim trailing clauses like "showing X, Y and Z"
    subject = re.split(r"\s+showing\b|\s+with\b|\s+including\b", subject)[0]
    return subject.strip().rstrip(".,")


def _select_constraints(constraints: list[str], subject: str) -> list[str]:
    """
    Pick the most relevant constraints — prioritise ones that mention
    the subject keywords, then fill up to MAX_CONSTRAINTS.
    """
    subject_words = set(subject.lower().split())
    scored = []
    for c in constraints:
        hits = sum(1 for w in subject_words if w in c.lower())
        scored.append((hits, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:MAX_CONSTRAINTS]]


class SpatialArchitectAgent:
    @traceable(name="spatial_architect_build_prompt", run_type="chain")
    def build_prompt(
        self,
        query: str,
        constraints: list[str],
        violations: list[str] | None = None,
    ) -> dict:
        """
        Returns a dict with:
          - 'prompt'          : the positive SD prompt
          - 'negative_prompt' : the negative SD prompt
          - 'subject'         : extracted biological subject
          - 'constraints_used': which constraints were injected
        """
        subject = _extract_subject(query)

        # On retry: put violated constraints first so they survive token-limit truncation
        if violations:
            priority = violations[:MAX_CONSTRAINTS]
            rest = [c for c in constraints if c not in priority]
            reordered = priority + rest
        else:
            reordered = constraints
        selected = _select_constraints(reordered, subject)

        # Build the structural detail string from constraints
        if selected:
            # Convert each constraint into a compact phrase
            detail_phrases = []
            for c in selected:
                # Remove "must", "has", "is" preambles for tighter phrasing
                phrase = re.sub(
                    r"^(a |an |the )?([\w\s]+?)\s+(must have|has|have|contains?|is)\s+",
                    r"\2 with ",
                    c,
                    flags=re.IGNORECASE,
                ).strip()
                detail_phrases.append(phrase)
            details = ", ".join(detail_phrases)
            prompt = (
                f"A highly detailed scientific cross-section diagram of {subject}, "
                f"{details}, {STYLE_SUFFIX}"
            )
        else:
            prompt = (
                f"A highly detailed scientific diagram of {subject}, "
                f"{STYLE_SUFFIX}"
            )

        return {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "subject": subject,
            "constraints_used": selected,
        }
