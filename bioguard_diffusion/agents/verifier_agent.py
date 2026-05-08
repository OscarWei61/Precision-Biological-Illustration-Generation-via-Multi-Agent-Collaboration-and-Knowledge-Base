"""
Verifier Agent — cross-checks generated images against biological constraints.

Two-stage verification:
  Stage 1 — CLIP-SAS (fast, local):
    1. Extract key biological terms from each constraint sentence.
    2. Score image vs. keyword query using CLIP ViT-B/32.
    3. SAS = mean score across all constraints.
    A constraint is "passed" if its score >= PASS_THRESHOLD (0.22).

  Stage 2 — LLM-as-Judge (semantic, vision-capable):
    1. Encode image as base64.
    2. Send image + query + constraints to claude-haiku-4-5 (vision).
    3. LLM returns structured JSON: score/10, present/missing structures,
       improvement_suggestions for next img2img attempt.
    4. Passes if llm_score >= LLM_PASS_THRESHOLD (6.0).

Combined pass = CLIP passed AND LLM passed.

Why CLIP ViT-B/32 over BioCLIP:
  BioCLIP is trained on Tree-of-Life taxonomy labels and handles single-word
  species names well, but its embedding space does not align with multi-word
  structural constraint sentences.  CLIP ViT-B/32 provides better text-image
  alignment for descriptive anatomical phrases.

  Empirical calibration on AI2D mitochondria image (3288.png):
    Correct keywords  (cristae / inner membrane): 0.268 – 0.317
    Wrong keywords    (heart / neuron / flower):  0.147 – 0.224
    Chosen threshold: 0.22
"""

import re
import json
import base64
import torch
import open_clip
import anthropic
from io import BytesIO
from PIL import Image

try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_anthropic as _wrap_anthropic
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator
    def _wrap_anthropic(client): return client

PASS_THRESHOLD     = 0.22
LLM_PASS_THRESHOLD = 6.0   # out of 10
LLM_MODEL          = "claude-haiku-4-5-20251001"

_LLM_SYSTEM = """\
You are a biology education expert evaluating AI-generated scientific diagrams.
Assess whether the image accurately depicts the requested biological subject with
the required anatomical structures. The image should NOT contain any text, labels, or annotations.

Respond ONLY with valid JSON matching this exact schema (no markdown, no explanation):
{
  "score": <float 0-10>,
  "passed": <true|false>,
  "present_structures": [<string>, ...],
  "missing_structures": [<string>, ...],
  "improvement_suggestions": "<one or two sentences of specific prompt additions>"
}

Scoring guide:
  9-10: All structures clearly present and visually distinct
  7-8:  Most structures present, minor inaccuracies
  5-6:  Core subject recognisable, several structures missing
  3-4:  Subject recognisable but major structures absent
  0-2:  Wrong subject or unrecognisable as scientific diagram
"""

# Common stop-words to strip when building the keyword query
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "into",
    "from", "as", "or", "and", "but", "not", "only", "also", "its",
    "it", "this", "that", "these", "those", "which", "where", "when",
    "how", "what", "who", "whose", "called", "known", "typically",
    "often", "usually", "located", "found", "made", "used", "called",
    "during", "through", "between", "within", "along", "up", "down",
    "each", "all", "both", "per", "if", "such", "so",
}


def _extract_keywords(constraint: str, max_words: int = 6) -> str:
    """Return a compact keyword string from a constraint sentence."""
    words = re.sub(r"[^a-zA-Z0-9\-/]", " ", constraint).split()
    keywords = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]
    return " ".join(keywords[:max_words])


def _encode_image_b64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


class VerifierAgent:
    def __init__(self, api_key: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(self.device).eval()
        self.model_name = "CLIP-ViT-B/32 (keyword extraction)"
        self._llm = _wrap_anthropic(anthropic.Anthropic(api_key=api_key))
        print(f"Verifier: {self.model_name} + LLM-judge ({LLM_MODEL}) loaded, "
              f"thresholds CLIP={PASS_THRESHOLD} LLM={LLM_PASS_THRESHOLD}")

    @torch.no_grad()
    def _img_feat(self, image: Image.Image) -> torch.Tensor:
        t = self.preprocess(image).unsqueeze(0).to(self.device)
        f = self.model.encode_image(t)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _score(self, img_feat: torch.Tensor, text: str) -> float:
        tokens = self.tokenizer([text]).to(self.device)
        f = self.model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        return float((img_feat @ f.T).squeeze())

    @traceable(name="verifier_clip_sas", run_type="chain")
    def verify(
        self,
        image: Image.Image,
        constraints: list[str],
        entity: str = "",
    ) -> dict:
        """
        Returns:
          passed        bool   — True if SAS >= PASS_THRESHOLD
          sas           float  — Scientific Accuracy Score (mean score, 0–1)
          per_constraint list  — (constraint, keyword_query, score, passed) tuples
          violations    list   — constraint sentences that did NOT pass
          feedback      str    — prompt hint for retry
        """
        if not constraints:
            return {
                "passed": True, "sas": 1.0,
                "per_constraint": [], "violations": [], "feedback": "",
            }

        img_feat = self._img_feat(image)
        per = []
        for c in constraints:
            kw    = _extract_keywords(c)
            score = self._score(img_feat, kw)
            ok    = score >= PASS_THRESHOLD
            per.append((c, kw, round(score, 4), ok))

        sas        = sum(s for _, _, s, _ in per) / len(per)
        violations = [c for c, _, _, ok in per if not ok]
        passed     = sas >= PASS_THRESHOLD

        if violations:
            feedback = (
                "The previous image was missing or incorrectly depicted: "
                + "; ".join(violations[:3])
                + ". Please emphasise these anatomical details in the prompt."
            )
        else:
            feedback = ""

        return {
            "passed": passed,
            "sas": round(sas, 4),
            "per_constraint": [(c, kw, s, ok) for c, kw, s, ok in per],
            "violations": violations,
            "feedback": feedback,
        }

    @traceable(name="verifier_llm_judge", run_type="llm")
    def verify_with_llm(
        self,
        image: Image.Image,
        query: str,
        constraints: list[str],
    ) -> dict:
        """
        LLM-as-judge: sends image + query + constraints to claude-haiku-4-5 vision.

        Returns:
          passed                bool
          llm_score             float  0–10
          present_structures    list[str]
          missing_structures    list[str]
          improvement_suggestions  str  — actionable text for next img2img prompt
        """
        constraint_block = "\n".join(f"  - {c}" for c in constraints) if constraints else "  (none)"
        user_text = (
            f"Query: {query}\n\n"
            f"Required biological structures/constraints:\n{constraint_block}\n\n"
            "Evaluate the image."
        )

        try:
            resp = self._llm.messages.create(
                model=LLM_MODEL,
                max_tokens=512,
                system=_LLM_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _encode_image_b64(image),
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }],
            )
            raw = resp.content[0].text.strip()
            # Strip accidental markdown fences
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            return {
                "passed": False,
                "llm_score": 0.0,
                "present_structures": [],
                "missing_structures": [],
                "improvement_suggestions": f"LLM judge error: {e}",
                "raw_response": str(e),
            }

        score   = float(data.get("score", 0))
        passed  = data.get("passed", score >= LLM_PASS_THRESHOLD)
        return {
            "passed":                  passed,
            "llm_score":               round(score, 2),
            "present_structures":      data.get("present_structures", []),
            "missing_structures":      data.get("missing_structures", []),
            "improvement_suggestions": data.get("improvement_suggestions", ""),
        }

    @traceable(name="verifier_combined", run_type="chain")
    def verify_combined(
        self,
        image: Image.Image,
        query: str,
        constraints: list[str],
        entity: str = "",
    ) -> dict:
        """
        Runs both CLIP-SAS and LLM-as-judge.
        Overall pass = clip_passed AND llm_passed.

        Returns merged dict with all sub-results plus:
          passed_combined  bool
          feedback         str  — combined feedback for next img2img prompt
        """
        clip_result = self.verify(image, constraints, entity=entity)
        llm_result  = self.verify_with_llm(image, query, constraints)

        passed_combined = clip_result["passed"] and llm_result["passed"]

        # Build unified feedback string for img2img prompt injection
        parts = []
        if clip_result["violations"]:
            parts.append(
                "CLIP violations: " + "; ".join(clip_result["violations"][:3])
            )
        if llm_result["improvement_suggestions"]:
            parts.append(llm_result["improvement_suggestions"])
        feedback = " | ".join(parts) if parts else ""

        return {
            # CLIP
            "sas":              clip_result["sas"],
            "clip_passed":      clip_result["passed"],
            "per_constraint":   clip_result["per_constraint"],
            "violations":       clip_result["violations"],
            # LLM
            "llm_score":        llm_result["llm_score"],
            "llm_passed":       llm_result["passed"],
            "present_structures":       llm_result["present_structures"],
            "missing_structures":       llm_result["missing_structures"],
            "improvement_suggestions":  llm_result["improvement_suggestions"],
            # Combined
            "passed":           passed_combined,
            "passed_combined":  passed_combined,
            "feedback":         feedback,
        }
