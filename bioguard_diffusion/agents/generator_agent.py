"""
Generator Agent — wraps FluxGGUFPipeline + PromptSplitter into a trackable agent.

Each generate() / refine() call:
  1. Injects LLM feedback from the verifier into the full prompt
  2. Splits into CLIP + T5 sub-prompts (tracked)
  3. Calls the diffusion pipeline
  4. Appends a full prompt record to prompt_history

prompt_history is the authoritative log of exactly what went into each generation.
"""

import time
from pathlib import Path
from PIL import Image

try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

IMG2IMG_STRENGTH_BASE = 0.60
IMG2IMG_STRENGTH_DECAY = 0.10
IMG2IMG_STRENGTH_MIN   = 0.30


class GeneratorAgent:
    """
    Single-responsibility: generate images from prompts + feedback.

    Usage:
        agent = GeneratorAgent()

        # attempt 1 — text2img
        result = agent.generate(full_prompt, seed=42, steps=20)
        image  = result["image"]

        # attempt 2+ — img2img, pass verifier feedback
        result = agent.refine(
            image=best_image,
            full_prompt=full_prompt,
            llm_feedback=verifier_suggestions,
            attempt=2,
            seed=42,
            steps=20,
        )

    Attributes:
        prompt_history  list[dict]  — one entry per attempt, contains all prompts
    """

    def __init__(self, api_key: str = None):
        from bioguard_diffusion.generation.flux_gguf_pipeline import FluxGGUFPipeline
        from bioguard_diffusion.agents.prompt_splitter import PromptSplitter

        self.pipe     = FluxGGUFPipeline()
        self.splitter = PromptSplitter(api_key=api_key)
        self.prompt_history: list[dict] = []

    def _split(self, full_prompt: str, llm_feedback: str) -> dict:
        """Merge feedback into prompt, split for CLIP + T5 encoders."""
        combined = (full_prompt + " " + llm_feedback).strip() if llm_feedback else full_prompt
        return self.splitter.split(combined)

    @traceable(name="generator_agent_text2img", run_type="chain")
    def generate(
        self,
        full_prompt: str,
        llm_feedback: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
    ) -> dict:
        """
        Text-to-image (attempt 1).

        Returns dict:
          image         PIL.Image
          mode          "text2img"
          full_prompt   str
          llm_feedback  str   (empty on first attempt)
          clip_prompt   str
          t5_prompt     str
          seed          int
          gen_time_s    float
        """
        split = self._split(full_prompt, llm_feedback)
        clip_prompt = split["clip_prompt"]
        t5_prompt   = split["t5_prompt"]

        print(f"  [GeneratorAgent] text2img  seed={seed}")
        print(f"    [CLIP] {clip_prompt[:70]}...")
        print(f"    [T5]   {t5_prompt[:90]}...")

        t0 = time.time()
        image = self.pipe.generate(
            prompt=clip_prompt,
            prompt_2=t5_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )
        gen_time = round(time.time() - t0, 1)

        record = {
            "mode":         "text2img",
            "full_prompt":  full_prompt,
            "llm_feedback": llm_feedback,
            "clip_prompt":  clip_prompt,
            "t5_prompt":    t5_prompt,
            "seed":         seed,
            "gen_time_s":   gen_time,
        }
        self.prompt_history.append(record)
        return {"image": image, **record}

    @traceable(name="generator_agent_img2img", run_type="chain")
    def refine(
        self,
        image: Image.Image,
        full_prompt: str,
        llm_feedback: str = "",
        attempt: int = 2,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
        clip_prompt_override: str | None = None,
        t5_prompt_override: str | None = None,
        strength: float | None = None,
    ) -> dict:
        """
        Image-to-image refinement (attempt 2+).

        clip_prompt_override / t5_prompt_override: bypass PromptSplitter.
        strength: if None, uses default decay schedule; otherwise caller-controlled
                  (used by adaptive strength + rollback logic in run_single).
        """
        if strength is None:
            strength = max(
                IMG2IMG_STRENGTH_BASE - IMG2IMG_STRENGTH_DECAY * (attempt - 2),
                IMG2IMG_STRENGTH_MIN,
            )

        if clip_prompt_override is not None and t5_prompt_override is not None:
            clip_prompt = clip_prompt_override
            t5_prompt   = t5_prompt_override
        else:
            split = self._split(full_prompt, llm_feedback)
            clip_prompt = split["clip_prompt"]
            t5_prompt   = split["t5_prompt"]

        mode = f"img2img_str{strength:.2f}"
        print(f"  [GeneratorAgent] {mode}  seed={seed}")
        print(f"    [CLIP] {clip_prompt[:70]}...")
        print(f"    [T5]   {t5_prompt[:90]}...")

        t0 = time.time()
        refined = self.pipe.refine(
            image=image,
            prompt=clip_prompt,
            prompt_2=t5_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        gen_time = round(time.time() - t0, 1)

        record = {
            "mode":         mode,
            "full_prompt":  full_prompt,
            "llm_feedback": llm_feedback,
            "clip_prompt":  clip_prompt,
            "t5_prompt":    t5_prompt,
            "strength":     strength,
            "seed":         seed,
            "gen_time_s":   gen_time,
        }
        self.prompt_history.append(record)
        return {"image": refined, **record}

    def write_prompt_log(self, run_dir: Path, query: str, constraints: list[str]) -> None:
        """Write full prompt_history to prompt_log.txt in run_dir."""
        lines = [f"Query: {query}\n", f"Constraints ({len(constraints)}):\n"]
        for c in constraints:
            lines.append(f"  - {c}\n")
        lines.append("\n")

        for i, rec in enumerate(self.prompt_history, 1):
            lines.append(f"--- Attempt {i} ({rec['mode']}) ---\n")
            lines.append(f"Seed:        {rec['seed']}\n")
            if rec.get("llm_feedback"):
                lines.append(f"LLM feedback: {rec['llm_feedback']}\n")
            lines.append(f"Full prompt: {rec['full_prompt']}\n")
            lines.append(f"CLIP prompt: {rec['clip_prompt']}\n")
            lines.append(f"T5 prompt:   {rec['t5_prompt']}\n")
            if "strength" in rec:
                lines.append(f"Strength:    {rec['strength']}\n")
            lines.append(f"Gen time:    {rec['gen_time_s']}s\n\n")

        with open(run_dir / "prompt_log.txt", "w") as f:
            f.writelines(lines)
