"""
Generator Agent V4 — wraps FluxControlNetPipeline with ControlNet support.

Drop-in companion to generator_agent.py for the v4 ControlNet pipeline.
Accepts an optional control_image on every generate() / refine() call.
When control_image is None or has_spatial_data=False, conditioning_scale is
forced to 0.0 so ControlNet has zero effect (degrades to vanilla FLUX).

prompt_history tracks all CLIP + T5 prompts plus ControlNet scale per attempt.
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

# ControlNet conditioning scale schedule
CONTROLNET_SCALE_T2I     = 0.70   # text2img: strong guidance
CONTROLNET_SCALE_I2I_HIGH = 0.50  # img2img, high LLM score: lighter touch
CONTROLNET_SCALE_I2I_LOW  = 0.65  # img2img, low LLM score: keep structure


class GeneratorAgentV4:
    """
    ControlNet-enhanced generator.

    Usage:
        agent = GeneratorAgentV4()

        # attempt 1 — text2img with blueprint
        result = agent.generate(
            full_prompt=full_prompt,
            control_image=blueprint,
            has_spatial_data=True,
            seed=42, steps=20,
        )

        # attempt 2+ — img2img + ControlNet
        result = agent.refine(
            image=best_image,
            full_prompt=full_prompt,
            control_image=updated_blueprint,
            has_spatial_data=True,
            ...
        )
    """

    def __init__(self, api_key: str = None):
        from bioguard_diffusion.generation.flux_controlnet_pipeline import (
            FluxControlNetPipelineWrapper,
        )
        from bioguard_diffusion.agents.prompt_splitter import PromptSplitter

        self.pipe     = FluxControlNetPipelineWrapper()
        self.splitter = PromptSplitter(api_key=api_key)
        self.prompt_history: list[dict] = []

    def _split(self, full_prompt: str, llm_feedback: str) -> dict:
        combined = (full_prompt + " " + llm_feedback).strip() if llm_feedback else full_prompt
        return self.splitter.split(combined)

    @traceable(name="generator_v4_text2img", run_type="chain")
    def generate(
        self,
        full_prompt: str,
        control_image: Image.Image | None = None,
        has_spatial_data: bool = False,
        llm_feedback: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
        controlnet_scale: float | None = None,
    ) -> dict:
        """Text-to-image with ControlNet edge guidance."""
        split = self._split(full_prompt, llm_feedback)
        clip_prompt = split["clip_prompt"]
        t5_prompt   = split["t5_prompt"]

        # Determine effective ControlNet scale
        if not has_spatial_data or control_image is None:
            eff_scale = 0.0
        else:
            eff_scale = controlnet_scale if controlnet_scale is not None else CONTROLNET_SCALE_T2I

        print(f"  [GeneratorV4] text2img  seed={seed}  cn_scale={eff_scale:.2f}")
        print(f"    [CLIP] {clip_prompt[:70]}...")
        print(f"    [T5]   {t5_prompt[:90]}...")

        # Use all-black control image when no spatial data (zero effect)
        ctrl_img = control_image if (has_spatial_data and control_image is not None) \
                   else Image.new("RGB", (width, height), (0, 0, 0))

        t0 = time.time()
        image = self.pipe.generate(
            prompt=clip_prompt,
            prompt_2=t5_prompt,
            control_image=ctrl_img,
            controlnet_conditioning_scale=eff_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )
        gen_time = round(time.time() - t0, 1)

        record = {
            "mode":             "text2img",
            "full_prompt":      full_prompt,
            "llm_feedback":     llm_feedback,
            "clip_prompt":      clip_prompt,
            "t5_prompt":        t5_prompt,
            "seed":             seed,
            "gen_time_s":       gen_time,
            "controlnet_scale": eff_scale,
            "has_spatial_data": has_spatial_data,
        }
        self.prompt_history.append(record)
        return {"image": image, **record}

    @traceable(name="generator_v4_img2img", run_type="chain")
    def refine(
        self,
        image: Image.Image,
        full_prompt: str,
        control_image: Image.Image | None = None,
        has_spatial_data: bool = False,
        llm_feedback: str = "",
        attempt: int = 2,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
        clip_prompt_override: str | None = None,
        t5_prompt_override: str | None = None,
        strength: float | None = None,
        controlnet_scale: float | None = None,
    ) -> dict:
        """Image-to-image refinement with ControlNet edge guidance."""
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

        if not has_spatial_data or control_image is None:
            eff_scale = 0.0
        else:
            eff_scale = controlnet_scale if controlnet_scale is not None else CONTROLNET_SCALE_I2I_HIGH

        mode = f"img2img_cn{eff_scale:.2f}_str{strength:.2f}"
        print(f"  [GeneratorV4] {mode}  seed={seed}")
        print(f"    [CLIP] {clip_prompt[:70]}...")
        print(f"    [T5]   {t5_prompt[:90]}...")

        ctrl_img = control_image if (has_spatial_data and control_image is not None) \
                   else Image.new("RGB", (image.width, image.height), (0, 0, 0))

        t0 = time.time()
        refined = self.pipe.refine(
            image=image,
            prompt=clip_prompt,
            prompt_2=t5_prompt,
            control_image=ctrl_img,
            strength=strength,
            controlnet_conditioning_scale=eff_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        gen_time = round(time.time() - t0, 1)

        record = {
            "mode":             mode,
            "full_prompt":      full_prompt,
            "llm_feedback":     llm_feedback,
            "clip_prompt":      clip_prompt,
            "t5_prompt":        t5_prompt,
            "strength":         strength,
            "seed":             seed,
            "gen_time_s":       gen_time,
            "controlnet_scale": eff_scale,
            "has_spatial_data": has_spatial_data,
        }
        self.prompt_history.append(record)
        return {"image": refined, **record}

    def write_prompt_log(self, run_dir: Path, query: str, constraints: list[str]) -> None:
        lines = [f"Query: {query}\n", f"Constraints ({len(constraints)}):\n"]
        for c in constraints:
            lines.append(f"  - {c}\n")
        lines.append("\n")

        for i, rec in enumerate(self.prompt_history, 1):
            lines.append(f"--- Attempt {i} ({rec['mode']}) ---\n")
            lines.append(f"Seed:              {rec['seed']}\n")
            lines.append(f"ControlNet scale:  {rec['controlnet_scale']:.2f}  "
                         f"(spatial_data={rec['has_spatial_data']})\n")
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
