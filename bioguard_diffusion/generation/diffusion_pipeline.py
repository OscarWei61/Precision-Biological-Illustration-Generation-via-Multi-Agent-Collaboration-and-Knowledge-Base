"""
Generation Layer — Stable Diffusion pipeline wrapper.
Supports MPS (Apple Silicon), CUDA, and CPU fallback.
"""

import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BioGenerationPipeline:
    MODEL_ID = "runwayml/stable-diffusion-v1-5"

    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or self.MODEL_ID
        self.device = device or get_device()
        print(f"Loading SD model on {self.device}...")

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD model ready.")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, cartoon, deformed, ugly, low quality, watermark",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
        )
        return result.images[0]
