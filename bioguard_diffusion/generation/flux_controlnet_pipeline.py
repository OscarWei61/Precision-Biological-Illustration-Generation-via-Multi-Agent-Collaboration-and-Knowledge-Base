"""
FLUX.1-dev ControlNet Pipeline (GGUF transformer + Canny ControlNet)

Loads:
  Transformer  : city96/FLUX.1-dev-gguf  Q5_K_S.gguf  (~7 GB, same as v3)
  ControlNet   : InstantX/FLUX.1-dev-Controlnet-Canny
  Encoders/VAE : black-forest-labs/FLUX.1-schnell  (same weights as v3)

Mirrors the exact loading pattern from flux_gguf_pipeline.py so both
pipelines can be run on the same machine without re-downloading.

When no spatial data is available (control_image is all-black + scale=0.0),
the ControlNet has zero effect and the pipeline degrades to vanilla FLUX.
"""

import torch
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import (
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)

try:
    from diffusers import FluxControlNetImg2ImgPipeline
    _HAS_IMG2IMG_CONTROLNET = True
except ImportError:
    _HAS_IMG2IMG_CONTROLNET = False

GGUF_REPO      = "city96/FLUX.1-dev-gguf"
GGUF_FILE      = "flux1-dev-Q5_K_S.gguf"
CONTROLNET_REPO = "InstantX/FLUX.1-dev-Controlnet-Canny"
BASE_REPO      = "black-forest-labs/FLUX.1-schnell"


def _get_device() -> str:
    if torch.cuda.is_available():             return "cuda"
    if torch.backends.mps.is_available():     return "mps"
    return "cpu"


def _make_generator(seed: int, device: str) -> torch.Generator:
    gen_device = "cpu" if device == "mps" else device
    return torch.Generator(device=gen_device).manual_seed(seed)


class FluxControlNetPipelineWrapper:
    """
    FLUX.1-dev + ControlNet Canny.

    generate()  text-to-image  with ControlNet edge guidance
    refine()    image-to-image with ControlNet edge guidance
                Falls back to FluxControlNetPipeline with strength param if
                FluxControlNetImg2ImgPipeline is unavailable in this diffusers build.
    """

    def __init__(self):
        self.device = _get_device()
        self.dtype  = torch.bfloat16
        print(f"[FluxControlNetPipeline] device={self.device}")

        # ── Transformer (GGUF, same as v3) ───────────────────────────────────
        print("  Downloading GGUF transformer...")
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)

        print("  Downloading architecture config...")
        config_path = hf_hub_download(
            repo_id=BASE_REPO,
            subfolder="transformer",
            filename="config.json",
        )
        config_dir = str(Path(config_path).parent)

        print("  Initialising GGUF transformer...")
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            config=config_dir,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            torch_dtype=self.dtype,
        )

        # ── ControlNet ────────────────────────────────────────────────────────
        print(f"  Loading ControlNet ({CONTROLNET_REPO})...")
        controlnet = FluxControlNetModel.from_pretrained(
            CONTROLNET_REPO,
            torch_dtype=self.dtype,
        )

        # ── Assemble text2img pipeline ────────────────────────────────────────
        print("  Assembling FluxControlNetPipeline...")
        self.pipe = FluxControlNetPipeline.from_pretrained(
            BASE_REPO,
            transformer=transformer,
            controlnet=controlnet,
            torch_dtype=self.dtype,
        )
        try:
            self.pipe.enable_model_cpu_offload()
            print("  CPU offload enabled (text2img).")
        except Exception as e:
            print(f"  CPU offload unavailable ({e}) — loading to {self.device}")
            self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        # ── Assemble img2img ControlNet pipeline ──────────────────────────────
        if _HAS_IMG2IMG_CONTROLNET:
            self.img2img_pipe = FluxControlNetImg2ImgPipeline(
                scheduler=self.pipe.scheduler,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer_2=self.pipe.tokenizer_2,
                vae=self.pipe.vae,
                transformer=self.pipe.transformer,
                controlnet=self.pipe.controlnet,
            )
            try:
                self.img2img_pipe.enable_model_cpu_offload()
            except Exception:
                pass
            self.img2img_pipe.set_progress_bar_config(disable=True)
            print("  FluxControlNetImg2ImgPipeline ready.")
        else:
            self.img2img_pipe = None
            print("  FluxControlNetImg2ImgPipeline not available — "
                  "refine() will re-use text2img pipe with conditioning only.")

        print("FluxControlNetPipeline ready.")

    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        prompt_2: str,
        control_image: "Image.Image",
        controlnet_conditioning_scale: float = 0.7,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
    ) -> "Image.Image":
        generator = _make_generator(seed, self.device)
        result = self.pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
        )
        return result.images[0]

    def refine(
        self,
        image: "Image.Image",
        prompt: str,
        prompt_2: str,
        control_image: "Image.Image",
        strength: float = 0.55,
        controlnet_conditioning_scale: float = 0.5,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
    ) -> "Image.Image":
        generator = _make_generator(seed, self.device)

        if self.img2img_pipe is not None:
            result = self.img2img_pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                image=image,
                control_image=control_image,
                strength=strength,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            # Fallback: ControlNet text2img only (ignores input image, but
            # ControlNet spatial guidance still applies)
            result = self.pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                control_image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=image.width,
                height=image.height,
            )

        return result.images[0]
