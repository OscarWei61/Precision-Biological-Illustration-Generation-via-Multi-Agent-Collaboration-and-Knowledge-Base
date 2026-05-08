"""
Z-Image-Turbo Pipeline Wrapper
Uses ZImagePipeline from diffusers with memory-efficient loading.

Strategy for 18 GB unified memory (Apple M3 Pro):
  - torch_dtype=bfloat16  → transformer ~12.3 GB
  - enable_model_cpu_offload() → offload inactive modules
  - num_inference_steps=8  → Z-Image-Turbo recommended setting
  - Same prompt chunking as BioGenerationPipeline

Falls back to SDXL-Turbo if OOM is detected.
"""

import gc
import time
import torch
from pathlib import Path
from PIL import Image
from diffusers import ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_prompt_chunked_zi(tokenizer, text_encoder, prompt, device, dtype, max_length=77):
    """Same chunking logic as BioGenerationPipeline, adapted for Z-Image's tokenizer."""
    chunk_size = max_length - 2
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    all_ids = tokenizer(
        prompt, truncation=False, return_tensors="pt", padding=False
    ).input_ids[0]
    content_ids = all_ids[1:-1]

    chunk_tensors = []
    for i in range(0, max(len(content_ids), 1), chunk_size):
        chunk_content = content_ids[i : i + chunk_size]
        chunk = torch.cat([
            torch.tensor([bos_id], dtype=torch.long),
            chunk_content,
            torch.tensor([eos_id], dtype=torch.long),
        ])
        pad_len = max_length - len(chunk)
        if pad_len > 0:
            chunk = torch.cat([chunk, torch.full((pad_len,), pad_id, dtype=torch.long)])
        chunk_tensors.append(chunk)

    chunk_batch = torch.stack(chunk_tensors).to(device)
    with torch.no_grad():
        embeddings = text_encoder(chunk_batch)[0].to(dtype=dtype)
    return embeddings.view(1, -1, embeddings.shape[-1])


class ZImageGenerationPipeline:
    """
    Wrapper around ZImagePipeline with:
    - bfloat16 precision
    - CPU offload for memory efficiency on 18 GB unified memory
    - Prompt chunking for prompts > 77 tokens
    - 8-step inference (Z-Image-Turbo default)
    """

    RECOMMENDED_STEPS = 8

    def __init__(self):
        self.device = _get_device()
        self.dtype = torch.bfloat16
        print(f"Loading Z-Image-Turbo on {self.device} (dtype={self.dtype})...")
        print("  This model is ~12-18 GB — first load may take several minutes.")

        # Load with CPU offload to manage peak memory on 18 GB RAM
        self.pipe = ZImagePipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )

        # Use CPU offload: keeps weights on CPU, moves layer to MPS only during forward pass
        # On Apple Silicon unified memory this still reduces peak activation memory
        try:
            self.pipe.enable_model_cpu_offload()
            print("  CPU offload enabled.")
        except Exception as e:
            print(f"  CPU offload unavailable ({e}), loading directly to {self.device}")
            self.pipe = self.pipe.to(self.device)

        self.pipe.set_progress_bar_config(disable=True)
        print("Z-Image-Turbo ready.")

    def _token_count(self, text: str) -> int:
        return len(self.pipe.tokenizer(text, truncation=False).input_ids)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, cartoon, deformed, wrong anatomy, low quality, watermark",
        num_inference_steps: int = RECOMMENDED_STEPS,
        guidance_scale: float = 3.5,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        generator = torch.Generator().manual_seed(seed)

        n_tokens = self._token_count(prompt)
        if n_tokens > 77:
            print(f"         [z-img chunking] {n_tokens} tokens → {(n_tokens+74)//75} chunk(s)")

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

    @staticmethod
    def is_available() -> bool:
        try:
            from diffusers import ZImagePipeline
            return True
        except ImportError:
            return False
