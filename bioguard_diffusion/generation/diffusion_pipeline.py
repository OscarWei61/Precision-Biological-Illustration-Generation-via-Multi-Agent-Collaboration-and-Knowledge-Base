"""
Generation Layer — Stable Diffusion pipeline wrapper.
Supports MPS (Apple Silicon), CUDA, and CPU fallback.

Prompt Chunking:
  SD-v1.5's CLIP text encoder has a hard limit of 77 tokens (75 content +
  BOS + EOS). Prompts longer than 75 tokens are normally silently truncated.

  This module implements chunking to preserve the full prompt:
    1. Tokenise without truncation to get all token IDs.
    2. Split content tokens into chunks of 75.
    3. Prepend BOS / append EOS / pad each chunk to length 77.
    4. Encode each chunk independently through the CLIP text encoder.
    5. Concatenate along the sequence dimension → [1, n_chunks×77, 768].
    6. Pass the combined tensor to the U-Net via `prompt_embeds`.

  Example: a 150-token prompt → 2 chunks → 154×768 tensor to U-Net
  (2 chunks × 77 tokens = 154 positions, each 768-dim).
"""

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_prompt_chunked(
    tokenizer,
    text_encoder,
    prompt: str,
    device: str,
    dtype: torch.dtype,
    max_length: int = 77,
) -> torch.Tensor:
    """
    Encode a prompt of arbitrary length via chunking.
    Returns shape: [1, n_chunks * max_length, hidden_dim]
    """
    chunk_size = max_length - 2  # 75 content tokens per chunk

    bos_id = tokenizer.bos_token_id                             # 49406
    eos_id = tokenizer.eos_token_id                             # 49407
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # Tokenise without truncation — includes BOS at [0] and EOS at [-1]
    all_ids = tokenizer(
        prompt,
        truncation=False,
        return_tensors="pt",
        padding=False,
    ).input_ids[0]

    content_ids = all_ids[1:-1]  # strip the auto-added BOS / EOS

    # Build chunks
    chunk_tensors = []
    n_content = len(content_ids)

    # Always produce at least one chunk (handles empty prompt too)
    for i in range(0, max(n_content, 1), chunk_size):
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

    # Stack → [n_chunks, 77]
    chunk_batch = torch.stack(chunk_tensors).to(device)

    # Encode each chunk through CLIP
    with torch.no_grad():
        embeddings = text_encoder(chunk_batch)[0]  # [n_chunks, 77, 768]

    embeddings = embeddings.to(dtype=dtype)

    # Flatten chunks into one long sequence → [1, n_chunks*77, 768]
    return embeddings.view(1, -1, embeddings.shape[-1])


class BioGenerationPipeline:
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    CHUNK_THRESHOLD = 77  # token count above which chunking activates

    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or self.MODEL_ID
        self.device = device or get_device()
        self.dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        print(f"Loading SD model on {self.device}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD model ready.")

    def _token_count(self, text: str) -> int:
        return len(self.pipe.tokenizer(text, truncation=False).input_ids)

    def _get_embeddings(self, prompt: str, negative_prompt: str):
        """
        Return (prompt_embeds, negative_prompt_embeds).
        Uses chunking for any text whose token count exceeds CHUNK_THRESHOLD.
        """
        pos_count = self._token_count(prompt)
        neg_count = self._token_count(negative_prompt)
        needs_chunking = (pos_count > self.CHUNK_THRESHOLD) or (neg_count > self.CHUNK_THRESHOLD)

        if not needs_chunking:
            return None, None  # let diffusers handle short prompts normally

        pos_embeds = _encode_prompt_chunked(
            self.pipe.tokenizer,
            self.pipe.text_encoder,
            prompt,
            self.device,
            self.dtype,
        )
        neg_embeds = _encode_prompt_chunked(
            self.pipe.tokenizer,
            self.pipe.text_encoder,
            negative_prompt,
            self.device,
            self.dtype,
        )

        # Pad shorter sequence to match the longer one so U-Net sees equal lengths
        max_seq = max(pos_embeds.shape[1], neg_embeds.shape[1])
        hidden = pos_embeds.shape[2]

        def _pad(t: torch.Tensor) -> torch.Tensor:
            gap = max_seq - t.shape[1]
            if gap == 0:
                return t
            return torch.cat([t, torch.zeros(1, gap, hidden, dtype=self.dtype, device=self.device)], dim=1)

        return _pad(pos_embeds), _pad(neg_embeds)

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

        prompt_embeds, negative_prompt_embeds = self._get_embeddings(prompt, negative_prompt)

        if prompt_embeds is not None:
            # Long prompt — use pre-computed chunked embeddings
            pos_tok = self._token_count(prompt)
            n_chunks = (pos_tok + 74) // 75  # ceil division
            print(f"         [chunking] {pos_tok} tokens → {n_chunks} chunk(s), embed shape {tuple(prompt_embeds.shape)}")
            result = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=width,
                height=height,
            )
        else:
            # Short prompt — standard path, no chunking overhead
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
