import torch
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)

GGUF_REPO = "city96/FLUX.1-dev-gguf"
GGUF_FILE = "flux1-dev-Q5_K_S.gguf"
BASE_REPO = "black-forest-labs/FLUX.1-schnell"  # 使用 Schnell 的 Text Encoders

def _get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _make_generator(seed: int, device: str) -> torch.Generator:
    # MPS doesn't support seeded Generator — use CPU generator which diffusers handles correctly
    gen_device = "cpu" if device == "mps" else device
    return torch.Generator(device=gen_device).manual_seed(seed)

class FluxGGUFPipeline:
    RECOMMENDED_STEPS = 20

    def __init__(self):
        self.device = _get_device()
        self.dtype = torch.bfloat16

        print(f"Loading FLUX.1-dev GGUF on {self.device} (dtype={self.dtype})...")
        
        # 1. 下載 GGUF 權重文件
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)

        # 2. 強制下載 config.json 到本地，解決路徑解析問題
        # 這樣做可以確保我們拿到正確的本地路徑，避開 404 錯誤
        print("  Downloading architecture config...")
        config_path = hf_hub_download(
            repo_id=BASE_REPO, 
            subfolder="transformer", 
            filename="config.json"
        )
        # 獲取 config.json 所在的資料夾路徑 (diffusers 需要的是資料夾路徑字串)
        config_dir = str(Path(config_path).parent)

        # 3. 準備量化設定
        gguf_config = GGUFQuantizationConfig(compute_dtype=self.dtype)

        # 4. 初始化 Transformer
        # 關鍵：config 傳入的是資料夾路徑字串，這能滿足它的驗證邏輯
        print("  Initializing transformer from GGUF...")
        transformer = FluxTransformer2DModel.from_single_file(
            gguf_path,
            config=config_dir, 
            quantization_config=gguf_config,
            torch_dtype=self.dtype
        )

        # 5. 加載完整的 Pipeline
        self.pipe = FluxPipeline.from_pretrained(
            BASE_REPO,
            transformer=transformer,
            torch_dtype=self.dtype,
        )

        try:
            self.pipe.enable_model_cpu_offload()
            print("  CPU offload enabled.")
        except Exception as e:
            print(f"  CPU offload unavailable ({e}), loading to {self.device}")
            self.pipe = self.pipe.to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        # Build img2img pipeline sharing all weights — no extra memory cost.
        # Construct directly from components to avoid from_pipe() dtype-cast
        # error on GGUF quantized transformer.
        self.img2img_pipe = FluxImg2ImgPipeline(
            scheduler=self.pipe.scheduler,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            text_encoder_2=self.pipe.text_encoder_2,
            tokenizer_2=self.pipe.tokenizer_2,
            vae=self.pipe.vae,
            transformer=self.pipe.transformer,
        )
        self.img2img_pipe.set_progress_bar_config(disable=True)

        print("FLUX.1-dev GGUF ready.")

    def generate(
        self,
        prompt: str,
        prompt_2: str = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        """
        prompt   → CLIP encoder (≤77 tokens, visual keywords)
        prompt_2 → T5 encoder (full detail); falls back to prompt if None
        """
        generator = _make_generator(seed, self.device)
        result = self.pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
        )
        return result.images[0]

    def refine(
        self,
        image: Image.Image,
        prompt: str,
        prompt_2: str = None,
        strength: float = 0.55,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = 42,
    ) -> Image.Image:
        """
        img2img refinement with dual-encoder prompts.
        prompt   → CLIP (≤77 tokens)
        prompt_2 → T5 (full detail)
        strength=0.5–0.6: preserve structure, correct violated details.
        """
        generator = _make_generator(seed, self.device)
        result = self.img2img_pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]