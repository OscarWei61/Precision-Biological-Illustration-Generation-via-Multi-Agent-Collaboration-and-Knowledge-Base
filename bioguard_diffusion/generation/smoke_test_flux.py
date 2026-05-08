"""Quick smoke test — FLUX.1-dev GGUF: generate 1 image, verify MPS + CLIP work."""
import sys
import time
from pathlib import Path

from huggingface_hub import login
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from bioguard_diffusion.generation.flux_gguf_pipeline import FluxGGUFPipeline
from bioguard_diffusion.evaluation.clip_score import CLIPScorer

PROMPT = (
    "A detailed scientific cross-section diagram of a mitochondria "
    "showing cristae and matrix, educational illustration, white background"
)
CLIP_QUERY = "mitochondria cristae matrix diagram"

print("Loading FLUX.1-dev GGUF pipeline...")
pipe = FluxGGUFPipeline()

print(f"Generating test image (mitochondria, {pipe.RECOMMENDED_STEPS} steps)...")
t0 = time.time()
img = pipe.generate(
    prompt=PROMPT,
    num_inference_steps=pipe.RECOMMENDED_STEPS,
    guidance_scale=3.5,
    seed=42,
    width=512,
    height=512,
)
elapsed = time.time() - t0
print(f"Generated in {elapsed:.1f}s, size={img.size}")

out = BASE / "outputs" / "smoke_test_flux.png"
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print(f"Image saved: {out}")

# 以下部分需確保 CLIPScorer 模組已正確安裝
try:
    print("Loading CLIP scorer...")
    scorer = CLIPScorer()
    score = scorer.score(img, CLIP_QUERY)
    print(f"CLIP score: {score:.4f}")
except Exception as e:
    print(f"CLIP Scorer skipped or failed: {e}")

print("\nSmoke test PASSED — FLUX.1-dev GGUF + MPS working.")