"""Quick smoke test — generate 1 image to verify SD + MPS + CLIP work."""
import sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from bioguard_diffusion.generation.diffusion_pipeline import BioGenerationPipeline
from bioguard_diffusion.evaluation.clip_score import CLIPScorer

print("Loading SD pipeline...")
pipe = BioGenerationPipeline()

print("Generating test image (mitochondria)...")
t0 = time.time()
img = pipe.generate(
    prompt="A detailed scientific cross-section diagram of a mitochondria showing cristae and matrix, educational illustration, white background",
    num_inference_steps=20,
    seed=42,
)
elapsed = time.time() - t0
print(f"Generated in {elapsed:.1f}s, size={img.size}")

out = BASE / "outputs" / "smoke_test.png"
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print(f"Image saved: {out}")

print("Loading CLIP scorer...")
scorer = CLIPScorer()
score = scorer.score(img, "mitochondria cristae matrix diagram")
print(f"CLIP score: {score:.4f}")

print("\nSmoke test PASSED — SD + MPS + CLIP all working.")
