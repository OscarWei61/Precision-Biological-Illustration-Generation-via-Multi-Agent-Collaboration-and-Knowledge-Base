"""
Stage 2 — Config 1: One-shot Baseline
Generates images for all 10 test queries using the raw user query.
No RAG, no agents.

Run: conda run -n agent python bioguard_diffusion/generation/run_config1_baseline.py
"""

import json
import time
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from bioguard_diffusion.generation.diffusion_pipeline import BioGenerationPipeline
from bioguard_diffusion.evaluation.clip_score import CLIPScorer

QUERIES_FILE = BASE / "results" / "test_queries.json"
OUT_DIR = BASE / "outputs" / "config1"
RESULTS_FILE = BASE / "results" / "config1_results.json"

NEGATIVE_PROMPT = (
    "blurry, cartoon, deformed, ugly, low quality, watermark, text, "
    "mislabeled, incorrect anatomy, extra limbs"
)


def main():
    print("=" * 60)
    print("Stage 2 — Config 1: One-shot Baseline Generation")
    print("=" * 60)

    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading models...")
    pipe = BioGenerationPipeline()
    scorer = CLIPScorer()
    print("Models ready.\n")

    records = []

    for i, q in enumerate(queries, 1):
        qid = q["id"]
        query = q["query"]
        print(f"[{i:02d}/10] {qid} | {query[:60]}...")

        t0 = time.time()
        image = pipe.generate(
            prompt=query,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            seed=42,
        )
        elapsed = time.time() - t0

        # Save image
        img_path = OUT_DIR / f"{qid}.png"
        image.save(img_path)

        # CLIP score
        clip = scorer.score(image, query)

        record = {
            "id": qid,
            "domain": q["domain"],
            "query": query,
            "image_file": str(img_path),
            "clip_score": round(clip, 4),
            "generation_time_s": round(elapsed, 1),
            "config": "config1_baseline",
        }
        records.append(record)
        print(f"         CLIP={clip:.4f}  time={elapsed:.1f}s  saved→ {img_path.name}")

    # Save results JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(records, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("Config 1 Results Summary")
    print(f"{'='*60}")
    print(f"{'ID':<6} {'Domain':<18} {'CLIP':>6}  Query")
    print("-" * 60)
    for r in records:
        print(f"{r['id']:<6} {r['domain']:<18} {r['clip_score']:>6.4f}  {r['query'][:40]}")

    scores = [r["clip_score"] for r in records]
    print(f"\nMean CLIP-Score: {sum(scores)/len(scores):.4f}")
    print(f"Min: {min(scores):.4f}  Max: {max(scores):.4f}")
    print(f"\nResults saved: {RESULTS_FILE}")
    print(f"Images saved:  {OUT_DIR}/")
    print("\nConfig 1 COMPLETE — ready for Stage 3 (Config 2: RAG + SD)")


if __name__ == "__main__":
    main()
