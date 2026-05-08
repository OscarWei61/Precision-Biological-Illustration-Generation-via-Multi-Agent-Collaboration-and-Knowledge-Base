"""
Z-Image-Turbo Experiment: Config 1 Baseline (same 10 queries, same seed)
Run: conda run -n agent python bioguard_diffusion/generation/run_zimage_baseline.py

Generates images using raw user queries (no RAG, no agents) with Z-Image-Turbo.
Results saved alongside SD-v1.5 Config 1 results for direct comparison.
"""

import json
import time
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

QUERIES_FILE = BASE / "results" / "test_queries.json"
C1_FILE      = BASE / "results" / "config1_results.json"   # SD-v1.5 baseline
OUT_DIR      = BASE / "outputs" / "zimage_config1"
RESULTS_FILE = BASE / "results" / "zimage_config1_results.json"

NEGATIVE = (
    "blurry, cartoon, deformed, extra limbs, wrong anatomy, mislabeled, "
    "missing structures, low quality, watermark, text overlay"
)


def main():
    print("=" * 60)
    print("Z-Image-Turbo — Config 1 Baseline Generation")
    print("=" * 60)
    print("Model: Tongyi-MAI/Z-Image-Turbo  |  Steps: 8  |  Seed: 42")

    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    with open(C1_FILE) as f:
        sd_c1 = {r["id"]: r for r in json.load(f)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading Z-Image-Turbo (this will take a while on first run)...")
    try:
        from bioguard_diffusion.generation.zimage_pipeline import ZImageGenerationPipeline
        from bioguard_diffusion.evaluation.clip_score import CLIPScorer
        pipe = ZImageGenerationPipeline()
        scorer = CLIPScorer()
    except Exception as e:
        print(f"\n[ERROR] Failed to load Z-Image-Turbo: {e}")
        print("Possible cause: insufficient RAM (need ≥ 24 GB for safe operation on 18 GB Mac).")
        sys.exit(1)

    print("Models ready.\n")

    records = []

    for i, q in enumerate(queries, 1):
        qid   = q["id"]
        query = q["query"]
        print(f"[{i:02d}/10] {qid} | {query[:55]}...")

        t0 = time.time()
        try:
            image = pipe.generate(
                prompt=query,
                negative_prompt=NEGATIVE,
                num_inference_steps=8,
                seed=42,
                width=512,
                height=512,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"         [OOM] Skipping {qid} — not enough RAM.")
                records.append({
                    "id": qid, "domain": q["domain"], "query": query,
                    "image_file": None, "clip_score": None,
                    "generation_time_s": None, "config": "zimage_config1",
                    "error": "OOM"
                })
                continue
            raise
        elapsed = time.time() - t0

        img_path = OUT_DIR / f"{qid}.png"
        image.save(img_path)
        clip = scorer.score(image, query)

        sd_clip = sd_c1[qid]["clip_score"]
        delta   = clip - sd_clip
        print(f"         CLIP={clip:.4f}  (SD-v1.5={sd_clip:.4f}, {'↑' if delta>0 else '↓'}{abs(delta):.4f})  time={elapsed:.1f}s")

        records.append({
            "id": qid,
            "domain": q["domain"],
            "query": query,
            "image_file": str(img_path),
            "clip_score": round(clip, 4),
            "clip_delta_vs_sd": round(delta, 4),
            "generation_time_s": round(elapsed, 1),
            "config": "zimage_config1",
        })

    # ── Save & print ─────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(records, f, indent=2)

    valid = [r for r in records if r["clip_score"] is not None]
    print(f"\n{'='*60}")
    print("Z-Image-Turbo vs SD-v1.5 — Baseline Comparison")
    print(f"{'='*60}")
    print(f"{'ID':<6} {'Domain':<18} {'SD-v1.5':>8} {'Z-Image':>8} {'Delta':>8}")
    print("-" * 55)
    for r in valid:
        sd_c = sd_c1[r["id"]]["clip_score"]
        d    = r["clip_delta_vs_sd"]
        print(f"{r['id']:<6} {r['domain']:<18} {sd_c:>8.4f} {r['clip_score']:>8.4f} {'↑' if d>0 else '↓'}{abs(d):>7.4f}")

    if valid:
        zi_mean  = sum(r["clip_score"] for r in valid) / len(valid)
        sd_mean  = sum(sd_c1[r["id"]]["clip_score"] for r in valid) / len(valid)
        print("-" * 55)
        print(f"{'Mean':<6} {'':18} {sd_mean:>8.4f} {zi_mean:>8.4f} {'↑' if zi_mean>sd_mean else '↓'}{abs(zi_mean-sd_mean):>7.4f}")
        zi_steps = sum(r["generation_time_s"] for r in valid) / len(valid)
        print(f"\nAvg generation time (Z-Image 8-step): {zi_steps:.1f}s")
        print(f"Avg generation time (SD-v1.5 30-step): ~30.3s")

    print(f"\nResults saved: {RESULTS_FILE}")
    print(f"Images saved:  {OUT_DIR}/")


if __name__ == "__main__":
    main()
