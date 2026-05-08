"""
Stage 3 — Config 2: RAG + Spatial Architect → SD
Generates images for all 10 test queries using:
  Biologist Agent (RAG retrieval) → Spatial Architect → Stable Diffusion

Run: conda run -n agent python bioguard_diffusion/generation/run_config2_rag.py
"""

import json
import time
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from bioguard_diffusion.agents.biologist_agent import BiologistAgent
from bioguard_diffusion.agents.spatial_architect import SpatialArchitectAgent
from bioguard_diffusion.generation.diffusion_pipeline import BioGenerationPipeline
from bioguard_diffusion.evaluation.clip_score import CLIPScorer

QUERIES_FILE = BASE / "results" / "test_queries.json"
C1_FILE = BASE / "results" / "config1_results.json"
OUT_DIR = BASE / "outputs" / "config2"
RESULTS_FILE = BASE / "results" / "config2_results.json"
PROMPTS_FILE = BASE / "results" / "config2_prompts.json"


def main():
    print("=" * 60)
    print("Stage 3 — Config 2: RAG + Spatial Architect Generation")
    print("=" * 60)

    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    with open(C1_FILE) as f:
        c1_results = {r["id"]: r for r in json.load(f)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load agents and models ──────────────────────────────────────────────
    print("\nInitialising agents...")
    biologist = BiologistAgent()
    architect = SpatialArchitectAgent()

    print("\nLoading generation & evaluation models...")
    pipe = BioGenerationPipeline()
    scorer = CLIPScorer()
    print("All models ready.\n")

    records = []
    prompt_log = []

    for i, q in enumerate(queries, 1):
        qid = q["id"]
        query = q["query"]
        print(f"[{i:02d}/10] {qid} | {query[:55]}...")

        # Step 1: Retrieve biological constraints
        constraints = biologist.retrieve(query, top_k=8)
        print(f"         Retrieved {len(constraints)} constraints")

        # Step 2: Build structured prompt
        result = architect.build_prompt(query, constraints)
        prompt = result["prompt"]
        neg_prompt = result["negative_prompt"]
        print(f"         Subject: {result['subject']}")
        print(f"         Prompt ({len(prompt)} chars): {prompt[:80]}...")

        # Step 3: Generate image (same seed as Config 1 for fair comparison)
        t0 = time.time()
        image = pipe.generate(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=30,
            seed=42,
        )
        elapsed = time.time() - t0

        # Step 4: Save image + score
        img_path = OUT_DIR / f"{qid}.png"
        image.save(img_path)
        clip = scorer.score(image, query)

        c1_clip = c1_results[qid]["clip_score"]
        delta = clip - c1_clip
        trend = "↑" if delta > 0 else "↓"

        print(f"         CLIP={clip:.4f}  (Config1={c1_clip:.4f}, {trend}{abs(delta):.4f})  time={elapsed:.1f}s")

        records.append({
            "id": qid,
            "domain": q["domain"],
            "query": query,
            "image_file": str(img_path),
            "clip_score": round(clip, 4),
            "clip_delta_vs_c1": round(delta, 4),
            "generation_time_s": round(elapsed, 1),
            "config": "config2_rag",
        })
        prompt_log.append({
            "id": qid,
            "original_query": query,
            "subject": result["subject"],
            "constraints_retrieved": constraints,
            "constraints_used": result["constraints_used"],
            "structured_prompt": prompt,
            "negative_prompt": neg_prompt,
        })

    # ── Save outputs ────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(records, f, indent=2)
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompt_log, f, indent=2, ensure_ascii=False)

    # ── Print comparison table ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Config 1 vs Config 2 — CLIP-Score Comparison")
    print(f"{'='*65}")
    print(f"{'ID':<6} {'Domain':<18} {'C1 CLIP':>8} {'C2 CLIP':>8} {'Delta':>7}")
    print("-" * 55)

    c1_scores, c2_scores = [], []
    for r in records:
        c1 = c1_results[r["id"]]["clip_score"]
        c2 = r["clip_score"]
        delta = r["clip_delta_vs_c1"]
        arrow = "↑" if delta > 0 else "↓"
        print(f"{r['id']:<6} {r['domain']:<18} {c1:>8.4f} {c2:>8.4f} {arrow}{abs(delta):>6.4f}")
        c1_scores.append(c1)
        c2_scores.append(c2)

    c1_mean = sum(c1_scores) / len(c1_scores)
    c2_mean = sum(c2_scores) / len(c2_scores)
    mean_delta = c2_mean - c1_mean
    improved = sum(1 for r in records if r["clip_delta_vs_c1"] > 0)

    print("-" * 55)
    print(f"{'Mean':<6} {'':18} {c1_mean:>8.4f} {c2_mean:>8.4f} {'↑' if mean_delta>0 else '↓'}{abs(mean_delta):>6.4f}")
    print(f"\nImproved queries: {improved}/10")
    print(f"Results saved:  {RESULTS_FILE}")
    print(f"Prompts saved:  {PROMPTS_FILE}")
    print("\nConfig 2 COMPLETE — ready for Stage 4 (Config 3: Full BioGuard)")


if __name__ == "__main__":
    main()
