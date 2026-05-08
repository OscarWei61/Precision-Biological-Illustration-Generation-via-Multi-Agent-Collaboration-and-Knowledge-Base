"""
Stage 4 — Config 3: Full BioGuard-Diffusion
Pipeline: Biologist Agent → Spatial Architect → SD → Verifier → [retry ≤3]

Run: conda run -n agent python bioguard_diffusion/generation/run_config3_full.py
"""

import json
import time
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from bioguard_diffusion.agents.biologist_agent   import BiologistAgent
from bioguard_diffusion.agents.spatial_architect  import SpatialArchitectAgent
from bioguard_diffusion.agents.verifier_agent     import VerifierAgent
from bioguard_diffusion.generation.diffusion_pipeline import BioGenerationPipeline
from bioguard_diffusion.evaluation.clip_score     import CLIPScorer

QUERIES_FILE = BASE / "results" / "test_queries.json"
KB_FILE      = BASE / "results" / "knowledge_base.json"
C1_FILE      = BASE / "results" / "config1_results.json"
C2_FILE      = BASE / "results" / "config2_results.json"
OUT_DIR      = BASE / "outputs" / "config3"
RESULTS_FILE = BASE / "results" / "config3_results.json"

MAX_RETRIES   = 3
SAS_THRESHOLD = 0.20   # same as verifier PASS_THRESHOLD


def _get_constraints_for_query(kb: list[dict], image_name: str) -> list[str]:
    for entry in kb:
        if entry["image"] == image_name:
            return entry.get("constraints", [])
    return []


def main():
    print("=" * 65)
    print("Stage 4 — Config 3: Full BioGuard-Diffusion")
    print("=" * 65)

    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    with open(KB_FILE) as f:
        kb = json.load(f)
    with open(C1_FILE) as f:
        c1 = {r["id"]: r for r in json.load(f)}
    with open(C2_FILE) as f:
        c2 = {r["id"]: r for r in json.load(f)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all agents & models ─────────────────────────────────────────────
    print("\nInitialising agents...")
    biologist = BiologistAgent()
    architect = SpatialArchitectAgent()
    verifier  = VerifierAgent()

    print("\nLoading SD pipeline & CLIP scorer...")
    pipe   = BioGenerationPipeline()
    scorer = CLIPScorer()
    print("All models ready.\n")

    records      = []
    verify_log   = []

    for i, q in enumerate(queries, 1):
        qid         = q["id"]
        query       = q["query"]
        constraints = _get_constraints_for_query(kb, q["image"])

        print(f"[{i:02d}/10] {qid} | {query[:55]}...")
        print(f"         Constraints to verify: {len(constraints)}")

        # Step 1: RAG retrieval
        retrieved = biologist.retrieve(query, top_k=8)

        best_image      = None
        best_sas        = -1.0
        best_clip       = 0.0
        retry_history   = []
        violations      = []
        total_time      = 0.0

        for attempt in range(1, MAX_RETRIES + 1):
            # Step 2: Build structured prompt (violations → priority on retry)
            arch_result = architect.build_prompt(
                query, retrieved,
                violations=violations if attempt > 1 else None,
            )
            prompt     = arch_result["prompt"]
            neg_prompt = arch_result["negative_prompt"]

            # Step 3: Generate (vary seed on retry to get different image)
            seed = 42 + (attempt - 1) * 7
            t0   = time.time()
            image = pipe.generate(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=30,
                seed=seed,
            )
            gen_time = time.time() - t0
            total_time += gen_time

            # Step 4: Verify with BioCLIP
            v = verifier.verify(image, constraints, entity=qid)
            clip = scorer.score(image, query)

            print(f"         Attempt {attempt}: SAS={v['sas']:.4f}  "
                  f"CLIP={clip:.4f}  violations={len(v['violations'])}  "
                  f"time={gen_time:.1f}s")
            if v["violations"]:
                print(f"           ↳ Violated: {v['violations'][:2]}")

            retry_history.append({
                "attempt": attempt, "seed": seed,
                "sas": v["sas"], "clip_score": clip,
                "violations": v["violations"],
                "prompt_len": len(prompt),
            })

            # Keep the best image seen so far
            if v["sas"] > best_sas:
                best_sas   = v["sas"]
                best_image = image
                best_clip  = clip

            if v["passed"]:
                print(f"         ✓ Passed on attempt {attempt}")
                violations = []
                break

            violations = v["violations"]   # feed back into next retry

        # Save best image
        img_path = OUT_DIR / f"{qid}.png"
        best_image.save(img_path)

        # Final verify result on best image
        final_v = verifier.verify(best_image, constraints, entity=qid)

        records.append({
            "id":                  qid,
            "domain":              q["domain"],
            "query":               query,
            "image_file":          str(img_path),
            "clip_score":          round(best_clip, 4),
            "sas":                 round(final_v["sas"], 4),
            "sas_passed":          final_v["passed"],
            "violations":          final_v["violations"],
            "attempts_used":       len(retry_history),
            "generation_time_s":   round(total_time, 1),
            "clip_delta_vs_c1":    round(best_clip - c1[qid]["clip_score"], 4),
            "clip_delta_vs_c2":    round(best_clip - c2[qid]["clip_score"], 4),
            "config":              "config3_full",
        })
        verify_log.append({
            "id": qid, "constraints": constraints,
            "per_constraint": final_v["per_constraint"],
            "retry_history": retry_history,
        })

    # ── Save outputs ─────────────────────────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(records, f, indent=2)
    with open(BASE / "results" / "config3_verify_log.json", "w") as f:
        json.dump(verify_log, f, indent=2, ensure_ascii=False)

    # ── Final 3-way comparison ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Final 3-way Comparison: Config 1 vs Config 2 vs Config 3")
    print(f"{'='*70}")
    print(f"{'ID':<6} {'Domain':<18} {'C1':>7} {'C2':>7} {'C3':>7} {'SAS':>7} {'Tries':>6}")
    print("-" * 65)

    c1s, c2s, c3s, sass = [], [], [], []
    for r in records:
        c1c = c1[r["id"]]["clip_score"]
        c2c = c2[r["id"]]["clip_score"]
        print(f"{r['id']:<6} {r['domain']:<18} "
              f"{c1c:>7.4f} {c2c:>7.4f} {r['clip_score']:>7.4f} "
              f"{r['sas']:>7.4f} {r['attempts_used']:>6}")
        c1s.append(c1c); c2s.append(c2c)
        c3s.append(r["clip_score"]); sass.append(r["sas"])

    print("-" * 65)
    print(f"{'Mean':<6} {'':18} "
          f"{sum(c1s)/10:>7.4f} {sum(c2s)/10:>7.4f} {sum(c3s)/10:>7.4f} "
          f"{sum(sass)/10:>7.4f}")

    total_retries = sum(r["attempts_used"] for r in records) - 10  # subtract base attempts
    passed_count  = sum(1 for r in records if r["sas_passed"])
    print(f"\nVerifier triggered retries: {total_retries}")
    print(f"Queries passing SAS threshold: {passed_count}/10")
    print(f"\nResults:  {RESULTS_FILE}")
    print(f"Verify log: {BASE / 'results' / 'config3_verify_log.json'}")
    print("\nConfig 3 COMPLETE")


if __name__ == "__main__":
    main()
