"""
BioGuard-Diffusion — Ablation Baseline Runners
===============================================
Two one-shot baselines for comparison against v3 (retry+RAG) and v4 (ControlNet):

  flux_direct     : raw query → PromptSplitter → FLUX (no RAG, no agents, no retry)
  flux_rag_noretry: BiologistAgent + SpatialArchitect → PromptSplitter → FLUX one-shot
                    (has RAG retrieval but NO verification retry loop)

Both modes score every image with CLIPScorer + VerifierAgent (LLM-as-judge) for
apples-to-apples comparison with v3/v4.

Usage:
  python bioguard_diffusion/generation/run_all_flux_baselines.py --mode flux_direct
  python bioguard_diffusion/generation/run_all_flux_baselines.py --mode flux_rag_noretry
  python bioguard_diffusion/generation/run_all_flux_baselines.py --mode both  (default)

Output:
  outputs/run_all_flux_baselines/{timestamp}_{mode}/
    q01_cell_biology/
      attempt_01_text2img.png
      prompt_log.txt
      summary.json
    all_results.json
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from dotenv import load_dotenv as _load_dotenv
try:
    _load_dotenv(BASE / ".env")
except Exception:
    pass

MAX_RETRIES   = 1   # baselines are one-shot only
STEPS         = int(os.environ.get("STEPS", 20))
SEED          = int(os.environ.get("SEED", 42))
VERIFIER_MODE = os.environ.get("VERIFIER", "combined")

QUERIES_FILE = BASE / "results" / "test_queries.json"
KB_FILE      = BASE / "results" / "knowledge_base.json"
IMAGES_DIR   = BASE / "data" / "ai2d" / "images"
OUT_ROOT     = BASE / "outputs" / "run_all_flux_baselines"

from bioguard_diffusion.agents.biologist_agent   import BiologistAgent
from bioguard_diffusion.agents.spatial_architect  import SpatialArchitectAgent
from bioguard_diffusion.agents.verifier_agent     import VerifierAgent
from bioguard_diffusion.agents.prompt_splitter    import PromptSplitter
from bioguard_diffusion.evaluation.clip_score     import CLIPScorer
from bioguard_diffusion.generation.flux_gguf_pipeline import FluxGGUFPipeline
from bioguard_diffusion.generation.run_single import (
    VERIFIER_CLIP,
    VERIFIER_COMBINED,
)

PIPELINE_VERSION_DIRECT = {
    "version": "flux_direct",
    "description": "No RAG, no agents — raw query → PromptSplitter → FLUX one-shot",
}

PIPELINE_VERSION_RAG_NORETRY = {
    "version": "flux_rag_noretry",
    "description": "BiologistAgent RAG + SpatialArchitect → PromptSplitter → FLUX one-shot (no retry)",
}

LLM_PASS_THRESHOLD = 6.0
CLIP_PASS_THRESHOLD = 0.22


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _get_constraints(kb: list[dict], image_name: str) -> list[str]:
    for entry in kb:
        if entry["image"] == image_name:
            return entry.get("constraints", [])
    return []


def _score_image(image, query: str, constraints: list[str], scorer: CLIPScorer,
                 verifier: VerifierAgent, verifier_mode: str) -> dict:
    """Run CLIP + optional LLM scoring. Returns score dict."""
    clip_score = scorer.score(image, query)

    llm_score  = 0.0
    llm_passed = False
    missing    = []
    suggestions = ""

    if verifier_mode == VERIFIER_COMBINED:
        result = verifier.verify_with_llm(image, query, constraints)
        llm_score   = result.get("llm_score", 0.0)
        llm_passed  = llm_score >= LLM_PASS_THRESHOLD
        missing     = result.get("missing_structures", [])
        suggestions = result.get("improvement_suggestions", "")

    clip_passed = clip_score >= CLIP_PASS_THRESHOLD
    if verifier_mode == VERIFIER_COMBINED:
        passed = clip_passed and llm_passed
        combined = 0.5 * clip_score + 0.5 * (llm_score / 10.0)
    else:
        passed   = clip_passed
        combined = clip_score

    return {
        "clip_score":    round(clip_score, 4),
        "llm_score":     round(llm_score, 4),
        "combined":      round(combined, 4),
        "clip_passed":   clip_passed,
        "llm_passed":    llm_passed,
        "passed":        passed,
        "missing":       missing,
        "suggestions":   suggestions,
    }


def _run_one_shot(
    query: str,
    constraints: list[str],
    image_name: str,
    run_dir: Path,
    pipeline: "FluxGGUFPipeline",
    splitter: PromptSplitter,
    scorer: CLIPScorer,
    verifier: VerifierAgent,
    verifier_mode: str,
    full_prompt: str,
    mode_tag: str,
) -> dict:
    """
    Generate exactly one image, score it, save artifacts.

    full_prompt : already-built prompt string (caller decides content)
    mode_tag    : 'flux_direct' or 'flux_rag_noretry' (for filenames)
    """
    split = splitter.split(full_prompt)
    clip_prompt = split["clip_prompt"]
    t5_prompt   = split["t5_prompt"]

    print(f"  [CLIP] {clip_prompt[:70]}...")
    print(f"  [T5]   {t5_prompt[:90]}...")

    t0    = time.time()
    image = pipeline.generate(
        prompt               = clip_prompt,
        prompt_2             = t5_prompt,
        num_inference_steps  = STEPS,
        guidance_scale       = 3.5,
        seed                 = SEED,
        width                = 512,
        height               = 512,
    )
    gen_time = round(time.time() - t0, 1)
    print(f"  Generated in {gen_time}s")

    img_path = run_dir / "attempt_01_text2img.png"
    image.save(img_path)

    scores = _score_image(image, query, constraints, scorer, verifier, verifier_mode)
    print(f"  CLIP={scores['clip_score']:.4f}  LLM={scores['llm_score']:.1f}  "
          f"passed={scores['passed']}")

    # prompt log
    with open(run_dir / "prompt_log.txt", "w") as f:
        f.write(f"Mode:        {mode_tag}\n")
        f.write(f"Query:       {query}\n")
        f.write(f"Constraints ({len(constraints)}):\n")
        for c in constraints:
            f.write(f"  - {c}\n")
        f.write("\n--- Attempt 1 (text2img) ---\n")
        f.write(f"Full prompt:  {full_prompt}\n")
        f.write(f"CLIP prompt:  {clip_prompt}\n")
        f.write(f"T5 prompt:    {t5_prompt}\n")
        f.write(f"Seed:         {SEED}\n")
        f.write(f"Steps:        {STEPS}\n")
        f.write(f"Gen time:     {gen_time}s\n")

    return {
        "attempts_used":  1,
        "best_clip":      scores["clip_score"],
        "best_llm":       scores["llm_score"],
        "best_combined":  scores["combined"],
        "passed":         scores["passed"],
        "gen_time_s":     gen_time,
        "full_prompt":    full_prompt,
        "clip_prompt":    clip_prompt,
        "t5_prompt":      t5_prompt,
        "missing":        scores["missing"],
    }


def _run_mode(
    mode: str,
    queries: list[dict],
    kb: list[dict],
    run_dir: Path,
    pipeline: "FluxGGUFPipeline",
    splitter: PromptSplitter,
    scorer: CLIPScorer,
    verifier: VerifierAgent,
    biologist: "BiologistAgent | None",
    architect: "SpatialArchitectAgent | None",
    verifier_mode: str,
) -> list[dict]:
    """Run all queries in the given mode, return per-query result list."""
    all_results = []

    for i, q in enumerate(queries, 1):
        qid        = q["id"]
        query      = q["query"]
        domain     = q["domain"]
        image_name = q["image"]
        constraints = _get_constraints(kb, image_name)

        print(f"\n{'=' * 70}")
        print(f"[{mode}] Query {i}/{len(queries)} — [{qid}] {domain}")
        print(f"  {query}")
        print(f"  Image: {image_name}  Constraints: {len(constraints)}")
        print("=" * 70)

        query_dir = run_dir / f"{qid}_{domain}"
        query_dir.mkdir(parents=True, exist_ok=True)

        # Copy reference image for visual comparison
        ref_src = IMAGES_DIR / image_name
        if ref_src.exists():
            shutil.copy2(ref_src, query_dir / "reference.png")

        q_start = time.time()
        try:
            if mode == "flux_direct":
                # No RAG — use raw query as the prompt (PromptSplitter handles CLIP/T5 split)
                full_prompt = query
            else:
                # flux_rag_noretry — RAG + SpatialArchitect
                retrieved = biologist.retrieve(query, top_k=10)
                full_prompt = architect.build_prompt(query, retrieved)["prompt"]

            result = _run_one_shot(
                query         = query,
                constraints   = constraints,
                image_name    = image_name,
                run_dir       = query_dir,
                pipeline      = pipeline,
                splitter      = splitter,
                scorer        = scorer,
                verifier      = verifier,
                verifier_mode = verifier_mode,
                full_prompt   = full_prompt,
                mode_tag      = mode,
            )
            q_wall = round(time.time() - q_start, 1)

            summary = {
                "query_id":      qid,
                "domain":        domain,
                "query":         query,
                "reference":     image_name,
                "wall_time_s":   q_wall,
                "verifier_mode": verifier_mode,
                "mode":          mode,
                "params": {
                    "steps": STEPS,
                    "seed":  SEED,
                },
                **result,
            }
            with open(query_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            all_results.append(summary)
            print(f"\n  [{qid}] done — CLIP={result['best_clip']:.4f}  "
                  f"LLM={result['best_llm']:.1f}  passed={result['passed']}  "
                  f"time={q_wall}s")

        except Exception as e:
            import traceback
            print(f"\n  [{qid}] ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "query_id": qid, "domain": domain, "query": query,
                "error": str(e), "mode": mode,
            })

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["flux_direct", "flux_rag_noretry", "both"],
        default="both",
        help="Which baseline(s) to run",
    )
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    modes = (
        ["flux_direct", "flux_rag_noretry"] if args.mode == "both"
        else [args.mode]
    )

    verifier_mode = VERIFIER_COMBINED if VERIFIER_MODE == "combined" else VERIFIER_CLIP

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    print("=" * 70)
    print("BioGuard-Diffusion — Ablation Baselines")
    print("=" * 70)
    print(f"Modes      : {', '.join(modes)}")
    print(f"Steps      : {STEPS}")
    print(f"Seed       : {SEED}")
    print(f"Verifier   : {verifier_mode}")
    if args.note:
        print(f"Note       : {args.note}")
    print("=" * 70)

    queries = _load_json(QUERIES_FILE)
    kb      = _load_json(KB_FILE)

    print("\nLoading FLUX pipeline (may take several minutes)...")
    pipeline = FluxGGUFPipeline()
    splitter = PromptSplitter()
    scorer   = CLIPScorer()
    verifier = VerifierAgent()
    print("FLUX pipeline ready.\n")

    # Only load RAG agents if needed
    biologist = None
    architect = None
    if "flux_rag_noretry" in modes:
        print("Loading RAG agents...")
        biologist = BiologistAgent()
        architect = SpatialArchitectAgent()
        print("RAG agents ready.\n")

    batch_start = time.time()
    all_mode_results = {}

    for mode in modes:
        mode_run_dir = OUT_ROOT / f"{ts}_{mode}"
        mode_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#' * 70}")
        print(f"# Running mode: {mode}")
        print(f"{'#' * 70}")

        results = _run_mode(
            mode          = mode,
            queries       = queries,
            kb            = kb,
            run_dir       = mode_run_dir,
            pipeline      = pipeline,
            splitter      = splitter,
            scorer        = scorer,
            verifier      = verifier,
            biologist     = biologist,
            architect     = architect,
            verifier_mode = verifier_mode,
        )

        n_passed = sum(1 for r in results if r.get("passed"))
        clips    = [r.get("best_clip", 0) for r in results if "best_clip" in r]
        avg_clip = round(sum(clips) / max(1, len(clips)), 4)
        llms     = [r.get("best_llm", 0) for r in results if "best_llm" in r]
        avg_llm  = round(sum(llms) / max(1, len(llms)), 2)

        pipeline_info = (
            PIPELINE_VERSION_DIRECT if mode == "flux_direct"
            else PIPELINE_VERSION_RAG_NORETRY
        )

        combined_result = {
            "run_timestamp": ts,
            "pipeline":      pipeline_info,
            "run_note":      args.note,
            "mode":          mode,
            "verifier_mode": verifier_mode,
            "total_wall_s":  round(time.time() - batch_start, 1),
            "n_queries":     len(queries),
            "n_passed":      n_passed,
            "avg_clip":      avg_clip,
            "avg_llm":       avg_llm,
            "params": {"steps": STEPS, "seed": SEED},
            "results":       results,
        }

        combined_path = mode_run_dir / "all_results.json"
        with open(combined_path, "w") as f:
            json.dump(combined_result, f, indent=2, ensure_ascii=False)

        all_mode_results[mode] = combined_result

        print(f"\n[{mode}] Passed: {n_passed}/{len(queries)}  "
              f"Avg CLIP: {avg_clip:.4f}  Avg LLM: {avg_llm:.2f}")
        print(f"  Results: {combined_path}")

    batch_wall = round(time.time() - batch_start, 1)
    print(f"\n{'=' * 70}")
    print("All baseline runs complete")
    print("=" * 70)
    print(f"Total time : {batch_wall}s")
    for mode, res in all_mode_results.items():
        print(f"  {mode:<22} passed={res['n_passed']}/{res['n_queries']}  "
              f"CLIP={res['avg_clip']:.4f}  LLM={res['avg_llm']:.2f}")


if __name__ == "__main__":
    main()
