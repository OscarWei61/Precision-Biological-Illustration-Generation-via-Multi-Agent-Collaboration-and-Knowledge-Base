"""
BioGuard-Diffusion v4 — Full 10-query batch runner (ControlNet)
================================================================
Loads all agents ONCE, runs every query sequentially with ControlNet spatial
guidance from AI2D annotations.

Usage:
  python bioguard_diffusion/generation/run_all_flux_v4.py
  python bioguard_diffusion/generation/run_all_flux_v4.py --note "v4 baseline"
  python bioguard_diffusion/generation/run_all_flux_v4.py --no-controlnet  # compare

Options (env vars or flags):
  MAX_RETRIES  default 3
  STEPS        default 20
  SEED         default 42
  VERIFIER     default combined

Output:
  outputs/run_all_flux_v4/{timestamp}/
    q01_cell_biology/
      blueprint_initial.png   ← spatial layout control image
      blueprint_att02.png     ← adaptive highlight (if missing structures found)
      attempt_01_*.png
      best.png
      reference.png
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

MAX_RETRIES   = int(os.environ.get("MAX_RETRIES", 3))
STEPS         = int(os.environ.get("STEPS", 20))
SEED          = int(os.environ.get("SEED", 42))
VERIFIER_MODE = os.environ.get("VERIFIER", "combined")

QUERIES_FILE    = BASE / "results" / "test_queries.json"
KB_FILE         = BASE / "results" / "knowledge_base.json"
IMAGES_DIR      = BASE / "data" / "ai2d" / "images"
ANNOTATIONS_DIR = BASE / "data" / "ai2d" / "annotations"
OUT_ROOT        = BASE / "outputs" / "run_all_flux_v4"

from bioguard_diffusion.agents.biologist_agent      import BiologistAgent
from bioguard_diffusion.agents.spatial_architect     import SpatialArchitectAgent
from bioguard_diffusion.agents.verifier_agent        import VerifierAgent
from bioguard_diffusion.agents.visual_planner_agent  import VisualPlannerAgent
from bioguard_diffusion.evaluation.clip_score        import CLIPScorer
from bioguard_diffusion.generation.blueprint_generator import BlueprintGenerator
from bioguard_diffusion.generation.run_single import (
    VERIFIER_CLIP,
    VERIFIER_COMBINED,
)
from bioguard_diffusion.generation.run_single_v4 import (
    _run_flux_v4,
    PIPELINE_VERSION_V4,
    _get_constraints,
)


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note",           type=str, default="")
    parser.add_argument("--no-controlnet",  action="store_true")
    parser.add_argument("--blueprint-mode", choices=["lineart", "segmentation"],
                        default="lineart")
    args = parser.parse_args()

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = OUT_ROOT / ts
    run_dir.mkdir(parents=True, exist_ok=False)

    queries       = _load_json(QUERIES_FILE)
    kb            = _load_json(KB_FILE)
    verifier_mode = VERIFIER_COMBINED if VERIFIER_MODE == "combined" else VERIFIER_CLIP

    print("=" * 70)
    print("BioGuard-Diffusion v4 — Full 10-query batch (ControlNet)")
    print("=" * 70)
    print(f"Queries    : {len(queries)}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Steps      : {STEPS}")
    print(f"Seed       : {SEED}")
    print(f"Verifier   : {verifier_mode}")
    print(f"ControlNet : {'DISABLED' if args.no_controlnet else args.blueprint_mode.upper()}")
    print(f"Pipeline   : {PIPELINE_VERSION_V4['version']}")
    if args.note:
        print(f"Note       : {args.note}")
    print(f"Output     : {run_dir}")
    print("=" * 70)

    print("\nLoading agents (FLUX + ControlNet — may take several minutes)...")
    biologist      = BiologistAgent()
    architect      = SpatialArchitectAgent()
    verifier_agent = VerifierAgent()
    scorer         = CLIPScorer()
    visual_planner = VisualPlannerAgent(ANNOTATIONS_DIR, IMAGES_DIR)
    blueprint_gen  = BlueprintGenerator()
    print("All agents ready.\n")

    all_results = []
    batch_start = time.time()

    for i, q in enumerate(queries, 1):
        qid         = q["id"]
        query       = q["query"]
        domain      = q["domain"]
        image_name  = q["image"]
        constraints = _get_constraints(kb, image_name)

        print(f"\n{'=' * 70}")
        print(f"Query {i}/{len(queries)} — [{qid}] {domain}")
        print(f"  {query}")
        print(f"  Image: {image_name}  Constraints: {len(constraints)}")
        print("=" * 70)

        query_dir = run_dir / f"{qid}_{domain}"
        query_dir.mkdir(parents=True, exist_ok=True)

        # Copy AI2D reference for visual comparison (NOT used as generation input)
        ref_src = IMAGES_DIR / image_name
        if ref_src.exists():
            shutil.copy2(ref_src, query_dir / "reference.png")

        q_start = time.time()
        try:
            result = _run_flux_v4(
                query          = query,
                constraints    = constraints,
                image_name     = image_name,
                run_dir        = query_dir,
                max_retries    = MAX_RETRIES,
                steps          = STEPS,
                seed           = SEED,
                verifier_mode  = verifier_mode,
                biologist      = biologist,
                architect      = architect,
                verifier       = verifier_agent,
                scorer         = scorer,
                visual_planner = visual_planner,
                blueprint_gen  = blueprint_gen,
                text2img_only  = False,
                use_controlnet = not args.no_controlnet,
                blueprint_mode = args.blueprint_mode,
            )
            q_wall = round(time.time() - q_start, 1)

            summary = {
                "query_id":         qid,
                "domain":           domain,
                "query":            query,
                "reference":        image_name,
                "wall_time_s":      q_wall,
                "verifier_mode":    verifier_mode,
                "controlnet_enabled": not args.no_controlnet,
                "params": {
                    "max_retries": MAX_RETRIES,
                    "steps":       STEPS,
                    "seed":        SEED,
                },
                **result,
            }
            with open(query_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            all_results.append(summary)
            print(f"\n  [{qid}] done — attempts={result['attempts_used']}  "
                  f"passed={result['passed']}  "
                  f"CLIP={result['best_clip']:.4f}  "
                  f"spatial={result['has_spatial_data']}  "
                  f"time={q_wall}s")

        except Exception as e:
            import traceback
            print(f"\n  [{qid}] ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "query_id": qid, "domain": domain, "query": query,
                "error": str(e),
            })

    batch_wall = round(time.time() - batch_start, 1)
    n_passed   = sum(1 for r in all_results if r.get("passed"))
    clips      = [r.get("best_clip", 0) for r in all_results if "best_clip" in r]
    avg_clip   = round(sum(clips) / max(1, len(clips)), 4)

    combined = {
        "run_timestamp":    ts,
        "pipeline":         PIPELINE_VERSION_V4,
        "run_note":         args.note,
        "controlnet": {
            "enabled":       not args.no_controlnet,
            "blueprint_mode": args.blueprint_mode,
        },
        "total_wall_s":     batch_wall,
        "n_queries":        len(queries),
        "n_passed":         n_passed,
        "avg_clip":         avg_clip,
        "results":          all_results,
    }
    combined_path = run_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("Batch complete")
    print("=" * 70)
    print(f"Total time : {batch_wall}s")
    print(f"Passed     : {n_passed}/{len(queries)}")
    print(f"Avg CLIP   : {avg_clip:.4f}")
    print(f"Results    : {combined_path}")


if __name__ == "__main__":
    main()
