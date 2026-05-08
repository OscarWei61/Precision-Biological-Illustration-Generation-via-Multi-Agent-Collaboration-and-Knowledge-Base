"""
BioGuard-Diffusion — Full 10-query batch runner (FLUX backend)
==============================================================
Loads all agents ONCE, then runs every query in test_queries.json sequentially.
Reference images are copied into each output folder for side-by-side comparison.

Usage:
  python bioguard_diffusion/generation/run_all_flux.py

Options (env vars or edit defaults below):
  MAX_RETRIES   default 3
  STEPS         default 20
  SEED          default 42
  VERIFIER      default combined   (clip | combined)

Output layout:
  outputs/run_all_flux/{timestamp}/
    q01_cell_biology/
      attempt_01_text2img_seed42.png
      attempt_02_img2img_str0.60_seed42.png
      best.png
      reference.png          ← AI2D reference image
      prompt_log.txt
      summary.json
    q02_cell_biology/
      ...
    all_results.json          ← combined results for all queries
"""

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

# ── Settings ──────────────────────────────────────────────────────────────────
MAX_RETRIES   = int(os.environ.get("MAX_RETRIES", 3))
STEPS         = int(os.environ.get("STEPS", 20))
SEED          = int(os.environ.get("SEED", 42))
VERIFIER_MODE = os.environ.get("VERIFIER", "combined")   # "clip" | "combined"

QUERIES_FILE  = BASE / "results" / "test_queries.json"
KB_FILE       = BASE / "results" / "knowledge_base.json"
IMAGES_DIR    = BASE / "data" / "ai2d" / "images"
OUT_ROOT      = BASE / "outputs" / "run_all_flux"

# ── Imports ───────────────────────────────────────────────────────────────────
from bioguard_diffusion.agents.biologist_agent  import BiologistAgent
from bioguard_diffusion.agents.spatial_architect import SpatialArchitectAgent
from bioguard_diffusion.agents.verifier_agent   import VerifierAgent
from bioguard_diffusion.evaluation.clip_score   import CLIPScorer
from bioguard_diffusion.generation.run_single   import (
    _run_flux,
    VERIFIER_CLIP,
    VERIFIER_COMBINED,
    PIPELINE_VERSION,
)


def _load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def _get_constraints(kb: list[dict], image_name: str) -> list[str]:
    for entry in kb:
        if entry["image"] == image_name:
            return entry.get("constraints", [])
    return []


def main():
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="BioGuard-Diffusion batch runner")
    _parser.add_argument("--note", type=str, default="",
                         help="Free-text note saved into all_results.json for comparison tracking")
    _args = _parser.parse_args()

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = OUT_ROOT / ts
    run_dir.mkdir(parents=True, exist_ok=False)   # fail loudly if collision

    queries = _load_json(QUERIES_FILE)
    kb      = _load_json(KB_FILE)

    verifier_mode = VERIFIER_COMBINED if VERIFIER_MODE == "combined" else VERIFIER_CLIP

    print("=" * 70)
    print("BioGuard-Diffusion — Full 10-query batch (FLUX)")
    print("=" * 70)
    print(f"Queries    : {len(queries)}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Steps      : {STEPS}")
    print(f"Seed       : {SEED}")
    print(f"Verifier   : {verifier_mode}")
    print(f"Pipeline   : {PIPELINE_VERSION['version']}")
    if _args.note:
        print(f"Note       : {_args.note}")
    print(f"Output     : {run_dir}")
    print("=" * 70)

    # ── Load agents ONCE ──────────────────────────────────────────────────────
    print("\nLoading agents (this loads FLUX model — may take several minutes)...")
    biologist      = BiologistAgent()
    architect      = SpatialArchitectAgent()
    verifier_agent = VerifierAgent()
    scorer         = CLIPScorer()
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
        print(f"  Constraints: {len(constraints)}")
        print("=" * 70)

        query_dir = run_dir / f"{qid}_{domain}"
        query_dir.mkdir(parents=True, exist_ok=True)

        # Copy reference image for visual comparison
        ref_src = IMAGES_DIR / image_name
        if ref_src.exists():
            shutil.copy2(ref_src, query_dir / "reference.png")
            print(f"  Reference: {image_name} → reference.png")
        else:
            print(f"  Reference image not found: {ref_src}")

        q_start = time.time()
        try:
            result = _run_flux(
                query        = query,
                constraints  = constraints,
                run_dir      = query_dir,
                max_retries  = MAX_RETRIES,
                steps        = STEPS,
                seed         = SEED,
                verifier_mode= verifier_mode,
                biologist    = biologist,
                architect    = architect,
                verifier     = verifier_agent,
                scorer       = scorer,
                text2img_only= False,
            )
            q_wall = round(time.time() - q_start, 1)

            summary = {
                "query_id":      qid,
                "domain":        domain,
                "query":         query,
                "reference":     image_name,
                "wall_time_s":   q_wall,
                "verifier_mode": verifier_mode,
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
                  f"time={q_wall}s")

        except Exception as e:
            print(f"\n  [{qid}] ERROR: {e}")
            all_results.append({
                "query_id": qid, "domain": domain, "query": query,
                "error": str(e),
            })

    # ── Save combined results ─────────────────────────────────────────────────
    batch_wall = round(time.time() - batch_start, 1)
    combined = {
        "run_timestamp":  ts,
        "pipeline":       PIPELINE_VERSION,
        "run_note":       _args.note,
        "total_wall_s":   batch_wall,
        "n_queries":      len(queries),
        "n_passed":       sum(1 for r in all_results if r.get("passed")),
        "avg_clip":       round(
            sum(r.get("best_clip", 0) for r in all_results if "best_clip" in r)
            / max(1, sum(1 for r in all_results if "best_clip" in r)), 4
        ),
        "results": all_results,
    }
    combined_path = run_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("Batch complete")
    print("=" * 70)
    print(f"Total time : {batch_wall}s")
    print(f"Passed     : {combined['n_passed']}/{len(queries)}")
    print(f"Avg CLIP   : {combined['avg_clip']:.4f}")
    print(f"Results    : {combined_path}")


if __name__ == "__main__":
    main()
