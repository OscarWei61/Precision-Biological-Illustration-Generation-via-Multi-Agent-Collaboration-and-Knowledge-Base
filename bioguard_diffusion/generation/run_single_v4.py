"""
BioGuard-Diffusion v4 — ControlNet Spatial-Guided Runner
=========================================================
Adds VisualPlannerAgent + BlueprintGenerator on top of the v3 pipeline.

New features vs v3
------------------
  1. VisualPlannerAgent reads AI2D blob annotations → layout JSON with
     polygon coordinates for each labelled biological structure.
  2. BlueprintGenerator renders a lineart control image from that layout.
  3. GeneratorAgentV4 uses FluxControlNetPipeline (GGUF transformer +
     InstantX/FLUX.1-dev-Controlnet-Canny).
  4. Adaptive blueprint: when Verifier reports missing_structures, the
     blueprint is regenerated with those regions highlighted in red, giving
     ControlNet stronger guidance toward them on the next attempt.
  5. ControlNet scale is adaptive: 0.70 for text2img, 0.50 for img2img
     when LLM score is high (fine-tune), 0.65 when score is low (keep structure).

Fallback behaviour
------------------
  If no AI2D annotation exists for the query image (has_spatial_data=False),
  the pipeline degrades gracefully to v3 behaviour (ControlNet scale = 0.0).

Usage
-----
  # Interactive
  python bioguard_diffusion/generation/run_single_v4.py

  # CLI
  python bioguard_diffusion/generation/run_single_v4.py --query-id q02
  python bioguard_diffusion/generation/run_single_v4.py --query-id q07 \\
      --blueprint-mode segmentation --controlnet-scale 0.8

  # Skip ControlNet (compare against v3 baseline)
  python bioguard_diffusion/generation/run_single_v4.py --query-id q01 \\
      --no-controlnet

Output dir: outputs/single_runs_v4/{timestamp}_{label}/
  blueprint_initial.png        initial ControlNet control image
  blueprint_att{N}.png         adaptive blueprint used on attempt N (if updated)
  attempt_{N}_{mode}_seed{S}.png
  best.png
  prompt_log.txt
  summary.json                 includes layout + per-attempt controlnet_scale
"""

import argparse
import datetime
import json
import os
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

_LANGSMITH_KEY     = os.environ.get("LANGCHAIN_API_KEY", "")
_LANGSMITH_ENABLED = bool(_LANGSMITH_KEY and _LANGSMITH_KEY != "YOUR_LANGSMITH_API_KEY_HERE")

try:
    from langsmith import traceable
    if not _LANGSMITH_ENABLED:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

from bioguard_diffusion.agents.biologist_agent    import BiologistAgent
from bioguard_diffusion.agents.spatial_architect   import SpatialArchitectAgent
from bioguard_diffusion.agents.verifier_agent      import VerifierAgent
from bioguard_diffusion.agents.visual_planner_agent import VisualPlannerAgent
from bioguard_diffusion.evaluation.clip_score      import CLIPScorer
from bioguard_diffusion.generation.blueprint_generator import BlueprintGenerator

# Re-use verifier mode constants from v3
from bioguard_diffusion.generation.run_single import (
    VERIFIER_CLIP,
    VERIFIER_COMBINED,
    PIPELINE_VERSION as _V3_PIPELINE_VERSION,
    _adaptive_strength,
)

QUERIES_FILE     = BASE / "results" / "test_queries.json"
KB_FILE          = BASE / "results" / "knowledge_base.json"
ANNOTATIONS_DIR  = BASE / "data" / "ai2d" / "annotations"
IMAGES_DIR       = BASE / "data" / "ai2d" / "images"
OUT_ROOT         = BASE / "outputs" / "single_runs_v4"

DEFAULT_STEPS       = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_SEED        = 42
DEFAULT_VERIFIER    = VERIFIER_COMBINED

PIPELINE_VERSION_V4 = {
    "version": "v4",
    "base":    _V3_PIPELINE_VERSION["version"],
    "changes": [
        "visual_planner: reads AI2D blob annotations → spatial layout JSON",
        "blueprint_generator: polygon outlines → ControlNet lineart control image",
        "generator_v4: FluxControlNetPipeline (GGUF + InstantX Canny ControlNet)",
        "adaptive_blueprint: missing structures highlighted red in control image",
        "controlnet_scale: 0.70 t2i / adaptive 0.50–0.65 i2i",
        "fallback: no annotation → cn_scale=0.0 (degrades to v3 behaviour)",
    ] + _V3_PIPELINE_VERSION["changes"],
}


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _make_run_dir(label: str) -> Path:
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = label.replace(" ", "_")[:40]
    run_dir    = OUT_ROOT / f"{ts}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _get_constraints(kb: list[dict], image_name: str) -> list[str]:
    for entry in kb:
        if entry["image"] == image_name:
            return entry.get("constraints", [])
    return []


def _controlnet_scale_for_attempt(
    llm_score: float,
    attempt: int,
    has_spatial_data: bool,
) -> float:
    """Dynamic ControlNet scale — tighter guidance when score is low."""
    if not has_spatial_data:
        return 0.0
    if attempt == 1:
        return 0.70
    # img2img: higher scale when image is far from target
    if llm_score < 5.0:
        return 0.65   # more structure guidance needed
    return 0.50       # near threshold, light touch


# ───────────────────────────────────────────────────────────────────────────
# FLUX v4 runner (ControlNet)
# ───────────────────────────────────────────────────────────────────────────

@traceable(name="bioguard_pipeline_flux_v4", run_type="chain")
def _run_flux_v4(
    query: str,
    constraints: list[str],
    image_name: str | None,
    run_dir: Path,
    max_retries: int,
    steps: int,
    seed: int,
    verifier_mode: str,
    biologist: BiologistAgent,
    architect: SpatialArchitectAgent,
    verifier: VerifierAgent,
    scorer: CLIPScorer,
    visual_planner: VisualPlannerAgent,
    blueprint_gen: BlueprintGenerator,
    text2img_only: bool = False,
    use_controlnet: bool = True,
    blueprint_mode: str = "lineart",
    run_note: str = "",
) -> dict:
    from bioguard_diffusion.agents.generator_agent_v4 import GeneratorAgentV4
    from bioguard_diffusion.agents.prompt_refiner_agent import PromptRefinerAgent

    generator = GeneratorAgentV4()
    refiner   = PromptRefinerAgent()

    print("Step 1 — RAG retrieval...")
    retrieved = biologist.retrieve(query, top_k=8)
    print(f"  Retrieved {len(retrieved)} constraints")

    # ── Spatial layout + initial blueprint ──────────────────────────────────
    layout = {"has_spatial_data": False, "layout": []}
    if use_controlnet and image_name:
        subject = query.split("of")[-1].strip() if "of" in query else query
        layout  = visual_planner.plan(image_name, subject)

    has_spatial = layout.get("has_spatial_data", False)
    current_blueprint = blueprint_gen.generate(layout, mode=blueprint_mode)
    current_blueprint.save(run_dir / "blueprint_initial.png")
    print(f"  Blueprint: {'spatial layout (' + str(len(layout.get('layout', []))) + ' regions)' if has_spatial else 'no spatial data (ControlNet disabled)'}")

    if text2img_only:
        print("  Mode: text2img-only")

    # ── State tracking (same as v3) ──────────────────────────────────────────
    best_image          = None
    best_combined       = -1.0
    best_clip           = 0.0
    best_clip_prompt    = None
    best_t5_prompt      = None
    violations          = []
    llm_feedback        = ""
    present_structures  = []
    missing_structures  = []
    refined_clip_prompt = None
    refined_t5_prompt   = None
    attempt_records     = []
    total_time          = 0.0
    next_gen_mode       = "text2img"
    next_strength       = None

    for attempt in range(1, max_retries + 1):
        attempt_seed = seed + (attempt - 1) * 7
        used_seed    = attempt_seed if (text2img_only or attempt == 1) else seed
        print(f"\nAttempt {attempt}/{max_retries}  (seed={used_seed})")

        arch        = architect.build_prompt(query, retrieved, violations=None)
        full_prompt = arch["prompt"]

        # ── Adaptive ControlNet scale ────────────────────────────────────────
        prev_llm = attempt_records[-1].get("llm_score", 5.0) if attempt_records else 5.0
        cn_scale = _controlnet_scale_for_attempt(prev_llm, attempt, has_spatial)

        # ── Generation ───────────────────────────────────────────────────────
        force_text2img = text2img_only or attempt == 1 or next_gen_mode == "text2img"

        if force_text2img:
            if attempt > 1:
                print(f"  [Adaptive] Restarting text2img (LLM score too low)")
            gen = generator.generate(
                full_prompt=full_prompt,
                control_image=current_blueprint,
                has_spatial_data=has_spatial,
                llm_feedback="",
                num_inference_steps=steps,
                guidance_scale=3.5,
                seed=attempt_seed,
                controlnet_scale=cn_scale,
            )
        else:
            gen = generator.refine(
                image=best_image,
                full_prompt=full_prompt,
                control_image=current_blueprint,
                has_spatial_data=has_spatial,
                llm_feedback="",
                attempt=attempt,
                num_inference_steps=steps,
                guidance_scale=3.5,
                seed=seed,
                clip_prompt_override=refined_clip_prompt,
                t5_prompt_override=refined_t5_prompt,
                strength=next_strength,
                controlnet_scale=cn_scale,
            )

        image      = gen["image"]
        mode       = gen["mode"]
        gen_time   = gen["gen_time_s"]
        total_time += gen_time

        img_path = run_dir / f"attempt_{attempt:02d}_{mode}_seed{used_seed}.png"
        image.save(img_path)
        print(f"  Saved: {img_path.name}  ({gen_time}s)  cn_scale={cn_scale:.2f}")

        # ── Verify ───────────────────────────────────────────────────────────
        if verifier_mode == VERIFIER_COMBINED:
            v    = verifier.verify_combined(image, query, constraints)
            clip = scorer.score(image, query)
            sas                = v["sas"]
            passed             = v["passed_combined"]
            violations         = v["violations"]
            llm_feedback       = v.get("improvement_suggestions", "")
            present_structures = v.get("present_structures", [])
            missing_structures = v.get("missing_structures", [])
            print(f"  SAS={sas:.4f} LLM={v['llm_score']:.1f}/10  CLIP={clip:.4f}  "
                  f"clip_pass={v['clip_passed']}  llm_pass={v['llm_passed']}")
        else:
            v    = verifier.verify(image, constraints)
            clip = scorer.score(image, query)
            sas                = v["sas"]
            passed             = v["passed"]
            violations         = v["violations"]
            llm_feedback       = v.get("feedback", "")
            present_structures = []
            missing_structures = []
            print(f"  SAS={sas:.4f}  CLIP={clip:.4f}  passed={passed}")

        if violations:
            print(f"  Violations: {violations[:2]}")

        # ── Score + Rollback ──────────────────────────────────────────────────
        llm_norm       = v.get("llm_score", 0) / 10.0
        combined_score = 0.5 * sas + 0.5 * llm_norm
        improved       = combined_score > best_combined

        if improved:
            best_combined    = combined_score
            best_image       = image
            best_clip        = clip
            best_clip_prompt = gen["clip_prompt"]
            best_t5_prompt   = gen["t5_prompt"]

        rollback_triggered = not improved and attempt > 1
        if rollback_triggered:
            print(f"  ↩ Rollback: {combined_score:.4f} < best {best_combined:.4f}")

        rec = {
            "attempt":          attempt,
            "seed":             used_seed,
            "mode":             mode,
            "gen_time_s":       gen_time,
            "sas":              round(sas, 4),
            "clip_score":       round(clip, 4),
            "passed":           passed,
            "violations":       violations,
            "full_prompt":      full_prompt,
            "clip_prompt":      gen["clip_prompt"],
            "t5_prompt":        gen["t5_prompt"],
            "image_file":       img_path.name,
            "combined_score":   round(combined_score, 4),
            "improved":         improved,
            "rollback":         rollback_triggered,
            "controlnet_scale": cn_scale,
            "has_spatial_data": has_spatial,
        }
        if verifier_mode == VERIFIER_COMBINED:
            rec["llm_score"]               = v["llm_score"]
            rec["present_structures"]      = present_structures
            rec["missing_structures"]      = missing_structures
            rec["improvement_suggestions"] = llm_feedback
        attempt_records.append(rec)

        # ── Break condition ───────────────────────────────────────────────────
        fully_passed = passed and not missing_structures
        if fully_passed:
            print(f"  ✓ Fully passed on attempt {attempt}")
            break
        elif passed and missing_structures:
            print(f"  ~ Scores passed but {len(missing_structures)} structure(s) missing: "
                  f"{missing_structures[:3]}")

        if attempt >= max_retries:
            break

        if not passed:
            print(f"  ✗ Failed — {max_retries - attempt} retry(s) remaining")

        # ── Adaptive strength for next attempt ────────────────────────────────
        next_strength, next_gen_mode = _adaptive_strength(
            llm_score     = v.get("llm_score", 0.0),
            sas           = sas,
            verifier_mode = verifier_mode,
        )
        print(f"  [Adaptive] next={next_gen_mode}  "
              f"strength={next_strength if next_strength else 'N/A'}")

        # ── Adaptive blueprint: highlight missing structures ───────────────────
        if has_spatial and missing_structures:
            current_blueprint = blueprint_gen.highlight_missing(
                layout         = layout,
                missing_names  = missing_structures,
            )
            bp_path = run_dir / f"blueprint_att{attempt + 1:02d}.png"
            current_blueprint.save(bp_path)
            print(f"  [Blueprint] Updated — {len(missing_structures)} missing structures highlighted → {bp_path.name}")
        else:
            bp_path = None

        # ── PromptRefiner ─────────────────────────────────────────────────────
        base_clip = best_clip_prompt
        base_t5   = best_t5_prompt
        print("  Refining CLIP+T5 prompts...")
        refined = refiner.refine(
            clip_prompt             = base_clip,
            t5_prompt               = base_t5,
            present_structures      = present_structures,
            missing_structures      = missing_structures,
            violations              = violations,
            improvement_suggestions = llm_feedback,
        )
        refined_clip_prompt = refined["refined_clip_prompt"]
        refined_t5_prompt   = refined["refined_t5_prompt"]

        rec["refined_clip_for_next"] = refined_clip_prompt
        rec["refined_t5_for_next"]   = refined_t5_prompt
        rec["refiner_changes"]       = refined["changes_made"]
        rec["next_strength"]         = next_strength
        rec["next_gen_mode"]         = next_gen_mode
        rec["blueprint_updated"]     = bp_path is not None
        if bp_path:
            rec["blueprint_file"]    = bp_path.name

        print(f"  Changes: {refined['changes_made']}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    best_path = run_dir / "best.png"
    best_image.save(best_path)
    generator.write_prompt_log(run_dir, query, constraints)

    return {
        "backend":          "flux_v4_controlnet",
        "query":            query,
        "constraints":      constraints,
        "retrieved":        retrieved,
        "image_name":       image_name,
        "has_spatial_data": has_spatial,
        "layout_regions":   len(layout.get("layout", [])),
        "attempts":         attempt_records,
        "best_clip":        round(best_clip, 4),
        "total_gen_time_s": round(total_time, 1),
        "attempts_used":    len(attempt_records),
        "passed":           attempt_records[-1]["passed"],
        "fully_passed":     fully_passed,
        "final_missing":    missing_structures,
    }


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BioGuard-Diffusion v4 — ControlNet spatial-guided runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query-id",       type=str, help="Test query ID (q01–q10)")
    parser.add_argument("--prompt",         type=str, help="Custom prompt text")
    parser.add_argument("--verifier",       choices=[VERIFIER_CLIP, VERIFIER_COMBINED],
                        default=None)
    parser.add_argument("--max-retries",    type=int, default=None)
    parser.add_argument("--steps",          type=int, default=None)
    parser.add_argument("--seed",           type=int, default=None)
    parser.add_argument("--text2img-only",  action="store_true")
    parser.add_argument("--no-controlnet",  action="store_true",
                        help="Disable ControlNet (compare against v3)")
    parser.add_argument("--blueprint-mode", choices=["lineart", "segmentation"],
                        default="lineart")
    parser.add_argument("--controlnet-scale", type=float, default=None,
                        help="Override adaptive ControlNet scale (0.0–1.0)")
    parser.add_argument("--note",           type=str, default="",
                        help="Free-text note saved in summary.json")
    args = parser.parse_args()

    queries = _load_json(QUERIES_FILE)
    kb      = _load_json(KB_FILE)

    if args.query_id or args.prompt:
        if args.prompt:
            query       = args.prompt
            constraints = []
            image_name  = None
            label       = "custom"
        else:
            qid   = args.query_id.lower()
            match = next((q for q in queries if q["id"] == qid), None)
            if not match:
                print(f"Unknown query ID '{qid}'. Available: {[q['id'] for q in queries]}")
                sys.exit(1)
            query       = match["query"]
            image_name  = match["image"]
            constraints = _get_constraints(kb, image_name)
            label       = f"{match['id']}_{match['domain']}"
    else:
        # Interactive
        print("\n" + "=" * 60)
        print("BioGuard-Diffusion v4 — Interactive Mode")
        print("=" * 60)
        for q in queries:
            print(f"  [{q['id']}] {q['domain']:<18} {q['query'][:50]}...")
        print("  [custom] Custom prompt")
        while True:
            choice = input("\nQuery ID or 'custom': ").strip().lower()
            if choice == "custom":
                query      = input("Prompt: ").strip()
                label      = "custom"
                image_name = None
                constraints = []
                break
            match = next((q for q in queries if q["id"] == choice), None)
            if match:
                query       = match["query"]
                image_name  = match["image"]
                constraints = _get_constraints(kb, image_name)
                label       = f"{match['id']}_{match['domain']}"
                break
            print("  Invalid.")

        vc = input("Verifier [1=combined(default) / 2=clip-only]: ").strip()
        verifier = VERIFIER_CLIP if vc == "2" else VERIFIER_COMBINED
        mr = input(f"Max retries [{DEFAULT_MAX_RETRIES}]: ").strip()
        max_retries = int(mr) if mr.isdigit() else DEFAULT_MAX_RETRIES
        st = input(f"Steps [{DEFAULT_STEPS}]: ").strip()
        steps = int(st) if st.isdigit() else DEFAULT_STEPS
        sd = input(f"Seed [{DEFAULT_SEED}]: ").strip()
        seed = int(sd) if sd.isdigit() else DEFAULT_SEED
        args.verifier      = verifier
        args.max_retries   = max_retries
        args.steps         = steps
        args.seed          = seed
        args.text2img_only = False
        args.no_controlnet = False
        args.blueprint_mode = "lineart"
        args.controlnet_scale = None
        args.note = ""

    verifier    = args.verifier    or DEFAULT_VERIFIER
    max_retries = args.max_retries if args.max_retries is not None else DEFAULT_MAX_RETRIES
    steps       = args.steps       if args.steps       is not None else DEFAULT_STEPS
    seed        = args.seed        if args.seed        is not None else DEFAULT_SEED

    run_dir = _make_run_dir(f"v4_{label}")

    print("\n" + "=" * 70)
    print("BioGuard-Diffusion v4 — ControlNet Spatial Runner")
    print("=" * 70)
    print(f"Query:          {query}")
    print(f"Image:          {image_name or 'N/A (custom prompt)'}")
    print(f"Verifier:       {verifier}")
    print(f"Max retries:    {max_retries}")
    print(f"Steps:          {steps}")
    print(f"ControlNet:     {'DISABLED' if args.no_controlnet else args.blueprint_mode.upper()}")
    if args.note:
        print(f"Note:           {args.note}")
    print(f"Output dir:     {run_dir}")
    print("=" * 70 + "\n")

    print("Loading agents...")
    biologist      = BiologistAgent()
    architect      = SpatialArchitectAgent()
    verifier_agent = VerifierAgent()
    scorer         = CLIPScorer()
    visual_planner = VisualPlannerAgent(ANNOTATIONS_DIR, IMAGES_DIR)
    blueprint_gen  = BlueprintGenerator()
    print("Agents ready.\n")

    run_start = time.time()
    result = _run_flux_v4(
        query          = query,
        constraints    = constraints,
        image_name     = image_name,
        run_dir        = run_dir,
        max_retries    = max_retries,
        steps          = steps,
        seed           = seed,
        verifier_mode  = verifier,
        biologist      = biologist,
        architect      = architect,
        verifier       = verifier_agent,
        scorer         = scorer,
        visual_planner = visual_planner,
        blueprint_gen  = blueprint_gen,
        text2img_only  = args.text2img_only,
        use_controlnet = not args.no_controlnet,
        blueprint_mode = args.blueprint_mode,
        run_note       = args.note,
    )
    wall_time = round(time.time() - run_start, 1)

    summary = {
        "run_dir":        str(run_dir),
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "wall_time_s":    wall_time,
        "verifier_mode":  verifier,
        "pipeline":       PIPELINE_VERSION_V4,
        "run_note":       args.note,
        "controlnet": {
            "enabled":       not args.no_controlnet,
            "blueprint_mode": args.blueprint_mode,
            "has_spatial_data": result.get("has_spatial_data", False),
            "layout_regions": result.get("layout_regions", 0),
        },
        "params": {
            "max_retries":   max_retries,
            "steps":         steps,
            "seed":          seed,
            "text2img_only": args.text2img_only,
        },
        **result,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Run complete")
    print("=" * 70)
    print(f"Attempts:       {result['attempts_used']}/{max_retries}")
    print(f"Passed:         {result['passed']}")
    print(f"Best CLIP:      {result['best_clip']:.4f}")
    print(f"Spatial data:   {result['has_spatial_data']} ({result['layout_regions']} regions)")
    print(f"Total gen time: {result['total_gen_time_s']}s")
    print(f"Wall time:      {wall_time}s")
    print()
    for p in sorted(run_dir.iterdir()):
        print(f"  {p.name:<45} {p.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    main()
