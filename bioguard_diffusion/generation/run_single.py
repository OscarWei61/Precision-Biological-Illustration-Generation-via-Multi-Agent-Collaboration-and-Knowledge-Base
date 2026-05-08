"""
BioGuard-Diffusion — Single Run Script
=======================================
Interactive or CLI-driven single query runner.

Usage examples:
  # Interactive menu
  python bioguard_diffusion/generation/run_single.py

  # Pick a test case
  python bioguard_diffusion/generation/run_single.py --query-id q03

  # Custom prompt
  python bioguard_diffusion/generation/run_single.py --prompt "Draw a labeled diagram of a chloroplast"

  # FLUX backend
  python bioguard_diffusion/generation/run_single.py --query-id q07 --backend flux

  # More retries, fewer steps (faster)
  python bioguard_diffusion/generation/run_single.py --query-id q02 --max-retries 3 --steps 20

  # Use combined CLIP+LLM verifier (slower, richer feedback)
  python bioguard_diffusion/generation/run_single.py --query-id q04 --verifier combined

Each run saves to: outputs/single_runs/{timestamp}_{label}/
  attempt_01_{mode}.png        — image from each attempt
  summary.json                 — full run metadata
  prompt_log.txt               — all prompts used

All attempt images are saved so you can compare the retry evolution.
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

# ── Load env vars from .env if present ─────────────────────────────────────
from dotenv import load_dotenv as _load_dotenv
try:
    _load_dotenv(BASE / ".env")
except Exception:
    pass

# ── LangSmith: enable if API key is set ────────────────────────────────────
_LANGSMITH_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
_LANGSMITH_ENABLED = bool(_LANGSMITH_KEY and _LANGSMITH_KEY != "YOUR_LANGSMITH_API_KEY_HERE")

try:
    from langsmith import traceable
    if not _LANGSMITH_ENABLED:
        # Silently disable tracing — decorator becomes no-op
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
except ImportError:
    def traceable(**kwargs):
        def decorator(fn): return fn
        return decorator

# ── Project imports ─────────────────────────────────────────────────────────
from bioguard_diffusion.agents.biologist_agent   import BiologistAgent
from bioguard_diffusion.agents.spatial_architect  import SpatialArchitectAgent
from bioguard_diffusion.agents.verifier_agent     import VerifierAgent
from bioguard_diffusion.evaluation.clip_score     import CLIPScorer

QUERIES_FILE = BASE / "results" / "test_queries.json"
KB_FILE      = BASE / "results" / "knowledge_base.json"
OUT_ROOT     = BASE / "outputs" / "single_runs"

# ── Verifier modes ──────────────────────────────────────────────────────────
VERIFIER_CLIP     = "clip"      # fast: CLIP-SAS only
VERIFIER_COMBINED = "combined"  # CLIP-SAS + LLM-as-judge (claude-haiku)

# ── Default params ──────────────────────────────────────────────────────────
DEFAULT_STEPS      = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_SEED        = 42
DEFAULT_BACKEND    = "flux" # "sd" | "flux"
DEFAULT_VERIFIER   = VERIFIER_COMBINED

# img2img strength constants are now managed inside GeneratorAgent

# ── Pipeline version — updated whenever key behaviour changes ───────────────
PIPELINE_VERSION = {
    "version": "v3",
    "changes": [
        "rollback: always refine from best-scoring prompts, not latest attempt",
        "adaptive_strength: img2img strength driven by LLM score (0.28/0.55/0.75) or text2img restart if score<3",
        "combined_score: equal-weight 0.5*SAS + 0.5*(LLM/10)",
        "prompt_refiner: CLIP≤55w / T5≤150w hard limits, direct CLIP+T5 output (no PromptSplitter re-compression)",
        "seed: fixed for img2img refinement, varies per attempt only for text2img",
        "no_labels: all prompt templates strip text/label generation instructions",
    ],
}


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _load_test_queries() -> list[dict]:
    with open(QUERIES_FILE) as f:
        return json.load(f)


def _load_kb() -> list[dict]:
    with open(KB_FILE) as f:
        return json.load(f)


def _get_constraints(kb: list[dict], image_name: str) -> list[str]:
    for entry in kb:
        if entry["image"] == image_name:
            return entry.get("constraints", [])
    return []


def _make_run_dir(label: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = label.replace(" ", "_")[:40]
    run_dir = OUT_ROOT / f"{ts}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=False)   # fail loudly if collision
    return run_dir


def _print_header(query: str, backend: str, verifier: str, max_retries: int):
    print("\n" + "=" * 70)
    print("BioGuard-Diffusion — Single Run")
    print("=" * 70)
    print(f"Query:     {query}")
    print(f"Backend:   {backend.upper()}")
    print(f"Verifier:  {verifier}")
    print(f"Max retries: {max_retries}")
    if _LANGSMITH_ENABLED:
        project = os.environ.get("LANGCHAIN_PROJECT", "bioguard-diffusion")
        print(f"LangSmith: ENABLED (project: {project})")
    else:
        print("LangSmith: disabled (set LANGCHAIN_API_KEY in .env to enable)")
    print("=" * 70 + "\n")


def _interactive_menu() -> tuple[str, str, str, str, int, int, int, bool]:
    """
    Returns (query_str, label, backend, verifier, max_retries, steps, seed, text2img_only)
    """
    queries = _load_test_queries()

    print("\n" + "=" * 60)
    print("BioGuard-Diffusion — Interactive Mode")
    print("=" * 60)
    print("\nTest queries:")
    for q in queries:
        print(f"  [{q['id']}] {q['domain']:<18} {q['query'][:52]}...")
    print(f"  [custom]  Enter a custom prompt")
    print()

    while True:
        choice = input("Select query ID (e.g. q03) or 'custom': ").strip().lower()
        if choice == "custom":
            query = input("Enter prompt: ").strip()
            if query:
                label = "custom"
                break
        else:
            match = next((q for q in queries if q["id"] == choice), None)
            if match:
                query = match["query"]
                label = f"{match['id']}_{match['domain']}"
                break
        print("  Invalid — try again.")

    print("\nBackend:")
    print("  [1] FLUX.1-dev GGUF  (slow, ~430s/image, better quality)")
    print("  [2] SD-v1.5  (fast, ~30s/image)")
    bc = input("Select [1/2, default=1]: ").strip()
    backend = "sd" if bc == "2" else "flux"

    print("\nVerifier:")
    print("  [1] CLIP-SAS + LLM-judge  (default, richer feedback)")
    print("  [2] CLIP-SAS only  (fast, no API call)")
    vc = input("Select [1/2, default=1]: ").strip()
    verifier = VERIFIER_CLIP if vc == "2" else VERIFIER_COMBINED

    mr = input(f"\nMax retries [default={DEFAULT_MAX_RETRIES}]: ").strip()
    max_retries = int(mr) if mr.isdigit() else DEFAULT_MAX_RETRIES

    st = input(f"Inference steps [default={DEFAULT_STEPS}]: ").strip()
    steps = int(st) if st.isdigit() else DEFAULT_STEPS

    sd = input(f"Seed [default={DEFAULT_SEED}]: ").strip()
    seed = int(sd) if sd.isdigit() else DEFAULT_SEED

    t2i = input("\nText2img-only mode (skip img2img refinement)? [y/N]: ").strip().lower()
    text2img_only = t2i == "y"

    return query, label, backend, verifier, max_retries, steps, seed, text2img_only


# ───────────────────────────────────────────────────────────────────────────
# SD Pipeline Runner
# ───────────────────────────────────────────────────────────────────────────

@traceable(name="bioguard_pipeline_sd", run_type="chain")
def _run_sd(
    query: str,
    constraints: list[str],
    run_dir: Path,
    max_retries: int,
    steps: int,
    seed_base: int,
    verifier_mode: str,
    biologist: BiologistAgent,
    architect: SpatialArchitectAgent,
    verifier: VerifierAgent,
    scorer: CLIPScorer,
) -> dict:
    from bioguard_diffusion.generation.diffusion_pipeline import BioGenerationPipeline

    pipe = BioGenerationPipeline()

    prompt_log_lines = [f"Query: {query}\n", f"Constraints ({len(constraints)}):\n"]
    for c in constraints:
        prompt_log_lines.append(f"  - {c}\n")
    prompt_log_lines.append("\n")

    # Step 1: RAG retrieval
    print("Step 1 — RAG retrieval...")
    retrieved = biologist.retrieve(query, top_k=8)
    print(f"  Retrieved {len(retrieved)} constraints from vector store")

    best_image      = None
    best_sas        = -1.0
    best_clip       = 0.0
    violations      = []
    attempt_records = []
    total_time      = 0.0

    for attempt in range(1, max_retries + 1):
        seed = seed_base + (attempt - 1) * 7
        print(f"\nAttempt {attempt}/{max_retries}  (seed={seed})")

        # Step 2: Build prompt
        arch = architect.build_prompt(
            query, retrieved,
            violations=violations if attempt > 1 else None,
        )
        prompt     = arch["prompt"]
        neg_prompt = arch["negative_prompt"]

        prompt_log_lines.append(f"--- Attempt {attempt} ---\n")
        prompt_log_lines.append(f"Seed: {seed}\n")
        prompt_log_lines.append(f"Prompt: {prompt}\n")
        prompt_log_lines.append(f"Negative: {neg_prompt}\n\n")

        print(f"  Prompt ({len(prompt)} chars): {prompt[:80]}...")

        # Step 3: Generate
        t0 = time.time()
        image = pipe.generate(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            seed=seed,
        )
        gen_time = round(time.time() - t0, 1)
        total_time += gen_time

        # Save attempt image
        img_path = run_dir / f"attempt_{attempt:02d}_sd_seed{seed}.png"
        image.save(img_path)
        print(f"  Saved: {img_path.name}  ({gen_time}s)")

        # Step 4: Verify
        if verifier_mode == VERIFIER_COMBINED:
            v = verifier.verify_combined(image, query, constraints)
            clip = scorer.score(image, query)
            sas        = v["sas"]
            passed     = v["passed_combined"]
            violations = v["violations"]
            llm_score  = v["llm_score"]
            feedback   = v.get("improvement_suggestions", "")
            print(f"  SAS={sas:.4f} LLM={llm_score:.1f}/10  CLIP={clip:.4f}  "
                  f"clip_pass={v['clip_passed']}  llm_pass={v['llm_passed']}")
        else:
            v = verifier.verify(image, constraints)
            clip       = scorer.score(image, query)
            sas        = v["sas"]
            passed     = v["passed"]
            violations = v["violations"]
            feedback   = v.get("feedback", "")
            print(f"  SAS={sas:.4f}  CLIP={clip:.4f}  passed={passed}")

        if violations:
            print(f"  Violations: {violations[:2]}")

        if sas > best_sas:
            best_sas   = sas
            best_image = image
            best_clip  = clip

        rec = {
            "attempt":       attempt,
            "seed":          seed,
            "mode":          "text2img",
            "gen_time_s":    gen_time,
            "sas":           round(sas, 4),
            "clip_score":    round(clip, 4),
            "passed":        passed,
            "violations":    violations,
            "prompt":        prompt,
            "image_file":    img_path.name,
        }
        if verifier_mode == VERIFIER_COMBINED:
            rec["llm_score"]             = v["llm_score"]
            rec["present_structures"]    = v.get("present_structures", [])
            rec["missing_structures"]    = v.get("missing_structures", [])
            rec["improvement_suggestions"] = feedback
        attempt_records.append(rec)

        if passed:
            print(f"  ✓ Passed on attempt {attempt}")
            break
        if attempt < max_retries:
            print(f"  ✗ Failed — {max_retries - attempt} retry(s) remaining")
            if feedback:
                print(f"  Feedback: {feedback[:80]}...")

    # Save best image separately
    best_path = run_dir / "best.png"
    best_image.save(best_path)

    # Write prompt log
    with open(run_dir / "prompt_log.txt", "w") as f:
        f.writelines(prompt_log_lines)

    return {
        "backend":         "sd",
        "query":           query,
        "constraints":     constraints,
        "retrieved":       retrieved,
        "attempts":        attempt_records,
        "best_sas":        round(best_sas, 4),
        "best_clip":       round(best_clip, 4),
        "total_gen_time_s": round(total_time, 1),
        "attempts_used":   len(attempt_records),
        "passed":          attempt_records[-1]["passed"],
    }


# ───────────────────────────────────────────────────────────────────────────
# Adaptive strength + rollback helpers
# ───────────────────────────────────────────────────────────────────────────

def _adaptive_strength(
    llm_score: float,
    sas: float,
    verifier_mode: str,
) -> tuple[float | None, str]:
    """
    Returns (strength, gen_mode):
      gen_mode = 'text2img' : restart fresh (strength ignored)
      gen_mode = 'img2img'  : refine from best_image with returned strength

    Combined mode thresholds (LLM score 0–10):
      < 3.0  → restart text2img  (completely wrong subject/structure)
      3–5    → strength 0.75     (far from target, allow large change)
      5–6.5  → strength 0.55     (approaching, moderate change)
      ≥ 6.5  → strength 0.28     (near/above threshold, fine-tune only)

    CLIP-only mode thresholds (SAS):
      < 0.20 → strength 0.75
      0.20–0.24 → strength 0.55
      ≥ 0.24 → strength 0.30
    """
    if verifier_mode == VERIFIER_COMBINED:
        if llm_score < 3.0:
            return None, "text2img"
        elif llm_score < 5.0:
            return 0.75, "img2img"
        elif llm_score < 6.5:
            return 0.55, "img2img"
        else:
            return 0.28, "img2img"
    else:
        if sas < 0.20:
            return 0.75, "img2img"
        elif sas < 0.24:
            return 0.55, "img2img"
        else:
            return 0.30, "img2img"


# ───────────────────────────────────────────────────────────────────────────
# FLUX Pipeline Runner
# ───────────────────────────────────────────────────────────────────────────

@traceable(name="bioguard_pipeline_flux", run_type="chain")
def _run_flux(
    query: str,
    constraints: list[str],
    run_dir: Path,
    max_retries: int,
    steps: int,
    seed: int,
    verifier_mode: str,
    biologist: BiologistAgent,
    architect: SpatialArchitectAgent,
    verifier: VerifierAgent,
    scorer: CLIPScorer,
    text2img_only: bool = False,
) -> dict:
    from bioguard_diffusion.agents.generator_agent      import GeneratorAgent
    from bioguard_diffusion.agents.prompt_refiner_agent import PromptRefinerAgent

    generator = GeneratorAgent()
    refiner   = PromptRefinerAgent()

    print("Step 1 — RAG retrieval...")
    retrieved = biologist.retrieve(query, top_k=8)
    print(f"  Retrieved {len(retrieved)} constraints")
    if text2img_only:
        print("  Mode: text2img-only (no img2img refinement, seed varies per attempt)")

    # ── State tracking ────────────────────────────────────────────────────────
    best_image          = None
    best_combined       = -1.0
    best_clip           = 0.0
    # Best CLIP+T5 prompts (rollback target if a retry regresses)
    best_clip_prompt    = None
    best_t5_prompt      = None
    violations          = []
    llm_feedback        = ""
    present_structures  = []
    missing_structures  = []
    # PromptRefiner output for next attempt (always based on best prompts)
    refined_clip_prompt = None
    refined_t5_prompt   = None
    attempt_records     = []
    total_time          = 0.0
    # Next-attempt generation mode (may be overridden to text2img by adaptive strength)
    next_gen_mode       = "text2img"   # always text2img on attempt 1
    next_strength       = None         # set by adaptive strength after each verify

    for attempt in range(1, max_retries + 1):
        attempt_seed = seed + (attempt - 1) * 7
        used_seed    = attempt_seed if (text2img_only or attempt == 1) else seed
        print(f"\nAttempt {attempt}/{max_retries}  (seed={used_seed})")

        arch        = architect.build_prompt(query, retrieved, violations=None)
        full_prompt = arch["prompt"]

        # ── Generation ───────────────────────────────────────────────────────
        force_text2img = text2img_only or attempt == 1 or next_gen_mode == "text2img"

        if force_text2img:
            if attempt > 1:
                print(f"  [Adaptive] Restarting text2img (LLM score too low for img2img)")
            gen = generator.generate(
                full_prompt=full_prompt,
                llm_feedback="",
                num_inference_steps=steps,
                guidance_scale=3.5,
                seed=attempt_seed,   # fresh seed on text2img restart for diversity
            )
        else:
            gen = generator.refine(
                image=best_image,
                full_prompt=full_prompt,
                llm_feedback="",
                attempt=attempt,
                num_inference_steps=steps,
                guidance_scale=3.5,
                seed=seed,
                clip_prompt_override=refined_clip_prompt,
                t5_prompt_override=refined_t5_prompt,
                strength=next_strength,   # adaptive strength from previous verify
            )

        image      = gen["image"]
        mode       = gen["mode"]
        gen_time   = gen["gen_time_s"]
        total_time += gen_time

        img_path  = run_dir / f"attempt_{attempt:02d}_{mode}_seed{used_seed}.png"
        image.save(img_path)
        print(f"  Saved: {img_path.name}  ({gen_time}s)")

        # Verify
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

        # ── Score + Rollback tracking ─────────────────────────────────────────
        # Equal-weight SAS (0–1) and LLM score (0–10 → 0–1) for fair comparison
        llm_norm      = v.get("llm_score", 0) / 10.0
        combined_score = 0.5 * sas + 0.5 * llm_norm
        improved       = combined_score > best_combined

        if improved:
            best_combined     = combined_score
            best_image        = image
            best_clip         = clip
            best_clip_prompt  = gen["clip_prompt"]
            best_t5_prompt    = gen["t5_prompt"]

        rollback_triggered = not improved and attempt > 1
        if rollback_triggered:
            print(f"  ↩ Rollback: score {combined_score:.4f} < best {best_combined:.4f} "
                  f"— will refine from best prompts next attempt")

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
            print(f"  ✓ Fully passed on attempt {attempt} — no missing structures")
            break
        elif passed and missing_structures:
            print(f"  ~ Scores passed but {len(missing_structures)} structure(s) still missing: "
                  f"{missing_structures[:3]}")

        if attempt < max_retries:
            if not passed:
                print(f"  ✗ Failed — {max_retries - attempt} retry(s) remaining")

            # ── Adaptive strength for next attempt ────────────────────────────
            next_strength, next_gen_mode = _adaptive_strength(
                llm_score     = v.get("llm_score", 0.0),
                sas           = sas,
                verifier_mode = verifier_mode,
            )
            print(f"  [Adaptive] next={next_gen_mode}  "
                  f"strength={next_strength if next_strength else 'N/A'}")

            # ── PromptRefiner: base on BEST prompts (rollback if regressed) ───
            base_clip = best_clip_prompt   # always refine from best, not latest
            base_t5   = best_t5_prompt
            print("  Refining CLIP+T5 prompts based on verifier feedback...")
            refined = refiner.refine(
                clip_prompt            = base_clip,
                t5_prompt              = base_t5,
                present_structures     = present_structures,
                missing_structures     = missing_structures,
                violations             = violations,
                improvement_suggestions= llm_feedback,
            )
            refined_clip_prompt = refined["refined_clip_prompt"]
            refined_t5_prompt   = refined["refined_t5_prompt"]
            rec["refined_clip_for_next"] = refined_clip_prompt
            rec["refined_t5_for_next"]   = refined_t5_prompt
            rec["refiner_changes"]       = refined["changes_made"]
            rec["next_strength"]         = next_strength
            rec["next_gen_mode"]         = next_gen_mode
            print(f"  Changes: {refined['changes_made']}")
            print(f"  Refined CLIP: {refined_clip_prompt[:70]}...")
            print(f"  Refined T5:   {refined_t5_prompt[:90]}...")

    best_path = run_dir / "best.png"
    best_image.save(best_path)

    # prompt_log.txt written by GeneratorAgent (includes full_prompt + CLIP + T5 + feedback)
    generator.write_prompt_log(run_dir, query, constraints)

    return {
        "backend":          "flux",
        "query":            query,
        "constraints":      constraints,
        "retrieved":        retrieved,
        "attempts":         attempt_records,
        "best_clip":        round(best_clip, 4),
        "total_gen_time_s": round(total_time, 1),
        "attempts_used":    len(attempt_records),
        "passed":           attempt_records[-1]["passed"],
        "fully_passed":     fully_passed,
        "final_missing":    missing_structures,
    }


# ───────────────────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BioGuard-Diffusion single-query runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query-id",    type=str, help="Test query ID (q01–q10)")
    parser.add_argument("--prompt",      type=str, help="Custom prompt text")
    parser.add_argument("--backend",     choices=["sd", "flux"], default=None,
                        help="Diffusion backend: sd (default) or flux")
    parser.add_argument("--verifier",    choices=[VERIFIER_CLIP, VERIFIER_COMBINED],
                        default=None, help="Verifier mode (default: clip)")
    parser.add_argument("--max-retries",    type=int, default=None)
    parser.add_argument("--steps",          type=int, default=None)
    parser.add_argument("--seed",           type=int, default=None)
    parser.add_argument("--text2img-only",  action="store_true",
                        help="Flux: skip img2img refinement, generate fresh each attempt")
    parser.add_argument("--note", type=str, default="",
                        help="Free-text note saved into summary.json for comparison tracking")
    args = parser.parse_args()

    queries = _load_test_queries()
    kb      = _load_kb()

    # ── Resolve query and settings ──────────────────────────────────────────
    if args.query_id or args.prompt:
        # Non-interactive
        if args.prompt:
            query       = args.prompt
            constraints = []
            label       = "custom"
        else:
            qid   = args.query_id.lower()
            match = next((q for q in queries if q["id"] == qid), None)
            if not match:
                print(f"Unknown query ID '{qid}'. Available: {[q['id'] for q in queries]}")
                sys.exit(1)
            query       = match["query"]
            constraints = _get_constraints(kb, match["image"])
            label       = f"{match['id']}_{match['domain']}"

        backend        = args.backend   or DEFAULT_BACKEND
        verifier       = args.verifier  or DEFAULT_VERIFIER
        max_retries    = args.max_retries if args.max_retries is not None else DEFAULT_MAX_RETRIES
        steps          = args.steps       if args.steps       is not None else DEFAULT_STEPS
        seed           = args.seed        if args.seed        is not None else DEFAULT_SEED
        text2img_only  = args.text2img_only
        run_note       = args.note
    else:
        # Interactive
        query, label, backend, verifier, max_retries, steps, seed, text2img_only = _interactive_menu()
        # Resolve constraints for test queries
        match = next((q for q in queries if q["query"] == query), None)
        constraints = _get_constraints(kb, match["image"]) if match else []
        run_note = ""

    run_dir = _make_run_dir(f"{backend}_{label}")
    _print_header(query, backend, verifier, max_retries)
    print(f"Output dir: {run_dir}\n")

    # ── Load shared agents ──────────────────────────────────────────────────
    print("Loading agents...")
    biologist = BiologistAgent()
    architect = SpatialArchitectAgent()
    verifier_agent = VerifierAgent()
    scorer = CLIPScorer()
    print("Agents ready.\n")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    run_start = time.time()

    if backend == "flux":
        result = _run_flux(
            query=query,
            constraints=constraints,
            run_dir=run_dir,
            max_retries=max_retries,
            steps=steps,
            seed=seed,
            verifier_mode=verifier,
            biologist=biologist,
            architect=architect,
            verifier=verifier_agent,
            scorer=scorer,
            text2img_only=text2img_only,
        )
    else:
        result = _run_sd(
            query=query,
            constraints=constraints,
            run_dir=run_dir,
            max_retries=max_retries,
            steps=steps,
            seed_base=seed,
            verifier_mode=verifier,
            biologist=biologist,
            architect=architect,
            verifier=verifier_agent,
            scorer=scorer,
        )

    wall_time = round(time.time() - run_start, 1)

    # ── Save summary JSON ─────────────────────────────────────────────────
    summary = {
        "run_dir":        str(run_dir),
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "wall_time_s":    wall_time,
        "verifier_mode":  verifier,
        "pipeline":       PIPELINE_VERSION,
        "run_note":       run_note,
        "params": {
            "max_retries":   max_retries,
            "steps":         steps,
            "seed":          seed,
            "text2img_only": text2img_only if backend == "flux" else None,
        },
        "langsmith_enabled": _LANGSMITH_ENABLED,
        **result,
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Run complete")
    print("=" * 70)
    print(f"Attempts used:  {result['attempts_used']}/{max_retries}")
    print(f"Final passed:   {result['passed']}")
    if backend == "flux":
        print(f"Best CLIP:      {result['best_clip']:.4f}")
    else:
        print(f"Best CLIP:      {result['best_clip']:.4f}")
        print(f"Best SAS:       {result.get('best_sas', 'n/a')}")
    print(f"Total gen time: {result['total_gen_time_s']}s")
    print(f"Wall time:      {wall_time}s")
    print()
    print("Saved files:")
    for p in sorted(run_dir.iterdir()):
        size = p.stat().st_size
        print(f"  {p.name:<45} {size:>8,} bytes")
    if _LANGSMITH_ENABLED:
        project = os.environ.get("LANGCHAIN_PROJECT", "bioguard-diffusion")
        print(f"\nLangSmith trace → https://smith.langchain.com  (project: {project})")


if __name__ == "__main__":
    main()
