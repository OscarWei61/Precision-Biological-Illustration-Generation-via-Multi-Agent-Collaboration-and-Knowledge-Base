"""
Config 3 (FLUX variant v2) — BioGuard-Diffusion with FLUX.1-dev GGUF
Pipeline: Biologist → Spatial Architect → PromptSplitter (CLIP+T5)
          → FLUX text2img → VerifierCombined (CLIP-SAS + LLM-judge)
          → [img2img retry ≤3: LLM feedback injected into T5 prompt]

Retry strategy:
  - Attempt 1: text2img, CLIP prompt (≤77 tok) + T5 prompt (full detail)
  - Attempt 2+: img2img, T5 prompt augmented with LLM improvement_suggestions
    * strength 0.60 → 0.50 → 0.40 (progressive conservation)
    * same seed (deterministic refinement chain)
  - Pass condition: CLIP-SAS ≥ 0.20 AND LLM score ≥ 6.0

VERSION stamps all output dirs and JSON files for experiment tracking.

Run: ANTHROPIC_API_KEY=sk-ant-... conda run -n agent python bioguard_diffusion/generation/run_config3_flux.py
"""

import json
import os
import time
import datetime
import sys
from pathlib import Path

# ── API Keys ──────────────────────────────────────────────────────────────────
from huggingface_hub import login as hf_login

os.environ.setdefault("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from bioguard_diffusion.agents.biologist_agent       import BiologistAgent
from bioguard_diffusion.agents.spatial_architect      import SpatialArchitectAgent
from bioguard_diffusion.agents.verifier_agent         import VerifierAgent
from bioguard_diffusion.agents.prompt_splitter        import PromptSplitter
from bioguard_diffusion.generation.flux_gguf_pipeline import FluxGGUFPipeline
from bioguard_diffusion.evaluation.clip_score         import CLIPScorer

# ── Version tag — stamps all output paths and JSON records ────────────────────
VERSION = "v2_llm_judge"

QUERIES_FILE = BASE / "results" / "test_queries.json"
KB_FILE      = BASE / "results" / "knowledge_base.json"
C1_FILE      = BASE / "results" / "config1_results.json"
C3_FILE      = BASE / "results" / "config3_results.json"
OUT_DIR      = BASE / "outputs" / f"config3_flux_{VERSION}"
RESULTS_FILE = BASE / "results" / f"config3_flux_{VERSION}_results.json"
VERIFY_LOG   = BASE / "results" / f"config3_flux_{VERSION}_verify_log.json"

# MAX_RETRIES = total loop iterations (1 text2img + 2 img2img retries = 3)
# The feedback loop runs at most MAX_RETRIES-1 = 2 img2img retry cycles.
MAX_RETRIES      = 3
SAS_THRESHOLD    = 0.20
IMG2IMG_STRENGTH = 0.60
STRENGTH_DECAY   = 0.10


def _get_constraints(kb: list[dict], image_name: str) -> list[str]:
    for entry in kb:
        if entry["image"] == image_name:
            return entry.get("constraints", [])
    return []


def main():
    run_start     = time.time()
    run_timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    print("=" * 70)
    print(f"Config 3 FLUX [{VERSION}] — BioGuard-Diffusion + LLM-judge + img2img")
    print(f"Started: {run_timestamp}")
    print("=" * 70)

    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    with open(KB_FILE) as f:
        kb = json.load(f)

    # Load comparison baselines if available
    c1 = {}
    c3_sd = {}
    try:
        with open(C1_FILE) as f:
            c1 = {r["id"]: r for r in json.load(f)}
    except FileNotFoundError:
        pass
    try:
        with open(C3_FILE) as f:
            c3_sd = {r["id"]: r for r in json.load(f)}
    except FileNotFoundError:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nInitialising agents...")
    biologist = BiologistAgent()
    architect = SpatialArchitectAgent()
    verifier  = VerifierAgent()
    splitter  = PromptSplitter()

    print("\nLoading FLUX.1-dev GGUF pipeline & CLIP scorer...")
    pipe   = FluxGGUFPipeline()
    scorer = CLIPScorer()
    print("All models ready.\n")

    records    = []
    verify_log = []

    for i, q in enumerate(queries, 1):
        qid         = q["id"]
        query       = q["query"]
        constraints = _get_constraints(kb, q["image"])

        print(f"[{i:02d}/10] {qid} | {query[:55]}...")
        print(f"         Constraints: {len(constraints)}")

        retrieved = biologist.retrieve(query, top_k=8)

        best_image       = None
        best_sas         = -1.0
        best_llm_score   = 0.0
        best_clip        = 0.0
        retry_history    = []
        violations       = []
        llm_feedback     = ""   # improvement suggestions from last LLM judge
        total_time       = 0.0

        for attempt in range(1, MAX_RETRIES + 1):
            arch = architect.build_prompt(
                query, retrieved,
                violations=violations if attempt > 1 else None,
            )
            full_prompt = arch["prompt"]

            # Split prompt for FLUX dual-encoder.
            # On retry: append LLM improvement suggestions to the base prompt
            # so the T5 encoder receives the feedback.
            prompt_for_split = (
                full_prompt + " " + llm_feedback if llm_feedback else full_prompt
            )
            split       = splitter.split(prompt_for_split)
            clip_prompt = split["clip_prompt"]
            t5_prompt   = split["t5_prompt"]
            print(f"         [CLIP] {clip_prompt[:80]}")
            print(f"         [T5]   {t5_prompt[:100]}...")

            seed = 42

            t0 = time.time()
            if attempt == 1:
                image = pipe.generate(
                    prompt=clip_prompt,
                    prompt_2=t5_prompt,
                    num_inference_steps=pipe.RECOMMENDED_STEPS,
                    guidance_scale=3.5,
                    seed=seed,
                )
                mode = "text2img"
            else:
                strength = max(
                    IMG2IMG_STRENGTH - STRENGTH_DECAY * (attempt - 2),
                    0.30,
                )
                image = pipe.refine(
                    image=best_image,
                    prompt=clip_prompt,
                    prompt_2=t5_prompt,
                    strength=strength,
                    num_inference_steps=pipe.RECOMMENDED_STEPS,
                    guidance_scale=3.5,
                    seed=seed,
                )
                mode = f"img2img(str={strength:.2f})"

            gen_time = time.time() - t0
            total_time += gen_time

            # Combined verification: CLIP-SAS + LLM-judge
            v    = verifier.verify_combined(image, query, constraints, entity=qid)
            clip = scorer.score(image, query)

            print(f"         Attempt {attempt} [{mode}]: "
                  f"SAS={v['sas']:.4f} LLM={v['llm_score']:.1f}/10 "
                  f"CLIP={clip:.4f}  time={gen_time:.1f}s")
            print(f"           CLIP-pass={v['clip_passed']}  LLM-pass={v['llm_passed']}  "
                  f"combined={v['passed_combined']}")
            if v["missing_structures"]:
                print(f"           ↳ Missing: {v['missing_structures'][:3]}")
            if v["improvement_suggestions"]:
                print(f"           ↳ LLM hint: {v['improvement_suggestions'][:80]}...")

            retry_history.append({
                "attempt":               attempt,
                "mode":                  mode,
                "seed":                  seed,
                "sas":                   v["sas"],
                "clip_passed":           v["clip_passed"],
                "llm_score":             v["llm_score"],
                "llm_passed":            v["llm_passed"],
                "passed_combined":       v["passed_combined"],
                "clip_score":            clip,
                "violations":            v["violations"],
                "missing_structures":    v["missing_structures"],
                "improvement_suggestions": v["improvement_suggestions"],
                "clip_prompt":           clip_prompt,
                "t5_prompt":             t5_prompt,
                "gen_time_s":            round(gen_time, 1),
            })

            # Keep best by combined SAS + LLM score
            combined_score = v["sas"] + v["llm_score"] / 100
            if combined_score > best_sas:
                best_sas       = combined_score
                best_image     = image
                best_clip      = clip
                best_llm_score = v["llm_score"]

            if v["passed_combined"]:
                print(f"         ✓ Passed (CLIP+LLM) on attempt {attempt} "
                      f"[{attempt-1} img2img retry loop(s) used]")
                violations   = []
                llm_feedback = ""
                break

            violations   = v["violations"]
            llm_feedback = v["improvement_suggestions"]
            remaining = MAX_RETRIES - attempt
            if remaining > 0:
                print(f"         ✗ Failed — {remaining} img2img retry loop(s) remaining")

        # Save best image with version-tagged filename
        img_path = OUT_DIR / f"{qid}.png"
        best_image.save(img_path)

        final_v = verifier.verify_combined(best_image, query, constraints, entity=qid)

        record = {
            "version":           VERSION,
            "id":                qid,
            "domain":            q["domain"],
            "query":             query,
            "image_file":        str(img_path),
            "clip_score":        round(best_clip, 4),
            "sas":               round(final_v["sas"], 4),
            "clip_passed":       final_v["clip_passed"],
            "llm_score":         final_v["llm_score"],
            "llm_passed":        final_v["llm_passed"],
            "passed_combined":   final_v["passed_combined"],
            "violations":        final_v["violations"],
            "missing_structures":       final_v["missing_structures"],
            "present_structures":       final_v["present_structures"],
            "attempts_used":     len(retry_history),
            "generation_time_s": round(total_time, 1),
            "config":            f"config3_flux_{VERSION}",
        }
        if c1:
            record["clip_delta_vs_c1"] = round(best_clip - c1[qid]["clip_score"], 4)
        if c3_sd:
            record["clip_delta_vs_c3_sd"] = round(best_clip - c3_sd[qid]["clip_score"], 4)

        records.append(record)
        verify_log.append({
            "version":        VERSION,
            "id":             qid,
            "constraints":    constraints,
            "per_constraint": final_v["per_constraint"],
            "retry_history":  retry_history,
        })

    run_end      = time.time()
    total_run_s  = round(run_end - run_start, 1)

    with open(RESULTS_FILE, "w") as f:
        json.dump(records, f, indent=2)
    with open(VERIFY_LOG, "w") as f:
        json.dump(verify_log, f, indent=2, ensure_ascii=False)

    # ── Aggregate performance metrics ─────────────────────────────────────────
    n            = len(records)
    clips        = [r["clip_score"]  for r in records]
    sass         = [r["sas"]         for r in records]
    llms         = [r["llm_score"]   for r in records]
    times        = [r["generation_time_s"] for r in records]
    attempts_all = [r["attempts_used"] for r in records]

    img2img_count = sum(
        1 for log in verify_log
        for h in log["retry_history"] if "img2img" in h["mode"]
    )
    clip_pass = sum(1 for r in records if r["clip_passed"])
    llm_pass  = sum(1 for r in records if r["llm_passed"])
    both_pass = sum(1 for r in records if r["passed_combined"])

    # Per-query retry stats from log
    retry_stats = []
    for log in verify_log:
        hist = log["retry_history"]
        retry_stats.append({
            "id":              log["id"],
            "attempts":        len(hist),
            "img2img_retries": sum(1 for h in hist if "img2img" in h["mode"]),
            "final_passed":    hist[-1]["passed_combined"],
            "sas_per_attempt":     [h["sas"]       for h in hist],
            "llm_per_attempt":     [h["llm_score"]  for h in hist],
            "clip_per_attempt":    [h["clip_score"] for h in hist],
            "time_per_attempt":    [h["gen_time_s"] for h in hist],
        })

    performance_summary = {
        "version":            VERSION,
        "run_timestamp":      run_timestamp,
        "run_duration_s":     total_run_s,
        "max_retries_config": MAX_RETRIES,
        "max_img2img_loops":  MAX_RETRIES - 1,
        "pass_thresholds": {
            "clip_sas": SAS_THRESHOLD,
            "llm_score": 6.0,
        },
        "aggregate": {
            "n_queries":           n,
            "mean_clip":           round(sum(clips) / n, 4),
            "mean_sas":            round(sum(sass)  / n, 4),
            "mean_llm_score":      round(sum(llms)  / n, 2),
            "min_clip":            round(min(clips), 4),
            "max_clip":            round(max(clips), 4),
            "min_llm":             round(min(llms),  2),
            "max_llm":             round(max(llms),  2),
            "queries_clip_passed": clip_pass,
            "queries_llm_passed":  llm_pass,
            "queries_both_passed": both_pass,
            "pass_rate_combined":  round(both_pass / n, 3),
        },
        "retry_performance": {
            "total_img2img_retries":  img2img_count,
            "queries_needing_retry":  sum(1 for s in retry_stats if s["img2img_retries"] > 0),
            "mean_attempts_per_query": round(sum(attempts_all) / n, 2),
            "max_attempts_used":      max(attempts_all),
        },
        "timing": {
            "total_run_s":         total_run_s,
            "mean_gen_time_s":     round(sum(times) / n, 1),
            "min_gen_time_s":      round(min(times), 1),
            "max_gen_time_s":      round(max(times), 1),
            "total_gen_time_s":    round(sum(times), 1),
        },
        "per_query": retry_stats,
        "baselines": {
            "c1_mean_clip":    round(sum(c1[r["id"]]["clip_score"] for r in records) / n, 4) if c1 else None,
            "c3_sd_mean_clip": round(sum(c3_sd[r["id"]]["clip_score"] for r in records) / n, 4) if c3_sd else None,
            "flux_mean_clip":  round(sum(clips) / n, 4),
            "delta_vs_c1":     round(sum(clips)/n - sum(c1[r["id"]]["clip_score"] for r in records)/n, 4) if c1 else None,
            "delta_vs_c3_sd":  round(sum(clips)/n - sum(c3_sd[r["id"]]["clip_score"] for r in records)/n, 4) if c3_sd else None,
        },
        "output_files": {
            "results":     str(RESULTS_FILE),
            "verify_log":  str(VERIFY_LOG),
            "images_dir":  str(OUT_DIR),
        },
    }

    perf_file = BASE / "results" / f"config3_flux_{VERSION}_performance.json"
    with open(perf_file, "w") as f:
        json.dump(performance_summary, f, indent=2)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"Config 3 FLUX [{VERSION}] Results")
    print(f"{'='*75}")
    headers = f"{'ID':<6} {'Domain':<18} {'CLIP':>7} {'SAS':>7} {'LLM':>6} {'Tries':>6} {'Retried':>8}"
    if c1:
        headers += f" {'ΔC1':>7}"
    if c3_sd:
        headers += f" {'ΔSD-C3':>8}"
    print(headers)
    print("-" * len(headers))

    for r, rs in zip(records, retry_stats):
        retried = "yes" if rs["img2img_retries"] > 0 else "no"
        line = (f"{r['id']:<6} {r['domain']:<18} "
                f"{r['clip_score']:>7.4f} {r['sas']:>7.4f} "
                f"{r['llm_score']:>5.1f} {r['attempts_used']:>6} {retried:>8}")
        if c1 and "clip_delta_vs_c1" in r:
            line += f" {r['clip_delta_vs_c1']:>+7.4f}"
        if c3_sd and "clip_delta_vs_c3_sd" in r:
            line += f" {r['clip_delta_vs_c3_sd']:>+8.4f}"
        print(line)

    print("-" * len(headers))
    print(f"{'Mean':<6} {'':18} "
          f"{sum(clips)/n:>7.4f} {sum(sass)/n:>7.4f} "
          f"{sum(llms)/n:>5.1f}")

    print(f"\n{'─'*45}")
    print(f"Version:                   {VERSION}")
    print(f"Run started:               {run_timestamp}")
    print(f"Total wall time:           {total_run_s:.0f}s ({total_run_s/60:.1f} min)")
    print(f"Avg gen time/query:        {sum(times)/n:.1f}s")
    print(f"Max feedback loops config: {MAX_RETRIES - 1} img2img retry loops per query")
    print(f"img2img retries triggered: {img2img_count}")
    print(f"Queries needing retry:     {sum(1 for s in retry_stats if s['img2img_retries']>0)}/{n}")
    print(f"Pass rate (CLIP):          {clip_pass}/{n}")
    print(f"Pass rate (LLM):           {llm_pass}/{n}")
    print(f"Pass rate (combined):      {both_pass}/{n}  ({both_pass/n*100:.0f}%)")
    if c1:
        delta_c1 = round(sum(clips)/n - sum(c1[r["id"]]["clip_score"] for r in records)/n, 4)
        print(f"CLIP delta vs C1:          {delta_c1:+.4f}")
    if c3_sd:
        delta_sd = round(sum(clips)/n - sum(c3_sd[r["id"]]["clip_score"] for r in records)/n, 4)
        print(f"CLIP delta vs C3-SD:       {delta_sd:+.4f}")

    print(f"\nResults:      {RESULTS_FILE}")
    print(f"Verify log:   {VERIFY_LOG}")
    print(f"Performance:  {perf_file}")
    print(f"Images:       {OUT_DIR}/")
    print(f"\nConfig 3 FLUX [{VERSION}] COMPLETE")


if __name__ == "__main__":
    main()
