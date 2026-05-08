# BioGuard-Diffusion

**Precision Biological Illustration Generation via Multi-Agent Collaboration and Knowledge Base**

ECE 598 Final Project — MingYi Wei (mingyi5@illinois.edu)

---

## Overview

BioGuard-Diffusion is a multi-agent pipeline that generates scientifically accurate biological diagrams from natural-language queries. A RAG knowledge base grounded in the AI2D dataset constrains a FLUX.1-dev diffusion backbone, while an iterative verify-refine loop corrects anatomical errors across up to three generation attempts.

The system ships in two pipeline versions:

| Version | Agents | Backbone | Spatial guidance |
|---------|--------|----------|-----------------|
| **v3** | 5 | FLUX.1-dev GGUF | Text-only |
| **v4** | 7 | FLUX.1-dev GGUF + ControlNet Canny | AI2D polygon annotations |

---

## Project Structure

```
bioguard_diffusion/
├── agents/
│   ├── biologist_agent.py        # RAG retrieval (ChromaDB)
│   ├── spatial_architect.py      # Structured prompt assembly
│   ├── prompt_splitter.py        # CLIP / T5 sub-prompt decomposition
│   ├── generator_agent.py        # FLUX text2img + img2img (v3)
│   ├── generator_agent_v4.py     # FLUX + ControlNet (v4)
│   ├── verifier_agent.py         # CLIP-SAS + LLM-as-judge
│   ├── prompt_refiner_agent.py   # Targeted prompt correction
│   └── visual_planner_agent.py   # AI2D annotation parser (v4)
├── generation/
│   ├── flux_gguf_pipeline.py     # FLUX.1-dev GGUF wrapper
│   ├── flux_controlnet_pipeline.py  # FLUX + ControlNet wrapper (v4)
│   ├── blueprint_generator.py    # Layout → ControlNet image (v4)
│   ├── run_all_flux.py           # v3 full run (10 queries)
│   ├── run_all_flux_v4.py        # v4 full run (10 queries)
│   ├── run_config1_baseline.py   # SD-v1.5 raw baseline
│   ├── run_config2_rag.py        # RAG + SpatialArchitect + SD
│   ├── run_config3_full.py       # Full BioGuard + SD
│   ├── run_config3_flux.py       # Full BioGuard + FLUX (v3)
│   ├── run_zimage_baseline.py    # Z-Image-Turbo baseline
│   ├── run_single.py             # Single-query v3 runner
│   └── run_single_v4.py          # Single-query v4 runner
├── data/
│   ├── extract_knowledge.py      # AI2D → knowledge_base.json
│   └── vector_store/             # ChromaDB persistent store
└── evaluation/
    └── clip_score.py             # CLIP ViT-B/32 scoring

data/
├── ai2d/
│   ├── images/                   # 4,903 AI2D diagrams
│   └── annotations/              # Per-image JSON (blobs, text, relationships)
results/
├── knowledge_base.json           # 2,603 entries, 8,642 unique labels
├── test_queries.json             # 10 fixed test queries
├── config1_results.json
├── config2_results.json
├── config3_results.json
├── config3_flux_results.json
└── config3_flux_v2_llm_judge_performance.json
outputs/                          # Generated images per config and query
```

---

## Agents

| Agent | Role |
|-------|------|
| **BiologistAgent** | Queries ChromaDB (2,673 docs, `all-MiniLM-L6-v2`) and returns top-8 biological constraint sentences for the user query |
| **SpatialArchitectAgent** | Assembles a structured FLUX prompt from retrieved constraints; reorders violated constraints to the front on retry |
| **PromptSplitter** | Calls `claude-haiku-4-5` to produce a CLIP sub-prompt (≤77 tokens) and a T5 sub-prompt (unlimited) from the assembled prompt |
| **GeneratorAgent** (v3) | Runs FLUX text2img (attempt 1) and img2img refinement (attempts 2–3) with adaptive denoising strength |
| **GeneratorAgentV4** (v4) | Extends GeneratorAgent with ControlNet conditioning (`controlnet_scale` 0.70 → 0.50/0.65 across attempts) |
| **VerifierAgent** | Stage 1: CLIP-SAS (threshold τ = 0.22); Stage 2: LLM-as-judge via `claude-haiku-4-5-20251001` (threshold 6.0/10) — both must pass |
| **PromptRefinerAgent** | Calls `claude-haiku-4-5` to surgically insert missing structures into best-so-far CLIP + T5 prompts |
| **VisualPlannerAgent** (v4) | Parses AI2D annotation JSON to extract scaled polygon layout at 512×512 |
| **BlueprintGenerator** (v4) | Renders layout to line-art ControlNet image; highlights missing structures in red after each failed attempt |

---

## Experimental Configurations

| Config | Pipeline | Mean CLIP | Mean SAS | Avg time/img |
|--------|----------|:---------:|:--------:|:------------:|
| **Config 1** — SD-v1.5 raw | Raw query → SD-v1.5 (30 steps) | 0.2774 | — | ~30 s |
| **Z-Image-Turbo** — baseline | Raw query → Z-Image-Turbo (8 steps) | 0.2459 | — | ~348 s |
| **Config 2** — RAG + SD | BiologistAgent + SpatialArchitect → SD-v1.5 | 0.2842 | — | ~30 s |
| **Config 3-SD** — Full BioGuard + SD | All agents + retry → SD-v1.5 | 0.2842 | 0.2543 | ~30 s |
| **Config 3-FLUX (v3)** — Full BioGuard + FLUX | All agents + retry → FLUX.1-dev | 0.2988 | 0.2711 | ~432 s |
| **Config 3-FLUX (v4)** — + ControlNet | v3 + VisualPlanner + BlueprintGenerator | 0.2983 | 0.2725 | ~1,322 s |

### Per-Query Results (CLIP Score)

| ID | Domain | SD-C1 | Z-Image | SD-C3 | FLUX-v3 | FLUX-v4 |
|----|--------|:-----:|:-------:|:-----:|:-------:|:-------:|
| q01 | Cell Biology (plant cell) | 0.3116 | 0.2105 | 0.2458 | 0.3061 | 0.2857 |
| q02 | Cell Biology (mitochondria) | 0.2297 | 0.2379 | **0.3216** | 0.2647 | 0.2491 |
| q03 | Neuroanatomy (neuron) | 0.2785 | 0.2311 | 0.2771 | **0.3068** | 0.3169 |
| q04 | Human Anatomy (eye) | 0.2717 | 0.2444 | 0.2917 | 0.2945 | 0.3212 |
| q05 | Human Anatomy (digestive) | 0.2556 | 0.2626 | **0.3052** | 0.2906 | 0.2963 |
| q06 | Human Anatomy (heart) | 0.2958 | 0.2988 | **0.3131** | 0.3088 | 0.3066 |
| q07 | Plant Biology (flower) | 0.3091 | 0.2243 | 0.2313 | **0.3317** | 0.3218 |
| q08 | Plant Biology (leaf) | 0.2949 | 0.2525 | 0.2867 | 0.3012 | 0.3206 |
| q09 | Biochemistry (photosynthesis) | 0.2723 | 0.2510 | 0.2525 | 0.3058 | 0.2753 |
| q10 | Zoology (insect) | 0.2549 | 0.2457 | **0.3165** | 0.2779 | 0.2899 |
| **Mean** | | 0.2774 | 0.2459 | 0.2842 | **0.2988** | 0.2983 |

### v3 vs v4 Summary

| Metric | v3 | v4 |
|--------|:--:|:--:|
| Mean CLIP | 0.2988 | 0.2983 |
| Mean SAS | 0.2711 | 0.2725 |
| Mean LLM score | 5.95 | 6.40 |
| Pass rate (CLIP+LLM) | 4 / 10 | 5 / 10 |
| Avg time per query | ~1,373 s | ~3,901 s |
| Queries where v4 > v3 | — | q02, q04, q09, q10 |
| Queries where v4 < v3 | — | q03 (morphology mismatch) |

---

## Setup

### Requirements

```
Python 3.12
conda env: agent

pip install torch diffusers==0.37.1 transformers accelerate==1.13.0
pip install open-clip-torch==3.3.0 sentence-transformers chromadb
pip install gguf>=0.10.0 pillow anthropic langsmith
```

### Models (downloaded automatically on first run)

| Model | Source | Size |
|-------|--------|------|
| FLUX.1-dev transformer | `city96/FLUX.1-dev-gguf` (Q5_K_S) | ~7 GB |
| CLIP-L + T5-XXL + VAE | `black-forest-labs/FLUX.1-schnell` | ~11 GB |
| ControlNet (v4) | `InstantX/FLUX.1-dev-Controlnet-Canny` | ~4 GB |
| CLIP ViT-B/32 | `openai` (via open-clip-torch) | ~0.6 GB |

### Environment Variables

```bash
# Required for LLM calls (PromptSplitter, PromptRefiner, LLM-as-judge)
export ANTHROPIC_API_KEY=your_key_here

# Optional: LangSmith tracing
export LANGCHAIN_API_KEY=your_key_here
export LANGCHAIN_TRACING_V2=true
```

---

## Running Experiments

### Single query (v3)

```bash
python -m bioguard_diffusion.generation.run_single \
  --query "Draw a detailed cross-section of a mitochondria showing cristae and matrix" \
  --output outputs/test/
```

### Single query (v4, with ControlNet)

```bash
python -m bioguard_diffusion.generation.run_single_v4 \
  --query "Draw a detailed cross-section of a mitochondria showing cristae and matrix" \
  --image_name 3288.png \
  --output outputs/test_v4/
```

### Full 10-query run

```bash
# v3
python -m bioguard_diffusion.generation.run_all_flux

# v4
python -m bioguard_diffusion.generation.run_all_flux_v4
```

### Baselines

```bash
# Config 1: SD-v1.5 raw
python -m bioguard_diffusion.generation.run_config1_baseline

# Config 2: RAG + SpatialArchitect + SD
python -m bioguard_diffusion.generation.run_config2_rag

# Z-Image-Turbo
python -m bioguard_diffusion.generation.run_zimage_baseline
```

---

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| RAG top-k | 8 | `BiologistAgent.retrieve()` |
| Max constraints injected | 5 | `SpatialArchitectAgent.MAX_CONSTRAINTS` |
| CLIP-SAS threshold | 0.22 | `VerifierAgent.PASS_THRESHOLD` |
| LLM-judge threshold | 6.0 / 10 | `VerifierAgent.LLM_PASS_THRESHOLD` |
| Combined score | 0.5×SAS + 0.5×(LLM/10) | rollback and best-tracking |
| Max retry attempts | 3 | `_run_flux()` / `_run_flux_v4()` |
| FLUX steps | 20 | `FluxGGUFPipeline.generate()` |
| FLUX guidance scale | 3.5 | `FluxGGUFPipeline.generate()` |
| SD steps | 30 | `BioGenerationPipeline.generate()` |
| SD guidance scale | 7.5 | `BioGenerationPipeline.generate()` |
| ControlNet scale (att 1) | 0.70 | `GeneratorAgentV4` |
| ControlNet scale (att 2+) | 0.50 / 0.65 | adaptive by LLM score |
| Seed | 42 | all configs, fixed for reproducibility |

---

## Dataset

- **Source:** AI2D (Allen Institute for AI Diagrams)
- **Total images:** 4,903
- **Biology-relevant images retained:** 2,724
- **Knowledge base entries:** 2,603 (8,642 unique biological labels)
- **Vector store documents:** 2,673 (individual constraint sentences + label summaries)
- **Test queries:** 10 fixed queries across 6 domains (cell biology, neuroanatomy, human anatomy, plant biology, biochemistry, zoology)

---

## Hardware Notes

All experiments run on Apple M3 Pro (18 GB unified memory) with Apple MPS acceleration.

- SD-v1.5: ~30 s/image
- FLUX.1-dev (v3, no retry): ~432 s/image
- FLUX.1-dev (v3, with retries): ~1,373 s/query average
- FLUX.1-dev + ControlNet (v4, with retries): ~3,901 s/query average

On a server GPU (H100/A100), FLUX inference is expected to run at ~2–4 s/image.

---

## Citation

```
@misc{bioguard2026,
  title  = {BioGuard-Diffusion: Precision Biological Illustration Generation
             via Multi-Agent Collaboration and Knowledge Base},
  author = {MingYi Wei},
  year   = {2026},
  note   = {ECE 598 Final Project, University of Illinois Urbana-Champaign}
}
```
