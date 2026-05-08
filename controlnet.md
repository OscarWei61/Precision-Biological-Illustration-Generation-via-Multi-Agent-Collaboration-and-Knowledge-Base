# BioGuard-Diffusion v4 — ControlNet Architecture

**ECE 598 Final Project — Architecture Detail Document**

---

## Overview

v4 extends the v3 multi-agent pipeline with a **spatial guidance layer** built on ControlNet Canny. Two new agents (Visual Planner, Blueprint Generator) sit upstream of the diffusion pipeline and convert AI2D dataset polygon annotations into a ControlNet control image that constrains where biological structures appear in the generated diagram.

---

## Full v4 Pipeline Architecture

```
AI2D Dataset
  ├── data/ai2d/annotations/{image}.json   ← blob polygons + text labels + relationships
  └── data/ai2d/images/{image}.png         ← original diagram (coordinate scaling only)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Spatial Pre-Processing Layer  (NEW in v4)            │
│                                                                             │
│  ┌──────────────────────────────┐                                           │
│  │   Visual Planner Agent       │  agents/visual_planner_agent.py           │
│  │                              │                                           │
│  │  Strategy A: intraObjectLabel│  blob polygon → named structure           │
│  │  Strategy B: intraObjectLinkage  text rect center → position hint        │ 
│  │              blob outline   → structural boundary                        │
│  │                              │                                           │
│  │  Output: layout dict         │                                           │
│  │    { name, center, bounds,   │                                           │
│  │      polygon (scaled 512px), │                                           │
│  │      is_label_hint }         │                                           │
│  │    has_spatial_data: bool    │                                           │
│  └──────────────┬───────────────┘                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────┐                                           │
│  │   Blueprint Generator        │  generation/blueprint_generator.py        │
│  │                              │                                           │
│  │  mode = "lineart" (default)  │  white bg + black polygon outlines        │
│  │  mode = "segmentation"       │  12-colour filled regions                 │
│  │                              │                                           │
│  │  generate(layout)  →  512×512 RGB control image (blueprint_initial.png)  │
│  │                              │                                           │
│  │  highlight_missing(layout,   │  missing structures → bold red (width=6)  │
│  │    missing_names)            │  others → thin black (width=2)            │
│  │                              │  blueprint_att{N}.png (per retry)         │
│  └──────────────┬───────────────┘                                           │
│                 │                                                           │
│                 │ has_spatial_data=False → black image (ControlNet disabled)│
│                 ▼                                                           │
│         ControlNet control image (current_blueprint)                        │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Multi-Agent Text Reasoning Layer  (from v3)             │
│                                                                             │
│  User Query ──► Biologist Agent  ──► Spatial Architect                     │
│                 (ChromaDB RAG,        (template prompt,                     │
│                  top-k=8)             violations re-prioritised)            │
│                                ─────────────────────────────────────        │
│                                       full_prompt                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                     ┌──────────────┴─────────────────┐
                     │        Attempt loop (max=3)     │
                     │                                 │
                     │  cn_scale = _controlnet_scale_  │
                     │    for_attempt(llm, attempt,    │
                     │    has_spatial)                 │
                     │                                 │
                     │  attempt 1 → 0.70  (text2img)  │
                     │  attempt 2+ (LLM < 5) → 0.65   │
                     │  attempt 2+ (LLM ≥ 5) → 0.50   │
                     │  no spatial data   → 0.00       │
                     └──────────────┬──────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Generator Agent v4   agents/generator_agent_v4.py        │
│                                                                             │
│   Attempt 1 (text2img)                                                      │
│   ─────────────────────────────────────────────────────────────             │
│   full_prompt  →  PromptSplitter (claude-haiku)                             │
│     ├── clip_prompt  (≤77 tokens, keyword style)                            │
│     └── t5_prompt    (full natural language, unlimited)                     │
│                                   │                                         │
│              FluxControlNetPipeline.generate(                               │
│                prompt=clip_prompt, prompt_2=t5_prompt,                      │
│                control_image=current_blueprint,                             │
│                controlnet_conditioning_scale=cn_scale,   ← 0.70            │
│                num_inference_steps=20, guidance_scale=3.5,                  │
│                seed=seed + (attempt-1)*7                 ← fresh seed       │
│              )                                                               │
│                                                                             │
│   Attempt 2–3 (img2img or text2img restart)                                 │
│   ─────────────────────────────────────────────────────────────             │
│   adaptive_strength logic (from v3):                                        │
│     LLM < 3.0  → text2img restart (fresh seed)                              │
│     LLM 3–5    → img2img strength=0.75                                      │
│     LLM 5–6.5  → img2img strength=0.55                                      │
│     LLM ≥ 6.5  → img2img strength=0.28                                      │
│                                                                             │
│   rollback: always refine from best_clip_prompt / best_t5_prompt,          │
│             not latest (prevents regression)                                │
│                                                                             │
│              FluxControlNetImg2ImgPipeline.generate(                        │
│                image=best_image,                                            │
│                prompt=refined_clip_prompt, prompt_2=refined_t5_prompt,      │
│                control_image=current_blueprint,                             │
│                controlnet_conditioning_scale=cn_scale,   ← 0.50 or 0.65    │
│                strength=adaptive_strength, seed=seed                        │
│              )                                                               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │  generated image (512 × 512 PNG)
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Verifier Agent  (two-stage)                          │
│                                                                             │
│  Stage 1 — CLIP-SAS (fast, local)                                           │
│    CLIP ViT-B/32 cosine-similarity vs keyword queries from constraints      │
│    SAS = mean score;  pass threshold = 0.22                                 │
│                                                                             │
│  Stage 2 — LLM-as-judge (claude-haiku-4-5 vision)                          │
│    image (base64) + query + constraints → JSON                              │
│    { score/10, present_structures, missing_structures,                      │
│      improvement_suggestions }                                              │
│    pass threshold = 6.0                                                     │
│                                                                             │
│  Combined pass = CLIP ≥ 0.22  AND  LLM ≥ 6.0                              │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
           ┌────────────────────┴─────────────────────────┐
           │  passed?                                      │  failed + retries remain?
           ▼                                               ▼
    Final image (best.png)                  Prompt Refiner Agent
                                            (claude-haiku-4-5)
                                              ├── preserve present_structures
                                              ├── strengthen missing_structures
                                              └── refined CLIP + T5 prompts
                                                          │
                                            Adaptive Blueprint Update:
                                              highlight_missing(missing_structures)
                                              → current_blueprint updated
                                              → saved as blueprint_att{N}.png
                                                          │
                                                          └──► next attempt
```

---

## ControlNet Pipeline Stack

```
FLUX.1-dev GGUF Transformer
  city96/FLUX.1-dev-gguf (Q5_K_S, ~7 GB)
  loaded via hf_hub_download → config_dir workaround
         │
         ├── Text Encoders + VAE
         │     black-forest-labs/FLUX.1-schnell
         │
         └── ControlNet Encoder
               InstantX/FLUX.1-dev-Controlnet-Canny
               FluxControlNetModel.from_pretrained(...)
                    │
                    ├── text2img: FluxControlNetPipeline
                    └── img2img:  FluxControlNetImg2ImgPipeline
                                  (fallback to FluxControlNetPipeline
                                   if img2img not available)
```

**MPS device note:** `torch.Generator` must be created on `"cpu"` for Apple Silicon, then passed to the pipeline. The pipeline itself runs on `"mps"` via `enable_model_cpu_offload()`.

---

## Blueprint Generation Detail

### Input: AI2D Annotation JSON Schema

```json
{
  "blobs": {
    "b1": { "polygon": [[x,y], ...] }
  },
  "text": {
    "t1": { "value": "Cristae", "rectangle": [[x1,y1],[x2,y2]] }
  },
  "relationships": {
    "r1": {
      "category": "intraObjectLinkage",
      "origin": "t1",
      "destination": "b1"
    }
  }
}
```

### Coordinate Scaling

```
Original image size  →  512 × 512 (target)
  sx(x) = int(x * 512 / orig_w)
  sy(y) = int(y * 512 / orig_h)
```

### Relationship Strategies

| Strategy | Relationship type | What is extracted | Output |
|---|---|---|---|
| A | `intraObjectLabel` | text IS blob name | Full blob polygon (structural) |
| B | `intraObjectLinkage` | text → arrow → blob | Text rect center (position hint) + blob outline |
| Fallback | none | Unlabelled blob | Blob polygon as `region_{id}` |

### Lineart Mode Output

- **White background** (255, 255, 255)
- **Structural polygons**: black outline, `line_width=3`
- **Label hints** (`is_label_hint=True`): grey filled dot (r=4 px) at text center
- **Adaptive highlight**: missing structures → red outline `width=6`; rest → black `width=2`

---

## ControlNet Scale Schedule

| Situation | cn_scale | Rationale |
|---|:---:|---|
| Attempt 1 (text2img) | **0.70** | Strong initial spatial guidance |
| Attempt 2+ AND prev LLM < 5.0 | **0.65** | Score low → keep more structure |
| Attempt 2+ AND prev LLM ≥ 5.0 | **0.50** | Near passing → light touch |
| No spatial data (any attempt) | **0.00** | Graceful fallback to v3 |

---

## Adaptive Blueprint Loop

```
After each failed attempt:
  1. Verifier returns missing_structures (e.g. ["Cristae", "Matrix"])
  2. blueprint_gen.highlight_missing(layout, missing_structures)
       → redraw: matching regions in red (width=6)
       → others in black (width=2)
  3. current_blueprint ← new control image
  4. saved as blueprint_att{N}.png
  5. Next attempt uses updated control image → ControlNet emphasises missing regions
```

Matching is case-insensitive substring: `"nucleus"` matches `"Cell Nucleus"`.

---

## Output Files Per Query

```
outputs/run_all_flux_v4/{timestamp}/{qid}_{domain}/
  blueprint_initial.png        ← initial AI2D polygon lineart
  blueprint_att02.png          ← adaptive highlight after attempt 1 (if missing found)
  blueprint_att03.png          ← updated again after attempt 2 (if still missing)
  attempt_01_text2img_seed42.png
  attempt_02_{mode}_seed42.png
  attempt_03_{mode}_seed42.png
  best.png                     ← highest combined_score image
  reference.png                ← AI2D original (visual comparison only, NOT generation input)
  prompt_log.txt               ← CLIP + T5 prompts per attempt + controlnet_scale
  summary.json                 ← all metrics, layout regions, per-attempt records
```

---

## Ablation Summary

| Config | RAG | ControlNet | Retry loop | Pass rate | Avg CLIP | Avg LLM | Time/query |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| flux_direct | ✗ | ✗ | ✗ | **7/10** | 0.2962 | 5.85 | ~400 s |
| flux_rag_noretry | ✓ | ✗ | ✗ | 6/10 | 0.2948 | 5.27 | ~401 s |
| v3 (rollback + adaptive) | ✓ | ✗ | ✓ | 4/10 | 0.2924 | 5.95 | ~1373 s |
| **v4 (ControlNet)** | ✓ | **✓** | ✓ | 5/10 | 0.2934 | **6.40** | ~3901 s |

**Key finding:** `flux_direct` (no RAG, no retry) achieves the highest pass rate because PromptSplitter's LLM-generated T5 prompt is richer than the SpatialArchitect template. ControlNet uniquely unlocks q02 (mitochondria) and q09 (photosynthesis) — the two queries where spatial polygon guidance overcomes FLUX's photorealistic bias — but adds 2.84× wall time.

---

## File Map

| File | Role |
|------|------|
| `agents/visual_planner_agent.py` | AI2D annotation → layout JSON |
| `generation/blueprint_generator.py` | layout JSON → ControlNet control image |
| `generation/flux_controlnet_pipeline.py` | GGUF transformer + ControlNet pipeline wrapper |
| `agents/generator_agent_v4.py` | GeneratorAgentV4 (wraps ControlNet pipeline) |
| `generation/run_single_v4.py` | Single-query v4 runner (CLI / interactive) |
| `generation/run_all_flux_v4.py` | Full 10-query batch runner (v4) |
| `generation/run_all_flux_baselines.py` | Ablation baselines (flux_direct, flux_rag_noretry) |
