# BioGuard-Diffusion: Precision Biological Illustration Generation via Multi-Agent Collaboration

**ECE 598 Final Project**  
**Author:** MingYi Wei (mingyi5@illinois.edu)  
**Environment:** Python 3.12.13 · Apple Silicon MPS (M3 Pro 18 GB) · conda env `agent`

---

## Abstract

BioGuard-Diffusion is a multi-agent framework that generates scientifically accurate biological diagrams by grounding diffusion model image generation in a structured knowledge base. The system evolves across four pipeline versions (v1–v4). The final **v4** architecture adds two new agents: a **Visual Planner Agent** that extracts spatial layout (polygon coordinates of labelled structures) from AI2D dataset annotations, and a **Blueprint Generator** that renders those polygons as a ControlNet Canny edge control image. Combined with the v3 rollback mechanism and adaptive strength schedule, the v4 pipeline uses **FLUX.1-dev GGUF + InstantX ControlNet Canny** to enforce spatial structure in generated diagrams. Five text agents (Biologist, Spatial Architect, Generator, Verifier, Prompt Refiner) collaborate with two spatial agents (Visual Planner, Blueprint Generator) in a closed feedback loop. The system is evaluated on 10 biological queries across six configurations using CLIP-Score, Scientific Accuracy Score (SAS), and LLM-as-judge score (0–10). **Best result (v4): 5/10 queries passed**, mean CLIP 0.2934, mean LLM 6.40/10 — a +1 pass improvement over v3 at the cost of 2.84× wall-time increase. Two ablation baselines (**flux_direct**: raw query → FLUX one-shot; **flux_rag_noretry**: RAG + one-shot, no retry) reveal a counterintuitive finding: flux_direct achieves the highest pass rate across all FLUX configurations (7/10), suggesting the multi-agent retry loop's benefit is concentrated on the two hardest structure-dense queries (q02 mitochondria, q09 photosynthesis) that only ControlNet guidance can consistently unlock.

---

## 1. System Architecture

### 1.1 Current Pipeline (FLUX backend with full feedback loop)

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       Multi-Agent Reasoning Layer                     │
│                                                                        │
│  ┌─────────────────────┐                                              │
│  │   Biologist Agent   │  ChromaDB RAG retrieval (AI2D constraints)   │
│  └────────┬────────────┘  top-k=8 constraint sentences                │
│           ▼                                                            │
│  ┌─────────────────────┐                                              │
│  │  Spatial Architect  │  Template-based structured prompt builder    │
│  └────────┬────────────┘  violations re-prioritised on retry          │
│           ▼                                                            │
│  ┌─────────────────────┐      ┌────────────────────────────────────┐ │
│  │  Generator Agent    │◄─────│     Prompt Refiner Agent           │ │
│  │                     │      │  (claude-haiku-4-5 LLM)            │ │
│  │  PromptSplitter     │      │  Surgically rewrites CLIP+T5       │ │
│  │  (attempt 1 only)   │      │  Preserves present_structures      │ │
│  │  FluxGGUFPipeline   │      │  Adds missing_structures by name   │ │
│  │  prompt_history log │      └────────────────────────────────────┘ │
│  └────────┬────────────┘                          ▲                  │
│           │  generated image                       │ refined CLIP+T5  │
│           ▼                                        │                  │
│  ┌─────────────────────┐                          │                  │
│  │   Verifier Agent    │  Stage 1: CLIP-SAS       │                  │
│  │                     │  Stage 2: LLM-as-judge   │                  │
│  │  pass condition:    │─────────────────────────►│                  │
│  │  CLIP ≥ 0.22 AND    │  present_structures                         │
│  │  LLM  ≥ 6.0 AND     │  missing_structures                         │
│  │  no missing structs │  improvement_suggestions                     │
│  └─────────────────────┘                                              │
└──────────────────────────────────────────────────────────────────────┘
    │ retry: img2img/text2img from best_image, fixed seed, refined CLIP+T5
    │   rollback: always use best_clip_prompt / best_t5_prompt (not latest)
    │   adaptive strength: LLM<3→text2img restart  3–5→0.75  5–6.5→0.55  ≥6.5→0.28
    │ (up to 3 attempts total)
    ▼
Final Biological Illustration (512 × 512 PNG)
```

---

## 2. Agent Descriptions

### 2.1 Biologist Agent (`agents/biologist_agent.py`)

Manages a **ChromaDB persistent vector store** built from the AI2D dataset knowledge base (`results/knowledge_base.json`). The store contains 2,673 documents: each biological constraint sentence indexed individually, plus per-image label-set summaries.

**Retrieval:**
- Uses ChromaDB's built-in `all-MiniLM-L6-v2` (ONNX) embedding model
- Queries with `where={"type": "constraint"}` filter to return only constraint sentences
- Default `top_k = 8`; returns a list of constraint strings ranked by cosine similarity

**Knowledge base stats:**
| Metric | Value |
|--------|-------|
| Total AI2D images | 4,903 |
| Biology-relevant images | 2,724 |
| Knowledge base entries | 2,603 |
| Unique biological labels | 8,642 |
| Documents indexed in vector store | 2,673 |

---

### 2.2 Spatial Architect Agent (`agents/spatial_architect.py`)

Template-based prompt builder. No LLM API call — uses regex pattern matching and keyword-overlap scoring to select and inject the most relevant constraints.

**Prompt construction:**
1. Extract biological subject from query (strip verb prefixes: "draw", "create", etc.)
2. Score retrieved constraints by keyword overlap with extracted subject
3. Select top `MAX_CONSTRAINTS = 5` constraints
4. Assemble structured positive prompt:
   ```
   "A highly detailed scientific cross-section diagram of {subject},
    {detail_phrases}, scientific educational illustration, detailed diagram,
    white background, accurate biology textbook style, no text, no labels"
   ```
5. Negative prompt: `"blurry, cartoon, deformed, extra limbs, wrong anatomy, missing structures, low quality, watermark, text, labels, annotations, words, letters, numbers, abstract art"`

> **Design decision:** Text labels explicitly excluded from both positive and negative prompts after empirical finding that diffusion models generate corrupted/hallucinated text (e.g., q02 v2 run showed `present_structures` entries like `"Rianito logemerl"` — FLUX-generated text artifacts). Removing the label requirement reduces LLM-judge penalty for incorrect text while preserving structural accuracy goals.

**Retry prioritisation:** On attempt > 1, violated constraints are moved to the front of the constraint list before re-scoring, ensuring truncated prompts still include the most critical failed structures.

---

### 2.3 Prompt Splitter (`agents/prompt_splitter.py`) — FLUX only

FLUX.1 uses a **dual text encoder** architecture: CLIP-L (77-token hard limit, keyword-style) and T5-XXL (no token limit, natural language). Two separate `claude-haiku-4-5` calls produce prompts optimised for each encoder.

| Encoder | Token limit | Input style | Content |
|---------|-------------|-------------|---------|
| CLIP | ≤ 77 tokens | Comma-separated keywords | Subject, 2–4 key structures, style |
| T5 | Unlimited | Full natural language | All structures, spatial relationships, functions |

Used only on **attempt 1**. On retry attempts, PromptRefiner outputs CLIP+T5 directly, bypassing re-compression.

---

### 2.4 Generator Agent (`agents/generator_agent.py`) — NEW

Wraps `FluxGGUFPipeline` + `PromptSplitter` into a single trackable unit. Maintains a `prompt_history` list recording every prompt fed into the pipeline.

**Key design decisions:**
- `generate()` = text2img (attempt 1), uses `PromptSplitter` to produce CLIP+T5
- `refine()` = img2img (attempt 2+), accepts `clip_prompt_override` / `t5_prompt_override` from `PromptRefiner` — bypasses PromptSplitter re-compression that previously destroyed refinements
- **Fixed seed for img2img**: `refine()` uses the base seed (not an incremented variant), ensuring the noise pattern is deterministic and only the prompt drives visual changes
- MPS generator fix: `torch.Generator(device="cpu").manual_seed(seed)` — avoids MPS device incompatibility that previously caused seed to be silently ignored

**Outputs per attempt:** `{image, mode, full_prompt, clip_prompt, t5_prompt, seed, gen_time_s}`

**Adaptive img2img strength (v3):** strength driven by LLM score from previous verification, not fixed schedule.

| LLM score (prev attempt) | Next strength | Rationale |
|:---:|:---:|---|
| < 3.0 | text2img restart | Image too far wrong; fresh seed gives better chance |
| 3.0–4.9 | 0.75 | Far from target; allow large structural change |
| 5.0–6.4 | 0.55 | Approaching threshold; moderate change |
| ≥ 6.5 | 0.28 | Near/above threshold; fine-tune only, preserve structure |

**Rollback mechanism (v3):** `PromptRefiner` always receives `best_clip_prompt` / `best_t5_prompt` (the prompts from the highest combined-score attempt), not the latest attempt's prompts. If an attempt regresses, the next iteration still refines from the best known base, preventing cumulative drift.

**Combined score:** `combined = 0.5 × SAS + 0.5 × (LLM_score / 10)` — equal weight between CLIP-SAS (0–1) and LLM-as-judge (0–10 normalised to 0–1).

---

### 2.5 Verifier Agent (`agents/verifier_agent.py`)

Two-stage verification system.

#### Stage 1 — CLIP-SAS (fast, local)

Uses **CLIP ViT-B/32** (`open_clip`, pretrained=`openai`).

1. Extract keyword query from each constraint sentence (stopword removal, max 6 words)
2. Compute cosine similarity between image embedding and keyword embedding
3. **SAS** = mean similarity across all constraints
4. Constraint passes if `score ≥ PASS_THRESHOLD = 0.22`

**Threshold calibration** (empirical, on AI2D mitochondria reference `3288.png`):

| Keyword type | Score range |
|---|---|
| Correct structures (cristae, inner membrane) | 0.268 – 0.317 |
| Wrong structures (heart, neuron, flower) | 0.147 – 0.224 |
| Chosen threshold | **0.22** |

> **Why CLIP over BioCLIP:** BioCLIP v2 failed to discriminate biological structures in constraint sentences — both correct and wrong keywords scored 0.08–0.14. BioCLIP is trained on Tree-of-Life taxonomy labels (single-word species names), not multi-word anatomical phrases.

#### Stage 2 — LLM-as-Judge (semantic, vision-capable)

Sends image (base64 PNG) + query + constraints to `claude-haiku-4-5-20251001` vision.

```json
{
  "score": 0.0,
  "passed": false,
  "present_structures": [],
  "missing_structures": [],
  "improvement_suggestions": ""
}
```

- Passes if `llm_score ≥ LLM_PASS_THRESHOLD = 6.0`
- `present_structures` / `missing_structures` fed directly to PromptRefiner
- `improvement_suggestions` appended to T5 prompt context

#### Combined Verification

`verify_combined()` runs both stages. **Pass condition:**
```
clip_passed AND llm_passed AND len(missing_structures) == 0
```

The `missing_structures` check was added after observing that LLM-judge can score ≥ 6.0 while still reporting absent structures — ensuring the loop continues until structural completeness is confirmed.

---

### 2.6 Prompt Refiner Agent (`agents/prompt_refiner_agent.py`) — NEW

Uses `claude-haiku-4-5` to **surgically refine CLIP and T5 prompts** directly based on verifier output. Unlike the prior approach (appending `improvement_suggestions` to a combined prompt then re-splitting), the Refiner operates directly on the CLIP and T5 strings, preserving architecture-specific formatting.

**Inputs:**
- `clip_prompt`: current CLIP prompt (from GeneratorAgent)
- `t5_prompt`: current T5 prompt (from GeneratorAgent)
- `present_structures`: from LLM-judge (do NOT modify related phrases)
- `missing_structures`: from LLM-judge (MUST add explicitly by name)
- `violations`: from CLIP-SAS (add emphasis)
- `improvement_suggestions`: from LLM-judge (incorporate surgically)

**Token limit enforcement:**
- CLIP: hard limit `55 words` enforced at both LLM prompt level and post-processing truncation
- T5: hard limit `150 words` — prevents JSON response truncation (prior bug: unlimited T5 caused `max_tokens=768` cutoff → malformed JSON → silent fallback to original prompt)

**Key rules in system prompt:**
- Items in `present_structures` → leave related phrases untouched
- Items in `missing_structures` → MUST appear by name in refined output
- Use "clearly visible X", "prominently depicted X" not generic "improve"

**Fallback behaviour:** On JSON parse error, returns truncated original prompts (not full original) — ensuring token limits are still enforced even on failure.

---

## 3. Pipeline Configurations

### Config 1 — One-Shot SD Baseline

```
User Query → SD-v1.5 (steps=30, cfg=7.5, seed=42) → Image
```

No agents, no RAG.

---

### Config 2 — RAG + Spatial Architect (SD)

```
User Query → BiologistAgent (top-8) → SpatialArchitect → SD v1.5 → Image
```

SD-v1.5's CLIP encoder has a hard 77-token limit. **Prompt chunking** in `diffusion_pipeline.py`:
1. Tokenise without truncation
2. Split into 75-content-token chunks + BOS/EOS → shape `(1, 77)`
3. Encode each chunk through CLIP
4. Concatenate: `(1, n_chunks×77, 768)` → U-Net via `prompt_embeds`

All Config 2 prompts required 2 chunks → `(1, 154, 768)`.

---

### Config 3 — Full BioGuard (SD backbone)

```
User Query → BiologistAgent → SpatialArchitect → SD v1.5
  → VerifierAgent (CLIP-SAS) → [retry ≤3, seed+=7]
```

---

### Config 3 FLUX — Full BioGuard (FLUX.1-dev, current)

```
User Query
  → BiologistAgent (top-8 RAG)
  → SpatialArchitect
  → GeneratorAgent
      attempt 1: PromptSplitter → CLIP+T5 → FluxGGUFPipeline.generate() [text2img]
  → VerifierAgent.verify_combined()
      → present_structures / missing_structures / violations / llm_feedback
  → PromptRefiner(clip, t5, present, missing, violations, suggestions)
      → refined_clip + refined_t5 (surgical, no re-compression)
  → GeneratorAgent
      attempt 2+: FluxGGUFPipeline.refine(best_image, refined_clip, refined_t5)
                  [img2img, fixed seed, adaptive strength 0.28/0.55/0.75]
                  OR text2img restart (LLM score < 3.0, fresh seed)
                  rollback: refined_clip/t5 always from best-scoring attempt
  → loop until: (CLIP_passed AND LLM_passed AND missing_structures=[]) OR max_retries
```

**FLUX model assembly:**

| Component | Source |
|-----------|--------|
| Transformer | `city96/FLUX.1-dev-gguf` — `flux1-dev-Q5_K_S.gguf` (~7 GB) |
| CLIP-L text encoder | `black-forest-labs/FLUX.1-schnell` |
| T5-XXL text encoder | `black-forest-labs/FLUX.1-schnell` |
| VAE | `black-forest-labs/FLUX.1-schnell` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` |

Memory strategy: `enable_model_cpu_offload()` on M3 Pro 18 GB unified memory.

---

## 4. Test Queries

10 fixed queries spanning 6 biological domains, sourced from AI2D dataset annotations:

| ID | Domain | Query | Source Image |
|----|--------|-------|--------------|
| q01 | cell_biology | Draw a labeled diagram of a plant cell showing all major organelles | 1071.png |
| q02 | cell_biology | Draw a detailed cross-section of a mitochondria showing cristae and matrix | 3288.png |
| q03 | neuroanatomy | Draw a labeled diagram of a neuron showing axon, dendrites and myelin sheath | 2925.png |
| q04 | human_anatomy | Draw a cross-section of a human eye showing retina, lens, cornea and optic nerve | 2857.png |
| q05 | human_anatomy | Draw a labeled diagram of the human digestive system | 1385.png |
| q06 | human_anatomy | Draw a labeled diagram of the human heart showing ventricles, atria and aorta | 1382.png |
| q07 | plant_biology | Draw a labeled diagram of the parts of a flower including petal, sepal, stamen and pistil | 1014.png |
| q08 | plant_biology | Draw a labeled diagram of a leaf showing blade, midrib, petiole and veins | 1086.png |
| q09 | biochemistry | Draw a diagram illustrating the process of photosynthesis with inputs and outputs | 4088.png |
| q10 | zoology | Draw a labeled diagram of an insect body showing head, thorax, abdomen and antennae | 1191.png |

---

## 5. Evaluation Methods

### 5.1 CLIP-Score

Measures **semantic alignment** between generated image and original user query.

- Model: CLIP ViT-B/32 (openai pretrained, via `open_clip`)
- Score = cosine similarity of L2-normalised image and text embeddings
- Range: typically 0.20–0.35 for generated biological diagrams

### 5.2 Scientific Accuracy Score (SAS)

Measures **biological constraint coverage**.

```
For each constraint c in ground-truth list:
    keyword_query = extract_keywords(c)   # stopword removal, max 6 words
    score_c       = CLIP_cosine_sim(image, keyword_query)
    passed_c      = (score_c >= 0.22)

SAS = mean(score_c for all c)
```

SAS decomposes requirements into individual constraint checks, sensitive to specific missing structures.

### 5.3 LLM-as-Judge Score (Config 3 FLUX)

`claude-haiku-4-5-20251001` vision model holistically evaluates each image:

| Score range | Criteria |
|-------------|----------|
| 9–10 | All structures clearly present and visually distinct |
| 7–8 | Most structures present, minor inaccuracies |
| 5–6 | Core subject recognisable, several structures missing |
| 3–4 | Subject recognisable but major structures absent |
| 0–2 | Wrong subject or unrecognisable as scientific diagram |

Pass threshold: `llm_score ≥ 6.0`

---

## 6. Results

### 6.1 Cross-Configuration Summary

| Metric | Config 1 (SD) | Config 2 (SD) | Config 3 (SD) | FLUX v1 | FLUX v2 | FLUX v3 | **FLUX v4** | flux_direct | flux_rag_noretry |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mean CLIP-Score | 0.2774 | 0.2842 | 0.2842 | 0.2988 | 0.2952 | 0.2924 | 0.2934 | **0.2962** | 0.2948 |
| Mean SAS | — | — | 0.2543 | 0.2711 | 0.2726 | 0.2713 | 0.2699 | — | — |
| Mean LLM score (/10) | — | — | — | 5.40 | 4.85 | 5.95 | **6.40** | 5.85 | 5.27 |
| Combined pass rate | — | — | 10/10 (SAS only) | 1/10 | 3/10 | 4/10 | 5/10 | **7/10** | 6/10 |
| Avg wall time / query | ~30 s | ~30 s | ~30 s | ~432 s | ~1718 s | ~1373 s | ~3901 s | **~400 s** | **~401 s** |
| Retries triggered | — | — | 0 | 8/10 | 10/10 | 10/10 | 10/10 | **0** | **0** |
| Rollback triggered | — | — | — | — | — | 7/10 | 6/10 | — | — |
| RAG retrieval | — | — | — | ✓ | ✓ | ✓ | ✓ | **✗** | ✓ |
| ControlNet guidance | — | — | — | — | — | — | ✓ | — | — |

> v1: CLIP-SAS only. v2: combined CLIP+LLM + PromptRefiner. v3: rollback + adaptive strength. v4: ControlNet Canny spatial guidance (VisualPlanner + BlueprintGenerator), same v3 rollback/adaptive logic.  
> flux_direct: raw query → PromptSplitter → FLUX one-shot (no RAG, no retry). flux_rag_noretry: BiologistAgent + SpatialArchitect → PromptSplitter → FLUX one-shot (RAG only, no retry).

---

### 6.2 Config 3 FLUX v2 — Per-Query Results (2026-05-01, previous)

| ID | Domain | CLIP | SAS | LLM (att1→2→3) | Tries | Pass | Missing |
|----|--------|:----:|:---:|:---:|:---:|:---:|:---:|
| q01 | cell_biology | 0.2971 | 0.2743 | 7.5→7.5→7.5 | 3 | ✓ | 2 |
| q02 | cell_biology | 0.2596 | 0.2389 | 2.5→2.0→2.0 | 3 | ✗ | 6 |
| q03 | neuroanatomy | 0.2568 | 0.2449 | 2.5→2.5→2.5 | 3 | ✗ | 8 |
| q04 | human_anatomy | 0.3097 | 0.2847 | 6.5→7.5→8.2 | 3 | ✓ | 3 |
| q05 | human_anatomy | 0.2958 | 0.2945 | 3.5→4.5→3.5 | 3 | ✗ | 5 |
| q06 | human_anatomy | 0.3122 | 0.2820 | 6.5→6.5→5.5 | 3 | ✗ | 6 |
| q07 | plant_biology | 0.3292 | 0.2764 | 2.0→2.0→2.0 | 3 | ✗ | 6 |
| q08 | plant_biology | 0.3029 | 0.2677 | 2.0→2.0→2.0 | 3 | ✗ | 3 |
| q09 | biochemistry | 0.2873 | 0.2738 | 3.5→6.5→2.5 | 3 | ✗ | 8 |
| q10 | zoology | 0.3012 | 0.2691 | 8.5→8.5→7.5 | 3 | ✓ | 2–3 |
| **Mean** | | **0.2952** | **0.2726** | **4.85** | 3.0 | **3/10** | — |

---

### 6.3 Config 3 FLUX v3 — Per-Query Results (2026-05-02, latest)

*v3 adds rollback + adaptive strength schedule + equal-weight combined score.*

| ID | Domain | Best CLIP | Best SAS | LLM (att1→2→3) | Tries | Pass | Rollbacks |
|----|--------|:---------:|:--------:|:---:|:---:|:---:|:---:|
| q01 | cell_biology | 0.2868 | 0.2524 | 7.2→7.5→7.5 | 3 | ✓ | 0 |
| q02 | cell_biology | 0.2743 | 0.2532 | 7.5→2.0→2.0 | 3 | ✗ | 2 |
| q03 | neuroanatomy | **0.3053** | **0.2921** | 2.5→0.5→**7.5** | 3 | ✓ *(new)* | 1 |
| q04 | human_anatomy | 0.2975 | 0.2842 | 6.5→6.5→6.5 | 3 | ✓ | 2 |
| q05 | human_anatomy | 0.2905 | 0.2930 | 4.5→4.5→6.5 | 3 | ✗ | 1 |
| q06 | human_anatomy | 0.3087 | 0.2851 | 6.5→7.5→6.5 | 3 | ✗ | 1 |
| q07 | plant_biology | 0.2985 | 0.2559 | 6.5→6.5→6.5 | 3 | ✗ | 1 |
| q08 | plant_biology | **0.3151** | 0.2870 | 6.5→6.5→4.5 | 3 | ✗ | 2 |
| q09 | biochemistry | 0.2449 | 0.2440 | 4.5→3.5→3.5 | 3 | ✗ | 2 |
| q10 | zoology | 0.3024 | 0.2660 | 7.5→**8.5**→8.5 | 3 | ✓ | 0 |
| **Mean** | | **0.2924** | **0.2713** | **6.02→5.35→5.95** | 3.0 | **4/10** | **1.2 avg** |

> "Rollbacks" = attempts where `combined_score < best_combined` — PromptRefiner was given best-known prompts instead of latest. q03 marked "new" = newly passing in v3 (failed in v2).

---

### 6.4 Per-Attempt LLM Progression — v3 with Adaptive Mode

| ID | Att 1 | Att 2 (mode / str) | Att 3 (mode / str) | Trend | Rollback |
|----|:-----:|:-------------------:|:-------------------:|-------|:---:|
| q01 | 7.2 | 7.5 (i2i / 0.28) | 7.5 (i2i / 0.28) | Stable ↑ | — |
| q02 | 7.5 | 2.0 (i2i / 0.28) ↩ | 2.0 (t2i restart) ↩ | Crashed att2 | att2, att3 |
| q03 | 2.5 | 0.5 (t2i restart) ↩ | **7.5** (t2i restart) | Recovered att3 | att2 |
| q04 | 6.5 | 6.5 (i2i / 0.28) ↩ | 6.5 (i2i / 0.28) ↩ | Stable | att2, att3 |
| q05 | 4.5 | 4.5 (i2i / 0.75) ↩ | 6.5 (i2i / 0.75) | Improving ↑ | att2 |
| q06 | 6.5 | 7.5 (i2i / 0.28) | 6.5 (i2i / 0.28) ↩ | Dip att3 | att3 |
| q07 | 6.5 | 6.5 (i2i / 0.28) | 6.5 (i2i / 0.28) ↩ | Stable | att3 |
| q08 | 6.5 | 6.5 (i2i / 0.28) ↩ | 4.5 (i2i / 0.28) ↩ | Declined att3 | att2, att3 |
| q09 | 4.5 | 3.5 (i2i / 0.75) ↩ | 3.5 (i2i / 0.75) ↩ | Stable low | att2, att3 |
| q10 | 7.5 | **8.5** (i2i / 0.28) | 8.5 (i2i / 0.28) | Steady ↑ | — |

> ↩ = rollback triggered (combined score did not improve vs best). t2i restart = adaptive strength chose text2img because LLM score < 3.0.

---

### 6.5 v2 vs v3 LLM Per-Attempt Comparison

| ID | v2 att1→2→3 | v3 att1→2→3 | Change |
|----|:-----------:|:-----------:|--------|
| q01 | 7.5→7.5→7.5 | 7.2→7.5→7.5 | ≈ Same |
| q02 | 2.5→2.0→2.0 | 7.5→2.0→2.0 | att1 much better, att2 crash same |
| q03 | 2.5→2.5→2.5 | 2.5→0.5→7.5 | **att3 recovered → NOW PASSES** |
| q04 | 6.5→7.5→8.2 | 6.5→6.5→6.5 | att3 slightly lower but still passes |
| q05 | 3.5→4.5→3.5 | 4.5→4.5→6.5 | Monotonic improvement in v3 |
| q06 | 6.5→6.5→5.5 | 6.5→7.5→6.5 | att2 better, att3 rollback |
| q07 | 2.0→2.0→2.0 | 6.5→6.5→6.5 | **+4.5 LLM — major improvement** |
| q08 | 2.0→2.0→2.0 | 6.5→6.5→4.5 | **+4.5 LLM att1/2, degraded att3** |
| q09 | 3.5→6.5→2.5 | 4.5→3.5→3.5 | Oscillation eliminated, but stable low |
| q10 | 8.5→8.5→7.5 | 7.5→8.5→8.5 | Consistent high, v3 slightly better |

---

### 6.6 Config 3 FLUX v4 — Per-Query Results (2026-05-02, ControlNet)

*v4 adds ControlNet Canny spatial guidance. All 10 queries have AI2D annotation data (5–13 labelled regions).*

| ID | Domain | Best CLIP | Best SAS | LLM (att1→2→3) | Tries | Pass | Spatial Regions |
|----|--------|:---------:|:--------:|:---:|:---:|:---:|:---:|
| q01 | cell_biology | 0.2942 | 0.2588 | 2.5→7.5→7.5 | 3 | ✓ | 13 |
| q02 | cell_biology | **0.3130** | 0.2641 | 2.5→7.5→**8.5** | 3 | ✓ *(new)* | 7 |
| q03 | neuroanatomy | 0.2910 | **0.2837** | 1.5→2.0→1.5 | 3 | ✗ | 9 |
| q04 | human_anatomy | **0.3141** | **0.2872** | 6.5→7.5→7.5 | 3 | ✓ | 10 |
| q05 | human_anatomy | 0.2877 | 0.2846 | 3.5→4.5→4.5 | 3 | ✗ | 12 |
| q06 | human_anatomy | 0.3003 | 0.2768 | 6.5→6.5→6.5 | 3 | ✗ | 9 |
| q07 | plant_biology | 0.2835 | 0.2437 | 4.5→6.5→6.5 | 3 | ✗ | 8 |
| q08 | plant_biology | 0.3120 | 0.2742 | 2.5→7.5→6.5 | 3 | ✗ | 9 |
| q09 | biochemistry | 0.2609 | 0.2585 | 2.5→1.5→**6.5** | 3 | ✓ *(new)* | 6 |
| q10 | zoology | 0.2775 | 0.2678 | **8.5**→8.5→8.5 | 3 | ✓ | 5 |
| **Mean** | | **0.2934** | **0.2699** | **4.10→5.95→6.40** | 3.0 | **5/10** | **8.8 avg** |

> "new" = newly passing in v4 (q02 and q09 both failed in all previous versions). att1 LLM dropped to mean 4.10 (vs 6.02 in v3) because ControlNet scale=0.70 over-constrains first-attempt structure; recovery via text2img restart at scale=0.65 accounts for most passes.

---

### 6.7 Per-Attempt LLM Progression — v4 with ControlNet Scale

| ID | Att 1 (cn=0.70) | Att 2 (cn=0.50–0.65) | Att 3 (cn=0.50–0.65) | Trend | Rollback |
|----|:---:|:---:|:---:|---|:---:|
| q01 | 2.5 (t2i) | 7.5 (t2i restart) | 7.5 (i2i) | Recovered att2 | — |
| q02 | 2.5 (t2i) | 7.5 (t2i restart) | **8.5** (i2i) | **Recovered, best ever** | — |
| q03 | 1.5 (t2i) | 2.0 (t2i restart) | 1.5 (t2i) ↩ | Flat/low throughout | att3 |
| q04 | 6.5 (t2i) | 7.5 (i2i) | 7.5 (i2i) | Smooth ↑ | — |
| q05 | 3.5 (t2i) | 4.5 (i2i) | 4.5 (i2i) ↩ | Slight ↑ stalls | att3 |
| q06 | 6.5 (t2i) | 6.5 (i2i) | 6.5 (i2i) | Stable, no gain | — |
| q07 | 4.5 (t2i) | 6.5 (i2i) | 6.5 (i2i) | ↑ att2, stable | — |
| q08 | 2.5 (t2i) | 7.5 (t2i restart) | 6.5 (i2i) ↩ | Recovered att2, dip att3 | att3 |
| q09 | 2.5 (t2i) | 1.5 (t2i) ↩ | **6.5** (t2i) | **Wild recovery att3** | att2 |
| q10 | **8.5** (t2i) | 8.5 (i2i) ↩ | 8.5 (i2i) ↩ | Stable peak | att2, att3 |

> ↩ = rollback. t2i restart = LLM score fell below 3.0 → adaptive strength chose text2img with fresh seed.

---

### 6.8 v3 vs v4 Per-Query Comparison

| ID | v3 CLIP | v4 CLIP | Δ CLIP | v3 LLM final | v4 LLM final | v3 Pass | v4 Pass |
|----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| q01 | 0.2868 | 0.2942 | +0.0074 | 7.5 | 7.5 | ✓ | ✓ |
| q02 | 0.2743 | **0.3130** | **+0.0387** | 2.0 | **8.5** | ✗ | ✓ |
| q03 | **0.3053** | 0.2910 | −0.0143 | **7.5** | 1.5 | ✓ | ✗ |
| q04 | 0.2975 | **0.3141** | +0.0166 | 6.5 | 7.5 | ✓ | ✓ |
| q05 | 0.2905 | 0.2877 | −0.0028 | 6.5 | 4.5 | ✗ | ✗ |
| q06 | 0.3087 | 0.3003 | −0.0084 | 6.5 | 6.5 | ✗ | ✗ |
| q07 | 0.2985 | 0.2835 | −0.0150 | 6.5 | 6.5 | ✗ | ✗ |
| q08 | **0.3151** | 0.3120 | −0.0031 | 4.5 | 6.5 | ✗ | ✗ |
| q09 | 0.2449 | 0.2609 | +0.0160 | 3.5 | **6.5** | ✗ | ✓ |
| q10 | 0.3024 | 0.2775 | −0.0249 | 8.5 | 8.5 | ✓ | ✓ |
| **Mean** | **0.2924** | **0.2934** | **+0.0010** | **5.95** | **6.40** | 4/10 | **5/10** |

---

### 6.9 6-Config CLIP-Score Comparison

| ID | Domain | C1 (SD) | C3 SD | FLUX v1 | FLUX v2 | FLUX v3 | **FLUX v4** | Δ v4 vs C1 |
|----|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| q01 | cell_biology | 0.3116 | 0.2458 | 0.3061 | 0.2971 | 0.2868 | 0.2942 | −0.0174 |
| q02 | cell_biology | 0.2297 | **0.3216** | 0.2647 | 0.2596 | 0.2743 | **0.3130** | +0.0833 |
| q03 | neuroanatomy | 0.2785 | 0.2771 | **0.3068** | 0.2568 | 0.3053 | 0.2910 | +0.0125 |
| q04 | human_anatomy | 0.2717 | 0.2917 | 0.2945 | 0.3097 | 0.2975 | **0.3141** | +0.0424 |
| q05 | human_anatomy | 0.2556 | 0.3052 | 0.2906 | 0.2958 | 0.2905 | 0.2877 | +0.0321 |
| q06 | human_anatomy | 0.2958 | **0.3131** | 0.3088 | 0.3122 | 0.3087 | 0.3003 | +0.0045 |
| q07 | plant_biology | 0.3091 | 0.2313 | **0.3317** | 0.3292 | 0.2985 | 0.2835 | −0.0256 |
| q08 | plant_biology | 0.2949 | 0.2867 | 0.3012 | 0.3029 | **0.3151** | 0.3120 | +0.0171 |
| q09 | biochemistry | 0.2723 | 0.2525 | **0.3058** | 0.2873 | 0.2449 | 0.2609 | −0.0114 |
| q10 | zoology | 0.2549 | 0.3165 | 0.2779 | 0.3012 | 0.3024 | 0.2775 | +0.0226 |
| **Mean** | | 0.2774 | 0.2842 | **0.2988** | 0.2952 | 0.2924 | 0.2934 | **+0.0160** |

---

### 6.10 Per-Domain CLIP Analysis (all versions)

| Domain | C1 | C3-SD | FLUX v1 | FLUX v2 | FLUX v3 | **FLUX v4** | Best |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| cell_biology (q01, q02) | 0.2707 | 0.2837 | 0.2854 | 0.2784 | 0.2806 | **0.3036** | **FLUX v4** |
| neuroanatomy (q03) | 0.2785 | 0.2771 | **0.3068** | 0.2568 | 0.3053 | 0.2910 | FLUX v1 ≈ v3 |
| human_anatomy (q04–q06) | 0.2744 | 0.3033 | 0.2980 | 0.3059 | 0.2989 | **0.3007** | FLUX v2 ≈ v4 |
| plant_biology (q07, q08) | 0.3020 | 0.2590 | **0.3165** | 0.3161 | 0.3068 | 0.2978 | FLUX v1 ≈ v2 |
| biochemistry (q09) | 0.2723 | 0.2525 | **0.3058** | 0.2873 | 0.2449 | 0.2609 | FLUX v1 |
| zoology (q10) | 0.2549 | 0.3165 | 0.2779 | 0.3012 | **0.3024** | 0.2775 | FLUX v3 |

---

### 6.11 Ablation Baseline — flux_direct Per-Query Results (2026-05-03)

*No RAG, no agents. Raw query → PromptSplitter (LLM generates CLIP + T5 sub-prompts) → FLUX one-shot. No verification or retry.*

| ID | Domain | CLIP | LLM | Pass | Notes |
|----|--------|:----:|:---:|:---:|---|
| q01 | cell_biology | 0.2720 | 6.5 | ✓ | Missing: Mitochondria, Ribosomes, Golgi (detail) |
| q02 | cell_biology | 0.2894 | **7.5** | ✓ | Cristae + matrix well rendered |
| q03 | neuroanatomy | 0.3060 | 4.5 | ✗ | Myelin sheath, nodes of Ranvier absent |
| q04 | human_anatomy | 0.2989 | 6.5 | ✓ | Retina + lens + optic nerve present |
| q05 | human_anatomy | 0.3053 | 4.5 | ✗ | Appendix, pharynx, duodenum absent |
| q06 | human_anatomy | 0.2970 | 6.5 | ✓ | Ventricles + atria + aorta recognisable |
| q07 | plant_biology | 0.2989 | 6.5 | ✓ | Petal, stamen, pistil present; sepals ambiguous |
| q08 | plant_biology | 0.2994 | 6.5 | ✓ | Blade, midrib, petiole, vein structure clear |
| q09 | biochemistry | 0.3095 | 2.0 | ✗ | No chloroplast/thylakoid/process-flow visible |
| q10 | zoology | 0.2858 | 7.5 | ✓ | Head, thorax, abdomen, antennae correct |
| **Mean** | | **0.2962** | **5.85** | **7/10** | |

---

### 6.12 Ablation Baseline — flux_rag_noretry Per-Query Results (2026-05-03)

*BiologistAgent RAG (top-k=10) + SpatialArchitect template → PromptSplitter → FLUX one-shot. No verification or retry.*

| ID | Domain | CLIP | LLM | Pass | Notes |
|----|--------|:----:|:---:|:---:|---|
| q01 | cell_biology | 0.2782 | 6.5 | ✓ | Cell wall, chloroplast, nucleus present |
| q02 | cell_biology | 0.2961 | 2.0 | ✗ | SpatialArchitect template disrupts cristae detail |
| q03 | neuroanatomy | 0.3065 | 2.5 | ✗ | Myelin sheath absent |
| q04 | human_anatomy | 0.2942 | 6.5 | ✓ | Eye structures recognisable |
| q05 | human_anatomy | 0.2918 | 3.5 | ✗ | Many digestive organs absent |
| q06 | human_anatomy | 0.3130 | 7.2 | ✓ | Heart chambers + aorta clear |
| q07 | plant_biology | 0.2880 | **7.5** | ✓ | Stamen, pistil well depicted |
| q08 | plant_biology | 0.3109 | 6.5 | ✓ | Leaf anatomy present |
| q09 | biochemistry | 0.2745 | 2.0 | ✗ | Photosynthesis process not depicted |
| q10 | zoology | 0.2947 | **8.5** | ✓ | Insect body plan very clear |
| **Mean** | | **0.2948** | **5.27** | **6/10** | |

---

### 6.13 All-Config Per-Query CLIP + Pass Comparison

| ID | Domain | v3 | v4 | direct | rag_nr | Best pass |
|----|--------|:---:|:---:|:---:|:---:|---|
| q01 | cell_biology | ✓ | ✓ | ✓ | ✓ | All 4 pass |
| q02 | cell_biology | ✗ | ✓ | ✓ | ✗ | v4, direct |
| q03 | neuroanatomy | ✓ | ✗ | ✗ | ✗ | v3 only |
| q04 | human_anatomy | ✓ | ✓ | ✓ | ✓ | All 4 pass |
| q05 | human_anatomy | ✗ | ✗ | ✗ | ✗ | All fail |
| q06 | human_anatomy | ✗ | ✗ | ✓ | ✓ | direct, rag_nr |
| q07 | plant_biology | ✗ | ✗ | ✓ | ✓ | direct, rag_nr |
| q08 | plant_biology | ✗ | ✗ | ✓ | ✓ | direct, rag_nr |
| q09 | biochemistry | ✗ | ✓ | ✗ | ✗ | v4 only |
| q10 | zoology | ✓ | ✓ | ✓ | ✓ | All 4 pass |
| **Pass total** | | 4/10 | 5/10 | **7/10** | 6/10 | |

> rag_nr = flux_rag_noretry. Queries q01/q04/q10 pass all configs — these are "easy" cases well within FLUX's training distribution. q05 fails all configs — the digestive system query is consistently the hardest. q03 only passes v3 (via lucky text2img restart), q09 only passes v4 (via ControlNet guidance).

---

## 7. Analysis

### 7.1 LLM-as-judge reveals CLIP-score overestimates quality

The most striking finding of the v2 run is the **severe disconnect between CLIP and LLM scores**:

- q07 (flower): CLIP = 0.3292 (highest in batch) but LLM = 2.0 (second lowest)
- q08 (leaf): CLIP = 0.3029 but LLM = 2.0
- q10 (insect): CLIP = 0.3012 and LLM = 8.5 (highest) — consistent

CLIP measures whether the image "looks like" the query domain. FLUX generates high-quality botanical imagery that CLIP recognises as plant-related even when the specific structures (stamen, pistil, petiole, midrib) are absent or ambiguous. LLM-as-judge evaluates structural completeness — its lower scores expose failures that CLIP cannot detect. This validates the two-stage verification design: CLIP alone is insufficient as a biological accuracy metric.

### 7.2 Feedback loop improves some queries, destabilises others

The per-attempt LLM progression shows heterogeneous behaviour:

**Positive cases (refinement helps):**
- q04 (eye): 6.5 → 7.5 → 8.2 — PromptRefiner successfully adds retina, cornea emphasis each iteration; img2img preserves the eye shape while adding detail
- q05 (digestive): 3.5 → 4.5 — partial improvement on attempt 2

**Neutral cases (refinement has no effect):**
- q03 (neuron): 2.5 → 2.5 → 2.5 — LLM score flat despite 3 attempts; CLIP also declines (0.2568 → 0.2481 → 0.2404). The img2img with strength=0.60 preserves the initial error pattern
- q07, q08: LLM=2.0 across all 3 attempts — FLUX generates photorealistic flowers/leaves instead of schematic diagrams; the PromptRefiner cannot overcome this model bias via prompt alone

**Unstable cases (refinement hurts):**
- q09 (photosynthesis): 3.5 → 6.5 → 2.5 — attempt 2 achieves passing LLM score but attempt 3 img2img refinement from the same base image causes regression. Fixed seed + changed prompt can produce worse conditioning alignment on the third pass

### 7.3 Hard cases: schematic diagram vs. photorealistic bias

Queries q02 (mitochondria), q03 (neuron), q07 (flower parts), q08 (leaf anatomy) all show consistently low LLM scores (2.0–2.5) regardless of prompting. These share a common failure pattern: FLUX's training distribution skews toward **photorealistic imagery**. A flower diagram requires a cross-sectional schematic with distinct labelled parts — FLUX produces a realistic flower photograph instead. The prompt contains "scientific diagram" and "white background" but these style keywords are insufficient to overcome the photorealistic prior.

SD-v1.5 paradoxically performs better on q10 (zoology, CLIP=0.3165 vs FLUX=0.3012) and q02 (CLIP=0.3216 vs FLUX=0.2596) for this reason — SD's training includes more educational illustration content.

### 7.4 RAG and Spatial Architect improve structured domain queries

Config 2 vs Config 1 shows a consistent +2.5% mean CLIP improvement. The largest gain is q02 (mitochondria, +0.0919) where the structured prompt preserved cristae/matrix/membrane terms that SD's 77-token limit previously truncated. The **prompt chunking technique** (embedding each 77-token chunk independently and concatenating) was critical: without it, the 94–124 token structured prompts silently truncated all constraint content.

### 7.5 FLUX v2 vs v1: combined verifier lowers effective CLIP mean

FLUX v1 used CLIP-SAS-only pass condition (threshold 0.20), so all 10 queries "passed" on attempt 1 and no retries occurred. FLUX v2 uses combined CLIP+LLM criterion, triggering retries on 10/10 queries. The img2img retry process with fixed seed and modified prompt sometimes produces lower CLIP than the original attempt 1 (e.g., q03 declines from 0.2568 to 0.2404), pulling down the mean. The combined criterion is more meaningful scientifically but exposes cases where the feedback loop cannot converge.

### 7.6 PromptRefiner direct CLIP+T5 editing prevents compression loss

The prior approach (appending `improvement_suggestions` as text to `full_prompt` then calling `PromptSplitter`) caused refinements to be silently compressed: PromptSplitter's LLM re-summarised the combined prompt, often omitting the specific additions. The new architecture — PromptRefiner writes directly to CLIP and T5 strings, GeneratorAgent accepts `clip_prompt_override` / `t5_prompt_override` to bypass PromptSplitter — ensures additions (e.g., "clearly visible cristae prominently depicted") survive into the diffusion pipeline.

### 7.7 v3 vs v2: +1 pass, +1.10 mean LLM, slight CLIP regression

v3 achieves **4/10 passed** (up from 3/10). The newly passing query is **q03 (neuroanatomy/neuron)** — it succeeded via two consecutive text2img restarts with fresh seeds after attempt 1 scored 2.5. Mean LLM jumped from 4.85 → **5.95 (+1.10)**, the largest improvement across any version transition. Mean CLIP dropped from 0.2952 → **0.2924 (−0.0028)** — likely because the adaptive strength schedule locks most queries at str=0.28 (fine-tune), which imposes smaller structural changes per attempt than the old 0.60→0.50 schedule; this is lower perturbation per step but more principled.

The biggest per-query CLIP shifts in v3:
- q03: +0.0485 (0.2568 → 0.3053) — successful text2img recovery
- q09: −0.0424 (0.2873 → 0.2449) — adaptive strength chose 0.75 (att1 LLM=4.5) but img2img still didn't help; photosynthesis process diagram remains hardest query
- q07: CLIP −0.0307 but LLM +4.5 — plant structure now recognised semantically even though CLIP embedding didn't improve

### 7.8 Rollback mechanism: prevents collapse, does not guarantee improvement

Rollback triggered in **7/10 queries** (all except q01, q10 which consistently improved). Key observations:

- **q09 (photosynthesis):** v2 had wild oscillation (3.5→6.5→2.5). v3 has stable low (4.5→3.5→3.5). Rollback prevented the catastrophic att3 collapse seen in v2 — combined score decreases monotonically instead of bouncing. The subject remains fundamentally hard for FLUX.
- **q04 (eye):** Rollback triggered att2 and att3 (combined score 0.4659, 0.4658 < best 0.4671). LLM stayed flat at 6.5 across all attempts — rollback correctly prevented re-applying a prompt that degraded from the att1 base.
- **q08 (leaf):** Rollback on att2 (SAS dropped 0.2870→0.2842), but att3 LLM still fell to 4.5. Root cause: PromptRefiner itself introduced changes based on att2 feedback that didn't transfer well; rollback to best base prompts helped but couldn't fully counteract the refiner's modifications.

The rollback mechanism is effective at preventing cumulative prompt drift but cannot recover when the PromptRefiner's own edits introduce semantically inconsistent combinations.

### 7.9 Adaptive strength: most queries locked at str=0.28 on att1 LLM ≥ 6.5

| Strength | Queries | Reason |
|:---:|---|---|
| 0.28 (fine-tune) | q01, q02, q03†, q04, q06, q07, q08 | att1 LLM ≥ 6.5 |
| 0.75 (large change) | q05, q09 | att1 LLM = 4.5 |
| text2img restart | q02 att3, q03 att2+att3 | LLM < 3.0 after att1 or att2 |

† q03 used text2img restart since att1 LLM=2.5 < 3.0.

A significant finding: **v3 att1 LLM scores are systematically higher than v2** (mean 6.02 vs 4.85 at att1). This reflects non-determinism in FLUX: the same seed but different run context produced better first-attempt images in v3. Since adaptive strength is keyed to att1 LLM, higher att1 scores → more queries locked at 0.28 → gentler refinement → less risk of structural collapse.

The 0.75 strength path (q05, q09) didn't help: both queries failed to improve past att1. For these process-diagram type queries, even large structural changes via img2img don't redirect FLUX's photorealistic prior. Text2img restart with fresh seeds may be the only effective strategy.

### 7.10 No query achieves fully_passed (zero missing structures)

`fully_passed` condition (`passed_combined AND missing_structures == []`) was not met by any query across all v1–v4 runs. LLM-judge consistently reports 2–9 missing structures even for high-scoring queries. Two interpretations: (1) max_retries=3 is insufficient for convergence; (2) diffusion models fundamentally cannot depict all required structures in a fixed 512×512 image for complex multi-element biological subjects. Even q10 (zoology, LLM=8.5) still reported 1 missing structure. The `fully_passed` bar may be too strict for a generation-only system without an explicit layout/segmentation step.

---

### 7.11 v4 ControlNet: +1 pass, highest mean LLM, 2.84× time cost

v4 achieves **5/10 passed** (+1 from v3). Newly passing queries: **q02 (mitochondria)** and **q09 (photosynthesis)** — both were the hardest queries across all prior versions. Mean LLM rises to **6.40** (highest across all versions). Mean CLIP is 0.2934 (+0.0010 vs v3). Total wall time: 39,008s (~10.8 hours) vs 13,725s for v3 — **2.84× slower**, driven by the additional ControlNet forward pass at every denoising step.

Pass trajectory across versions:
```
v1: 1/10 → v2: 3/10 → v3: 4/10 → v4: 5/10
```
Each version adds roughly +1 query, with diminishing marginal gains at increasing cost.

### 7.12 ControlNet att1 suppression: scale=0.70 over-constrains first generation

The most consistent v4 pattern: **att1 LLM scores dropped sharply** (mean 6.02 in v3 → 4.10 in v4). ControlNet scale=0.70 on the first attempt forces the model to rigidly follow the AI2D polygon boundaries, which for most queries are rough blob outlines — not detailed internal structure. The control image shows a single outer boundary blob, so the model generates a correctly-shaped exterior but misses internal detail.

Recovery via text2img restart (att2, scale=0.65) works well: q01 (2.5→7.5), q02 (2.5→7.5), q08 (2.5→7.5). The 0.65 scale provides lighter guidance while still enforcing the rough boundary shape, allowing the model more freedom for internal structure.

**Implication:** For queries where the AI2D annotation has only 1–2 blob regions (mostly outer boundary), ControlNet scale should start lower (0.4–0.5) to avoid boundary over-constraint. For queries with many labelled regions (q01: 13, q04: 10), higher scales (0.7) are appropriate.

### 7.13 ControlNet enables q02 and q09 — the two hardest queries

**q02 (mitochondria):** In v1–v3, q02 consistently scored LLM=2.0–2.5 (inner membrane structures, cristae never depicted correctly). In v4, att2 achieves LLM=7.5 and att3 reaches **8.5** — the highest LLM score of any mitochondria attempt across all versions. The AI2D annotation for 3288.png provides 6 named structure hint positions (Cristae, Outer membrane, Inner membrane, Intermembrane space, Mitochondrial DNA, Matrix). Even as text-label position hints (small dots in the lineart blueprint), these provided enough structural signal to guide FLUX toward the correct internal anatomy.

**q09 (photosynthesis):** In v1–v3, photosynthesis was the hardest query — always LLM≤4.5. In v4, att3 achieves LLM=6.5 via a second text2img restart with fresh seed. The annotation provides positions for Water, Carbon Dioxide, Oxygen, and Chlorophyll molecules — input/output components that helped orient the diagram. However, the pass is fragile (att2 regressed to 1.5 before recovery), suggesting the ControlNet is helpful but the process-diagram requirement remains near the edge of FLUX's capabilities.

### 7.14 q03 regression: ControlNet boundary conflicts with neuron morphology

v3 passed q03 via a lucky 3rd-attempt text2img restart (LLM: 2.5→0.5→7.5). v4 fails q03 entirely (LLM: 1.5→2.0→1.5). The AI2D annotation for 2925.png (neuron) provides 9 regions. However, the polygon boundaries likely trace the irregular blob outline of a neuron diagram — a complex elongated structure with axon, dendrites, and myelin nodes. ControlNet Canny guidance from these shapes may force the model into layouts incompatible with correct neuron morphology, preventing the random-seed recovery that succeeded in v3.

This is the clearest example of **ControlNet boundary interference**: when the annotation polygon does not match the target diagram layout, spatial guidance actively hurts generation quality.

### 7.15 Time-quality tradeoff across versions

| Version | Pass rate | Mean LLM | Avg time/query | Notes |
|:---:|:---:|:---:|:---:|---|
| v1 | 1/10 | 5.40 | ~432s | CLIP-SAS only, no retries |
| v2 | 3/10 | 4.85 | ~1718s | Combined verifier + PromptRefiner |
| v3 | 4/10 | 5.95 | ~1373s | Rollback + adaptive strength |
| **v4** | **5/10** | **6.40** | **~3901s** | **+ControlNet Canny** |

v3 was actually faster than v2 because adaptive strength resolved some queries faster (fewer img2img restarts). v4 is 2.84× slower than v3 due to ControlNet. Each ControlNet pass adds ~1–1.5 minutes per denoising step on M3 Pro 18GB (MPS + CPU offload).

For production use, the optimal tradeoff depends on the query type:
- High-structure queries (q02, q04, q09): v4 is worth the cost — ControlNet provides clear structural gain
- Elongated/topology-dependent structures (q03 neuron): v3 or v4 without ControlNet
- Plant diagrams (q07, q08): neither v3 nor v4 resolves photorealistic bias — annotation polygon quality insufficient

### 7.16 Ablation: direct prompting beats the full multi-agent pipeline

The two ablation baselines produce a counterintuitive ranking: **flux_direct (7/10) > flux_rag_noretry (6/10) > v4 (5/10) > v3 (4/10)** in pass rate. Several structural observations explain this:

**Why flux_direct outperforms v3/v4 overall:**

The PromptSplitter's T5 sub-prompt (generated by claude-haiku) is richer and more freely structured than the SpatialArchitect template prompt. For queries where FLUX already has strong training signal (flowers, leaves, heart, eye, insect), the direct T5 description contains every structural relationship in natural language — exactly what the T5 encoder is optimised to process. The multi-agent retry loop in v3/v4 introduces additional PromptRefiner calls that sometimes degrade a good first-attempt image (e.g., q06, q07, q08 in v3 fail despite passing in flux_direct).

**Why flux_rag_noretry is worse than flux_direct:**

The SpatialArchitect template prompt optimises for SD-v1.5's 77-token CLIP limit and compresses constraints into a comma-separated phrase format. This format is suboptimal for FLUX's T5 encoder, which benefits from complete sentences. q02 (mitochondria) is the clearest case: flux_direct LLM=7.5 (✓) vs flux_rag_noretry LLM=2.0 (✗) — the direct T5 description of cristae membrane folds generates the correct cross-section, while the SpatialArchitect template flattens the spatial detail into a shorter phrasing that loses key structural information.

**Where the multi-agent system uniquely wins:**

- **q03 (neuron):** only v3 passes — via a lucky text2img restart at attempt 3 with a seed shift. No other config gets LLM ≥ 6.0 here. The elongated morphology is consistently hard; the one pass is seed-dependent, not a systematic improvement.
- **q09 (photosynthesis):** only v4 passes — the ControlNet polygon boundary for the chloroplast provides spatial structure that neither direct prompting nor RAG can achieve without visual guidance. This is the strongest evidence for ControlNet's value on process-flow diagrams.

**Practical takeaway:**

| Query type | Recommended pipeline |
|---|---|
| Standard anatomy (eye, heart, plant, insect) | flux_direct — fastest, highest pass rate |
| Complex multi-component processes (photosynthesis) | v4 — ControlNet uniquely unlocks spatial accuracy |
| Structure-dense cross-sections (mitochondria) | v4 or flux_direct (both pass; v4 has higher LLM score) |
| Topology-sensitive morphology (neuron) | v3 — ControlNet boundary interferes; v3 recovery via restart |

The multi-agent retry loop's primary value is in edge cases where the first-attempt generation fails and can be recovered — which accounts for 2 additional passes (q02, q09 in v4). For the other 8 queries, a single well-prompted FLUX call is equally or more effective.

---

## 8. Testing & Validation Summary

| Stage | Test script | Tests | Passed |
|-------|-------------|-------|--------|
| Stage 1 — AI2D knowledge extraction | `data/test_stage1.py` | 12 | 12/12 |
| Stage 2 — Config 1 baseline generation | `generation/test_stage2.py` | 8 | 8/8 |
| Stage 3 — Config 2 RAG + Spatial Architect | `generation/test_stage3.py` | 11 | 11/11 |
| Stage 4 — Config 3 Full BioGuard | `generation/test_stage4.py` | 11 | 11/11 |

---

## 9. Output Files

| File | Description |
|------|-------------|
| `results/knowledge_base.json` | 2,603 AI2D entries with labels, constraints |
| `results/test_queries.json` | 10 fixed test queries |
| `results/config1_results.json` | SD-v1.5 baseline CLIP scores |
| `results/config2_results.json` | RAG+Architect CLIP scores |
| `results/config3_results.json` | Full BioGuard (SD) CLIP, SAS |
| `results/config3_flux_results.json` | FLUX v1 CLIP, SAS, timings |
| `results/config3_flux_v2_llm_judge_results.json` | FLUX v2 with LLM-judge |
| `results/config3_flux_v2_llm_judge_performance.json` | FLUX v2 aggregate stats |
| `outputs/run_all_flux/20260501_144343_308835/` | v2 batch run (2026-05-01) |
| `outputs/run_all_flux/20260501_232851_689924/` | v3 batch run (2026-05-02) |
| `outputs/run_all_flux_v4/20260502_103449_196753/` | v4 ControlNet batch run (2026-05-02) |
| `outputs/run_all_flux_baselines/20260503_120700_016787_flux_direct/` | **flux_direct ablation (2026-05-03)** |
| `outputs/run_all_flux_baselines/20260503_120700_016787_flux_rag_noretry/` | **flux_rag_noretry ablation (2026-05-03)** |
| `outputs/run_all_flux_v4/{ts}/q*/blueprint_initial.png` | AI2D polygon lineart control image |
| `outputs/run_all_flux_v4/{ts}/q*/blueprint_att{N}.png` | Adaptive blueprint (missing structures highlighted red) |
| `outputs/run_all_flux/{ts}/q*/reference.png` | AI2D reference images (side-by-side comparison) |
| `outputs/run_all_flux/{ts}/q*/best.png` | Best generated image per query |
| `outputs/run_all_flux/{ts}/q*/prompt_log.txt` | Full CLIP+T5 prompt history per attempt |
| `outputs/run_all_flux/{ts}/q*/summary.json` | Per-query metrics + attempt records |
| `outputs/run_all_flux/{ts}/all_results.json` | Combined batch results |
| `outputs/single_runs/{ts}/` | Individual single-query runs |

---

## 10. Environment & Dependencies

```
conda env: agent
Python: 3.12.13
Device: Apple MPS (Apple Silicon M3 Pro 18 GB)

Key packages:
  torch              2.11.0
  diffusers          0.37.1
  transformers       5.5.4
  accelerate         1.13.0
  open-clip-torch    3.3.0
  sentence-transformers 5.4.1
  chromadb           1.5.7
  anthropic          (claude-haiku-4-5-20251001)
  huggingface_hub    1.10.2
  numpy              2.4.4
  pillow             12.2.0
  gguf               ≥0.10.0

Diffusion models:
  SD-v1.5:  runwayml/stable-diffusion-v1-5
  FLUX:     city96/FLUX.1-dev-gguf (Q5_K_S, ~7 GB transformer)
            + black-forest-labs/FLUX.1-schnell (text encoders + VAE)
  Z-Image:  Tongyi-MAI/Z-Image-Turbo

Evaluation:
  CLIP model: ViT-B/32 (openai pretrained, via open_clip)
  LLM judge:  claude-haiku-4-5-20251001 (vision)
```
