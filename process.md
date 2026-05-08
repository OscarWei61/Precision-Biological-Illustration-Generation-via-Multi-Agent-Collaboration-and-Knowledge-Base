# BioGuard-Diffusion — Pipeline Process Description

**v3 vs v4 step-by-step walkthrough and difference analysis**

---

## v3 Pipeline — Step-by-Step

v3 is the base FLUX multi-agent pipeline with rollback and adaptive strength. It has **5 active agents** in a closed feedback loop.

---

### Pre-run (once, shared across all queries)

Before the first query runs, the system loads all agents into memory once:

- **BiologistAgent** builds or loads a ChromaDB persistent vector store from `knowledge_base.json`. The store contains 2,673 documents — each biological constraint sentence indexed individually using `all-MiniLM-L6-v2` embeddings.
- **SpatialArchitectAgent** is stateless (no loading cost).
- **VerifierAgent** loads CLIP ViT-B/32 into memory and initialises the Anthropic client for LLM-as-judge calls.
- **CLIPScorer** loads a second CLIP ViT-B/32 for standalone CLIP-score evaluation.
- **FluxGGUFPipeline** downloads and loads the FLUX.1-dev Q5_K_S GGUF transformer (~7 GB) plus the text encoders (CLIP-L, T5-XXL) and VAE from `black-forest-labs/FLUX.1-schnell`. This is the heaviest step — approximately 3–5 minutes on first run.

---

### Per-query execution (runs for each of the 10 test queries)

#### Step 1 — RAG Retrieval

The user query (e.g. "Draw a detailed cross-section of a mitochondria showing cristae and matrix") is passed to **BiologistAgent.retrieve(query, top_k=8)**. ChromaDB embeds the query and returns the 8 most semantically relevant constraint sentences from the knowledge base. For a mitochondria query these might include "The inner membrane of a mitochondrion folds into structures called cristae" and "The matrix is the space enclosed by the inner membrane."

These constraints serve two roles: they inform the SpatialArchitect's prompt, and they are used by the Verifier to score the generated image.

#### Step 2 — Prompt Construction

**SpatialArchitectAgent.build_prompt(query, retrieved_constraints)** assembles the positive FLUX prompt. It strips instruction verbs from the query to extract the biological subject, selects up to 5 of the most relevant constraints, converts each into a compact phrase, and concatenates them with a fixed style suffix ("scientific educational illustration, white background, no text, no labels"). The result is a single structured prompt string.

This prompt is then passed to **PromptSplitter**, which calls `claude-haiku-4-5` twice:
- First call produces the **CLIP sub-prompt** (≤77 tokens, keyword style) tuned for FLUX's CLIP-L encoder.
- Second call produces the **T5 sub-prompt** (unlimited length, full natural language sentences) tuned for FLUX's T5-XXL encoder.

#### Step 3 — Image Generation (Attempt 1)

**GeneratorAgent.generate()** calls `FluxGGUFPipeline.generate(prompt=clip_prompt, prompt_2=t5_prompt, seed=base_seed, steps=20)`. This produces a 512×512 PNG using text-to-image diffusion. The generation takes approximately 400–450 seconds on Apple M3 Pro MPS. The result is saved as `attempt_01_text2img_seed42.png`.

#### Step 4 — Two-Stage Verification

**VerifierAgent.verify_combined()** runs two independent evaluations:

**Stage 1 — CLIP-SAS:** For each constraint sentence, the verifier extracts keywords (stripping stop-words), then computes CLIP cosine similarity between the generated image and that keyword string. The Scientific Accuracy Score (SAS) is the mean across all constraints. A constraint passes if its score ≥ 0.22.

**Stage 2 — LLM-as-judge:** The image is base64-encoded and sent to `claude-haiku-4-5` (vision mode) along with the original query and full constraint list. The model returns a JSON object with: `score` (0–10), `present_structures` (list), `missing_structures` (list), and `improvement_suggestions` (text). Passing threshold is 6.0/10.

Combined pass condition: CLIP-SAS ≥ 0.22 **AND** LLM score ≥ 6.0. If both conditions are met and `missing_structures` is empty, the loop exits immediately with this image as the final result.

#### Step 5 — Rollback Decision

After verification, the system computes a **combined score** = 0.5 × SAS + 0.5 × (LLM/10). This score is compared to the best combined score seen so far.

If the current attempt's combined score is higher than the previous best, the system records this attempt's CLIP prompt and T5 prompt as the new best prompts (`best_clip_prompt`, `best_t5_prompt`).

If the score is lower (regression), a **rollback** is triggered: the next refinement will be based on the best-known prompts, not the current attempt's prompts. This prevents a bad retry from compounding errors.

#### Step 6 — Adaptive Strength Decision

Based on the LLM score from the current attempt, the system decides how to approach the next attempt:

| LLM score | Next mode | Strength |
|:---:|:---:|:---:|
| < 3.0 | Text-to-image restart | — (fresh generation, new seed) |
| 3.0–5.0 | Image-to-image | 0.75 (large change allowed) |
| 5.0–6.5 | Image-to-image | 0.55 (moderate refinement) |
| ≥ 6.5 | Image-to-image | 0.28 (fine-tune only) |

For text2img restarts, the seed is `base_seed + (attempt-1) × 7` to get a different but reproducible random starting point. For img2img refinement, the fixed base seed is used to keep generation stable.

#### Step 7 — Prompt Refinement

**PromptRefinerAgent.refine()** calls `claude-haiku-4-5` with the best-known CLIP and T5 prompts plus the verifier's structured feedback (`present_structures`, `missing_structures`, `improvement_suggestions`). The refiner makes targeted edits: it preserves terms for structures that are already present and inserts stronger emphasis on structures that are missing by name. It respects hard token limits (CLIP ≤ 55 words, T5 ≤ 150 words) to avoid truncation.

The refined CLIP and T5 prompts are stored as `refined_clip_prompt` and `refined_t5_prompt` for use in the next generation call.

#### Step 8 — Retry (Steps 3–7 repeat)

The system repeats Steps 3–7 for up to 3 total attempts. If the image passes at any attempt, the loop exits. After 3 attempts, the image with the highest combined score is saved as `best.png`.

---

## v4 Pipeline — Step-by-Step

v4 adds a **Spatial Pre-Processing Layer** upstream of the generation loop. Two new agents (VisualPlannerAgent and BlueprintGenerator) run before the first attempt and produce a ControlNet control image that is fed into every generation call. Everything else in Steps 1–8 is identical to v3.

---

### Pre-run additions (v4 only)

In addition to all v3 agents, v4 loads:

- **VisualPlannerAgent** — stateless, reads AI2D annotation JSON files at query time.
- **BlueprintGenerator** — stateless PIL image renderer.
- **FluxControlNetPipeline** replaces `FluxGGUFPipeline`. It loads the same GGUF transformer and FLUX text encoders, plus the ControlNet model (`InstantX/FLUX.1-dev-Controlnet-Canny`, ~4 GB). Both a text2img pipeline (`FluxControlNetPipeline`) and an img2img pipeline (`FluxControlNetImg2ImgPipeline`) are assembled. Loading takes approximately 5–8 minutes on first run.

---

### Per-query execution — v4 additions before the attempt loop

#### Step 0A — Spatial Layout Extraction (NEW)

**VisualPlannerAgent.plan(image_name, subject)** reads the AI2D annotation JSON for the query's reference image. The annotation contains three types of data: `blobs` (polygon coordinates of biological regions), `text` (label rectangles with text values), and `relationships` (connections between labels and blobs).

The agent handles two relationship strategies:

**Strategy A — `intraObjectLabel`**: The text label is directly embedded on its blob. The blob's full polygon is extracted and recorded as a named structure (e.g. "Nucleus" → full polygon outline).

**Strategy B — `intraObjectLinkage`** (most common in AI2D): A text label connects via an arrow relationship to a blob. The text rectangle's centre point is used as a position hint for that structure. The linked blob's polygon is also extracted as an unlabelled structural boundary. For example, a label "Cristae" pointing via arrow to a fold polygon produces a small square hint polygon at the text position plus the blob polygon as a boundary.

All coordinates are scaled from the original image dimensions to 512×512 using linear scaling (`sx(x) = int(x × 512 / orig_w)`).

The output is a `layout` dict with a list of items (name, centre, bounds, polygon, is_label_hint) and a `has_spatial_data` flag. If no annotation file exists for the image, `has_spatial_data` is False.

#### Step 0B — Blueprint Rendering (NEW)

**BlueprintGenerator.generate(layout, mode="lineart")** renders the layout dict into a 512×512 RGB control image:

In **lineart mode** (default): white background, black polygon outlines (width=3). Label hint items are drawn as small grey filled dots (radius=4 px) at the text centre. Structural polygon items are drawn as outlines. This image is saved as `blueprint_initial.png` and used as the ControlNet conditioning signal.

If `has_spatial_data` is False, the generator returns a fully black image. A black control image with ControlNet scale=0.0 has zero effect on generation — the pipeline gracefully degrades to v3 behaviour.

---

### Per-attempt changes in v4

#### Step 3 — Generation with ControlNet (CHANGED)

Every generation call now passes two additional arguments: `control_image=current_blueprint` and `controlnet_conditioning_scale=cn_scale`.

The ControlNet scale is determined per-attempt by `_controlnet_scale_for_attempt(prev_llm, attempt, has_spatial)`:

- Attempt 1: scale = **0.70** — strong spatial guidance on the first generation.
- Attempt 2+ with previous LLM < 5.0: scale = **0.65** — keep strong guidance because the image is still far from target.
- Attempt 2+ with previous LLM ≥ 5.0: scale = **0.50** — lighter touch to allow refinement without over-constraining.
- No spatial data: scale = **0.00** — ControlNet has no effect.

Internally, `FluxControlNetPipeline` runs the ControlNet encoder on `current_blueprint` at the beginning of each denoising step to produce residual activations. These residuals are injected into the FLUX transformer's attention layers at each step. This is the main source of v4's 2.84× time overhead — every denoising step now runs both the ControlNet encoder and the FLUX transformer instead of just the transformer.

#### Step 7B — Adaptive Blueprint Update (NEW)

After the PromptRefiner runs (Step 7), if `has_spatial_data` is True and `missing_structures` is non-empty, the blueprint is regenerated using **BlueprintGenerator.highlight_missing(layout, missing_structures)**. This redraws the same polygon layout, but polygons whose names match any entry in `missing_structures` are drawn in bold red (width=6, colour RGB(200,0,0)) while all other polygons remain thin black (width=2). The matching is case-insensitive substring: "cristae" matches "Cristae" or "inner membrane cristae".

The updated blueprint is saved as `blueprint_att{N}.png` and replaces `current_blueprint` for the next attempt. The effect is that the ControlNet signal now emphasises the missing regions with a stronger visual cue, nudging the next generation to place those structures more prominently.

---

## Differences Summary

| Aspect | v3 | v4 |
|--------|----|----|
| **Agents** | 5 (Biologist, SpatialArchitect, Generator, Verifier, PromptRefiner) | 7 (+ VisualPlanner, BlueprintGenerator) |
| **Diffusion backbone** | `FluxGGUFPipeline` (FLUX GGUF only) | `FluxControlNetPipeline` (FLUX GGUF + InstantX ControlNet Canny) |
| **Spatial input** | None — purely text-driven | AI2D polygon annotations → lineart control image |
| **ControlNet scale** | N/A | 0.70 (att1) → 0.50–0.65 (att2+) → 0.00 (no annotation) |
| **First-attempt seed** | `base_seed + (attempt-1) × 7` | Same |
| **Control image update** | N/A | `highlight_missing()` after each failed attempt |
| **Rollback logic** | `combined_score < best_combined` → use best prompts | Identical |
| **Adaptive strength** | LLM-driven (identical thresholds) | Identical |
| **Pass condition** | CLIP ≥ 0.22 AND LLM ≥ 6.0 | Identical |
| **Time per query** | ~1373 s | ~3901 s (2.84× slower) |
| **Pass rate (10 queries)** | 4/10 | 5/10 |
| **Mean LLM score** | 5.95 | 6.40 |
| **Queries newly enabled** | — | q02 (mitochondria), q09 (photosynthesis) |
| **Queries regressed** | — | q03 (neuron) — ControlNet polygon conflicts with elongated morphology |

---

## Decision Logic Comparison — Attempt 2 Branching

The clearest structural difference between v3 and v4 is what happens after attempt 1 fails. The logic is identical except for two new v4 branches:

```
After attempt 1 verification:
  ├── SAME in v3 and v4 ──────────────────────────────────────────────────
  │   Compute combined_score = 0.5 × SAS + 0.5 × (LLM/10)
  │   If improved → update best_image, best_clip_prompt, best_t5_prompt
  │   Else        → rollback flag set (next refine uses best prompts)
  │
  │   Decide next generation mode via _adaptive_strength(LLM, SAS):
  │     LLM < 3   → text2img restart (fresh seed)
  │     LLM 3–5   → img2img strength=0.75
  │     LLM 5–6.5 → img2img strength=0.55
  │     LLM ≥ 6.5 → img2img strength=0.28
  │
  │   Run PromptRefiner on best_clip_prompt + best_t5_prompt
  │     → refined_clip_prompt, refined_t5_prompt for next attempt
  │
  ├── NEW in v4 only ──────────────────────────────────────────────────────
  │   Compute cn_scale for attempt 2:
  │     If LLM < 5  → cn_scale = 0.65
  │     If LLM ≥ 5  → cn_scale = 0.50
  │
  │   If has_spatial_data AND missing_structures is non-empty:
  │     blueprint_gen.highlight_missing(layout, missing_structures)
  │     → current_blueprint updated with red-highlighted missing regions
  │     → saved as blueprint_att02.png
  │
  └── Proceed to attempt 2 generation
        v3: FluxGGUFPipeline(refined prompts, img2img/t2i)
        v4: FluxControlNetPipeline(refined prompts, current_blueprint,
                                   cn_scale, img2img/t2i)
```

---

## Why v4 Does Not Always Outperform v3

The ControlNet polygon data comes from AI2D annotations, which were drawn for the original reference diagram — not for the subject FLUX would naturally generate. Two failure modes arise:

**Mismatched morphology (q03 — neuron):** The AI2D annotation for the neuron reference image contains polygons from a 2D top-down or cross-section perspective. FLUX's natural generation of a neuron is a long horizontal cell body with branching dendrites and an axon. The ControlNet polygon (e.g. a rough oval for the soma, short blobs for dendrites) is spatially incompatible with this morphology. The ControlNet guidance actively conflicts with FLUX's natural output, preventing the lucky text2img restart that succeeded in v3. In v4 all 3 attempts stay stuck in the low-LLM range (1.5–2.0).

**Outer boundary only (q05–q08):** Most AI2D annotations contain only the outer boundary blob of the subject (one large polygon enclosing everything) plus a few text-linked blobs for major sub-structures. For queries like flower parts or digestive system, this outer boundary provides insufficient spatial discrimination — FLUX is constrained to "put the content inside this shape" but gets no guidance on the internal arrangement of stamen vs pistil, or small intestine vs large intestine. The ControlNet signal is essentially a bounding shape, which FLUX already handles well without guidance.

The two queries where ControlNet uniquely helps are q02 (mitochondria) and q09 (photosynthesis). Both have AI2D annotations with multiple labelled blobs at different spatial positions. For mitochondria, the cristae fold polygons provide clear edge patterns that overlap with the structural detail FLUX needs to draw. For photosynthesis, the annotation provides spatial separation between light-reaction and Calvin-cycle regions that FLUX alone cannot infer from text.
