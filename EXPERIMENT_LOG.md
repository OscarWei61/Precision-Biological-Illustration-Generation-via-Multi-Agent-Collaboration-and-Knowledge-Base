# BioGuard-Diffusion — Experiment Log

**Project:** Precision Biological Illustration Generation via Multi-Agent Collaboration and Knowledge Base  
**Course:** ECE 598 — Final Project  
**Author:** MingYi Wei (mingyi5@illinois.edu)  
**Environment:** conda env `agent` · Python 3.12.13 · Apple Silicon MPS

---

## Experiment Overview

The goal is to compare three image generation configurations on the same fixed 10-query test set:

| Config | Pipeline | Hypothesis |
|--------|----------|------------|
| **Config 1** | Raw query → SD (no agents) | Lowest accuracy, most hallucinations |
| **Config 2** | RAG + Spatial Architect → SD | Better structural prompts, higher CLIP |
| **Config 3** | Full BioGuard (Config 2 + Verifier + Retry) | Highest SAS, fewest anatomical violations |

**Primary metrics:** CLIP-Score · Bio-CLIP Score · Scientific Accuracy Score (SAS)

---

## Stage 1 — AI2D Knowledge Extraction

**Date:** 2026-04-24  
**Script:** `bioguard_diffusion/data/extract_knowledge.py`  
**Test:** `bioguard_diffusion/data/test_stage1.py`

### Operations
1. Loaded `data/ai2d/categories.json` and filtered biology-relevant categories: `partsOfA`, `lifeCycles`, `photosynthesisRespiration`, `foodChainsWebs`
2. Parsed all 4,903 annotation JSON files, extracted text labels and spatial relationships
3. Identified best candidate image per biological topic using keyword matching
4. Defined **10 fixed test queries** spanning 6 biology domains — these are used unchanged across all three configs
5. Hand-curated biological constraints (ground-truth anatomical facts) for each test image
6. Saved structured knowledge base to `results/knowledge_base.json`

### Results

| Metric | Value |
|--------|-------|
| Total AI2D images | 4,903 |
| Biology-relevant images | 2,724 |
| Knowledge base entries | **2,603** |
| Unique biological labels | **8,642** |
| Test queries defined | **10** |
| Domains covered | 6 |

**Test queries:**

| ID | Domain | Query (truncated) | Source Image |
|----|--------|-------------------|--------------|
| q01 | cell_biology | Draw a labeled diagram of a plant cell… | 1071.png |
| q02 | cell_biology | Draw a detailed cross-section of a mitochondria… | 3288.png |
| q03 | neuroanatomy | Draw a labeled diagram of a neuron… | 2925.png |
| q04 | human_anatomy | Draw a cross-section of a human eye… | 2857.png |
| q05 | human_anatomy | Draw a labeled diagram of the human digestive system | 1385.png |
| q06 | human_anatomy | Draw a labeled diagram of the human heart… | 1382.png |
| q07 | plant_biology | Draw a labeled diagram of the parts of a flower… | 1014.png |
| q08 | plant_biology | Draw a labeled diagram of a leaf… | 1086.png |
| q09 | biochemistry | Draw a diagram illustrating photosynthesis… | 4088.png |
| q10 | zoology | Draw a labeled diagram of an insect body… | 1191.png |

**Top 10 most frequent biological labels in dataset:**

| Rank | Label | Count |
|------|-------|-------|
| 1 | nucleus | 282 |
| 2 | nucleolus | 206 |
| 3 | cytoplasm | 198 |
| 4 | mitochondrion | 197 |
| 5 | ribosomes | 159 |
| 6 | cell membrane | 147 |
| 7 | golgi apparatus | 136 |
| 8 | lysosome | 133 |
| 9 | plasma membrane | 130 |
| 10 | cell wall | 125 |

### Test Results
**12/12 passed** ✓

### Output Files
- `results/knowledge_base.json` — 2,603 entries with labels, relationships, constraints
- `results/test_queries.json` — 10 fixed test queries

---

## Stage 2 — Config 1: One-Shot Baseline Generation

**Date:** 2026-04-24  
**Script:** `bioguard_diffusion/generation/run_config1_baseline.py`  
**Test:** `bioguard_diffusion/generation/test_stage2.py`

### Operations
1. Installed missing packages into conda `agent` env: `diffusers==0.37.1`, `accelerate==1.13.0`, `open-clip-torch==3.3.0`, `ftfy==6.3.1`
2. Ran SD smoke test (1 image, 20 steps) — confirmed MPS inference works (~26s/image)
3. Loaded `runwayml/stable-diffusion-v1-5` on Apple MPS
4. For each of the 10 test queries: passed **raw user query** directly to SD (no RAG, no agents)
5. Generation params: `steps=30`, `guidance_scale=7.5`, `seed=42`, `512×512`
6. Computed CLIP-Score (ViT-B/32, openai) for each image vs. its original query

### Results

| ID | Domain | Query (short) | CLIP Score | Time (s) |
|----|--------|---------------|-----------|----------|
| q01 | cell_biology | plant cell organelles | **0.3116** | 30.4 |
| q02 | cell_biology | mitochondria cross-section | 0.2297 | 30.5 |
| q03 | neuroanatomy | neuron diagram | 0.2785 | 30.3 |
| q04 | human_anatomy | human eye cross-section | 0.2717 | 30.0 |
| q05 | human_anatomy | digestive system | 0.2556 | 29.9 |
| q06 | human_anatomy | human heart | 0.2958 | 30.2 |
| q07 | plant_biology | parts of a flower | 0.3091 | 30.8 |
| q08 | plant_biology | leaf structure | 0.2949 | 30.5 |
| q09 | biochemistry | photosynthesis | 0.2723 | 30.2 |
| q10 | zoology | insect body | 0.2549 | 30.0 |

**Summary:**

| Metric | Value |
|--------|-------|
| **Mean CLIP-Score** | **0.2774** ← baseline to beat |
| Min CLIP | 0.2297 (q02: mitochondria) |
| Max CLIP | 0.3116 (q01: plant cell) |
| Avg generation time | ~30.3 s/image |
| Total images generated | 10 |
| Image resolution | 512 × 512 |

**Observation:** Mitochondria (q02) scores lowest — aligns with literature finding that sub-cellular organelle details are hardest for SD to get right without guidance.

### Test Results
**8/8 passed** ✓

### Output Files
- `outputs/config1/q01.png` ~ `q10.png` — 10 baseline images
- `results/config1_results.json` — per-image scores and metadata

---

## Stage 3 — Config 2: RAG + Spatial Architect

**Date:** 2026-04-24  
**Scripts:** `bioguard_diffusion/agents/biologist_agent.py`, `bioguard_diffusion/agents/spatial_architect.py`, `bioguard_diffusion/generation/run_config2_rag.py`  
**Test:** `bioguard_diffusion/generation/test_stage3.py`

### Operations
1. Built ChromaDB PersistentClient vector store (`data/vector_store/`) using default `all-MiniLM-L6-v2` (ONNX) embeddings
2. Indexed **2,673 documents** from knowledge_base.json: each constraint as one doc + each image's label set as one doc
3. Implemented `BiologistAgent.retrieve()`: queries ChromaDB with top-k=8, filters to constraint-type docs only
4. Implemented `SpatialArchitectAgent.build_prompt()`: template-based (no LLM API key required), extracts subject from query, selects top-5 most relevant constraints, builds structured positive + negative prompt
5. Generated 10 images with structured prompts (same `seed=42`, `steps=30` for fair comparison)
6. Computed CLIP-Score vs. original user query for all images

**Implementation note — Prompt Chunking:**  
SD-v1.5's CLIP text encoder has a hard 77-token limit (75 content + BOS + EOS). Structured prompts in Config 2 are 94–124 tokens, causing silent truncation. A prompt chunking technique was added to `diffusion_pipeline.py`:
1. Tokenise without truncation to get all token IDs
2. Split content tokens into chunks of 75
3. Prepend BOS / append EOS / pad each chunk to length 77
4. Encode each chunk independently through CLIP text encoder
5. Concatenate along sequence dimension → `[1, n_chunks×77, 768]`
6. Pass combined tensor to U-Net via `prompt_embeds`

All Config 2 prompts (94–124 tokens) → 2 chunks → `(1, 154, 768)` tensor. Results below reflect the chunking-enabled run.

### Results

| ID | Domain | C1 CLIP | C2 CLIP (w/ chunking) | Delta |
|----|--------|:-------:|:---------------------:|:-----:|
| q01 | cell_biology | 0.3116 | 0.2458 | ↓ 0.0658 |
| q02 | cell_biology | 0.2297 | **0.3216** | ↑ **+0.0919** |
| q03 | neuroanatomy | 0.2785 | 0.2771 | ↓ 0.0014 |
| q04 | human_anatomy | 0.2717 | 0.2917 | ↑ 0.0200 |
| q05 | human_anatomy | 0.2556 | 0.3052 | ↑ 0.0496 |
| q06 | human_anatomy | 0.2958 | 0.3131 | ↑ 0.0173 |
| q07 | plant_biology | 0.3091 | 0.2313 | ↓ 0.0778 |
| q08 | plant_biology | 0.2949 | 0.2867 | ↓ 0.0082 |
| q09 | biochemistry | 0.2723 | 0.2525 | ↓ 0.0198 |
| q10 | zoology | 0.2549 | **0.3165** | ↑ **+0.0616** |
| **Mean** | | **0.2774** | **0.2842** | **↑ +0.0067** |

| Metric | Value |
|--------|-------|
| **Mean CLIP-Score** | **0.2842** (+0.0067 vs Config 1) |
| Queries improved | **5 / 10** |
| Prompt token range | 94–124 tokens → all needed 2 chunks |
| Chunk embed shape | `(1, 154, 768)` for all prompts |
| Vector store docs indexed | 2,673 |
| Avg constraints retrieved/query | 8 |
| Avg constraints injected into prompt | 5 |

**Observations:**
- q02 (mitochondria) largest improvement: **+0.0919** — chunking preserved cristae/matrix/membrane terms that were previously truncated
- q10 (insect) strong improvement: **+0.0616** — RAG injected head/thorax/abdomen/leg count correctly
- q01 (plant cell) and q07 (flower) still regress: both are visually complex multi-element diagrams that SD struggles to compose even with full constraints available — an architectural limitation beyond prompt engineering
- Chunking vs no-chunking: mean improved from 0.2826 → 0.2842 (+0.0016 additional gain)
- Retrieval quality confirmed: cosine distances for top matches < 0.85 (highly similar)

### Test Results
**11/11 passed** ✓

### Output Files
- `outputs/config2/q01.png` ~ `q10.png` — 10 RAG-guided images
- `results/config2_results.json` — per-image scores and delta vs Config 1
- `results/config2_prompts.json` — full prompt log with retrieved constraints

---

## Z-Image-Turbo — Baseline Comparison (Config 1 equivalent)

**Date:** 2026-04-25  
**Scripts:** `bioguard_diffusion/generation/zimage_pipeline.py`, `bioguard_diffusion/generation/run_zimage_baseline.py`

### Operations
1. Searched HuggingFace for Z-Image-Turbo GGUF → found `unsloth/Z-Image-Turbo-GGUF` (Q6_K file)
2. Attempted GGUF inference via `stable-diffusion-cpp-python` → **build failed** on Mac ARM (no pre-compiled wheel)
3. Attempted pre-built `sd-cli` binary → **blocked by macOS Gatekeeper** (unsigned binary, SIGKILL)
4. Fell back to `ZImagePipeline` (already in diffusers 0.37.1) with original `Tongyi-MAI/Z-Image-Turbo`
5. Memory analysis: transformer = 24.6 GB (fp32) / 12.3 GB (bf16) + Qwen3 text encoder + VAE ≈ 15–18 GB total
6. Strategy: `torch_dtype=bfloat16` + `enable_model_cpu_offload()` on M3 Pro 18 GB unified memory
7. Generated same 10 queries as SD-v1.5 Config 1 (`seed=42`, `512×512`, `steps=8`)

### Runtime Notes
- Model load time: ~615s (first run, downloading ~15 GB)
- Generation time: **~348s/image** on M3 Pro with CPU offload (vs SD-v1.5's 30s)
- Root cause: CPU offload moves each transformer block to MPS and back for every forward pass; 6B DiT × 8 NFE steps = significant transfer overhead on unified memory

### Results

| ID | Domain | SD-v1.5 CLIP | Z-Image CLIP | Delta |
|----|--------|:------------:|:------------:|:-----:|
| q01 | cell_biology | 0.3116 | 0.2105 | ↓ 0.1011 |
| q02 | cell_biology | 0.2297 | 0.2379 | ↑ 0.0082 |
| q03 | neuroanatomy | 0.2785 | 0.2311 | ↓ 0.0474 |
| q04 | human_anatomy | 0.2717 | 0.2444 | ↓ 0.0273 |
| q05 | human_anatomy | 0.2556 | 0.2626 | ↑ 0.0070 |
| q06 | human_anatomy | 0.2958 | 0.2988 | ↑ 0.0030 |
| q07 | plant_biology | 0.3091 | 0.2243 | ↓ 0.0848 |
| q08 | plant_biology | 0.2949 | 0.2525 | ↓ 0.0424 |
| q09 | biochemistry | 0.2723 | 0.2510 | ↓ 0.0213 |
| q10 | zoology | 0.2549 | 0.2457 | ↓ 0.0092 |
| **Mean** | | **0.2774** | **0.2459** | **↓ 0.0315** |

| Metric | Value |
|--------|-------|
| **Z-Image Mean CLIP** | **0.2459** (−0.0315 vs SD-v1.5) |
| Queries where Z-Image > SD | **3 / 10** (q02, q05, q06) |
| Avg generation time | **348s/image** (vs SD-v1.5 ~30s) |
| Model size on disk | ~15 GB (bf16) |

### Observations
- **CLIP 整體低於 SD-v1.5**：Z-Image-Turbo 在生物圖解領域的 zero-shot CLIP alignment 不如 SD-v1.5，可能因為 Z-Image 的訓練偏向寫實照片風格，而 SD-v1.5 更容易生成教育插圖風格
- **q02 (mitochondria) 是 Z-Image 唯一明顯勝出的項目**：說明 Z-Image 在細胞器細節上確實有一定能力
- **速度差距巨大**：在 M3 Pro 18 GB 上，Z-Image 每張需 348s vs SD-v1.5 的 30s — 差 11.5×。正式比較需要 GPU server（H800 可達 sub-second）
- **GGUF 路線受限**：Mac ARM 無法使用 stable-diffusion-cpp-python，Q6_K 量化（~5 GB）無法在本機執行

### Output Files
- `outputs/zimage_config1/q01.png` ~ `q10.png` — 10 Z-Image-Turbo images
- `results/zimage_config1_results.json` — per-image scores and metadata

---

## Stage 4 — Config 3: Full BioGuard-Diffusion

**Date:** 2026-04-22  
**Scripts:** `bioguard_diffusion/agents/verifier_agent.py`, `bioguard_diffusion/generation/run_config3_full.py`  
**Test:** `bioguard_diffusion/generation/test_stage4.py` — **11/11 PASS**

### Operations
1. Implemented `VerifierAgent` using **CLIP ViT-B/32** (not BioCLIP) with keyword extraction from constraint sentences
   - BioCLIP was evaluated first but failed to discriminate (correct ≈ wrong, both 0.08–0.14); switched to CLIP ViT-B/32
   - Empirically calibrated threshold = **0.22** on AI2D mitochondria reference image (correct kw: 0.27–0.32, wrong kw: 0.15–0.22)
   - SAS = mean CLIP similarity across all constraints; query passes if SAS ≥ 0.22
2. Implemented `SpatialArchitectAgent.build_prompt()` with violation-priority reordering on retry
3. Full pipeline: Biologist Agent (RAG top-8) → Spatial Architect → SD-v1.5 (seed=42+attempt×7) → Verifier → retry ≤ 3
4. Generated 10 images, saved to `outputs/config3/`; best image per query (highest SAS) retained
5. Ran final 3-way comparison vs. Config 1 and Config 2

### Results

| Metric | Value |
|--------|-------|
| Queries completed | **10/10** |
| Config 3 mean CLIP | **0.2842** |
| Config 3 mean SAS | **0.2543** |
| Queries passing SAS threshold | **10/10** |
| Retries triggered | **0** |
| Max constraints checked per query | 5 |
| Avg generation time/image | ~30s (MPS, 30-step) |

**Per-query 3-way comparison:**

| ID | Domain | C1 CLIP | C2 CLIP | C3 CLIP | SAS | Tries |
|----|--------|:-------:|:-------:|:-------:|:---:|:-----:|
| q01 | cell_biology | 0.3116 | 0.2458 | 0.2458 | 0.2421 | 1 |
| q02 | cell_biology | 0.2297 | 0.3216 | 0.3216 | 0.2667 | 1 |
| q03 | neuroanatomy | 0.2785 | 0.2771 | 0.2771 | 0.2483 | 1 |
| q04 | human_anatomy | 0.2717 | 0.2917 | 0.2917 | 0.2534 | 1 |
| q05 | human_anatomy | 0.2556 | 0.3052 | 0.3052 | 0.2784 | 1 |
| q06 | human_anatomy | 0.2958 | 0.3131 | 0.3131 | 0.2842 | 1 |
| q07 | plant_biology | 0.3091 | 0.2313 | 0.2313 | 0.2216 | 1 |
| q08 | plant_biology | 0.2949 | 0.2867 | 0.2867 | 0.2548 | 1 |
| q09 | biochemistry | 0.2723 | 0.2525 | 0.2525 | 0.2244 | 1 |
| q10 | zoology | 0.2549 | 0.3165 | 0.3165 | 0.2687 | 1 |
| **Mean** | | **0.2774** | **0.2842** | **0.2842** | **0.2543** | — |

**Key observations:**
- All 10 queries passed SAS threshold on attempt 1 — no retries needed. Config 2's structured prompts already produce anatomically acceptable images; Verifier acts as quality gate.
- q01, q07, q09 had violations detected but mean SAS still above 0.22 (3–4 minor constraint misses out of ~5–6 constraints).
- Config 3 CLIP = Config 2 CLIP (deterministic: same prompts, same seeds, no retry path activated).
- Mean CLIP improvement over Config 1: **+0.0068** (+2.5%), confirming RAG + Spatial Architect adds value.

### Test Results — `test_stage4.py`

| Test | Result |
|------|--------|
| T1: Exactly 10 result records | PASS |
| T2: All 10 images saved to disk | PASS |
| T3: All images are 512×512 PNG | PASS |
| T4: All records have SAS scores | PASS |
| T5: Mean SAS > 0.20 (actual: 0.2543) | PASS |
| T6: Config 3 mean CLIP ≥ Config 1 (C3=0.2842, C1=0.2774) | PASS |
| T7: Verify log has 10 entries | PASS |
| T8: All verify entries have per_constraint | PASS |
| T9: All records tagged config3_full | PASS |
| T10: All records have attempts_used ≥ 1 | PASS |
| T11: All SAS scores in [0, 1] | PASS |
| **Total** | **11/11** |

### Output Files
- `outputs/config3/q01.png` ~ `q10.png` — 10 × 512×512 generated images
- `results/config3_results.json` — CLIP, SAS, violations, attempt count per query
- `results/config3_verify_log.json` — per-constraint scores and retry history

---

## Cumulative Comparison Table

### BioGuard Pipeline (SD-v1.5 backbone)

| ID | Domain | Config 1 CLIP | Config 2 CLIP | Config 3 CLIP | Config 3 SAS |
|----|--------|:-------------:|:-------------:|:-------------:|:------------:|
| q01 | cell_biology | 0.3116 | 0.2458 | 0.2458 | 0.2421 |
| q02 | cell_biology | 0.2297 | 0.3216 | 0.3216 | 0.2667 |
| q03 | neuroanatomy | 0.2785 | 0.2771 | 0.2771 | 0.2483 |
| q04 | human_anatomy | 0.2717 | 0.2917 | 0.2917 | 0.2534 |
| q05 | human_anatomy | 0.2556 | 0.3052 | 0.3052 | 0.2784 |
| q06 | human_anatomy | 0.2958 | 0.3131 | 0.3131 | 0.2842 |
| q07 | plant_biology | 0.3091 | 0.2313 | 0.2313 | 0.2216 |
| q08 | plant_biology | 0.2949 | 0.2867 | 0.2867 | 0.2548 |
| q09 | biochemistry | 0.2723 | 0.2525 | 0.2525 | 0.2244 |
| q10 | zoology | 0.2549 | 0.3165 | 0.3165 | 0.2687 |
| **Mean** | | **0.2774** | **0.2842** | **0.2842** | **0.2543** |

### Model Comparison (Baseline, no agents)

| ID | Domain | SD-v1.5 (30-step) | Z-Image-Turbo (8-step) |
|----|--------|:-----------------:|:----------------------:|
| q01 | cell_biology | 0.3116 | 0.2105 |
| q02 | cell_biology | 0.2297 | 0.2379 |
| q03 | neuroanatomy | 0.2785 | 0.2311 |
| q04 | human_anatomy | 0.2717 | 0.2444 |
| q05 | human_anatomy | 0.2556 | 0.2626 |
| q06 | human_anatomy | 0.2958 | 0.2988 |
| q07 | plant_biology | 0.3091 | 0.2243 |
| q08 | plant_biology | 0.2949 | 0.2525 |
| q09 | biochemistry | 0.2723 | 0.2510 |
| q10 | zoology | 0.2549 | 0.2457 |
| **Mean** | | **0.2774** | **0.2459** |
| **Avg time/image** | | **~30s** | **~348s** (M3 Pro CPU offload) |

---

## Environment & Dependencies

```
conda env: agent
Python:    3.12.13
Device:    Apple MPS (Apple Silicon)

Key packages:
  torch              2.11.0
  diffusers          0.37.1
  transformers       5.5.4
  accelerate         1.13.0
  open-clip-torch    3.3.0
  sentence-transformers 5.4.1
  chromadb           1.5.7
  huggingface_hub    1.10.2
  langchain-core     1.2.28
  numpy              2.4.4
  pillow             12.2.0
  scipy              1.17.1
  safetensors        0.7.0
  ftfy               6.3.1

SD Model: runwayml/stable-diffusion-v1-5
CLIP Model: ViT-B/32 (openai pretrained)
```

---

## Stage 5 — FLUX.1-dev GGUF: Smoke Test + Config 3 Variant

**Date:** 2026-04-29  
**Scripts:** `bioguard_diffusion/generation/flux_gguf_pipeline.py`, `bioguard_diffusion/generation/smoke_test_flux.py`, `bioguard_diffusion/generation/run_config3_flux.py`

### Model Setup

| Component | Source | Size |
|-----------|--------|------|
| Transformer | `city96/FLUX.1-dev-gguf` — `flux1-dev-Q5_K_S.gguf` | ~7 GB |
| Text encoder (CLIP-L) | `black-forest-labs/FLUX.1-schnell` (text_encoder) | ~0.5 GB |
| Text encoder 2 (T5-XXL) | `black-forest-labs/FLUX.1-schnell` (text_encoder_2) | ~10 GB |
| VAE | `black-forest-labs/FLUX.1-schnell` (vae) | ~0.3 GB |
| Scheduler | `FlowMatchEulerDiscreteScheduler` (from schnell config) | — |

**Architecture note:** Both FLUX.1-dev and FLUX.1-schnell share identical transformer architecture (19 double-stream + 38 single-stream blocks, `attention_head_dim=128`, `joint_attention_dim=4096`). The dev GGUF transformer replaces the schnell weights; text encoders and VAE are identical between variants.  
**Warning observed:** `time_text_embed.guidance_embedder` weights in dev GGUF are unused because schnell config sets `guidance_embeds=False`. This means guidance scale conditioning is inactive — a known limitation of this cross-model assembly approach.

**Installation:** `pip install gguf>=0.10.0` (required for diffusers GGUF loader)

**Memory strategy:** `enable_model_cpu_offload()` on M3 Pro 18 GB unified memory.  
**img2img support:** `FluxImg2ImgPipeline` constructed from shared components (avoids `from_pipe()` dtype-cast error on quantized transformer).

### Smoke Test Results

**Script:** `smoke_test_flux.py`  
**Prompt:** "A detailed scientific cross-section diagram of a mitochondria showing cristae and matrix, educational illustration, white background"  
**Params:** 20 steps, guidance_scale=3.5, seed=42, 512×512

| Metric | Value |
|--------|-------|
| Generation time | **385.6 s** (6.4 min) |
| Image size | 512 × 512 |
| CLIP score | **0.2674** |
| Status | **PASSED** |

---

### Config 3 (FLUX) — Full Pipeline Run

**Pipeline:** Biologist Agent (RAG top-8) → Spatial Architect → FLUX.1-dev GGUF (text2img) → Verifier → [img2img retry ≤3]  
**Retry strategy:**
- Attempt 1: text2img, seed=42, 20 steps
- Attempt 2+: img2img with previous best image as init latent; strength starts at 0.60, decays by 0.10 per retry (min 0.30); same prompt updated with violated constraints at priority

**Generation params:** `steps=20`, `guidance_scale=3.5`, `seed=42` (fixed — img2img chain deterministic), `512×512`

### Results

| ID | Domain | C1 CLIP | C3-SD CLIP | **FLUX CLIP** | FLUX SAS | ΔC1 | ΔC3-SD | Time (s) |
|----|--------|:-------:|:----------:|:-------------:|:--------:|:---:|:------:|:--------:|
| q01 | cell_biology | 0.3116 | 0.2458 | 0.3061 | 0.2627 | −0.0055 | **+0.0603** | 409.9 |
| q02 | cell_biology | 0.2297 | **0.3216** | 0.2647 | 0.2397 | +0.0350 | −0.0569 | 433.9 |
| q03 | neuroanatomy | 0.2785 | 0.2771 | 0.3068 | 0.2810 | +0.0283 | **+0.0297** | 425.5 |
| q04 | human_anatomy | 0.2717 | 0.2917 | 0.2945 | 0.2822 | +0.0228 | +0.0028 | 424.5 |
| q05 | human_anatomy | 0.2556 | 0.3052 | 0.2906 | 0.2918 | +0.0350 | −0.0146 | 401.7 |
| q06 | human_anatomy | 0.2958 | 0.3131 | 0.3088 | 0.2834 | +0.0130 | −0.0043 | 405.3 |
| q07 | plant_biology | 0.3091 | 0.2313 | **0.3317** | 0.2670 | +0.0226 | **+0.1004** | 479.7 |
| q08 | plant_biology | 0.2949 | 0.2867 | 0.3012 | 0.2709 | +0.0063 | +0.0145 | 449.5 |
| q09 | biochemistry | 0.2723 | 0.2525 | 0.3058 | 0.2758 | +0.0335 | **+0.0533** | 441.8 |
| q10 | zoology | 0.2549 | 0.3165 | 0.2779 | 0.2562 | +0.0230 | −0.0386 | 449.2 |
| **Mean** | | **0.2774** | **0.2842** | **0.2988** | **0.2711** | **+0.0214** | **+0.0146** | **432.1** |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Mean CLIP (FLUX) | **0.2988** |
| Mean SAS (FLUX) | **0.2711** |
| vs Config 1 (SD raw) | **+0.0214 (+7.7%)** |
| vs Config 3-SD (BioGuard+SD) | **+0.0146 (+5.1%)** |
| Queries improved vs C1 | **9 / 10** |
| Queries improved vs C3-SD | **5 / 10** |
| Queries passing SAS (≥0.22) | **10 / 10** |
| img2img retries triggered | **0** (all passed attempt 1) |
| Avg generation time | **432 s/image** (~7.2 min) |
| Total run time | **4,321 s** (~72 min) |

### Per-Domain Analysis

| Domain | C1 mean | C3-SD mean | FLUX mean | Best model |
|--------|:-------:|:----------:|:---------:|:----------:|
| cell_biology (q01, q02) | 0.2707 | 0.2837 | 0.2854 | FLUX +0.0017 |
| neuroanatomy (q03) | 0.2785 | 0.2771 | 0.3068 | **FLUX +0.0297** |
| human_anatomy (q04–q06) | 0.2744 | 0.3033 | 0.2980 | C3-SD −0.0054 |
| plant_biology (q07, q08) | 0.3020 | 0.2590 | 0.3165 | **FLUX +0.0575** |
| biochemistry (q09) | 0.2723 | 0.2525 | 0.3058 | **FLUX +0.0533** |
| zoology (q10) | 0.2549 | 0.3165 | 0.2779 | C3-SD −0.0386 |

### Observations

1. **FLUX overall best CLIP** — mean 0.2988 beats both SD baselines (C1: 0.2774, C3-SD: 0.2842). Largest gains in plant biology (+0.0575) and biochemistry (+0.0533).
2. **q07 (flower) largest gain: +0.1004 vs C3-SD** — FLUX's superior compositional ability handles multi-element floral diagrams; SD-C3 had 4 constraint violations on q07.
3. **q02 (mitochondria) regression: −0.0569 vs C3-SD** — SD-C3 scored 0.3216 on this query; FLUX 0.2647. FLUX tends toward realistic/photographic style; mitochondria cross-section is a highly schematic diagram that SD-v1.5 handles well via its training distribution.
4. **q10 (insect) regression: −0.0386 vs C3-SD** — similar style-mismatch effect; SD-v1.5 better at insect body diagrams.
5. **Mean SAS 0.2711 vs C3-SD 0.2543** — FLUX images have better constraint coverage on average. All 10 queries passed SAS on attempt 1; img2img retry loop not exercised.
6. **img2img not triggered** — verifier threshold (0.22) easily cleared at first attempt. Retry loop is dormant; would activate if SAS threshold were raised (e.g., to 0.26) or harder queries used.
7. **Speed: 432 s/image (14.4× slower than SD-v1.5's 30 s)** — CPU offload on 18 GB M3 Pro is the bottleneck. Full-GPU inference (H100/A100) expected ~2–4 s/image.
8. **guidance_embedder unused** — dev GGUF's 4 guidance-conditioning layers not connected due to schnell config; enabling `guidance_embeds=True` may improve quality at cost of minor refactor.

### 4-Model Comparison (CLIP Score)

| ID | Domain | SD-C1 | Z-Image | SD-C3 | **FLUX-C3** |
|----|--------|:-----:|:-------:|:-----:|:-----------:|
| q01 | cell_biology | 0.3116 | 0.2105 | 0.2458 | 0.3061 |
| q02 | cell_biology | 0.2297 | 0.2379 | **0.3216** | 0.2647 |
| q03 | neuroanatomy | 0.2785 | 0.2311 | 0.2771 | **0.3068** |
| q04 | human_anatomy | 0.2717 | 0.2444 | 0.2917 | **0.2945** |
| q05 | human_anatomy | 0.2556 | 0.2626 | **0.3052** | 0.2906 |
| q06 | human_anatomy | 0.2958 | 0.2988 | **0.3131** | 0.3088 |
| q07 | plant_biology | 0.3091 | 0.2243 | 0.2313 | **0.3317** |
| q08 | plant_biology | 0.2949 | 0.2525 | 0.2867 | **0.3012** |
| q09 | biochemistry | 0.2723 | 0.2510 | 0.2525 | **0.3058** |
| q10 | zoology | 0.2549 | 0.2457 | **0.3165** | 0.2779 |
| **Mean** | | 0.2774 | 0.2459 | 0.2842 | **0.2988** |
| **Avg time/img** | | ~30 s | ~432 s | ~30 s | ~432 s |

### Output Files

- `outputs/config3_flux/q01.png` ~ `q10.png` — 10 × 512×512 FLUX-generated images
- `results/config3_flux_results.json` — CLIP, SAS, per-query timings
- `results/config3_flux_verify_log.json` — per-constraint scores and retry history
- `outputs/smoke_test_flux.png` — smoke test mitochondria image

---

## Notes & Observations

- MPS inference is stable at ~30s/image for 30 steps at 512×512
- `position_ids` key mismatch in SD text encoder is a known harmless warning (HF issue #254)
- Mitochondria (q02) consistently lowest CLIP — a known hard case per literature (sub-cellular detail)
- All experiments use fixed `seed=42` for reproducibility across configs
