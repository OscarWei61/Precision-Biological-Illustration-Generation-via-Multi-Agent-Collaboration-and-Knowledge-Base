# BioGuard-Diffusion — Agent Reference

每個 agent 的完整處理步驟、輸入輸出、fallback 機制與 feedback loop 說明。

---

## 系統中的所有 Agents

| Agent | 檔案 | 版本 | 角色 |
|-------|------|------|------|
| BiologistAgent | `agents/biologist_agent.py` | v3, v4 | RAG 知識庫檢索 |
| SpatialArchitectAgent | `agents/spatial_architect.py` | v3, v4 | 結構化 prompt 組裝 |
| PromptSplitter | `agents/prompt_splitter.py` | v3, v4 | CLIP / T5 子 prompt 分割 |
| GeneratorAgent | `agents/generator_agent.py` | v3 | FLUX 圖像生成（無 ControlNet）|
| GeneratorAgentV4 | `agents/generator_agent_v4.py` | v4 | FLUX 圖像生成（含 ControlNet）|
| VerifierAgent | `agents/verifier_agent.py` | v3, v4 | 兩階段圖像驗證 |
| PromptRefinerAgent | `agents/prompt_refiner_agent.py` | v3, v4 | 外科式 prompt 修正 |
| VisualPlannerAgent | `agents/visual_planner_agent.py` | v4 only | AI2D 標注轉空間 layout |
| BlueprintGenerator | `generation/blueprint_generator.py` | v4 only | Layout → ControlNet 控制圖 |

---

## 1. BiologistAgent

**角色：** RAG 知識庫管理與生物約束條件檢索

### 初始化步驟

1. 建立或掛載 ChromaDB persistent vector store（`data/vector_store/`）。
2. 若 collection `bio_constraints` 不存在，從 `results/knowledge_base.json` 建立索引：
   - 每條 constraint 句子單獨作為一份文件（deduplicated by MD5 hash）。
   - 每張圖片的 label 集合合併為一份 label summary 文件（`biological diagram showing: ...`）。
   - 分批插入（每批 500 筆），總計約 2,673 份文件。
   - ChromaDB 內建 `all-MiniLM-L6-v2`（ONNX）自動做 embedding。
3. 若 collection 已存在，直接掛載（不重建）。

### 主要方法：`retrieve(query, top_k=8)`

**輸入：** 使用者的自然語言問題（e.g. "Draw a cross-section of mitochondria"）

**處理步驟：**
1. ChromaDB 對 query 做 embedding。
2. 在 collection 中做 cosine similarity 查詢，filter `where={"type": "constraint"}` 排除 label summary 文件。
3. 回傳相似度最高的 top_k 條 constraint 句子。

**輸出：** `list[str]` — 生物約束條件句子，按相似度由高到低排列

### Fallback
- 若 query 對應的 constraint 不足 top_k 筆，回傳實際有的數量（ChromaDB 不報錯）。
- 若 collection 完全空（不正常狀況），回傳空 list，下游 SpatialArchitect 會使用無 constraint 的 fallback prompt。

### Feedback Loop
- BiologistAgent 本身無 feedback loop，屬於 **靜態 retrieval**。
- 不隨 retry 更新：v3/v4 每次 attempt 都用同一批 retrieved constraints（並非重新 retrieve）。
- v3 SpatialArchitect 的 `violations` 參數會重排 constraints 優先級（不重新 retrieve）。

---

## 2. SpatialArchitectAgent

**角色：** 將 RAG 約束條件組裝成結構化的 FLUX positive prompt

### 主要方法：`build_prompt(query, constraints, violations=None)`

**輸入：**
- `query`：使用者問題
- `constraints`：BiologistAgent 回傳的 constraint 句子列表
- `violations`：（retry 時）上一次 CLIP 驗證未通過的 constraint（可選）

**處理步驟：**

**Step 1 — Subject extraction:**  
正規表達式去除問題中的指令動詞（draw / create / generate / show / illustrate 等）和介詞結構（"diagram of" / "cross-section of"），提取生物主詞。  
例：`"Draw a detailed cross-section of a mitochondria showing cristae"` → `"a mitochondria"`

**Step 2 — Constraint reordering（retry 時）:**  
若 `violations` 非空，將違規的 constraint 移至列表最前面，確保它們在 token 限制截斷前被保留在 prompt 中。

**Step 3 — Constraint selection:**  
從重排後的 constraints 中選最多 5 筆（`MAX_CONSTRAINTS=5`）。選擇標準：計算每條 constraint 包含幾個 subject 關鍵詞，分數高的優先。

**Step 4 — Phrase compaction:**  
對每條選中的 constraint 用正規表達式移除冗餘前綴（"a/the ... must have / has / contains"），轉為更緊湊的短語格式。  
例：`"The inner membrane has cristae"` → `"inner membrane with cristae"`

**Step 5 — Prompt assembly:**  
```
"A highly detailed scientific cross-section diagram of {subject},
{phrase1}, {phrase2}, ...,
scientific educational illustration, detailed diagram,
white background, accurate biology textbook style, no text, no labels"
```
若無任何 constraint，用無 constraint 的 fallback 模板。

**輸出：**
```python
{
  "prompt":            str,   # positive prompt
  "negative_prompt":   str,   # fixed negative prompt
  "subject":           str,   # extracted subject
  "constraints_used":  list[str],
}
```

### Fallback
- 無 constraint 時：`"A highly detailed scientific diagram of {subject}, {STYLE_SUFFIX}"` — 純風格 prompt，無結構約束。
- 無 LLM 依賴：純 regex + 排序，不會因 API 失敗而崩潰。

### Feedback Loop
- 每次 retry 時接收 `violations` 參數 → violation constraint 被移到 prompt 最前面 → 優先進入 token limit 範圍內。
- 這是一個**間接 feedback**：VerifierAgent → violations 列表 → SpatialArchitect constraint 排序調整。

---

## 3. PromptSplitter

**角色：** 將單一 prompt 拆分成 CLIP 子 prompt（≤77 tokens）和 T5 子 prompt（無限制）

### 主要方法：`split(prompt)`

**輸入：** SpatialArchitect 產生的完整 prompt 字串（或 PromptRefiner 精修後的版本）

**處理步驟：**

兩次獨立的 `claude-haiku-4-5` LLM 呼叫：

**呼叫 1 — CLIP prompt 生成:**  
System prompt 要求：≤77 tokens、逗號分隔的關鍵字格式、包含生物主詞 + 最多 2-4 個結構關鍵字 + 圖示風格詞。  
User prompt：`"Generate the CLIP encoder prompt (≤77 tokens, keyword style): {original_prompt}"`

**呼叫 2 — T5 prompt 生成:**  
System prompt 要求：完整自然語言句子、包含所有生物結構與空間關係、無 token 限制。  
User prompt：`"Generate the T5 encoder prompt (detailed natural language): {original_prompt}"`

**輸出：**
```python
{
  "clip_prompt": str,   # ≤77 tokens keyword style
  "t5_prompt":   str,   # full natural language
}
```

### Fallback
- 若 LLM API 失敗（exception），PromptSplitter 不做特別處理（exception 向上拋）。
- 在 v3 中，PromptSplitter 只在 **attempt 1** 呼叫（後續 attempt 由 PromptRefiner 直接輸出精修後的 CLIP+T5，不再呼叫 PromptSplitter）。

### Feedback Loop
- 本身無 feedback loop。
- v3/v4 attempt 2+ 的 prompt 由 PromptRefiner 直接輸出，**繞過 PromptSplitter**。

---

## 4. GeneratorAgent（v3）

**角色：** 管理 FLUX 圖像生成，支援 text2img 和 img2img 兩種模式

### 初始化
- 載入 `FluxGGUFPipeline`（FLUX.1-dev Q5_K_S GGUF transformer + FLUX.1-schnell text encoders + VAE）
- 載入 `PromptSplitter`

### 方法 1：`generate(full_prompt, seed, steps, ...)` — text2img

**處理步驟：**
1. 呼叫 `PromptSplitter.split(full_prompt)` → `clip_prompt`, `t5_prompt`
2. 呼叫 `FluxGGUFPipeline.generate(prompt=clip_prompt, prompt_2=t5_prompt, seed=seed, steps=steps)`
3. 記錄本次嘗試到 `prompt_history`
4. 回傳 `{image, mode, clip_prompt, t5_prompt, seed, gen_time_s}`

### 方法 2：`refine(image, full_prompt, attempt, strength, clip_prompt_override, t5_prompt_override, ...)` — img2img

**處理步驟：**
1. 若有 `clip_prompt_override` 和 `t5_prompt_override`（rollback 情況），直接使用，**不呼叫 PromptSplitter**。
2. 若無 override，呼叫 `PromptSplitter.split(full_prompt)` 重新生成。
3. 若未指定 `strength`，使用 decay schedule：`strength = max(0.60 - 0.10 × (attempt-2), 0.30)`
4. 呼叫 `FluxGGUFPipeline.refine(image=image, prompt=clip_prompt, prompt_2=t5_prompt, strength=strength)`
5. 記錄本次嘗試到 `prompt_history`

### Fallback
- 若 `strength` 不指定，使用 decay schedule fallback（但實際上 v3 pipeline 總是從 `_adaptive_strength()` 傳入 strength，不依賴 decay）。
- `prompt_history` 保存所有嘗試記錄，可用 `write_prompt_log()` 輸出完整日誌。

### Feedback Loop
- GeneratorAgent 本身只是執行生成，不直接參與 feedback。
- Feedback 由外層 `_run_flux()` 管理：VerifierAgent → PromptRefinerAgent → 精修後的 prompts 透過 `clip_prompt_override`/`t5_prompt_override` 傳入 `refine()`。

---

## 5. GeneratorAgentV4（v4）

**角色：** v3 GeneratorAgent 的 ControlNet 版本，新增 `control_image` 和 `controlnet_scale` 參數

### 初始化
- 載入 `FluxControlNetPipelineWrapper`（FLUX GGUF + InstantX/FLUX.1-dev-Controlnet-Canny）
- 載入 `PromptSplitter`

### 方法 1：`generate(full_prompt, control_image, has_spatial_data, controlnet_scale, ...)` — text2img + ControlNet

**處理步驟：**
1. 呼叫 `PromptSplitter.split(full_prompt)` → `clip_prompt`, `t5_prompt`
2. 決定有效 ControlNet scale：
   - 若 `has_spatial_data=False` 或 `control_image=None` → `eff_scale = 0.0`（ControlNet 無效）
   - 否則使用傳入的 `controlnet_scale`（預設 0.70）
3. 若無 spatial data，使用全黑 512×512 圖作為 control image（確保 ControlNet 無作用）
4. 呼叫 `FluxControlNetPipeline.generate(prompt, prompt_2, control_image, controlnet_conditioning_scale=eff_scale, ...)`
5. 記錄本次嘗試到 `prompt_history`（含 `controlnet_scale`, `has_spatial_data`）

### 方法 2：`refine(image, ..., control_image, has_spatial_data, controlnet_scale, ...)` — img2img + ControlNet

**處理步驟：**
1. 決定 strength（同 v3 logic 或 caller 指定）
2. 決定 CLIP+T5（使用 override 或重新 split，同 v3）
3. 決定 `eff_scale`：
   - `has_spatial_data=False` → 0.0
   - 否則使用傳入的 `controlnet_scale`（預設 0.50）
4. 呼叫 `FluxControlNetImg2ImgPipeline.generate(..., strength=strength, controlnet_conditioning_scale=eff_scale)`

### Fallback（v4 特有）
- 若 `has_spatial_data=False`（AI2D 標注不存在）→ `eff_scale=0.0`，自動降級為純 v3 行為。
- 若 `FluxControlNetImg2ImgPipeline` 不可用（import error）→ `FluxControlNetPipeline` 的 wrapper 中有 fallback 改用 text2img pipeline。
- 使用全黑圖而非 None 作為 control image，避免 pipeline 收到 NoneType 報錯。

### Feedback Loop
- 同 GeneratorAgent v3，feedback 由外層 `_run_flux_v4()` 管理。
- v4 額外的 feedback：`missing_structures` → BlueprintGenerator 更新 `current_blueprint` → 下次 `generate()/refine()` 使用新的 control image（blueprint → ControlNet → 空間強調缺失區域）。

---

## 6. VerifierAgent

**角色：** 兩階段圖像驗證 — CLIP-SAS（快速本地）+ LLM-as-judge（語意視覺）

### 初始化
- 載入 CLIP ViT-B/32（openai pretrained）到 MPS/CPU 裝置。
- 初始化 Anthropic client（`claude-haiku-4-5-20251001`）。

### 方法 1：`verify(image, constraints)` — CLIP-SAS 驗證

**處理步驟：**
1. 對圖像做 CLIP image embedding（`model.encode_image`，L2 normalized）。
2. 對每條 constraint 句子：
   - 用 `_extract_keywords(constraint, max_words=6)` 去除 stop-words，提取關鍵名詞。
   - 對關鍵字做 CLIP text embedding（L2 normalized）。
   - 計算 image 與 keyword text 的 cosine similarity score。
   - 若 score ≥ 0.22 → 此 constraint 通過。
3. SAS = 所有 constraint score 的平均值。
4. `passed = SAS ≥ 0.22`。
5. 收集未通過的 constraint → `violations`。
6. 組裝 feedback 字串（用於 img2img prompt injection）。

**輸出：**
```python
{
  "passed":          bool,
  "sas":             float,       # 0–1
  "per_constraint":  list[tuple], # (constraint, keywords, score, passed)
  "violations":      list[str],
  "feedback":        str,
}
```

**Pass threshold：** `PASS_THRESHOLD = 0.22`

### 方法 2：`verify_with_llm(image, query, constraints)` — LLM-as-judge

**處理步驟：**
1. 將圖像 base64 編碼（PNG format）。
2. 組裝 user message：query + constraints 條列 + "Evaluate the image."
3. 送給 `claude-haiku-4-5-20251001`（vision 模式），system prompt 要求回傳 JSON：
   ```json
   {
     "score": 0–10,
     "passed": true/false,
     "present_structures": [...],
     "missing_structures": [...],
     "improvement_suggestions": "..."
   }
   ```
4. 解析回應 JSON（剝除可能的 markdown fences）。
5. `passed = score ≥ 6.0`。

**輸出：**
```python
{
  "passed":                   bool,
  "llm_score":                float,      # 0–10
  "present_structures":       list[str],
  "missing_structures":       list[str],
  "improvement_suggestions":  str,
}
```

**Pass threshold：** `LLM_PASS_THRESHOLD = 6.0`

### 方法 3：`verify_combined(image, query, constraints)` — 合併驗證（v3/v4 預設）

**處理步驟：**
1. 依序執行 `verify()`（CLIP-SAS）和 `verify_with_llm()`（LLM-judge）。
2. `passed_combined = clip_passed AND llm_passed`
3. 合併 feedback：CLIP violations + LLM suggestions → 單一 feedback 字串。

**Combined pass condition：** CLIP SAS ≥ 0.22 **且** LLM score ≥ 6.0

### Fallback
- **LLM API 失敗（JSON parse error 或 exception）：** `verify_with_llm()` 的 except block 回傳 `llm_score=0.0, passed=False, missing_structures=[], improvement_suggestions="LLM judge error: {e}"`。不重試、不崩潰。
- **空 constraints：** `verify()` 直接回傳 `passed=True, sas=1.0`（無法驗證時視為通過，避免誤拒）。
- **Threshold 設定根據：** 在 AI2D 線粒體圖（3288.png）上實驗校準：正確關鍵字（cristae / inner membrane）得分 0.268–0.317，錯誤關鍵字（heart / neuron）得分 0.147–0.224，threshold 0.22 區分效果最佳。

### Feedback Loop
VerifierAgent 是整個系統 feedback loop 的**核心節點**：
1. `missing_structures` → PromptRefinerAgent（強化缺失結構的 prompt）
2. `missing_structures` → BlueprintGenerator.highlight_missing()（v4，更新 ControlNet 控制圖）
3. `violations` → SpatialArchitectAgent.build_prompt() 的 violations 參數（重排 constraints 優先級）
4. `improvement_suggestions` → PromptRefinerAgent（具體改進建議）
5. `llm_score` → `_adaptive_strength()`（決定下次 attempt 的 text2img/img2img 及 strength）
6. `combined_score = 0.5×SAS + 0.5×(LLM/10)` → rollback 機制（判斷是否要回滾到 best prompts）

---

## 7. PromptRefinerAgent

**角色：** 根據 VerifierAgent 的結構化 feedback，對 CLIP 和 T5 prompt 做外科式修正

### 主要方法：`refine(clip_prompt, t5_prompt, present_structures, missing_structures, violations, improvement_suggestions)`

**輸入：**
- `clip_prompt`：上一次（或 best）的 CLIP prompt
- `t5_prompt`：上一次（或 best）的 T5 prompt
- `present_structures`：已正確呈現的結構（不可刪除）
- `missing_structures`：缺失的結構（必須加入兩個 prompt）
- `violations`：CLIP 未通過的 constraint 句子
- `improvement_suggestions`：LLM-as-judge 給的建議文字

**處理步驟：**

**Step 1 — 快速檢查：**  
若 `missing_structures`, `violations`, `improvement_suggestions` 全為空，直接回傳截斷後的原始 prompts（不做 LLM 呼叫）。

**Step 2 — 輸入截斷：**  
將輸入的 `clip_prompt` 截斷到 55 words，`t5_prompt` 截斷到 150 words，避免超出 LLM 輸入限制。

**Step 3 — LLM 呼叫（`claude-haiku-4-5`）：**  
System prompt 明確規定 5 條規則：
1. 保留 `present_structures` 中所有關鍵詞
2. `missing_structures` 中每個結構必須以名稱出現在兩個 prompt 中
3. CLIP：缺失結構名稱插入前段（高優先位置）
4. T5：每個缺失結構加一句簡短描述
5. 不改寫已工作的部分，只做最小化補充

回應格式（strict JSON）：
```json
{
  "refined_clip_prompt": "...",
  "refined_t5_prompt":   "...",
  "changes_made":        ["added cristae to CLIP", "added matrix sentence to T5"]
}
```

**Step 4 — 輸出截斷（強制執行）：**  
無論 LLM 輸出什麼，一律在字詞邊界截斷：
- `clip_prompt` → 最多 55 words
- `t5_prompt` → 最多 150 words

**輸出：**
```python
{
  "refined_clip_prompt": str,      # ≤55 words
  "refined_t5_prompt":   str,      # ≤150 words
  "changes_made":        list[str],
}
```

### Fallback
- **JSON parse error（LLM 輸出格式不符）：** 使用截斷後的原始 prompts，`changes_made` 記錄錯誤訊息。
- **任何其他 exception：** 同上，fallback to truncated originals，不崩潰。
- **無論如何都做字詞截斷：** LLM 即使遵守了 word limit，仍對輸出再執行一次截斷，確保下游 CLIP encoder 不超 token 限制。

### Feedback Loop
PromptRefinerAgent 是 feedback loop 的**終端執行者**：
- 接收：VerifierAgent 的 structured feedback（missing, present, violations, suggestions）
- 輸出：改良後的 CLIP + T5 prompts
- 這兩個 prompts 作為 `clip_prompt_override`/`t5_prompt_override` 傳入 GeneratorAgent 的 `refine()`，直接影響下一次圖像生成。

Rollback 機制的關鍵：PromptRefiner 永遠接收的是 **best-so-far prompts**（非 latest），確保即使上一次 attempt 的 prompt 導致分數下降，refinement 仍基於已知最佳的起點。

---

## 8. VisualPlannerAgent（v4 only）

**角色：** 解析 AI2D 標注 JSON，提取生物結構的空間位置和多邊形座標

### 主要方法：`plan(image_name, subject, target_size=512)`

**輸入：**
- `image_name`：AI2D 圖片檔名（如 `"3288.png"`）
- `subject`：生物主詞（用於日誌）
- `target_size`：輸出座標空間（預設 512，對應 FLUX 生成尺寸）

**處理步驟：**

**Step 1 — 查找標注檔：**  
在 `data/ai2d/annotations/{image_name}.json` 尋找標注。若不存在，直接回傳 `{has_spatial_data: False, layout: []}`。

**Step 2 — 取得原始圖片尺寸（座標縮放用）：**  
若圖片存在，開啟讀取 `(orig_w, orig_h)`。若不存在，從標注 JSON 中所有 polygon 和 rectangle 座標推算最大邊界。

定義縮放函數：`sx(x) = int(x × 512 / orig_w)`，`sy(y) = int(y × 512 / orig_h)`

**Step 3 — 解析關係並分類（兩種策略）：**

讀取 `relationships` 欄位，對每條關係：

- **Strategy A — `intraObjectLabel`：** text label 直接標記在 blob 上。  
  → 從 `text[text_id].replacementText` 或 `.value` 取得名稱。  
  → 記錄到 `blob_to_labels[blob_id]`。

- **Strategy B — `intraObjectLinkage`：** text label 透過箭頭指向 blob（AI2D 最常見）。  
  → 記錄到 `text_linked[text_id] = blob_id`。

**Step 4 — 建立 layout items：**

**Strategy A blobs：** 對每個有標籤的 blob，提取完整多邊形，縮放座標，生成 layout item（`is_label_hint=False`）。

**Strategy B — text 位置 hint：** 對每個 `text_linked` 關係：
- 從 text rectangle `[[x1,y1],[x2,y2]]` 計算中心點 `(cx, cy)`
- 以 r=12 像素建立一個小正方形多邊形（標示文字位置）
- `is_label_hint=True`（後續 BlueprintGenerator 以小圓點而非輪廓渲染）

**Strategy B — blob 輪廓：** 對所有被 linkage 指向的 blob，若未在 Strategy A 中出現，提取完整多邊形作為無名稱的結構邊界（`name="boundary_{blob_id}"`，`is_label_hint=False`）。

**未被任何關係連結的 blob：** 一律提取為 `name="region_{blob_id}"`（純形狀 fallback）。

**輸出：**
```python
{
  "image":            str,
  "subject":          str,
  "layout": [
    {
      "name":          str,          # e.g. "Cristae" or "boundary_b3"
      "center":        [int, int],   # scaled to 512px
      "bounds":        [int, int, int, int],
      "polygon":       [[int,int], ...],
      "is_label_hint": bool,
    },
    ...
  ],
  "has_spatial_data": bool,
}
```

### Fallback
- **標注檔不存在：** 直接回傳 `has_spatial_data=False`，BlueprintGenerator 輸出全黑圖，GeneratorAgentV4 強制 `eff_scale=0.0`，降級為 v3 行為。
- **圖片不存在（只有標注）：** 使用 `_infer_size()` 從標注座標推算圖片尺寸（找所有 polygon 和 rectangle 的最大 x/y + margin 10px）。
- **多邊形點數少於 3：** 跳過該 blob（無法構成有效多邊形）。

### Feedback Loop
- VisualPlannerAgent 不直接參與 feedback loop，是 **預處理層**（每個 query 只執行一次）。
- 它產生的 `layout` dict 被 BlueprintGenerator 使用，並在 feedback loop 中透過 `highlight_missing()` 多次引用（不重新 parse，只重新 render）。

---

## 9. BlueprintGenerator（v4 only）

**角色：** 將 VisualPlannerAgent 的 layout dict 渲染成 ControlNet 控制圖

### 方法 1：`generate(layout, mode="lineart", line_width=3)`

**輸入：** VisualPlannerAgent 的 layout dict

**處理步驟：**

**Fallback 檢查：** 若 `has_spatial_data=False` 或 `layout` 為空，回傳全黑 512×512 圖。

**Lineart mode（預設）：**
1. 建立白色 512×512 畫布。
2. 對每個 layout item：
   - 若 `is_label_hint=True`：在 `center` 位置畫灰色實心小圓（半徑 4px，RGB=80,80,80）。
   - 否則：畫黑色多邊形輪廓（width=3，RGB=0,0,0）。

**Segmentation mode（視覺檢查用）：**
1. 建立白色畫布。
2. 對每個 item：用 12 色 palette 輪流填充不同顏色 + 黑色輪廓（width=2）。

**輸出：** 512×512 RGB PIL Image（`blueprint_initial.png`）

### 方法 2：`highlight_missing(layout, missing_names, base_line_width=2, highlight_line_width=6)`

**角色：** Feedback loop 中更新 ControlNet 控制圖，強調缺失結構

**輸入：**
- `layout`：與 `generate()` 相同的 layout dict
- `missing_names`：VerifierAgent 回報的 `missing_structures` 列表

**處理步驟：**
1. 建立白色畫布。
2. 對每個 layout item，判斷其名稱是否匹配任何 `missing_names`（**大小寫不敏感子字串匹配**）：
   - 匹配（缺失結構）→ 紅色粗輪廓（RGB=200,0,0，width=6）
   - 不匹配 → 黑色細輪廓（RGB=0,0,0，width=2）
3. 儲存為 `blueprint_att{N}.png`，替換 `current_blueprint`。

**Fallback：** 若 `has_spatial_data=False`，回傳全黑圖（同 `generate()` 的 fallback）。

### Feedback Loop
BlueprintGenerator 是 v4 **空間 feedback loop** 的執行端：
1. Attempt N 生成後，VerifierAgent 回傳 `missing_structures`。
2. `highlight_missing(layout, missing_structures)` 重繪控制圖，缺失結構變為紅色粗線。
3. 新控制圖用於 Attempt N+1 的 ControlNet conditioning。
4. 效果：ControlNet 在下次生成時對缺失區域施加更強的邊緣約束，引導 FLUX 在該位置產生對應結構。

---

## Agents 之間的 Feedback 流向圖

```
User Query
    │
    ▼
BiologistAgent ──────────────────────────────────────────────────────────────┐
  (retrieve constraints)                                                       │
    │                                                                          │
    ▼                                                                          │
SpatialArchitectAgent  ◄──── violations (from VerifierAgent, on retry)       │
  (build prompt)                                                               │
    │                                                                          │
    ▼                                                                          │
PromptSplitter ─────────────────────────────── (attempt 1 only)              │
  (clip_prompt, t5_prompt)                                                     │
    │                                                                          │
    ▼                                                                          │
GeneratorAgent / GeneratorAgentV4                                              │
  (generate / refine)  ◄─── clip_override, t5_override (from PromptRefiner)  │
    │                   ◄─── control_image (from BlueprintGenerator, v4)      │
    │                                                                          │
    ▼                                                                          │
  Generated Image                                                              │
    │                                                                          │
    ▼                                                                          │
VerifierAgent  ─────────────────────────────────────────────────────────────  │
  Stage 1: CLIP-SAS     ──► violations ──────────────────────► SpatialArch   │
  Stage 2: LLM-as-judge ──► missing_structures ─────────────► PromptRefiner  │
                         ──► missing_structures ─────────────► BlueprintGen   │ (v4)
                         ──► improvement_suggestions ────────► PromptRefiner  │
                         ──► llm_score ──────────────────────► _adaptive_str  │
                         ──► combined_score ─────────────────► rollback check │
    │                                                                          │
    │ passed?                                                                  │
    ├── YES → final image (best.png)                                           │
    └── NO  →                                                                  │
           PromptRefinerAgent                                                   │
             (refine best CLIP+T5 based on missing + violations)               │
             ──► refined_clip, refined_t5 ──► GeneratorAgent.refine()         │
                                                                               │
           BlueprintGenerator.highlight_missing()  (v4 only)                  │
             (redraw: missing structures in red)                               │
             ──► updated control_image ──► GeneratorAgentV4.refine()          │
                                                                               │
           _adaptive_strength(llm_score)                                       │
             ──► next_gen_mode (text2img / img2img)                            │
             ──► next_strength (0.28 / 0.55 / 0.75)                           │
                                                                               │
           rollback check (combined_score < best_combined)                     │
             ──► if rollback: use best_clip_prompt, best_t5_prompt             │
             ──► if improved: update best prompts                              │
                                                                               │
           → next attempt (up to max_retries=3 total)                         │
                                                                               │
VisualPlannerAgent (v4, runs ONCE before attempt loop) ────────────────────── ┘
  (parse AI2D annotation → layout dict → used by BlueprintGenerator)
```

---

## Fallback 機制總結

| Agent | 觸發條件 | Fallback 行為 |
|-------|---------|--------------|
| BiologistAgent | Constraints 不足 | 回傳實際有的筆數（不報錯） |
| SpatialArchitectAgent | 無任何 constraints | 使用純風格 prompt（無結構細節） |
| PromptSplitter | — | 無內建 fallback（exception 向上拋） |
| GeneratorAgentV4 | `has_spatial_data=False` | `eff_scale=0.0`，全黑 control image，降級為 v3 |
| VerifierAgent (LLM) | JSON parse error / API error | `llm_score=0.0, passed=False`，不崩潰 |
| VerifierAgent (CLIP) | 空 constraints | `passed=True, sas=1.0`（視為通過） |
| PromptRefinerAgent | JSON parse error / API error | 回傳截斷後的原始 prompts，不崩潰 |
| PromptRefinerAgent | 無 missing/violations | 直接回傳截斷原始 prompts（省略 LLM 呼叫） |
| VisualPlannerAgent | 標注檔不存在 | `has_spatial_data=False` |
| VisualPlannerAgent | 圖片不存在 | 用 `_infer_size()` 從標注座標推算尺寸 |
| BlueprintGenerator | `has_spatial_data=False` | 回傳全黑 512×512 圖 |
