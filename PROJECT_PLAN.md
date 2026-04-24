# BioGuard-Diffusion 專案執行計畫

**ECE 598 Final Project — MingYi Wei (mingyi5@illinois.edu)**  
**題目：Precision Biological Illustration Generation via Multi-Agent Collaboration and Knowledge Base**

---

## 系統架構總覽

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           Reasoning Layer (LangChain)        │
│                                             │
│  [Biologist Agent] ──► RAG Knowledge Base   │
│         │                (AI2D + Vector DB)  │
│         ▼                                   │
│  [Spatial Architect] ──► Layout Prompt       │
│         │                                   │
│         ▼                                   │
│  [Verifier Agent] ──► BioCLIP / BioMedBLIP  │
│         │             (cross-check draft)    │
└─────────┼───────────────────────────────────┘
          │ Structured Prompt + Constraints
          ▼
┌─────────────────────────────────────────────┐
│           Generation Layer                   │
│   Stable Diffusion (SD-v1.5 / Z-Image)      │
└─────────────────────────────────────────────┘
          │
          ▼
   Final Biological Illustration
```

---

## Phase 0：環境建置與資料準備

### 目標
建立完整開發環境，下載並預處理所有所需資料與模型。

### 實作步驟

#### 0.1 套件安裝
```bash
pip install langchain langchain-openai langchain-community
pip install chromadb faiss-cpu sentence-transformers
pip install diffusers transformers accelerate torch
pip install open-clip-torch pillow numpy pandas
pip install huggingface_hub datasets
```

#### 0.2 專案目錄結構
```
bioguard-diffusion/
├── data/
│   ├── ai2d/                  # AI2D dataset
│   ├── knowledge_base/        # 結構化生物知識文本
│   └── vector_store/          # ChromaDB / FAISS index
├── agents/
│   ├── biologist_agent.py
│   ├── spatial_architect.py
│   └── verifier_agent.py
├── generation/
│   └── diffusion_pipeline.py
├── evaluation/
│   ├── clip_score.py
│   ├── bioclip_score.py
│   └── sas_score.py
├── pipeline.py                # 主流程串接
├── baseline.py                # Baseline 比較實驗
└── config.yaml
```

#### 0.3 資料集下載
- **AI2D Dataset**：從官方來源下載，包含 5000+ 生物圖解與 QA annotations
- **生物知識文本**：從 AI2D 的 JSON annotations 中萃取結構化知識條目（`conceptmap.json`）
- **Diffusion 模型**：從 HuggingFace 下載 `runwayml/stable-diffusion-v1-5`

### Phase 0 Testing

| 測試項目 | 方法 | 通過標準 |
|---------|------|---------|
| 環境相依性 | `import` 所有套件，執行 `python -c "import torch; print(torch.cuda.is_available())"` | 無 ImportError，CUDA 可用 |
| AI2D 資料完整性 | 統計圖片數量與 annotation 數量 | 圖片 ≥ 4000 張，annotations 正確對應 |
| SD 模型載入 | 執行一次 dummy generation | 成功生成 512×512 圖片，無 OOM |
| 目錄結構 | `ls` 各子目錄 | 所有資料夾與初始檔案到位 |

---

## Phase 1：RAG 知識庫建構（Biologist Agent）

### 目標
從 AI2D dataset 萃取生物知識，建立可查詢的向量資料庫，實作 Biologist Agent。

### 實作步驟

#### 1.1 知識萃取（`data/knowledge_extractor.py`）
從 AI2D 的 `conceptmap.json` 與圖片 annotations 中提取結構化知識條目：
```python
# 每一個知識條目的格式範例
{
  "entity": "mitochondria",
  "constraints": [
    "must have inner folds called cristae",
    "has an outer membrane and inner membrane",
    "contains matrix with enzymes",
    "typically oval-shaped with diameter 0.5–10 μm"
  ],
  "spatial_relations": ["located in cytoplasm", "not in nucleus"],
  "source": "AI2D:cell_biology_003"
}
```

#### 1.2 向量資料庫建立（`data/build_vector_store.py`）
- Embedding model：`all-MiniLM-L6-v2`（輕量，適合本地）或 `text-embedding-ada-002`
- Vector store：ChromaDB（持久化儲存）
- Chunk strategy：每個生物實體一個 document，附加其 constraints 列表

#### 1.3 Biologist Agent 實作（`agents/biologist_agent.py`）
```python
# 核心功能
class BiologistAgent:
    def retrieve(self, user_query: str) -> list[str]:
        # 從向量庫檢索相關生物知識條目
        # 回傳：結構性約束清單 (biological constraints)
```
- 使用 LangChain `RetrievalQA` chain
- Top-k = 5 相似條目
- 輸出格式：JSON list of constraints

### Phase 1 Testing

| 測試項目 | 測試案例 | 通過標準 |
|---------|---------|---------|
| 知識萃取完整性 | 統計 AI2D 中被萃取的實體數量 | ≥ 500 個生物實體 |
| Embedding 品質 | 查詢 "mitochondria"，檢視 top-5 結果 | 所有返回條目均為細胞生物學相關 |
| Retrieval 精準度 | 設計 20 組 query-answer pair，計算 MRR | MRR ≥ 0.6 |
| Biologist Agent 輸出格式 | 輸入 5 種不同 query，驗證 JSON 格式 | 100% 輸出合法 JSON，constraints 非空 |
| 邊界條件 | 輸入與生物無關的 query（如 "car"） | 返回空列表或低相似度警告 |

---

## Phase 2：Spatial Architect Agent

### 目標
將 Biologist Agent 輸出的生物約束轉換為擴散模型可用的結構化空間描述 prompt。

### 實作步驟

#### 2.1 Spatial Architect 設計（`agents/spatial_architect.py`）
```python
class SpatialArchitectAgent:
    def build_layout_prompt(self, 
                            user_query: str, 
                            bio_constraints: list[str]) -> str:
        # 輸入：用戶查詢 + 生物約束條目
        # 輸出：結構化的 SD prompt（含空間關係描述）
```

- 使用 LLM（GPT-3.5-turbo 或本地 LLM）將約束轉換為描述性 prompt
- 參考 LayoutGPT 的思路：先生成 layout bounding box 描述，再轉換為文字 prompt
- Prompt 範例輸出：
  ```
  "A detailed scientific diagram of a mitochondria. 
   The organelle has an oval shape. The outer membrane 
   is smooth and surrounds the entire structure. The inner 
   membrane is folded into cristae (finger-like projections). 
   The central region contains the matrix. White background, 
   educational illustration style, labeled diagram."
  ```

#### 2.2 Constraint Injection 機制
- 強制在 prompt 中保留關鍵結構詞（negative prompt 排除常見錯誤）
- Negative prompt 範例：`"incorrect structure, missing cristae, wrong anatomy, blurry, cartoon"`

### Phase 2 Testing

| 測試項目 | 測試案例 | 通過標準 |
|---------|---------|---------|
| Prompt 完整性 | 對 10 種生物結構生成 prompt | 每個 prompt 包含 ≥ 3 個原始 bio_constraints 中的關鍵詞 |
| Prompt 可讀性（人工評估） | 3 位同學閱讀 prompt 並評分 1-5 | 平均分 ≥ 3.5 |
| 與 Biologist Agent 整合 | End-to-end: query → constraints → prompt | 無 exception，輸出格式符合 SD 輸入要求 |
| Negative prompt 生成 | 驗證 negative prompt 不含正確解剖詞彙 | 不出現 "cristae", "nucleus" 等正確結構名稱 |

---

## Phase 3：Generation Layer（Stable Diffusion 整合）

### 目標
接入 SD 模型，實作從結構化 prompt 到圖像的生成流程。

### 實作步驟

#### 3.1 Diffusion Pipeline 封裝（`generation/diffusion_pipeline.py`）
```python
from diffusers import StableDiffusionPipeline

class BioGenerationPipeline:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id)
    
    def generate(self, prompt: str, negative_prompt: str,
                 num_inference_steps=50, guidance_scale=7.5) -> Image:
        # 生成圖片，返回 PIL Image
```

#### 3.2 Baseline 建立（`baseline.py`）
- **Config 1（Baseline）**：直接用 user query 生成（無 RAG，無 agent）
- **Config 2**：RAG + Diffusion（Biologist Agent 輸出的 prompt 直接送 SD）
- **Config 3**：Full BioGuard-Diffusion（三個 agent + SD）

#### 3.3 圖片儲存與管理
- 每次生成附帶 metadata（query, prompt, config, seed）
- 儲存至 `outputs/{config}/{entity}_{seed}.png`

### Phase 3 Testing

| 測試項目 | 測試案例 | 通過標準 |
|---------|---------|---------|
| SD 模型推論 | 生成 5 張測試圖片（固定 seed=42） | 成功生成，解析度 512×512，無黑圖 |
| Prompt 長度限制 | 測試超過 77 token 的 prompt | 自動截斷或分段，不報錯 |
| 批量生成穩定性 | 連續生成 20 張圖 | 無 OOM、無 crash，完成率 100% |
| Baseline vs Structured prompt | 人工比較同一主題的兩種 prompt 生成結果 | Structured prompt 視覺上更具解剖標示 |

---

## Phase 4：Verifier Agent（VLM 驗證層）

### 目標
使用 VLM 對生成圖像進行自動化的科學準確性交叉驗證，並決定是否需要重新生成。

### 實作步驟

#### 4.1 Verifier Agent 設計（`agents/verifier_agent.py`）
```python
class VerifierAgent:
    def verify(self, image: Image, 
               bio_constraints: list[str],
               entity: str) -> dict:
        # 輸出：{
        #   "passed": bool,
        #   "score": float,  # 0~1
        #   "violations": list[str],  # 違反的約束
        #   "feedback": str           # 給 regeneration 的修正建議
        # }
```

- **方法一（輕量）**：使用 BioCLIP / CLIP 計算 image 與每個 constraint 的相似度分數
- **方法二（精確）**：使用 BioMedBLIP 或 GPT-4V 的 VQA 功能，針對每個 constraint 問 Yes/No question
  - 範例：*"Does this image show inner membrane folds (cristae) in the mitochondria?"*
- 若 score < threshold（如 0.6），觸發重新生成並附帶 feedback 給 Spatial Architect

#### 4.2 Retry Loop
```
generate → verify → [pass?] → output
                  ↓ [fail, ≤ 3 retries]
              refine prompt → generate again
```

### Phase 4 Testing

| 測試項目 | 測試案例 | 通過標準 |
|---------|---------|---------|
| BioCLIP 載入 | 載入 BioCLIP v2 模型 | 無錯誤，可對圖文進行 embedding |
| 正確圖像驗證 | 輸入真實生物教科書圖片 + 對應 constraints | score ≥ 0.7 |
| 錯誤圖像偵測 | 輸入故意錯誤的圖片（如缺少 cristae 的線粒體圖） | score < 0.5，violations 非空 |
| VQA 準確率 | 設計 30 個有/無特定結構的圖-問 pair | VQA 準確率 ≥ 70% |
| Retry Loop | 模擬 3 次 fail，驗證第 4 次強制輸出 | 無無限循環，正確終止 |

---

## Phase 5：全流程整合（Full Pipeline）

### 目標
將所有模組串接成完整的 BioGuard-Diffusion pipeline，實作三種 configuration 的比較架構。

### 實作步驟

#### 5.1 主流程（`pipeline.py`）
```python
def bioguard_pipeline(user_query: str, config: str) -> Image:
    if config == "baseline":
        return sd.generate(user_query)
    
    elif config == "rag_only":
        constraints = biologist_agent.retrieve(user_query)
        prompt = spatial_architect.build_layout_prompt(user_query, constraints)
        return sd.generate(prompt)
    
    elif config == "full":
        constraints = biologist_agent.retrieve(user_query)
        prompt = spatial_architect.build_layout_prompt(user_query, constraints)
        image = sd.generate(prompt)
        result = verifier.verify(image, constraints)
        # retry loop...
        return final_image
```

#### 5.2 LangChain Agent 協調
使用 LangChain `AgentExecutor` 或 `LangGraph` 協調三個 agents，設定：
- 每個 agent 的 tool description
- 執行順序（sequential）
- Retry 與 fallback 策略

### Phase 5 Testing

| 測試項目 | 測試案例 | 通過標準 |
|---------|---------|---------|
| 三種 config 全部可執行 | 對同一 query 跑三個 config | 均無 exception，均輸出圖片 |
| 端到端延遲測量 | 計時各 config 完成時間 | Baseline < 30s；Full pipeline < 3 分鐘 |
| Retry 機制驗證 | 強制第一次 verify fail | 自動重試並最終輸出 |
| 10 個多樣查詢測試 | 涵蓋細胞生物、神經解剖、骨骼系統 | 所有查詢完成，無 pipeline 崩潰 |

---

## Phase 6：評估實驗（Experiments & Evaluation）

### 目標
對三種 configuration 進行系統性定量評估，驗證 multi-agent 的必要性。

### 評估資料集設計
從 AI2D dataset 中選取 **50 個生物結構查詢**，涵蓋：
- 細胞生物（10 個）：mitochondria, nucleus, ribosome...
- 神經解剖（10 個）：gyri, sulci, cerebellum...
- 骨骼系統（10 個）：ribs, skull foramina, metacarpals...
- 心血管（10 個）：coronary arteries, aorta branching...
- 植物生物（10 個）：chloroplast, stomata, root hair...

### 評估指標實作

#### 6.1 CLIP-Score（`evaluation/clip_score.py`）
```python
# 量測生成圖像與 text prompt 的語意對齊程度
clip_score = cosine_similarity(
    clip.encode_image(image), 
    clip.encode_text(original_query)
)
```

#### 6.2 Bio-CLIP Score（`evaluation/bioclip_score.py`）
```python
# 使用 BioCLIP v2 計算圖像與生物分類標籤的對齊
# 量測是否正確呈現 taxonomy 上的正確生物結構
bioclip_score = bioclip.get_taxonomy_alignment(image, entity_label)
```

#### 6.3 Scientific Accuracy Score, SAS（`evaluation/sas_score.py`）
```python
# 依據 bio_constraints 列表，用 VLM VQA 計算通過率
# SAS = (passed_constraints / total_constraints)
# 可選：依照 taxonomic importance 加權
sas = sum(weights[c] * verify(image, c) for c in constraints) / sum(weights)
```

#### 6.4 人工評估（Human Evaluation）
- 邀請 2-3 位具備生物背景的評審（可為 TA 或同學）
- 評分維度：Anatomical Correctness (1-5)、Label Clarity (1-5)、Educational Value (1-5)
- Blind evaluation：評審不知道哪張圖是哪個 config 生成

### 實驗設計

| 實驗組 | 描述 | 預期假設 |
|-------|------|---------|
| Config 1：Baseline | 直接 one-shot SD 生成 | 最低 SAS，最多 hallucinations |
| Config 2：RAG + SD | Biologist Agent + Spatial Architect | 中等 SAS，改善 structural accuracy |
| Config 3：Full BioGuard | Config 2 + Verifier Agent + Retry | 最高 SAS，最少 violations |

### Phase 6 Testing / Validation

| 測試項目 | 方法 | 通過標準 |
|---------|------|---------|
| 指標一致性 | 同一張圖跑 3 次評估 | 評分標準差 < 0.05 |
| Config 1 vs 3 顯著性 | Paired t-test on SAS scores (n=50) | p-value < 0.05 |
| CLIP vs Bio-CLIP 相關性 | Pearson correlation | 驗證兩指標不完全相同（r < 0.9），說明 Bio-CLIP 有額外資訊 |
| Ablation：去掉 Verifier | Config 2 vs Config 3 SAS 差異 | Config 3 SAS ≥ Config 2 SAS + 5% |
| 人工評估一致性 | Inter-rater agreement (Cohen's κ) | κ ≥ 0.6（moderate agreement） |

---

## Phase 7：分析、撰寫與展示

### 目標
整理結果，完成 final report 與 demo。

### 實作步驟

#### 7.1 結果視覺化
- 三個 config 的 SAS / CLIP / BioCLIP 比較 bar chart
- 每個生物大類別的分析（細胞 vs 骨骼 vs 神經的難度差異）
- 失敗案例分析（Failure Analysis）：展示 Verifier 抓到的典型錯誤

#### 7.2 Qualitative 展示
- 同一查詢的三個 config 輸出圖片並排比較
- 展示 Retry Loop 的修正過程（Before / After）

#### 7.3 Demo Script
- 互動式 demo：使用者輸入生物名稱 → pipeline 輸出圖片 + SAS 分數

---

## 整體時間軸

```
Week 1-2  │ Phase 0 + Phase 1   │ 環境建置、AI2D 資料處理、RAG 向量庫建立
Week 3    │ Phase 2             │ Spatial Architect Agent
Week 4    │ Phase 3             │ SD Generation Layer + 3 Baselines
Week 5    │ Phase 4             │ Verifier Agent（BioCLIP / VQA）
Week 6    │ Phase 5             │ 全流程整合 + LangChain 協調
Week 7    │ Phase 6             │ 50 queries 評估實驗，計算所有指標
Week 8    │ Phase 7             │ 分析、撰寫報告、製作 Demo
```

---

## 技術選型總結

| 組件 | 選擇 | 備選 |
|-----|------|------|
| Multi-Agent Framework | LangChain / LangGraph | AutoGen |
| LLM (Agents) | GPT-3.5-turbo | Local LLaMA 3 |
| Vector DB | ChromaDB | FAISS |
| Embedding Model | all-MiniLM-L6-v2 | text-embedding-ada-002 |
| Diffusion Model | stable-diffusion-v1-5 | Z-Image, SDXL |
| Verifier VLM | BioCLIP v2 + CLIP VQA | GPT-4V, BioMedBLIP |
| Bio Knowledge Source | AI2D dataset | OpenStax Biology textbooks |

---

## 潛在風險與 Mitigation

| 風險 | 影響 | 對策 |
|-----|------|------|
| GPT API 費用過高 | 影響實驗規模 | 優先用本地 LLM；GPT 只用於 prompt building |
| SD 在解剖細節上效果差 | SAS 整體偏低 | 使用 negative prompts + ControlNet 輔助 |
| BioCLIP VQA 準確率不足 | Verifier 誤判 | 設計備用的 rule-based constraint checker |
| AI2D 知識庫覆蓋不足 | 部分查詢無法檢索 | 補充 OpenStax / Wikipedia 生物文本 |
| 人工評估招募困難 | Human eval 無法完成 | 用 GPT-4V 模擬人工評估作為 proxy |

---

## 定義「成功」的最低標準

- [ ] 三個 config 均可完整執行，無崩潰
- [ ] Full pipeline (Config 3) SAS ≥ Baseline (Config 1) SAS，差距 ≥ 10%
- [ ] 在 ≥ 3 種不同生物類別上有結果
- [ ] 完成定量評估表格（CLIP-Score + Bio-CLIP + SAS）
- [ ] 有至少 5 組 qualitative 圖像比較（3 configs 並排）
