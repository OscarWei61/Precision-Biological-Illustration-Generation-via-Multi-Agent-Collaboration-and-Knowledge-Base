"""
Stage 3 Tests — Config 2 RAG Pipeline Validation
Run: conda run -n agent python bioguard_diffusion/generation/test_stage3.py
"""
import json
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
C1_FILE = BASE / "results" / "config1_results.json"
C2_FILE = BASE / "results" / "config2_results.json"
PROMPTS_FILE = BASE / "results" / "config2_prompts.json"
OUT_DIR = BASE / "outputs" / "config2"
VECTOR_STORE = BASE / "data" / "vector_store"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    results.append(condition)

with open(C1_FILE) as f:
    c1 = {r["id"]: r for r in json.load(f)}
with open(C2_FILE) as f:
    c2_records = json.load(f)
with open(PROMPTS_FILE) as f:
    prompts = {p["id"]: p for p in json.load(f)}

print("=" * 58)
print("Stage 3 Tests: Config 2 RAG + Spatial Architect")
print("=" * 58)

# T1: Vector store exists on disk
check("Vector store directory exists", VECTOR_STORE.exists())

# T2: 10 result records
check("Exactly 10 result records", len(c2_records) == 10, f"actual: {len(c2_records)}")

# T3: All images saved
missing = [r["id"] for r in c2_records if not Path(r["image_file"]).exists()]
check("All 10 images saved to disk", len(missing) == 0,
      f"missing: {missing}" if missing else "")

# T4: All images are valid 512×512
bad = []
for r in c2_records:
    try:
        img = Image.open(r["image_file"])
        if img.size != (512, 512):
            bad.append((r["id"], img.size))
    except Exception as e:
        bad.append((r["id"], str(e)))
check("All images are 512×512 PNG", len(bad) == 0, str(bad) if bad else "")

# T5: All CLIP scores in range
bad_clip = [(r["id"], r["clip_score"]) for r in c2_records
            if not (0.1 <= r["clip_score"] <= 0.5)]
check("All CLIP scores in range [0.1, 0.5]", len(bad_clip) == 0,
      str(bad_clip) if bad_clip else "")

# T6: Config tag is correct
wrong = [r["id"] for r in c2_records if r.get("config") != "config2_rag"]
check("All records tagged config2_rag", len(wrong) == 0)

# T7: Prompts were generated for all 10 queries
check("Prompt log has 10 entries", len(prompts) == 10, f"actual: {len(prompts)}")

# T8: Every prompt is longer than the raw query (RAG added content)
shorter = [(pid, len(p["structured_prompt"]), len(p["original_query"]))
           for pid, p in prompts.items()
           if len(p["structured_prompt"]) <= len(p["original_query"])]
check("All structured prompts longer than raw query", len(shorter) == 0,
      str(shorter) if shorter else "")

# T9: Every prompt log has constraints_used > 0
no_constraints = [pid for pid, p in prompts.items() if not p.get("constraints_used")]
check("All prompts injected ≥ 1 constraint", len(no_constraints) == 0,
      f"empty: {no_constraints}" if no_constraints else "")

# T10: Mean CLIP of Config 2 ≥ Config 1
c1_mean = sum(c1[r["id"]]["clip_score"] for r in c2_records) / 10
c2_mean = sum(r["clip_score"] for r in c2_records) / 10
check("Config 2 mean CLIP ≥ Config 1 mean CLIP",
      c2_mean >= c1_mean,
      f"C2={c2_mean:.4f}, C1={c1_mean:.4f}, delta={c2_mean-c1_mean:+.4f}")

# T11: Majority of queries improved
improved = sum(1 for r in c2_records if r["clip_delta_vs_c1"] > 0)
check("≥ 5/10 queries improved vs Config 1", improved >= 5,
      f"improved: {improved}/10")

# Print full comparison table
print(f"\n{'ID':<6} {'Domain':<18} {'C1':>7} {'C2':>7} {'Delta':>8}")
print("-" * 50)
for r in c2_records:
    c1_clip = c1[r["id"]]["clip_score"]
    d = r["clip_delta_vs_c1"]
    arrow = "↑" if d > 0 else "↓"
    print(f"{r['id']:<6} {r['domain']:<18} {c1_clip:>7.4f} {r['clip_score']:>7.4f} {arrow}{abs(d):>7.4f}")
print(f"\nMean: C1={c1_mean:.4f}  C2={c2_mean:.4f}  delta={c2_mean-c1_mean:+.4f}")
print(f"Improved: {improved}/10")

# Print sample prompt
sample = prompts.get("q02", {})
if sample:
    print(f"\nSample structured prompt (q02 - mitochondria):")
    print(f"  {sample['structured_prompt'][:200]}...")

passed = sum(results)
total = len(results)
print(f"\n{'='*58}")
print(f"Result: {passed}/{total} tests passed")
if passed == total:
    print("Stage 3 COMPLETE — ready for Stage 4 (Full BioGuard)")
else:
    print("Fix failing tests before proceeding.")
print("=" * 58)
