"""
Stage 1 Tests — Knowledge Base Validation
Run: python test_stage1.py
"""

import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[2]
KB_FILE = BASE / "results" / "knowledge_base.json"
QUERIES_FILE = BASE / "results" / "test_queries.json"
AI2D_IMGS = BASE / "data" / "ai2d" / "images"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    results.append(condition)


# ── Load files ─────────────────────────────────────────────────────────────
with open(KB_FILE) as f:
    kb = json.load(f)
with open(QUERIES_FILE) as f:
    queries = json.load(f)

print("=" * 55)
print("Stage 1 Tests: Knowledge Base & Test Queries")
print("=" * 55)

# T1: KB has enough entries
check("KB has ≥ 500 entries", len(kb) >= 500, f"actual: {len(kb)}")

# T2: Every entry has required keys
required_keys = {"image", "category", "labels", "relationships", "constraints"}
bad = [e for e in kb if not required_keys.issubset(e.keys())]
check("All KB entries have required keys", len(bad) == 0, f"{len(bad)} entries missing keys")

# T3: Every entry has ≥ 1 label
no_labels = [e for e in kb if not e["labels"]]
check("All KB entries have ≥ 1 label", len(no_labels) == 0, f"{len(no_labels)} entries empty")

# T4: Exactly 10 test queries
check("Exactly 10 test queries", len(queries) == 10, f"actual: {len(queries)}")

# T5: Query IDs are unique
ids = [q["id"] for q in queries]
check("Query IDs are unique", len(ids) == len(set(ids)))

# T6: Query fields are complete
missing_fields = []
for q in queries:
    for key in ("id", "query", "image", "domain"):
        if not q.get(key):
            missing_fields.append((q.get("id"), key))
check("All queries have id/query/image/domain", len(missing_fields) == 0,
      str(missing_fields) if missing_fields else "")

# T7: All test images exist on disk
missing_imgs = [q["image"] for q in queries if not (AI2D_IMGS / q["image"]).exists()]
check("All 10 test images exist on disk", len(missing_imgs) == 0,
      f"missing: {missing_imgs}" if missing_imgs else "")

# T8: Test images have constraints in KB
kb_img_map = {e["image"]: e for e in kb}
no_constraints = [q["image"] for q in queries
                  if not kb_img_map.get(q["image"], {}).get("constraints")]
check("All 10 test images have constraints", len(no_constraints) == 0,
      f"missing constraints: {no_constraints}" if no_constraints else "")

# T9: Each test image has ≥ 5 constraints
few_constraints = [(q["image"], len(kb_img_map.get(q["image"], {}).get("constraints", [])))
                   for q in queries
                   if len(kb_img_map.get(q["image"], {}).get("constraints", [])) < 5]
check("All test images have ≥ 5 constraints", len(few_constraints) == 0,
      str(few_constraints) if few_constraints else "")

# T10: Domain coverage — at least 4 distinct domains
domains = {q["domain"] for q in queries}
check("Test queries span ≥ 4 domains", len(domains) >= 4,
      f"domains: {sorted(domains)}")

# T11: Unique labels across KB (vocabulary richness)
all_labels = [lbl.lower() for e in kb for lbl in e["labels"]]
unique_labels = len(set(all_labels))
check("KB has ≥ 1000 unique biological labels", unique_labels >= 1000,
      f"actual: {unique_labels}")

# T12: No empty query strings
empty_queries = [q["id"] for q in queries if not q["query"].strip()]
check("No empty query strings", len(empty_queries) == 0)

# ── Summary ────────────────────────────────────────────────────────────────
passed = sum(results)
total = len(results)
print(f"\n{'='*55}")
print(f"Result: {passed}/{total} tests passed")
if passed == total:
    print("Stage 1 COMPLETE — ready to proceed to Stage 2 (Config 1 Baseline)")
else:
    print("Fix failing tests before proceeding.")
print("=" * 55)
