"""
Stage 4 Tests — Config 3 Full BioGuard Validation
Run: conda run -n agent python bioguard_diffusion/generation/test_stage4.py
"""
import json
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
C1_FILE  = BASE / "results" / "config1_results.json"
C2_FILE  = BASE / "results" / "config2_results.json"
C3_FILE  = BASE / "results" / "config3_results.json"
VER_FILE = BASE / "results" / "config3_verify_log.json"
OUT_DIR  = BASE / "outputs" / "config3"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    results.append(condition)

with open(C1_FILE) as f: c1 = {r["id"]: r for r in json.load(f)}
with open(C2_FILE) as f: c2 = {r["id"]: r for r in json.load(f)}
with open(C3_FILE) as f: c3_records = json.load(f)
with open(VER_FILE) as f: ver_log = json.load(f)

print("=" * 60)
print("Stage 4 Tests: Config 3 Full BioGuard-Diffusion")
print("=" * 60)

# T1: 10 result records
check("Exactly 10 result records", len(c3_records) == 10)

# T2: All images exist
missing = [r["id"] for r in c3_records if not Path(r["image_file"]).exists()]
check("All 10 images saved to disk", len(missing) == 0,
      f"missing: {missing}" if missing else "")

# T3: All images valid 512×512
bad = []
for r in c3_records:
    try:
        if Image.open(r["image_file"]).size != (512, 512):
            bad.append(r["id"])
    except: bad.append(r["id"])
check("All images are 512×512 PNG", len(bad) == 0)

# T4: All records have SAS field
no_sas = [r["id"] for r in c3_records if r.get("sas") is None]
check("All records have SAS scores", len(no_sas) == 0)

# T5: Mean SAS > 0.20 (above threshold baseline)
mean_sas = sum(r["sas"] for r in c3_records) / 10
check("Mean SAS > 0.20", mean_sas > 0.20, f"actual: {mean_sas:.4f}")

# T6: Mean CLIP Config 3 ≥ Config 1
c3_mean = sum(r["clip_score"] for r in c3_records) / 10
c1_mean = sum(c1[r["id"]]["clip_score"] for r in c3_records) / 10
check("Config 3 mean CLIP ≥ Config 1", c3_mean >= c1_mean,
      f"C3={c3_mean:.4f}, C1={c1_mean:.4f}")

# T7: Verify log has 10 entries
check("Verify log has 10 entries", len(ver_log) == 10)

# T8: Every verify log entry has per_constraint
no_pc = [e["id"] for e in ver_log if not e.get("per_constraint")]
check("All verify entries have per_constraint", len(no_pc) == 0)

# T9: Config tagged correctly
wrong_tag = [r["id"] for r in c3_records if r.get("config") != "config3_full"]
check("All records tagged config3_full", len(wrong_tag) == 0)

# T10: attempts_used is always ≥ 1
bad_attempts = [r["id"] for r in c3_records if r.get("attempts_used", 0) < 1]
check("All records have attempts_used ≥ 1", len(bad_attempts) == 0)

# T11: SAS in valid range
bad_sas = [(r["id"], r["sas"]) for r in c3_records if not (0.0 <= r["sas"] <= 1.0)]
check("All SAS scores in [0, 1]", len(bad_sas) == 0)

# Print final 3-way table
c2_mean = sum(c2[r["id"]]["clip_score"] for r in c3_records) / 10
print(f"\n{'ID':<6} {'Domain':<18} {'C1':>7} {'C2':>7} {'C3':>7} {'SAS':>7} {'Tries':>6}")
print("-" * 62)
for r in c3_records:
    print(f"{r['id']:<6} {r['domain']:<18} "
          f"{c1[r['id']]['clip_score']:>7.4f} "
          f"{c2[r['id']]['clip_score']:>7.4f} "
          f"{r['clip_score']:>7.4f} "
          f"{r['sas']:>7.4f} "
          f"{r['attempts_used']:>6}")
print("-" * 62)
print(f"{'Mean':<6} {'':18} {c1_mean:>7.4f} {c2_mean:>7.4f} {c3_mean:>7.4f} {mean_sas:>7.4f}")

retries = sum(r["attempts_used"] for r in c3_records) - 10
passed_sas = sum(1 for r in c3_records if r["sas_passed"])
print(f"\nRetries triggered: {retries}")
print(f"Queries passing SAS: {passed_sas}/10")
print(f"Verifier model: CLIP-ViT-B/32 (keyword extraction, threshold=0.22)")

passed = sum(results)
total  = len(results)
print(f"\n{'='*60}")
print(f"Result: {passed}/{total} tests passed")
if passed == total:
    print("Stage 4 COMPLETE — all experiments finished!")
else:
    print("Fix failing tests before finalising.")
print("=" * 60)
