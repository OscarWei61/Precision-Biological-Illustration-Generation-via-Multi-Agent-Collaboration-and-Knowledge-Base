"""
Stage 2 Tests — Config 1 Baseline Validation
Run: conda run -n agent python bioguard_diffusion/generation/test_stage2.py
"""
import json
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
RESULTS_FILE = BASE / "results" / "config1_results.json"
OUT_DIR = BASE / "outputs" / "config1"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status} {name}" + (f" — {detail}" if detail else ""))
    results.append(condition)

with open(RESULTS_FILE) as f:
    records = json.load(f)

print("=" * 55)
print("Stage 2 Tests: Config 1 Baseline")
print("=" * 55)

# T1: 10 records produced
check("Exactly 10 result records", len(records) == 10, f"actual: {len(records)}")

# T2: All images saved on disk
missing = [r["id"] for r in records if not Path(r["image_file"]).exists()]
check("All 10 images saved to disk", len(missing) == 0,
      f"missing: {missing}" if missing else "")

# T3: All images are valid 512x512 PNG
bad_imgs = []
for r in records:
    try:
        img = Image.open(r["image_file"])
        if img.size != (512, 512):
            bad_imgs.append((r["id"], img.size))
    except Exception as e:
        bad_imgs.append((r["id"], str(e)))
check("All images are 512×512 PNG", len(bad_imgs) == 0,
      str(bad_imgs) if bad_imgs else "")

# T4: All records have CLIP scores
no_clip = [r["id"] for r in records if r.get("clip_score") is None]
check("All records have CLIP scores", len(no_clip) == 0)

# T5: CLIP scores in a reasonable range (0.1 – 0.5 for SD)
out_of_range = [(r["id"], r["clip_score"]) for r in records
                if not (0.1 <= r["clip_score"] <= 0.5)]
check("All CLIP scores in range [0.1, 0.5]", len(out_of_range) == 0,
      str(out_of_range) if out_of_range else "")

# T6: Results JSON has all required keys
required = {"id", "domain", "query", "image_file", "clip_score", "generation_time_s", "config"}
bad_keys = [r["id"] for r in records if not required.issubset(r.keys())]
check("All records have required keys", len(bad_keys) == 0)

# T7: Config field is correct
wrong_config = [r["id"] for r in records if r.get("config") != "config1_baseline"]
check("All records tagged config1_baseline", len(wrong_config) == 0)

# T8: Generation completed within 120s/image
slow = [(r["id"], r["generation_time_s"]) for r in records if r["generation_time_s"] > 120]
check("All images generated within 120s", len(slow) == 0,
      str(slow) if slow else "")

# Print score table
print(f"\n{'ID':<6} {'Domain':<18} {'CLIP':>6}  {'Time(s)':>7}")
print("-" * 45)
scores = []
for r in records:
    print(f"{r['id']:<6} {r['domain']:<18} {r['clip_score']:>6.4f}  {r['generation_time_s']:>7.1f}")
    scores.append(r["clip_score"])
print(f"\nMean CLIP: {sum(scores)/len(scores):.4f} | Min: {min(scores):.4f} | Max: {max(scores):.4f}")
print("(These are the baseline numbers. Config 2 should improve on mean CLIP.)")

passed = sum(results)
total = len(results)
print(f"\n{'='*55}")
print(f"Result: {passed}/{total} tests passed")
if passed == total:
    print("Stage 2 COMPLETE — baseline established, ready for Stage 3")
else:
    print("Fix failing tests before proceeding.")
print("=" * 55)
