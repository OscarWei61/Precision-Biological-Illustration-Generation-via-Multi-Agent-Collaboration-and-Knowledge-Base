"""
Stage 1: AI2D Knowledge Extraction
Parses AI2D annotations -> structured knowledge base JSON.
Run: python extract_knowledge.py
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
AI2D = BASE / "data" / "ai2d"
ANN_DIR = AI2D / "annotations"
IMG_DIR = AI2D / "images"
CATS_FILE = AI2D / "categories.json"
OUT_FILE = BASE / "results" / "knowledge_base.json"
QUERIES_FILE = BASE / "results" / "test_queries.json"

# ── Biology-relevant AI2D categories ───────────────────────────────────────
BIO_CATEGORIES = {
    "partsOfA",
    "lifeCycles",
    "photosynthesisRespiration",
    "foodChainsWebs",
}

# ── Handpicked test set (image → human query) ──────────────────────────────
# Selected to cover 5 biology domains; each image has ≥ 3 matched keywords.
TEST_QUERIES = [
    {
        "id": "q01",
        "image": "1071.png",
        "query": "Draw a labeled diagram of a plant cell showing all major organelles",
        "domain": "cell_biology",
    },
    {
        "id": "q02",
        "image": "3288.png",
        "query": "Draw a detailed cross-section of a mitochondria showing cristae and matrix",
        "domain": "cell_biology",
    },
    {
        "id": "q03",
        "image": "2925.png",
        "query": "Draw a labeled diagram of a neuron showing axon, dendrites and myelin sheath",
        "domain": "neuroanatomy",
    },
    {
        "id": "q04",
        "image": "2857.png",
        "query": "Draw a cross-section of a human eye showing retina, lens, cornea and optic nerve",
        "domain": "human_anatomy",
    },
    {
        "id": "q05",
        "image": "1385.png",
        "query": "Draw a labeled diagram of the human digestive system",
        "domain": "human_anatomy",
    },
    {
        "id": "q06",
        "image": "1382.png",
        "query": "Draw a labeled diagram of the human heart showing ventricles, atria and aorta",
        "domain": "human_anatomy",
    },
    {
        "id": "q07",
        "image": "1014.png",
        "query": "Draw a labeled diagram of the parts of a flower including petal, sepal, stamen and pistil",
        "domain": "plant_biology",
    },
    {
        "id": "q08",
        "image": "1086.png",
        "query": "Draw a labeled diagram of a leaf showing blade, midrib, petiole and veins",
        "domain": "plant_biology",
    },
    {
        "id": "q09",
        "image": "4088.png",
        "query": "Draw a diagram illustrating the process of photosynthesis with inputs and outputs",
        "domain": "biochemistry",
    },
    {
        "id": "q10",
        "image": "1191.png",
        "query": "Draw a labeled diagram of an insect body showing head, thorax, abdomen and antennae",
        "domain": "zoology",
    },
]


def extract_labels(annotation: dict) -> list[str]:
    """Return cleaned text labels from an AI2D annotation dict."""
    labels = []
    for tobj in annotation.get("text", {}).values():
        val = tobj.get("value", "").strip()
        if val and len(val) > 1:
            labels.append(val)
    return labels


def extract_relationships(annotation: dict, label_map: dict) -> list[str]:
    """Convert AI2D relationship keys (e.g. 'T0+A0+B0') to human-readable strings."""
    rels = []
    for rel_key in annotation.get("relationships", {}):
        parts = rel_key.split("+")
        names = []
        for p in parts:
            if p.startswith("T") and p in label_map:
                names.append(label_map[p])
        if len(names) >= 2:
            rels.append(f"{names[0]} → {names[1]}")
    return rels


def build_knowledge_base() -> list[dict]:
    print("Loading categories...")
    with open(CATS_FILE) as f:
        categories = json.load(f)

    bio_images = {
        img: cat for img, cat in categories.items() if cat in BIO_CATEGORIES
    }
    print(f"  Bio-relevant images: {len(bio_images)}")

    knowledge_base = []
    label_global_counter = Counter()

    for img_name, category in bio_images.items():
        ann_path = ANN_DIR / f"{img_name}.json"
        if not ann_path.exists():
            continue

        with open(ann_path) as f:
            ann = json.load(f)

        label_map = {
            tid: tobj.get("value", "").strip()
            for tid, tobj in ann.get("text", {}).items()
        }
        labels = [v for v in label_map.values() if v and len(v) > 1]
        relationships = extract_relationships(ann, label_map)

        if not labels:
            continue

        for lbl in labels:
            label_global_counter[lbl.lower()] += 1

        knowledge_base.append(
            {
                "image": img_name,
                "category": category,
                "labels": labels,
                "relationships": relationships,
                "label_count": len(labels),
            }
        )

    return knowledge_base, label_global_counter


def enrich_with_constraints(knowledge_base: list[dict]) -> list[dict]:
    """
    Attach manually-curated biological constraints for each test image.
    These become the ground-truth facts the Biologist Agent should retrieve.
    """
    constraints_map = {
        "1071.png": [
            "A plant cell has a rigid cell wall made of cellulose surrounding the cell membrane",
            "The nucleus is the control center and contains the cell's DNA",
            "Chloroplasts are found only in plant cells and perform photosynthesis",
            "The central vacuole stores water and maintains turgor pressure",
            "Mitochondria are the powerhouse of the cell, producing ATP via respiration",
            "Ribosomes are small organelles that synthesize proteins",
            "The Golgi apparatus packages and ships proteins",
            "Rough ER has ribosomes and is involved in protein synthesis",
            "Smooth ER is involved in lipid synthesis and detoxification",
        ],
        "3288.png": [
            "Mitochondria have an outer membrane and a folded inner membrane",
            "The inner membrane folds are called cristae, which increase surface area",
            "The space between outer and inner membrane is the intermembrane space",
            "The central fluid-filled region is called the matrix",
            "Mitochondrial DNA is located in the matrix",
            "ATP synthesis occurs along the inner membrane (cristae)",
            "Mitochondria are typically oval-shaped with diameter 0.5-10 micrometers",
        ],
        "2925.png": [
            "A neuron has a cell body (soma), dendrites, and an axon",
            "Dendrites receive signals from other neurons",
            "The axon transmits electrical signals away from the cell body",
            "The myelin sheath insulates the axon and speeds signal transmission",
            "Nodes of Ranvier are gaps in the myelin sheath",
            "Axon terminals release neurotransmitters into the synapse",
            "Schwann cells produce the myelin sheath in the peripheral nervous system",
        ],
        "2857.png": [
            "The cornea is the transparent front part of the eye that refracts light",
            "The iris controls the size of the pupil",
            "The lens focuses light onto the retina",
            "The retina contains photoreceptor cells (rods and cones)",
            "The optic nerve transmits signals from the retina to the brain",
            "The fovea is the area of sharpest vision in the retina",
            "The vitreous gel fills the space between the lens and retina",
        ],
        "1385.png": [
            "The digestive system begins at the mouth and ends at the anus",
            "The esophagus connects the mouth to the stomach",
            "The stomach uses acid and enzymes to break down food",
            "The liver produces bile to help digest fats",
            "The pancreas secretes digestive enzymes into the small intestine",
            "The small intestine absorbs nutrients",
            "The large intestine absorbs water and forms feces",
            "The appendix is a small pouch attached to the large intestine",
        ],
        "1382.png": [
            "The heart has four chambers: right atrium, left atrium, right ventricle, left ventricle",
            "The aorta carries oxygenated blood from the left ventricle to the body",
            "The pulmonary arteries carry deoxygenated blood from the right ventricle to the lungs",
            "The septum separates the left and right sides of the heart",
            "Valves prevent backflow of blood between chambers",
            "The heart's right side receives deoxygenated blood and pumps it to the lungs",
            "The heart's left side receives oxygenated blood and pumps it to the body",
        ],
        "1014.png": [
            "A flower has petals that attract pollinators",
            "Sepals are leaf-like structures that protect the flower bud",
            "The stamen is the male reproductive organ consisting of anther and filament",
            "The anther produces pollen grains",
            "The pistil is the female reproductive organ consisting of stigma, style, and ovary",
            "The stigma traps pollen during pollination",
            "The ovary contains egg cells that develop into seeds after fertilization",
        ],
        "1086.png": [
            "A leaf has a flat blade (lamina) that captures sunlight",
            "The midrib is the central vein running through the leaf",
            "Lateral veins branch off the midrib and transport water and nutrients",
            "The petiole is the stalk that attaches the leaf to the stem",
            "Stipules are small leaf-like appendages at the base of the petiole",
            "The leaf margin is the edge of the blade",
        ],
        "4088.png": [
            "Photosynthesis takes place in the chloroplasts of plant cells",
            "The inputs of photosynthesis are carbon dioxide (CO2), water (H2O), and sunlight",
            "The outputs of photosynthesis are glucose (C6H12O6) and oxygen (O2)",
            "Chlorophyll molecules in the leaves absorb radiant energy from sunlight",
            "The light-dependent reactions occur in the thylakoid membranes",
            "The Calvin cycle (light-independent reactions) occurs in the stroma",
        ],
        "1191.png": [
            "An insect body is divided into three main parts: head, thorax, and abdomen",
            "Insects have exactly 6 legs attached to the thorax",
            "The head contains compound eyes, antennae, and mouthparts",
            "Antennae are sensory organs used for smell and touch",
            "Wings, when present, are attached to the thorax",
            "The abdomen contains the digestive and reproductive organs",
        ],
    }

    for entry in knowledge_base:
        img = entry["image"]
        entry["constraints"] = constraints_map.get(img, [])

    return knowledge_base


def print_summary(knowledge_base: list[dict], label_counter: Counter):
    print(f"\n{'='*60}")
    print("KNOWLEDGE BASE SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries:        {len(knowledge_base)}")
    print(f"Unique bio labels:    {len(label_counter)}")
    print(f"\nTop 20 most frequent labels:")
    for label, cnt in label_counter.most_common(20):
        print(f"  {cnt:4d}  {label}")

    print(f"\n{'='*60}")
    print("TEST QUERIES (10 selected)")
    print(f"{'='*60}")
    domain_count = Counter(q["domain"] for q in TEST_QUERIES)
    for domain, cnt in domain_count.items():
        print(f"  {domain}: {cnt} queries")

    print(f"\nQuery list:")
    for q in TEST_QUERIES:
        print(f"  [{q['id']}] ({q['domain']}) {q['query'][:70]}...")


def main():
    print("Stage 1: AI2D Knowledge Extraction")
    print(f"AI2D path: {AI2D}")

    knowledge_base, label_counter = build_knowledge_base()
    knowledge_base = enrich_with_constraints(knowledge_base)

    # Save full knowledge base
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    print(f"\nKnowledge base saved: {OUT_FILE}")
    print(f"  Total entries: {len(knowledge_base)}")

    # Save test queries
    with open(QUERIES_FILE, "w") as f:
        json.dump(TEST_QUERIES, f, indent=2, ensure_ascii=False)
    print(f"Test queries saved:   {QUERIES_FILE}")

    print_summary(knowledge_base, label_counter)

    # Spot-check: show constraints for test images
    test_imgs = {q["image"] for q in TEST_QUERIES}
    print(f"\n{'='*60}")
    print("SPOT CHECK: Constraints for test images")
    print(f"{'='*60}")
    for entry in knowledge_base:
        if entry["image"] in test_imgs and entry.get("constraints"):
            print(f"\n[{entry['image']}] Labels: {entry['labels'][:5]}")
            print(f"  Constraints ({len(entry['constraints'])}):")
            for c in entry["constraints"][:3]:
                print(f"    - {c}")


if __name__ == "__main__":
    main()
