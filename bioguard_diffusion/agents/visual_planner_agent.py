"""
Visual Planner Agent — extracts spatial layout from AI2D annotations.

Reads the raw AI2D annotation JSON (blobs + text labels + relationships) for a
given image and returns a structured layout dict with polygon coordinates scaled
to the target generation size (512x512 by default).

Two relationship strategies are handled:
  intraObjectLabel   : text label IS the blob name (embedded label)
  intraObjectLinkage : text → arrow → blob  (external label, most common in AI2D)
                       In this case the text rectangle center is used as the
                       approximate position of the named structure.

Output schema:
  {
    "image":            "1071.png",
    "subject":          "plant cell",
    "layout": [
      {
        "name":         "Cristae",
        "center":       [256, 180],
        "bounds":       [200, 140, 310, 220],
        "polygon":      [[200,140], ...],
        "is_label_hint": False,   # True = text-rect approximation, not blob polygon
      },
      ...
    ],
    "has_spatial_data": True,
  }

When no annotation exists, layout is empty and has_spatial_data=False.
BlueprintGenerator falls back to a blank (no-guidance) control image.
"""

import json
from pathlib import Path
from PIL import Image


class VisualPlannerAgent:
    def __init__(self, annotations_dir: Path, images_dir: Path):
        self.annotations_dir = annotations_dir
        self.images_dir      = images_dir

    def plan(
        self,
        image_name: str,
        subject: str,
        target_size: int = 512,
    ) -> dict:
        """
        Extract and scale blob polygons + text labels from AI2D annotations.

        Parameters
        ----------
        image_name  : filename of the source image, e.g. "3288.png"
        subject     : human-readable subject label for the layout
        target_size : output coordinate space (default 512)

        Returns
        -------
        layout dict (see module docstring)
        """
        ann_path = self.annotations_dir / f"{image_name}.json"
        if not ann_path.exists():
            print(f"  [VisualPlanner] No annotation: {ann_path.name}")
            return {
                "image":            image_name,
                "subject":          subject,
                "layout":           [],
                "has_spatial_data": False,
            }

        with open(ann_path) as f:
            ann = json.load(f)

        # Determine original image size for coordinate scaling
        img_path = self.images_dir / image_name
        if img_path.exists():
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
        else:
            orig_w, orig_h = self._infer_size(ann)

        def sx(x): return int(x * target_size / orig_w)
        def sy(y): return int(y * target_size / orig_h)

        text_items    = ann.get("text", {})
        blob_items    = ann.get("blobs", {})
        relationships = ann.get("relationships", {})

        # Strategy A: intraObjectLabel (text directly labelling a blob)
        blob_to_labels: dict[str, list[str]] = {}
        # Strategy B: intraObjectLinkage (text → arrow → blob, most common)
        text_linked: dict[str, str] = {}  # text_id → blob_id

        for rel in relationships.values():
            cat     = rel.get("category", "")
            text_id = rel.get("origin", "")
            blob_id = rel.get("destination", "")

            if cat == "intraObjectLabel" and text_id in text_items and blob_id in blob_items:
                raw  = text_items[text_id]
                name = (raw.get("replacementText") or raw.get("value", "")).strip()
                if name:
                    blob_to_labels.setdefault(blob_id, []).append(name)

            elif cat == "intraObjectLinkage" and text_id in text_items and blob_id in blob_items:
                text_linked[text_id] = blob_id

        layout = []

        # ── Strategy A blobs (embedded label directly on blob) ────────────────
        for blob_id, names in blob_to_labels.items():
            blob = blob_items.get(blob_id)
            if not blob:
                continue
            poly = blob.get("polygon", [])
            if len(poly) < 3:
                continue
            layout.append(self._make_item(
                name         = ", ".join(names),
                poly         = poly,
                scale_x      = sx,
                scale_y      = sy,
                is_label_hint= False,
            ))

        # ── Strategy B: text-label positions + blob outline ───────────────────
        # The text rectangle center approximates where the structure is labelled.
        blob_ids_from_linkage: set[str] = set(text_linked.values())

        for text_id, blob_id in text_linked.items():
            txt  = text_items[text_id]
            rect = txt.get("rectangle", [])
            name = txt.get("value", "").strip() or txt.get("replacementText", text_id)
            if len(rect) == 2:
                x1, y1 = rect[0]
                x2, y2 = rect[1]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                r  = 12   # small square region around the label text
                layout.append({
                    "name":         name,
                    "center":       [sx(cx), sy(cy)],
                    "bounds":       [sx(cx - r), sy(cy - r), sx(cx + r), sy(cy + r)],
                    "polygon":      [
                        [sx(cx - r), sy(cy - r)], [sx(cx + r), sy(cy - r)],
                        [sx(cx + r), sy(cy + r)], [sx(cx - r), sy(cy + r)],
                    ],
                    "is_label_hint": True,
                })

        # Add blob outlines from linkage as structural boundaries (different from label hints)
        for blob_id in blob_ids_from_linkage:
            if blob_id in blob_to_labels:
                continue  # already added via Strategy A
            blob = blob_items.get(blob_id)
            if not blob:
                continue
            poly = blob.get("polygon", [])
            if len(poly) < 3:
                continue
            layout.append(self._make_item(
                name         = f"boundary_{blob_id}",
                poly         = poly,
                scale_x      = sx,
                scale_y      = sy,
                is_label_hint= False,
            ))

        # ── Unlabelled blobs with no linkage (pure shape fallback) ────────────
        linked_blob_ids = set(blob_to_labels.keys()) | blob_ids_from_linkage
        for blob_id, blob in blob_items.items():
            if blob_id in linked_blob_ids:
                continue
            poly = blob.get("polygon", [])
            if len(poly) < 3:
                continue
            layout.append(self._make_item(
                name         = f"region_{blob_id}",
                poly         = poly,
                scale_x      = sx,
                scale_y      = sy,
                is_label_hint= False,
            ))

        has_data = len(layout) > 0
        labelled = [i for i in layout if not i["name"].startswith(("region_", "boundary_"))]
        print(f"  [VisualPlanner] {image_name}: {len(layout)} regions "
              f"({len(labelled)} named structures)")

        return {
            "image":            image_name,
            "subject":          subject,
            "layout":           layout,
            "has_spatial_data": has_data,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_item(
        name: str,
        poly: list,
        scale_x,
        scale_y,
        is_label_hint: bool = False,
    ) -> dict:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))
        return {
            "name":         name,
            "center":       [scale_x(cx), scale_y(cy)],
            "bounds":       [scale_x(min(xs)), scale_y(min(ys)),
                             scale_x(max(xs)), scale_y(max(ys))],
            "polygon":      [[scale_x(p[0]), scale_y(p[1])] for p in poly],
            "is_label_hint": is_label_hint,
        }

    @staticmethod
    def _infer_size(ann: dict) -> tuple:
        max_x = max_y = 100
        for blob in ann.get("blobs", {}).values():
            for p in blob.get("polygon", []):
                max_x = max(max_x, p[0])
                max_y = max(max_y, p[1])
        for txt in ann.get("text", {}).values():
            for p in txt.get("rectangle", []):
                max_x = max(max_x, p[0])
                max_y = max(max_y, p[1])
        return max_x + 10, max_y + 10
