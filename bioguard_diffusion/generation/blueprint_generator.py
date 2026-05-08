"""
Blueprint Generator — converts VisualPlannerAgent layout into a ControlNet
control image.

Two modes
---------
lineart      (default)
    Draws blob polygon outlines (white background, black edges).
    Suitable as ControlNet Canny input — the outlines ARE the structural edges.

segmentation
    Fills each blob with a distinct colour.
    Useful for inspecting coverage but less suitable for Canny ControlNet.

Adaptive highlighting
---------------------
highlight_missing(layout, missing_names) redraws polygons whose names match
any entry in missing_names with thicker, higher-contrast edges.  Called by
the generation loop when the Verifier reports missing_structures, so the next
ControlNet pass emphasises those regions.
"""

from PIL import Image, ImageDraw
from pathlib import Path

# 12-colour palette for segmentation mode
_PALETTE = [
    (220,  50,  50),  # red
    ( 50, 100, 220),  # blue
    ( 50, 180,  80),  # green
    (220, 180,  50),  # yellow
    (180,  50, 220),  # purple
    ( 50, 200, 200),  # cyan
    (220, 120,  50),  # orange
    (200,  50, 130),  # pink
    (100, 220, 220),  # light-cyan
    (220, 220, 100),  # light-yellow
    (130, 100, 220),  # lavender
    (100, 180, 100),  # light-green
]


class BlueprintGenerator:
    def __init__(self, width: int = 512, height: int = 512):
        self.width  = width
        self.height = height

    # ──────────────────────────────────────────────────────────────────────────
    # Primary API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        layout: dict,
        mode: str = "lineart",
        line_width: int = 3,
    ) -> Image.Image:
        """
        Convert layout dict (from VisualPlannerAgent) to a control image.

        Returns a 512×512 RGB PIL image.
        If has_spatial_data=False, returns blank black (no-edge) image so
        ControlNet conditioning has zero effect when scale=0.
        """
        if not layout.get("has_spatial_data") or not layout.get("layout"):
            # No spatial data → black image (no edges)
            return Image.new("RGB", (self.width, self.height), (0, 0, 0))

        if mode == "segmentation":
            return self._segmentation(layout["layout"])
        return self._lineart(layout["layout"], line_width, normal_width=line_width)

    def highlight_missing(
        self,
        layout: dict,
        missing_names: list[str],
        base_line_width: int = 2,
        highlight_line_width: int = 6,
    ) -> Image.Image:
        """
        Regenerate the blueprint with missing structures drawn in bold red.
        Called after Verifier reports missing_structures to push ControlNet
        guidance toward those regions on the next attempt.

        matching is case-insensitive substring: layout item "Nucleus" matches
        missing "nucleus" or "cell nucleus".
        """
        if not layout.get("has_spatial_data") or not layout.get("layout"):
            return Image.new("RGB", (self.width, self.height), (0, 0, 0))

        canvas = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        draw   = ImageDraw.Draw(canvas)

        missing_lower = [m.lower() for m in missing_names]

        for item in layout["layout"]:
            poly = item.get("polygon", [])
            if len(poly) < 3:
                continue
            pts = [tuple(p) for p in poly]

            name_lower = item["name"].lower()
            is_missing = any(
                ml in name_lower or name_lower in ml
                for ml in missing_lower
            )

            if is_missing:
                draw.polygon(pts, outline=(200, 0, 0), width=highlight_line_width)
            else:
                draw.polygon(pts, outline=(0, 0, 0), width=base_line_width)

        return canvas

    def save(self, image: Image.Image, path: Path) -> None:
        image.save(path)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _lineart(
        self,
        items: list[dict],
        line_width: int,
        normal_width: int,
    ) -> Image.Image:
        canvas = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        draw   = ImageDraw.Draw(canvas)
        for item in items:
            poly = item.get("polygon", [])
            if len(poly) < 3:
                continue
            pts = [tuple(p) for p in poly]
            if item.get("is_label_hint"):
                # Text-position hint: draw as small filled dot (dark grey)
                cx, cy = item["center"]
                r = 4
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(80, 80, 80))
            else:
                # Structural polygon: draw outline
                draw.polygon(pts, outline=(0, 0, 0), width=line_width)
        return canvas

    def _segmentation(self, items: list[dict]) -> Image.Image:
        canvas = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        draw   = ImageDraw.Draw(canvas)
        for i, item in enumerate(items):
            poly = item.get("polygon", [])
            if len(poly) < 3:
                continue
            pts   = [tuple(p) for p in poly]
            color = _PALETTE[i % len(_PALETTE)]
            draw.polygon(pts, fill=color, outline=(0, 0, 0), width=2)
        return canvas
