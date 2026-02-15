"""Merge Dynamic World and VLM masks into final training masks.

Dynamic World provides baseline land cover at 10m native resolution.
VLM provides solar panel detection (which DW cannot distinguish from urban).
The merge uses DW as base and overlays VLM solar panel classifications.

Also exports colored visualizations and copies images to training directory.

Usage:
    python scripts/merge_masks.py
"""

import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import json
import re

PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
LABEL_DIR = PROJECT_DIR / 'data' / 'for_labeling'
MASK_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'masks'
IMAGES_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'images'
CLASSES_FILE = PROJECT_DIR / 'data' / 'training_dataset' / 'classes.json'
MASK_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Color map for visualizations (matches apply_segmentation.py)
CLASS_COLORS = {
    0: [0, 0, 0],        # background - black
    1: [255, 255, 0],    # agriculture - yellow
    2: [0, 128, 0],      # forest - green
    3: [0, 0, 255],      # water - blue
    4: [255, 0, 0],      # urban - red
    5: [128, 0, 128],    # solar_panels - purple
    6: [165, 42, 42],    # bare_land - brown
}

with open(CLASSES_FILE) as f:
    CLASSES = json.load(f)
CLASS_NAMES = {v: k for k, v in CLASSES.items()}


def parse_filename(name):
    """Parse period (pre/post) from filename."""
    match = re.match(r'(.+?)_1km_(\d{4})_(\d{2})_(pre|post)\.png', name)
    if match:
        return match.group(4)
    return None


def merge(dw_mask, vlm_mask, is_post):
    """Merge Dynamic World and VLM masks.

    Strategy:
    - Start with DW as base (higher spatial detail from 10m native resolution)
    - VLM overrides DW where it identifies solar panels (class 5)
    - For pre-construction: never allow solar panels
    - Where only one source exists, use that source
    """
    if dw_mask is not None and vlm_mask is not None:
        # Both available: DW base + VLM solar override
        merged = dw_mask.copy()

        if is_post:
            # VLM solar panel detections override DW
            solar_mask = vlm_mask == 5
            merged[solar_mask] = 5

            # Where DW says urban but VLM says bare_land near solar areas,
            # trust VLM (construction-related bare land)
            bare_mask = vlm_mask == 6
            urban_mask = dw_mask == 4
            merged[bare_mask & urban_mask] = 6
        else:
            # Pre-construction: ensure no solar panels
            merged[merged == 5] = 4  # remap any accidental solar to urban

        return merged

    elif dw_mask is not None:
        # DW only (no VLM)
        merged = dw_mask.copy()
        if not is_post:
            merged[merged == 5] = 4
        return merged

    elif vlm_mask is not None:
        # VLM only (no DW, e.g. missing coordinates)
        merged = vlm_mask.copy()
        if not is_post:
            merged[merged == 5] = 4
        return merged

    return None


def create_colored_mask(mask):
    """Create RGB visualization from class mask."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    return colored


def print_class_distribution(mask, prefix=""):
    """Print class distribution for a mask."""
    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size
    parts = []
    for v, c in zip(unique, counts):
        name = CLASS_NAMES.get(v, f"unknown({v})")
        pct = 100.0 * c / total
        parts.append(f"{name}:{pct:.0f}%")
    print(f"{prefix}{', '.join(parts)}")


def main():
    print("=" * 60)
    print("Merge Dynamic World + VLM Masks")
    print("=" * 60)

    # Find all 1km PNG images
    png_files = sorted(LABEL_DIR.glob('*_1km_*_*.png'))
    print(f"Found {len(png_files)} images to process\n")

    success = 0
    skipped = 0
    failed = 0

    for png_path in png_files:
        stem = png_path.stem
        period = parse_filename(png_path.name)
        is_post = period == "post"

        # Check for existing final mask
        final_mask_path = MASK_DIR / f"{stem}_mask.png"
        colored_path = MASK_DIR / f"{stem}_mask_colored.png"

        # Load available masks
        dw_path = MASK_DIR / f"{stem}_dw_mask.png"
        vlm_path = MASK_DIR / f"{stem}_vlm_mask.png"

        dw_mask = None
        vlm_mask = None

        if dw_path.exists():
            dw_mask = np.array(Image.open(dw_path))
        if vlm_path.exists():
            vlm_mask = np.array(Image.open(vlm_path))

        if dw_mask is None and vlm_mask is None:
            print(f"  {stem}: no masks available, skipping")
            failed += 1
            continue

        # Get image dimensions for validation
        img = Image.open(png_path)
        w, h = img.size

        # Validate mask dimensions match image
        for name, m in [("DW", dw_mask), ("VLM", vlm_mask)]:
            if m is not None and m.shape != (h, w):
                print(f"  {stem}: {name} mask size {m.shape} != image size ({h},{w}), resizing")
                m_img = Image.fromarray(m.astype(np.uint8))
                m_resized = m_img.resize((w, h), Image.NEAREST)
                if name == "DW":
                    dw_mask = np.array(m_resized)
                else:
                    vlm_mask = np.array(m_resized)

        # Merge
        merged = merge(dw_mask, vlm_mask, is_post)
        if merged is None:
            print(f"  {stem}: merge failed")
            failed += 1
            continue

        # Save final mask (single-channel, pixel value = class ID)
        Image.fromarray(merged.astype(np.uint8)).save(str(final_mask_path))

        # Save colored visualization
        colored = create_colored_mask(merged)
        Image.fromarray(colored).save(str(colored_path))

        # Copy source image to training images directory
        dest_img = IMAGES_DIR / png_path.name
        if not dest_img.exists():
            shutil.copy2(str(png_path), str(dest_img))

        # Print summary
        sources = []
        if dw_mask is not None:
            sources.append("DW")
        if vlm_mask is not None:
            sources.append("VLM")
        source_str = "+".join(sources)

        print(f"  {stem} ({source_str}, {'post' if is_post else 'pre'}): ", end="")
        print_class_distribution(merged)

        success += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success} merged, {skipped} skipped, {failed} failed")
    print(f"Final masks: {MASK_DIR}/*_mask.png")
    print(f"Visualizations: {MASK_DIR}/*_mask_colored.png")
    print(f"Training images: {IMAGES_DIR}/")
    print()

    # Print legend
    print("Color legend:")
    for class_id, color in sorted(CLASS_COLORS.items()):
        name = CLASS_NAMES.get(class_id, "unknown")
        print(f"  {class_id} = {name}: RGB({color[0]},{color[1]},{color[2]})")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
