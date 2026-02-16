"""Merge masks into final training labels (V3: DW base + GRW/VLM solar overlay).

Strategy:
  1. Start with Dynamic World mask as base (non-solar land cover at 10m detail)
  2. If GRW mask exists → overlay solar (class 5) pixels from GRW (pixel-perfect polygons)
  3. Else if VLM mask exists → overlay solar (class 5) pixels from VLM (blocky grid fallback)
  4. Pre-construction images: never allow class 5 (solar → bare_land/urban)

This replaces V2 (VLM-primary for all classes, DW gap-fill) with a cleaner split:
DW provides spatially detailed non-solar context, GRW provides precise solar boundaries
from actual polygon data. VLM is only used for solar on sites not covered by GRW.

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
    """Parse period (pre/post) from filename like 'manikganj_5km_2017_02_pre.png'"""
    match = re.match(r'(.+?)_(\d+)km_(\d{4})_(\d{2})_(pre|post)\.png', name)
    if match:
        return match.group(5)
    return None


def merge(dw_mask, grw_mask, vlm_mask, is_post):
    """Merge masks: DW base + GRW/VLM solar overlay (V3).

    Strategy:
    - DW provides the base (non-solar land cover classes)
    - GRW provides pixel-perfect solar polygons (preferred)
    - VLM provides blocky solar grid (fallback if no GRW)
    - Pre-construction: never allow class 5
    """
    # Determine base mask (DW preferred, fall back to VLM for non-solar)
    if dw_mask is not None:
        merged = dw_mask.copy()
        # Remove any solar from DW (DW doesn't map solar, but just in case)
        merged[merged == 5] = 0
    elif vlm_mask is not None:
        # No DW available — use VLM as base but strip solar (will re-add below)
        merged = vlm_mask.copy()
        merged[merged == 5] = 0
    else:
        return None

    # Overlay solar from best available source
    if is_post:
        if grw_mask is not None:
            # GRW: pixel-perfect polygon boundaries
            solar_pixels = grw_mask == 5
            merged[solar_pixels] = 5
        elif vlm_mask is not None:
            # VLM fallback: blocky 20x20 grid solar
            solar_pixels = vlm_mask == 5
            merged[solar_pixels] = 5

    # Pre-construction enforcement: no solar allowed
    if not is_post:
        merged[merged == 5] = 4  # solar → urban (nearest built class)

    return merged


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


def resize_mask(mask, target_w, target_h, name, stem):
    """Resize mask to target dimensions if needed."""
    if mask.shape != (target_h, target_w):
        print(f"  {stem}: {name} mask size {mask.shape} != image size ({target_h},{target_w}), resizing")
        m_img = Image.fromarray(mask.astype(np.uint8))
        return np.array(m_img.resize((target_w, target_h), Image.NEAREST))
    return mask


def main():
    print("=" * 60)
    print("Merge Masks V3: DW base + GRW/VLM solar overlay")
    print("=" * 60)

    # Find all PNG images (1km and 5km)
    png_files = sorted(LABEL_DIR.glob('*_*km_*_*.png'))
    print(f"Found {len(png_files)} images to process\n")

    success = 0
    skipped = 0
    failed = 0
    grw_count = 0
    vlm_count = 0

    for png_path in png_files:
        stem = png_path.stem
        period = parse_filename(png_path.name)
        is_post = period == "post"

        final_mask_path = MASK_DIR / f"{stem}_mask.png"
        colored_path = MASK_DIR / f"{stem}_mask_colored.png"

        # Load available masks
        dw_path = MASK_DIR / f"{stem}_dw_mask.png"
        vlm_path = MASK_DIR / f"{stem}_vlm_mask.png"
        grw_path = MASK_DIR / f"{stem}_grw_mask.png"

        dw_mask = np.array(Image.open(dw_path)) if dw_path.exists() else None
        vlm_mask = np.array(Image.open(vlm_path)) if vlm_path.exists() else None
        grw_mask = np.array(Image.open(grw_path)) if grw_path.exists() else None

        if dw_mask is None and vlm_mask is None:
            print(f"  {stem}: no base masks available, skipping")
            failed += 1
            continue

        # Get image dimensions for validation
        img = Image.open(png_path)
        w, h = img.size

        # Validate/resize mask dimensions
        if dw_mask is not None:
            dw_mask = resize_mask(dw_mask, w, h, "DW", stem)
        if vlm_mask is not None:
            vlm_mask = resize_mask(vlm_mask, w, h, "VLM", stem)
        if grw_mask is not None:
            grw_mask = resize_mask(grw_mask, w, h, "GRW", stem)

        # Merge
        merged = merge(dw_mask, grw_mask, vlm_mask, is_post)
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

        # Determine solar source for logging
        sources = []
        if dw_mask is not None:
            sources.append("DW")
        solar_source = "none"
        if is_post:
            if grw_mask is not None and np.any(grw_mask == 5):
                solar_source = "GRW"
                grw_count += 1
            elif vlm_mask is not None and np.any(vlm_mask == 5):
                solar_source = "VLM"
                vlm_count += 1
        sources.append(f"solar:{solar_source}")
        source_str = ", ".join(sources)

        print(f"  {stem} ({source_str}, {'post' if is_post else 'pre'}): ", end="")
        print_class_distribution(merged)

        success += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success} merged, {skipped} skipped, {failed} failed")
    print(f"Solar sources: {grw_count} GRW (polygon), {vlm_count} VLM (grid fallback)")
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
