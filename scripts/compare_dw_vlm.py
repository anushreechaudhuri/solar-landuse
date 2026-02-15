"""Compare Dynamic World vs VLM land cover classifications.

For each image that has both a DW mask and a VLM mask, computes:
- Per-class area percentages for both sources
- Pixel-level disagreement rate
- Side-by-side visualization (DW | VLM | Disagreement)

Outputs:
- results/dw_vs_vlm_comparison.csv  — summary table
- results/dw_vlm_comparison/        — per-image visualizations

Usage:
    python scripts/compare_dw_vlm.py
"""

import csv
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import re

PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
MASK_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'masks'
RESULTS_DIR = PROJECT_DIR / 'results'
VIZ_DIR = RESULTS_DIR / 'dw_vlm_comparison'
CSV_PATH = RESULTS_DIR / 'dw_vs_vlm_comparison.csv'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

CLASSES_FILE = PROJECT_DIR / 'data' / 'training_dataset' / 'classes.json'
with open(CLASSES_FILE) as f:
    CLASSES = json.load(f)
CLASS_NAMES = {v: k for k, v in CLASSES.items()}
NUM_CLASSES = len(CLASSES)

CLASS_COLORS = {
    0: [0, 0, 0],        # background
    1: [255, 255, 0],    # agriculture
    2: [0, 128, 0],      # forest
    3: [0, 0, 255],      # water
    4: [255, 0, 0],      # urban
    5: [128, 0, 128],    # solar_panels
    6: [165, 42, 42],    # bare_land
}

# Disagreement map colors
AGREE_COLOR = [40, 40, 40]       # dark gray = agreement
DISAGREE_COLOR = [255, 255, 0]   # yellow = disagreement


def colorize(mask):
    """Convert single-channel class mask to RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        rgb[mask == cid] = color
    return rgb


def class_percentages(mask):
    """Return dict of class_name -> percentage."""
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    pcts = {}
    for cid in range(NUM_CLASSES):
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        idx = np.where(unique == cid)[0]
        pcts[name] = 100.0 * (counts[idx[0]] / total) if len(idx) > 0 else 0.0
    return pcts


def parse_stem(stem):
    """Extract base stem from mask filename like 'teesta_5km_2024_01_post_dw_mask'."""
    m = re.match(r'(.+?)_(dw|vlm)_mask$', stem)
    return m.group(1) if m else None


def main():
    # Discover paired masks
    dw_masks = {f.stem: f for f in sorted(MASK_DIR.glob('*_dw_mask.png'))}
    vlm_masks = {f.stem: f for f in sorted(MASK_DIR.glob('*_vlm_mask.png'))}

    # Build base_stem -> {dw, vlm} mapping
    pairs = {}
    for stem, path in dw_masks.items():
        base = parse_stem(stem)
        if base:
            pairs.setdefault(base, {})['dw'] = path
    for stem, path in vlm_masks.items():
        base = parse_stem(stem)
        if base:
            pairs.setdefault(base, {})['vlm'] = path

    both = {k: v for k, v in pairs.items() if 'dw' in v and 'vlm' in v}
    print(f"Found {len(both)} images with both DW and VLM masks\n")

    if not both:
        print("Nothing to compare.")
        return

    rows = []

    for base_stem in sorted(both):
        dw_arr = np.array(Image.open(both[base_stem]['dw']))
        vlm_arr = np.array(Image.open(both[base_stem]['vlm']))

        # Ensure same shape
        if dw_arr.shape != vlm_arr.shape:
            h = min(dw_arr.shape[0], vlm_arr.shape[0])
            w = min(dw_arr.shape[1], vlm_arr.shape[1])
            dw_arr = np.array(Image.fromarray(dw_arr).resize((w, h), Image.NEAREST))
            vlm_arr = np.array(Image.fromarray(vlm_arr).resize((w, h), Image.NEAREST))

        disagree = dw_arr != vlm_arr
        disagree_pct = 100.0 * disagree.sum() / disagree.size

        dw_pcts = class_percentages(dw_arr)
        vlm_pcts = class_percentages(vlm_arr)

        row = {'image': base_stem, 'disagree_pct': round(disagree_pct, 1)}
        for name in CLASS_NAMES.values():
            row[f'dw_{name}'] = round(dw_pcts[name], 1)
            row[f'vlm_{name}'] = round(vlm_pcts[name], 1)
        rows.append(row)

        # Console output
        print(f"{base_stem}  disagree={disagree_pct:.1f}%")
        for name in CLASS_NAMES.values():
            d, v = dw_pcts[name], vlm_pcts[name]
            if d > 0.5 or v > 0.5:
                print(f"  {name:15s}  DW={d:5.1f}%  VLM={v:5.1f}%  diff={v-d:+.1f}%")

        # Visualization: DW | VLM | Disagreement
        dw_rgb = colorize(dw_arr)
        vlm_rgb = colorize(vlm_arr)
        diff_rgb = np.full_like(dw_rgb, AGREE_COLOR, dtype=np.uint8)
        diff_rgb[disagree] = DISAGREE_COLOR

        gap = 4
        h, w = dw_arr.shape
        canvas_w = w * 3 + gap * 2
        canvas = np.ones((h + 30, canvas_w, 3), dtype=np.uint8) * 255
        canvas[30:30+h, 0:w] = dw_rgb
        canvas[30:30+h, w+gap:2*w+gap] = vlm_rgb
        canvas[30:30+h, 2*w+2*gap:3*w+2*gap] = diff_rgb

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((w // 2 - 10, 8), "DW", fill=(0, 0, 0), font=font)
        draw.text((w + gap + w // 2 - 12, 8), "VLM", fill=(0, 0, 0), font=font)
        draw.text((2 * w + 2 * gap + w // 2 - 30, 8), f"Disagree {disagree_pct:.0f}%", fill=(0, 0, 0), font=font)

        img.save(str(VIZ_DIR / f"{base_stem}_comparison.png"))

    # Write CSV
    fieldnames = rows[0].keys()
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved: {CSV_PATH}")
    print(f"Visualizations: {VIZ_DIR}/")


if __name__ == "__main__":
    print("=" * 60)
    print("DW vs VLM Land Cover Comparison")
    print("=" * 60)
    main()
    print("=" * 60)
