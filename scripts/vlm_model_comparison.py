"""Compare 3 Gemini approaches for satellite LULC classification.

Tests on a single site-year image:
  1. Gemini 2.0 Flash — percentage-based JSON classification (current pipeline)
  2. Gemini 2.5 Flash — native segmentation masks (pixel-level)
  3. Gemini 3 Flash — agentic vision with code execution

Compares all three against Dynamic World ground truth.

Usage:
    python scripts/vlm_model_comparison.py
    python scripts/vlm_model_comparison.py --site teesta --year 2024
    python scripts/vlm_model_comparison.py --site manikganj --year 2023
"""
import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (
    apply_style, save_fig, FULL_WIDTH, LULC_COLORS, CLASS_LABELS,
    LULC_COLORS_RGB,
)

DATA_DIR = Path(__file__).parent.parent / "data"
IMG_DIR = DATA_DIR / "case_study_images"
DW_DIR = DATA_DIR / "case_study_dw_rasters"
VLM_DIR = DATA_DIR / "case_study_vlm"
FIG_DIR = Path(__file__).parent.parent / "docs" / "figures" / "case_studies"

# 10-class scheme
CLASSES = ["cropland", "trees", "shrub", "grassland", "flooded_veg",
           "built", "bare", "water", "snow", "solar"]
LULC_CLASSES_NO_SOLAR = [c for c in CLASSES if c != "solar"]

# DW raw label → our class name
DW_IDX_TO_CLASS = {
    0: "water", 1: "trees", 2: "grassland", 3: "flooded_veg",
    4: "cropland", 5: "shrub", 6: "built", 7: "bare", 8: "snow",
}

API_KEY = os.getenv("GOOGLE_AI_API_KEY")


# ── Approach 1: Gemini 2.0 Flash (percentage JSON) ───────────────────────────

def run_gemini_20_flash(img_path, site_name, year):
    """Current pipeline: percentage-based JSON classification."""
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Analyze this satellite image (4.77m resolution, ~4km x 4km).
Site: {site_name}, Year: {year}, Bangladesh.

Estimate the percentage breakdown of land cover classes:
cropland, trees, shrub, grassland, flooded_veg, built, bare, water, snow, solar

Return JSON:
{{"land_cover": {{"cropland": %, "trees": %, "shrub": %, "grassland": %, "flooded_veg": %, "built": %, "bare": %, "water": %, "snow": %, "solar": %}},
  "solar_visible": "yes" or "no",
  "solar_area_pct": 0-100,
  "description": "brief description"}}"""

    img = Image.open(img_path)
    response = model.generate_content(
        [prompt, img],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    result = json.loads(response.text)
    print(f"  2.0 Flash: {result.get('description', '')[:80]}")
    return {
        "model": "gemini-2.0-flash",
        "approach": "percentage_json",
        "result": result,
        "percentages": {k: float(v) for k, v in result.get("land_cover", {}).items()},
    }


# ── Approach 2: Gemini 2.5 Flash (segmentation masks) ────────────────────────

def run_gemini_25_flash_segmentation(img_path, site_name, year):
    """New: native segmentation masks with pixel-level output."""
    from google import genai as genai_new

    client = genai_new.Client(api_key=API_KEY)

    prompt = """Detect and segment all land cover regions in this satellite image.
For each region, provide a segmentation mask and label.

The land cover classes to detect are:
- solar (solar panels/arrays)
- cropland (agricultural fields)
- trees (forest/tree cover)
- shrub (shrub/scrub vegetation)
- grassland
- flooded_veg (wetland/flooded vegetation)
- built (buildings/roads/structures)
- bare (bare ground/sand/riverbed)
- water (rivers/ponds/water bodies)

Return the result as JSON list where each item has:
- "label": the land cover class name
- "box_2d": [y0, x0, y1, x1] normalized 0-1000
- "mask": base64 encoded PNG segmentation mask"""

    img = Image.open(img_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img],
        config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "thinking_config": {"thinking_budget": 0},
        },
    )

    raw_text = response.text
    segments = json.loads(raw_text)

    # Parse masks into numpy arrays
    img_w, img_h = img.size
    composite = np.zeros((img_h, img_w), dtype=np.uint8)  # class ID map
    class_to_id = {c: i for i, c in enumerate(CLASSES)}

    mask_count = 0
    for seg in segments:
        label = seg.get("label", "").lower().replace(" ", "_")
        if label not in class_to_id:
            # Try fuzzy match
            for cls in CLASSES:
                if cls in label or label in cls:
                    label = cls
                    break
            else:
                continue

        cid = class_to_id[label]
        box = seg.get("box_2d", [0, 0, 1000, 1000])
        mask_b64 = seg.get("mask", "")

        if not mask_b64:
            continue

        try:
            # Decode base64 PNG mask
            mask_bytes = base64.b64decode(mask_b64)
            mask_img = Image.open(BytesIO(mask_bytes)).convert("L")

            # Map box coordinates to pixel space
            y0 = int(box[0] / 1000 * img_h)
            x0 = int(box[1] / 1000 * img_w)
            y1 = int(box[2] / 1000 * img_h)
            x1 = int(box[3] / 1000 * img_w)

            # Resize mask to box dimensions
            box_w = max(x1 - x0, 1)
            box_h = max(y1 - y0, 1)
            mask_resized = mask_img.resize((box_w, box_h), Image.NEAREST)
            mask_arr = np.array(mask_resized)

            # Binarize at midpoint
            binary = (mask_arr > 127).astype(np.uint8)

            # Write to composite (later segments overwrite)
            composite[y0:y0+box_h, x0:x0+box_w][binary == 1] = cid
            mask_count += 1
        except Exception as e:
            print(f"    Mask decode error for {label}: {e}")

    # Compute percentages from mask
    total_pixels = composite.size
    percentages = {}
    for i, cls in enumerate(CLASSES):
        pct = 100.0 * np.sum(composite == i) / total_pixels
        percentages[cls] = round(pct, 1)

    # The "0" index captures unlabeled pixels — assign to most common non-zero
    unlabeled_pct = percentages.get(CLASSES[0], 0)

    print(f"  2.5 Flash Seg: {mask_count} masks, solar={percentages.get('solar', 0)}%")
    return {
        "model": "gemini-2.5-flash",
        "approach": "segmentation_masks",
        "mask_count": mask_count,
        "composite_mask": composite,
        "percentages": percentages,
        "raw_segments": len(segments),
    }


# ── Approach 2b: Gemini 2.5 Flash (percentage JSON for fair comparison) ──────

def run_gemini_25_flash_pct(img_path, site_name, year):
    """Gemini 2.5 Flash with same percentage prompt as 2.0 Flash."""
    from google import genai as genai_new

    client = genai_new.Client(api_key=API_KEY)

    prompt = f"""Analyze this satellite image (4.77m resolution, ~4km x 4km).
Site: {site_name}, Year: {year}, Bangladesh.

Estimate the percentage breakdown of land cover classes:
cropland, trees, shrub, grassland, flooded_veg, built, bare, water, snow, solar

Return JSON:
{{"land_cover": {{"cropland": %, "trees": %, "shrub": %, "grassland": %, "flooded_veg": %, "built": %, "bare": %, "water": %, "snow": %, "solar": %}},
  "solar_visible": "yes" or "no",
  "solar_area_pct": 0-100,
  "description": "brief description"}}"""

    img = Image.open(img_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img],
        config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    )

    result = json.loads(response.text)
    print(f"  2.5 Flash %: {result.get('description', '')[:80]}")
    return {
        "model": "gemini-2.5-flash",
        "approach": "percentage_json",
        "result": result,
        "percentages": {k: float(v) for k, v in result.get("land_cover", {}).items()},
    }


# ── Approach 3: Gemini 3 Flash (agentic vision) ──────────────────────────────

def run_gemini_3_flash(img_path, site_name, year):
    """Gemini 3 Flash with agentic vision (code execution for zoom/analysis)."""
    from google import genai as genai_new

    client = genai_new.Client(api_key=API_KEY)

    prompt = f"""You are analyzing a satellite image (4.77m resolution, ~4km x 4km).
Site: {site_name}, Year: {year}, Bangladesh.

Use code execution to carefully analyze this image. You may crop, zoom, and
examine different regions to accurately classify the land cover.

Estimate the percentage breakdown of land cover classes:
cropland, trees, shrub, grassland, flooded_veg, built, bare, water, snow, solar

After your analysis, return your final answer as JSON:
{{"land_cover": {{"cropland": %, "trees": %, "shrub": %, "grassland": %, "flooded_veg": %, "built": %, "bare": %, "water": %, "snow": %, "solar": %}},
  "solar_visible": "yes" or "no",
  "solar_area_pct": 0-100,
  "description": "detailed description of what you see and how you analyzed it"}}"""

    img = Image.open(img_path)

    # Enable code execution for agentic vision
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, img],
        config={
            "tools": [{"code_execution": {}}],
            "temperature": 0.1,
        },
    )

    # Extract the JSON from the response (may be embedded in text)
    full_text = response.text
    result = None
    # Try to find JSON in the response
    import re
    json_match = re.search(r'\{[\s\S]*"land_cover"[\s\S]*\}', full_text)
    if json_match:
        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    if not result:
        # Fallback: try the whole text
        try:
            result = json.loads(full_text)
        except json.JSONDecodeError:
            result = {"land_cover": {}, "description": full_text[:500]}

    desc = result.get("description", "")[:80]
    print(f"  3 Flash Agentic: {desc}")
    return {
        "model": "gemini-3-flash-preview",
        "approach": "agentic_vision",
        "result": result,
        "percentages": {k: float(v) for k, v in result.get("land_cover", {}).items()},
        "full_response": full_text[:2000],
    }


# ── Approach 3b: Gemini 3 Flash (plain percentage, no code execution) ────────

def run_gemini_3_flash_pct(img_path, site_name, year):
    """Gemini 3 Flash with plain percentage prompt (no code execution)."""
    from google import genai as genai_new

    client = genai_new.Client(api_key=API_KEY)

    prompt = f"""Analyze this satellite image (4.77m resolution, ~4km x 4km).
Site: {site_name}, Year: {year}, Bangladesh.

Estimate the percentage breakdown of land cover classes:
cropland, trees, shrub, grassland, flooded_veg, built, bare, water, snow, solar

Return JSON:
{{"land_cover": {{"cropland": %, "trees": %, "shrub": %, "grassland": %, "flooded_veg": %, "built": %, "bare": %, "water": %, "snow": %, "solar": %}},
  "solar_visible": "yes" or "no",
  "solar_area_pct": 0-100,
  "description": "brief description"}}"""

    img = Image.open(img_path)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, img],
        config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    )

    result = json.loads(response.text)
    print(f"  3 Flash %: {result.get('description', '')[:80]}")
    return {
        "model": "gemini-3-flash-preview",
        "approach": "percentage_json",
        "result": result,
        "percentages": {k: float(v) for k, v in result.get("land_cover", {}).items()},
    }


# ── Ground Truth: Dynamic World ──────────────────────────────────────────────

def load_dw_ground_truth(site, year):
    """Load DW raster as class percentages and colorized image."""
    npz_path = DW_DIR / f"{site}_{year}_dw.npz"
    if not npz_path.exists():
        return None, None

    data = np.load(npz_path)
    remapped = data["remapped"]  # 10-class IDs

    # Compute percentages
    total = remapped.size
    pcts = {}
    # 10-class IDs: 0=nodata, 1=cropland, 2=trees, 3=shrub, 4=grassland,
    #               5=flooded_veg, 6=built, 7=bare, 8=water, 9=snow
    id_to_class = {
        1: "cropland", 2: "trees", 3: "shrub", 4: "grassland",
        5: "flooded_veg", 6: "built", 7: "bare", 8: "water", 9: "snow",
    }
    for cid, cname in id_to_class.items():
        pcts[cname] = round(100.0 * np.sum(remapped == cid) / total, 1)
    pcts["solar"] = 0.0  # DW has no solar class

    # Colorize
    h, w = remapped.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    id_to_rgb = {
        0: (221, 221, 221), 1: (221, 204, 119), 2: (17, 119, 51),
        3: (153, 153, 51), 4: (68, 170, 153), 5: (51, 34, 136),
        6: (204, 102, 119), 7: (136, 34, 85), 8: (136, 204, 238),
        9: (245, 245, 245),
    }
    for cid, color in id_to_rgb.items():
        rgb[remapped == cid] = color

    return pcts, rgb


# ── Comparison Figure ─────────────────────────────────────────────────────────

def make_comparison_figure(site, year, results, dw_pcts, dw_rgb, seg_mask=None):
    """Create comparison figure across all approaches."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    apply_style()

    classes_display = ["cropland", "trees", "shrub", "grassland", "flooded_veg",
                       "built", "bare", "water", "solar"]

    has_seg = seg_mask is not None
    n_img_cols = 3 if has_seg else 2

    # Clean layout with GridSpec: top row = images, bottom = bar chart
    fig = plt.figure(figsize=(FULL_WIDTH + 1.5, 7.0))
    gs = GridSpec(2, n_img_cols, figure=fig, height_ratios=[1, 1.1],
                  hspace=0.35, wspace=0.15,
                  top=0.90, bottom=0.12, left=0.06, right=0.98)

    # Top row: satellite image and DW map
    img = Image.open(IMG_DIR / f"{site}_{year}.png")
    ax_sat = fig.add_subplot(gs[0, 0])
    ax_sat.imshow(img)
    ax_sat.set_title("Planet Satellite (4.77m)", fontsize=8, fontweight='bold')
    ax_sat.axis("off")

    ax_dw = fig.add_subplot(gs[0, 1])
    ax_dw.imshow(dw_rgb)
    ax_dw.set_title("Dynamic World (10m)", fontsize=8, fontweight='bold')
    ax_dw.axis("off")

    if has_seg:
        # Colorize segmentation mask
        h, w = seg_mask.shape
        seg_rgb = np.ones((h, w, 3), dtype=np.uint8) * 221
        class_to_id = {c: i for i, c in enumerate(CLASSES)}
        for cls, color_hex in LULC_COLORS.items():
            if cls in class_to_id:
                cid = class_to_id[cls]
                r, g, b = LULC_COLORS_RGB.get(cls, (221, 221, 221))
                seg_rgb[seg_mask == cid] = (r, g, b)
        solar_id = class_to_id.get("solar", -1)
        if solar_id >= 0:
            seg_rgb[seg_mask == solar_id] = (255, 107, 53)

        ax_seg = fig.add_subplot(gs[0, 2])
        ax_seg.imshow(seg_rgb)
        ax_seg.set_title("Gemini 2.5 Flash\nSegmentation Mask", fontsize=8,
                         fontweight='bold')
        ax_seg.axis("off")

    # Bottom row: grouped bar chart spanning full width
    ax_bar = fig.add_subplot(gs[1, :])

    x = np.arange(len(classes_display))
    n_bars = len(results) + 1  # +1 for DW
    width = 0.8 / n_bars

    # DW bars
    dw_vals = [dw_pcts.get(c, 0) for c in classes_display]
    ax_bar.bar(x - width * n_bars / 2 + width * 0.5, dw_vals, width,
               color='#4477AA', alpha=0.8, label='Dynamic World')

    # Model name → clean display label
    model_labels = {
        ("gemini-2.0-flash", "percentage_json"): "Gemini 2.0 Flash",
        ("gemini-2.5-flash", "percentage_json"): "Gemini 2.5 Flash",
        ("gemini-2.5-flash", "segmentation_masks"): "Gemini 2.5 Flash (seg.)",
        ("gemini-3-flash-preview", "percentage_json"): "Gemini 3 Flash",
        ("gemini-3-flash-preview", "agentic_vision"): "Gemini 3 Flash (agentic)",
    }
    colors = ['#CC6677', '#44AA99', '#DDCC77', '#882255', '#332288']
    for i, r in enumerate(results):
        vals = [r["percentages"].get(c, 0) for c in classes_display]
        label = model_labels.get((r['model'], r['approach']),
                                 f"{r['model']} ({r['approach']})")
        ax_bar.bar(x - width * n_bars / 2 + width * (i + 1.5), vals, width,
                   color=colors[i % len(colors)], alpha=0.8, label=label)

    # Use CLASS_LABELS for proper display names (capitalizes "Solar")
    display_labels = []
    for c in classes_display:
        display_labels.append(CLASS_LABELS.get(c, c.title()))

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(display_labels, fontsize=7, rotation=30, ha='right')
    ax_bar.set_ylabel("Coverage (%)", fontsize=8)
    ax_bar.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0.0, 1.0),
                  framealpha=0.9, edgecolor='none')

    fig.suptitle(f"VLM Model Comparison — {site.title()} {year}",
                 fontsize=10, fontweight='bold', y=0.96)

    out_path = FIG_DIR / f"vlm_model_comparison_{site}_{year}.png"
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare VLM models for LULC classification")
    parser.add_argument("--site", default="teesta", help="Site short name")
    parser.add_argument("--year", type=int, default=2024, help="Year")
    args = parser.parse_args()

    site = args.site
    year = args.year
    img_path = IMG_DIR / f"{site}_{year}.png"

    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    print(f"\n{'='*60}")
    print(f"VLM Model Comparison: {site} {year}")
    print(f"Image: {img_path}")
    print(f"{'='*60}\n")

    # Load DW ground truth
    print("Loading Dynamic World ground truth...")
    dw_pcts, dw_rgb = load_dw_ground_truth(site, year)
    if dw_pcts:
        print(f"  DW: cropland={dw_pcts.get('cropland',0)}%, bare={dw_pcts.get('bare',0)}%, "
              f"built={dw_pcts.get('built',0)}%")

    results = []
    seg_mask = None

    # 1. Gemini 2.0 Flash (current pipeline)
    print("\n--- Approach 1: Gemini 2.0 Flash (percentage JSON) ---")
    try:
        r1 = run_gemini_20_flash(img_path, site, year)
        results.append(r1)
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(2)

    # 2. Gemini 2.5 Flash (percentage for fair comparison)
    print("\n--- Approach 2a: Gemini 2.5 Flash (percentage JSON) ---")
    try:
        r2a = run_gemini_25_flash_pct(img_path, site, year)
        results.append(r2a)
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(2)

    # 3. Gemini 2.5 Flash (segmentation masks)
    print("\n--- Approach 2b: Gemini 2.5 Flash (segmentation masks) ---")
    try:
        r2b = run_gemini_25_flash_segmentation(img_path, site, year)
        results.append(r2b)
        seg_mask = r2b.get("composite_mask")
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(2)

    # 4. Gemini 3 Flash (percentage, no code exec)
    print("\n--- Approach 3a: Gemini 3 Flash (percentage JSON) ---")
    try:
        r3a = run_gemini_3_flash_pct(img_path, site, year)
        results.append(r3a)
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(2)

    # 5. Gemini 3 Flash (agentic vision with code execution)
    print("\n--- Approach 3b: Gemini 3 Flash (agentic vision) ---")
    try:
        r3b = run_gemini_3_flash(img_path, site, year)
        results.append(r3b)
    except Exception as e:
        print(f"  ERROR: {e}")

    # Save raw results
    out_json = FIG_DIR / f"vlm_model_comparison_{site}_{year}.json"
    save_data = {
        "site": site,
        "year": year,
        "dw_percentages": dw_pcts,
        "approaches": [],
    }
    for r in results:
        entry = {
            "model": r["model"],
            "approach": r["approach"],
            "percentages": r["percentages"],
        }
        if "result" in r:
            entry["raw_result"] = r["result"]
        if "full_response" in r:
            entry["agentic_response"] = r["full_response"]
        if "mask_count" in r:
            entry["mask_count"] = r["mask_count"]
            entry["raw_segments"] = r["raw_segments"]
        save_data["approaches"].append(entry)

    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nRaw results saved: {out_json}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Class':<15} {'DW':>6} ", end="")
    for r in results:
        label = f"{r['model'].split('gemini-')[-1][:10]}"
        print(f" {label:>12}", end="")
    print()
    print("-" * (22 + 13 * len(results)))

    for cls in ["cropland", "trees", "shrub", "grassland", "flooded_veg",
                "built", "bare", "water", "solar"]:
        dw_val = dw_pcts.get(cls, 0) if dw_pcts else 0
        print(f"{cls:<15} {dw_val:>5.1f}%", end="")
        for r in results:
            val = r["percentages"].get(cls, 0)
            print(f" {val:>11.1f}%", end="")
        print()

    # Generate figure
    if results and dw_pcts is not None:
        make_comparison_figure(site, year, results, dw_pcts, dw_rgb, seg_mask)


if __name__ == "__main__":
    main()
