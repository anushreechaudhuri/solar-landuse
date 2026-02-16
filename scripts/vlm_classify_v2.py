"""VLM-based land cover classification using Gemini 2.0 Flash (10-class scheme).

For each satellite image, asks Gemini to estimate percentage breakdown of
land cover classes. For post-construction images, overlays the solar polygon
boundary and tells Gemini what % is solar, asking it to classify the rest.

Output: JSON files with percentage breakdowns per image, compatible with
compare_lulc_datasets.py CSV format.

Usage:
    python scripts/vlm_classify_v2.py
    python scripts/vlm_classify_v2.py --force   # re-classify all (ignore cache)
"""

import google.generativeai as genai
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from shapely.geometry import shape as shp_shape, Polygon

load_dotenv()

PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
LABEL_DIR = PROJECT_DIR / 'data' / 'for_labeling'
RAW_DIR = PROJECT_DIR / 'data' / 'raw_images'
VLM_V2_DIR = PROJECT_DIR / 'data' / 'vlm_v2_responses'
GRW_PATH = PROJECT_DIR / 'data' / 'grw' / 'confirmed_matches.json'

VLM_V2_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 4  # seconds between API calls

# 10-class scheme (matching compare_lulc_datasets.py)
CLASS_NAMES = [
    'no_data', 'cropland', 'trees', 'shrub', 'grassland',
    'flooded_veg', 'built', 'bare', 'water', 'snow',
]

# Site metadata
SITES = {
    "teesta": {"name": "Teesta 200 MW", "lat": 25.628342, "lon": 89.541082, "mw": 200},
    "feni": {"name": "Feni 75 MW", "lat": 22.787567, "lon": 91.367187, "mw": 75},
    "manikganj": {"name": "Manikganj 35 MW", "lat": 23.780834, "lon": 89.824775, "mw": 35},
    "moulvibazar": {"name": "Moulvibazar 10 MW", "lat": 24.493896, "lon": 91.633043, "mw": 10},
    "pabna": {"name": "Pabna 64 MW", "lat": 23.961375, "lon": 89.159720, "mw": 64},
    "mymensingh": {"name": "Mymensingh 50 MW", "lat": 24.702233, "lon": 90.461730, "mw": 50},
    "tetulia": {"name": "Tetulia 8 MW", "lat": 26.482817, "lon": 88.410139, "mw": 8},
    "mongla": {"name": "Mongla 100 MW", "lat": 22.574239, "lon": 89.570388, "mw": 100},
    "sirajganj68": {"name": "Sirajganj 68 MW", "lat": 24.403976, "lon": 89.738849, "mw": 68},
    "teknaf": {"name": "Teknaf 20 MW", "lat": 20.981669, "lon": 92.256021, "mw": 20},
    "sirajganj6": {"name": "Sirajganj 6 MW", "lat": 24.386137, "lon": 89.748970, "mw": 6},
    "kaptai": {"name": "Kaptai 7.4 MW", "lat": 22.491471, "lon": 92.226588, "mw": 7.4},
    "sharishabari": {"name": "Sharishabari 3 MW", "lat": 24.772287, "lon": 89.842629, "mw": 3},
    "barishal": {"name": "Barishal 1 MW", "lat": 22.657015, "lon": 90.339194, "mw": 1},
    "lalmonirhat": {"name": "Lalmonirhat 30 MW", "lat": 25.997873, "lon": 89.154467, "mw": 30},
}


def parse_filename(name):
    """Parse site, buffer_km, year, month, period from filename."""
    match = re.match(r'(.+?)_(\d+)km_(\d{4})_(\d{2})_(pre|post)\.png', name)
    if match:
        return (match.group(1), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), match.group(5))
    return None, None, None, None, None


def load_grw_polygons():
    """Load confirmed GRW polygons."""
    with open(GRW_PATH) as f:
        return json.load(f)


def compute_solar_percentage(site_key, grw_data, tif_path):
    """Compute what % of the image area is covered by solar polygons.
    Uses the GeoTIFF bounds and polygon coordinates.
    Returns (solar_pct, polygon_pixel_coords_list) or (0, []).
    """
    if site_key not in grw_data:
        return 0.0, []

    polygons = grw_data[site_key].get('polygons', [])
    if not polygons:
        return 0.0, []

    if not tif_path.exists():
        return 0.0, []

    with rasterio.open(tif_path) as src:
        transform = src.transform
        height, width = src.shape
        bounds = src.bounds  # left, bottom, right, top

    # Compute image area in degrees^2
    img_area_deg2 = (bounds.right - bounds.left) * (bounds.top - bounds.bottom)

    total_poly_area_deg2 = 0.0
    pixel_coords_list = []

    for poly in polygons:
        geom = shp_shape(poly)
        if geom.is_empty:
            continue

        # Clip to image bounds
        img_box = Polygon([
            (bounds.left, bounds.bottom), (bounds.right, bounds.bottom),
            (bounds.right, bounds.top), (bounds.left, bounds.top),
        ])
        clipped = geom.intersection(img_box)
        if clipped.is_empty:
            continue

        total_poly_area_deg2 += clipped.area

        # Convert polygon coords to pixel coords for drawing
        coords = list(geom.exterior.coords)
        pixel_coords = []
        for lon, lat in coords:
            col = (lon - transform.c) / transform.a
            row = (lat - transform.f) / transform.e
            pixel_coords.append((int(col), int(row)))
        pixel_coords_list.append(pixel_coords)

    solar_pct = 100.0 * total_poly_area_deg2 / img_area_deg2 if img_area_deg2 > 0 else 0.0
    return solar_pct, pixel_coords_list


def draw_polygon_overlay(img, pixel_coords_list):
    """Draw red polygon outlines on image. Returns a copy with overlay."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    for coords in pixel_coords_list:
        if len(coords) >= 3:
            draw.polygon(coords, outline='red', fill=None)
            # Draw thicker outline
            draw.line(coords + [coords[0]], fill='red', width=3)
    return img_copy


def build_pre_prompt(site_key, year, month, buffer_km):
    """Prompt for pre-construction images (no solar)."""
    site = SITES.get(site_key, {})
    area_km = buffer_km * 2

    return f"""Analyze this satellite image from Bangladesh and estimate the percentage of each land cover class.

Site: {site.get('name', site_key)}
Location: {site.get('lat', 0):.4f}N, {site.get('lon', 0):.4f}E
Date: {year}/{month:02d} (PRE-construction, before any solar farm was built)
Area: approximately {area_km}x{area_km} km

Estimate the percentage of the total image area covered by each land cover class:

1. cropland - crop fields, rice paddies, fallow farmland (green/brown regular field patterns)
2. trees - trees, dense vegetation, forest patches (dark green irregular areas)
3. shrub - shrubs, scrubland, short woody vegetation
4. grassland - grass, pasture, open meadows (lighter green)
5. flooded_veg - wetlands, marshes, mangroves, flooded vegetation
6. built - buildings, roads, settlements, infrastructure (gray/white clusters)
7. bare - exposed soil, sand, cleared ground (light brown/tan)
8. water - rivers, ponds, canals, lakes (dark blue/black smooth areas)
9. snow - snow or ice cover (extremely unlikely in Bangladesh)

Context: Bangladesh is mostly flat agricultural land (rice paddies). Rivers and ponds are common. Settlements are scattered. This is a {area_km}x{area_km} km area.

There should be NO solar panels in this pre-construction image.

Return a JSON object with key "percentages" containing the percentage for each class.
All percentages should sum to approximately 100.
Example: {{"percentages": {{"cropland": 60.0, "trees": 15.0, "shrub": 2.0, "grassland": 5.0, "flooded_veg": 3.0, "built": 8.0, "bare": 5.0, "water": 2.0, "snow": 0.0}}}}"""


def build_post_prompt(site_key, year, month, buffer_km, solar_pct):
    """Prompt for post-construction images (with solar polygon overlay)."""
    site = SITES.get(site_key, {})
    area_km = buffer_km * 2
    remaining_pct = 100.0 - solar_pct

    return f"""Analyze this satellite image from Bangladesh. The image has a red polygon outline marking a solar farm.

Site: {site.get('name', site_key)}
Location: {site.get('lat', 0):.4f}N, {site.get('lon', 0):.4f}E
Date: {year}/{month:02d} (POST-construction, solar farm is operational)
Area: approximately {area_km}x{area_km} km
Solar farm: The red-outlined polygon covers approximately {solar_pct:.1f}% of the image.

For the REMAINING {remaining_pct:.1f}% of the image (everything OUTSIDE the red polygon), estimate the percentage of each land cover class:

1. cropland - crop fields, rice paddies, fallow farmland (green/brown regular field patterns)
2. trees - trees, dense vegetation, forest patches (dark green irregular areas)
3. shrub - shrubs, scrubland, short woody vegetation
4. grassland - grass, pasture, open meadows (lighter green)
5. flooded_veg - wetlands, marshes, mangroves, flooded vegetation
6. built - buildings, roads, settlements, infrastructure (gray/white clusters)
7. bare - exposed soil, sand, cleared ground (light brown/tan)
8. water - rivers, ponds, canals, lakes (dark blue/black smooth areas)
9. snow - snow or ice cover (extremely unlikely in Bangladesh)

IMPORTANT: Only classify the area OUTSIDE the red polygon. Ignore the solar farm area.
The percentages should represent the non-solar portion only and sum to approximately 100.

Context: Bangladesh is mostly flat agricultural land (rice paddies). Rivers and ponds are common.

Return a JSON object with key "percentages" containing the percentage for each class.
Example: {{"percentages": {{"cropland": 55.0, "trees": 18.0, "shrub": 2.0, "grassland": 5.0, "flooded_veg": 3.0, "built": 10.0, "bare": 5.0, "water": 2.0, "snow": 0.0}}}}"""


def classify_image(model, img, prompt):
    """Send image + prompt to Gemini, return percentages dict."""
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content([img, prompt])
            result = json.loads(response.text)

            pcts = result.get("percentages")
            if pcts is None:
                raise ValueError("No 'percentages' key in response")

            # Validate and normalize
            valid_classes = CLASS_NAMES[1:]  # skip no_data
            cleaned = {}
            total = 0
            for cn in valid_classes:
                val = float(pcts.get(cn, 0))
                val = max(0, min(100, val))
                cleaned[cn] = val
                total += val

            # Normalize to 100% if reasonably close
            if total > 0:
                factor = 100.0 / total
                for cn in cleaned:
                    cleaned[cn] = round(cleaned[cn] * factor, 1)

            cleaned['no_data'] = 0.0
            return cleaned

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 5
                print(f"  retry in {wait}s ({e})...", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise


def main():
    force = '--force' in sys.argv

    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_AI_API_KEY not set in .env")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )
    )

    grw_data = load_grw_polygons()
    print(f"Loaded GRW polygons for {len(grw_data)} sites")

    print("=" * 60)
    print("VLM V2 Classification (Gemini 2.0 Flash, 10-class)")
    print("=" * 60)

    # Get all images with pre/post pattern
    png_files = sorted(LABEL_DIR.glob('*_*km_*_*_*.png'))
    print(f"Found {len(png_files)} images")
    print(f"Rate limit delay: {DELAY_BETWEEN_CALLS}s between calls\n")

    success = 0
    skipped = 0
    failed = 0
    total_cost_est = 0.0

    for i, png_path in enumerate(png_files):
        site, buffer_km, year, month, period = parse_filename(png_path.name)
        if site is None or site not in SITES:
            print(f"  Skip: {png_path.name} (unknown site)")
            skipped += 1
            continue

        stem = png_path.stem
        json_path = VLM_V2_DIR / f'{stem}_vlm_v2.json'

        if json_path.exists() and not force:
            print(f"  {stem}: cached")
            skipped += 1
            continue

        img = Image.open(png_path).convert('RGB')
        w, h = img.size

        print(f"  [{i+1}/{len(png_files)}] {stem} ({w}x{h}px, {period})", end=" ", flush=True)

        try:
            tif_path = RAW_DIR / f'{stem}.tif'

            if period == 'post':
                solar_pct, pixel_coords = compute_solar_percentage(
                    site, grw_data, tif_path)
                if pixel_coords:
                    img_overlay = draw_polygon_overlay(img, pixel_coords)
                else:
                    img_overlay = img
                    solar_pct = 0.0
                    print(f"(no polygon overlay) ", end="", flush=True)

                prompt = build_post_prompt(site, year, month, buffer_km, solar_pct)
                pcts = classify_image(model, img_overlay, prompt)
                pcts['_solar_pct'] = round(solar_pct, 1)
            else:
                prompt = build_pre_prompt(site, year, month, buffer_km)
                pcts = classify_image(model, img, prompt)
                pcts['_solar_pct'] = 0.0

            # Save result
            result = {
                'site': site,
                'stem': stem,
                'buffer_km': buffer_km,
                'year': year,
                'month': month,
                'period': period,
                'solar_pct': pcts.pop('_solar_pct'),
                'percentages': pcts,
            }
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)

            # Print summary
            top = sorted(pcts.items(), key=lambda x: -x[1])[:4]
            summary = ", ".join(f"{k}:{v:.0f}%" for k, v in top if v > 0.5)
            solar_info = f" solar:{result['solar_pct']:.1f}%" if period == 'post' else ""
            print(f"OK ({summary}{solar_info})")
            success += 1

            # Rough cost estimate: ~$0.0001 per image for flash
            total_cost_est += 0.001

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        if i < len(png_files) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    print(f"\n{'=' * 60}")
    print(f"Done: {success} classified, {skipped} skipped, {failed} failed")
    print(f"Results: {VLM_V2_DIR}")
    print(f"Estimated cost: ~${total_cost_est:.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
