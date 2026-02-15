"""VLM-based grid classification using Gemini API.

Sends each satellite image to Gemini vision with a structured prompt,
gets per-cell land cover classification on a 20x20 grid, converts to
full-resolution mask PNGs.

Usage:
    python scripts/vlm_classify.py

Requires GOOGLE_AI_API_KEY in .env (free tier).
"""

import google.generativeai as genai
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
import os
import re
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
LABEL_DIR = PROJECT_DIR / 'data' / 'for_labeling'
MASK_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'masks'
VLM_JSON_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'vlm_responses'
MASK_DIR.mkdir(parents=True, exist_ok=True)
VLM_JSON_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZE = 20
MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 4  # seconds, for 15 RPM rate limit

# Site metadata for prompt context
SITES = {
    "teesta": {"name": "Teesta (Gaibandha/Beximco) 200 MW", "lat": 25.629209, "lon": 89.544870, "mw": 200},
    "feni": {"name": "Feni/Sonagazi 75 MW", "lat": 22.787567, "lon": 91.367187, "mw": 75},
    "manikganj": {"name": "Manikganj (Spectra) 35 MW", "lat": 23.780834, "lon": 89.824775, "mw": 35},
    "moulvibazar": {"name": "Moulvibazar 10 MW", "lat": 24.493896, "lon": 91.633043, "mw": 10},
    "pabna": {"name": "Pabna 100 MW", "lat": 23.826372, "lon": 89.606831, "mw": 100},
    "mymensingh": {"name": "Mymensingh (HDFC) 50 MW", "lat": 24.702233, "lon": 90.461730, "mw": 50},
    "tetulia": {"name": "Tetulia/Panchagarh (Sympa) 8 MW", "lat": 26.482817, "lon": 88.410139, "mw": 8},
    "mongla": {"name": "Mongla 100 MW", "lat": 22.574239, "lon": 89.570388, "mw": 100},
    "sirajganj68": {"name": "Sirajganj 68 MW", "lat": 24.403976, "lon": 89.738849, "mw": 68},
    "teknaf": {"name": "Teknaf (Joules) 20 MW", "lat": 20.981669, "lon": 92.256021, "mw": 20},
    "sirajganj6": {"name": "Sirajganj 6 MW", "lat": 24.386137, "lon": 89.748970, "mw": 6},
    "kaptai": {"name": "Kaptai 7.4 MW", "lat": 22.491471, "lon": 92.226588, "mw": 7.4},
    "sharishabari": {"name": "Sharishabari 3 MW", "lat": 24.772287, "lon": 89.842629, "mw": 3},
    "barishal": {"name": "Barishal 1 MW", "lat": 22.657015, "lon": 90.339194, "mw": 1},
    "lalmonirhat": {"name": "Lalmonirhat Rangpur (Intraco) 30 MW", "lat": 25.912, "lon": 89.445, "mw": 30},
}


def parse_filename(name):
    """Parse site, buffer_km, year, month, period from filename like 'manikganj_5km_2017_02_pre.png'"""
    match = re.match(r'(.+?)_(\d+)km_(\d{4})_(\d{2})_(pre|post)\.png', name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4)), match.group(5)
    return None, None, None, None, None


def build_prompt(site_key, year, month, period, buffer_km=1):
    """Build the VLM classification prompt."""
    site = SITES.get(site_key, {})
    site_name = site.get("name", site_key)
    lat = site.get("lat", 0)
    lon = site.get("lon", 0)
    mw = site.get("mw", 0)
    area_km = buffer_km * 2  # diameter of the AOI

    period_hint = ""
    if period == "post":
        period_hint = (
            f"This is a POST-construction image. A {mw} MW solar farm should be "
            f"visible near the image center as regular dark geometric rectangular arrays. "
            f"Look carefully for solar panel patterns."
        )
    else:
        period_hint = (
            "This is a PRE-construction image. There should be NO solar panels "
            "(class 5) in this image. Do not assign class 5 to any cell."
        )

    return f"""Analyze this satellite image of a solar energy project site in Bangladesh.

Site: {site_name}
Coordinates: {lat:.4f}N, {lon:.4f}E
Period: {"POST" if period == "post" else "PRE"}-construction ({year}/{month:02d})
Image covers approximately {area_km}x{area_km} km area.

Mentally divide this image into a {GRID_SIZE}x{GRID_SIZE} grid ({GRID_SIZE} rows, {GRID_SIZE} columns).
Classify each grid cell into exactly one of these land cover classes:

0 = background (clouds, shadows, unidentifiable areas)
1 = agriculture (crop fields, rice paddies, fallow farmland - green or brown regular field patterns)
2 = forest (trees, dense vegetation, shrubs - dark green irregular patches)
3 = water (rivers, ponds, canals, lakes - dark blue/black smooth areas)
4 = urban (buildings, roads, settlements - gray/white clustered patterns)
5 = solar_panels (photovoltaic arrays - dark blue/black regular geometric rectangular patterns, distinct from water)
6 = bare_land (exposed soil, cleared ground, construction sites - light brown/tan areas)

{period_hint}

Context for Bangladesh: Most land is flat agricultural (rice paddies). Rivers and ponds are common.
Settlements are scattered clusters. Forest is less common in central regions.

Return a JSON object with key "grid" containing a {GRID_SIZE}x{GRID_SIZE} 2D array of integers (0-6).
grid[0] is the TOP row, grid[{GRID_SIZE - 1}] is the BOTTOM row.
grid[row][0] is the LEFT column, grid[row][{GRID_SIZE - 1}] is the RIGHT column."""


def grid_to_mask(grid, target_h, target_w):
    """Convert a GRID_SIZE x GRID_SIZE grid to a full-resolution mask."""
    grid_arr = np.array(grid, dtype=np.uint8)
    mask_img = Image.fromarray(grid_arr)
    mask_resized = mask_img.resize((target_w, target_h), Image.NEAREST)
    return np.array(mask_resized)


def fix_grid(grid):
    """Fix grid dimensions if Gemini returns slightly wrong sizes."""
    # Fix row count
    while len(grid) < GRID_SIZE:
        grid.append(grid[-1][:])  # duplicate last row
    grid = grid[:GRID_SIZE]

    # Fix column counts
    for i in range(len(grid)):
        row = grid[i]
        while len(row) < GRID_SIZE:
            row.append(row[-1])  # duplicate last col
        grid[i] = row[:GRID_SIZE]

    # Clamp values to valid range
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            val = grid[i][j]
            if not isinstance(val, int) or val < 0 or val > 6:
                grid[i][j] = 0  # default to background

    return grid


def classify_image(model, img_path, site_key, year, month, period, buffer_km=1):
    """Send image to Gemini and get grid classification."""
    img = Image.open(img_path)
    prompt = build_prompt(site_key, year, month, period, buffer_km=buffer_km)

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content([img, prompt])
            result = json.loads(response.text)

            grid = result.get("grid")
            if grid is None:
                raise ValueError("No 'grid' key in response")

            # Allow slight size mismatches and fix them
            if abs(len(grid) - GRID_SIZE) > 3:
                raise ValueError(f"Expected ~{GRID_SIZE} rows, got {len(grid)}")

            grid = fix_grid(grid)

            return grid

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 5
                print(f"retry in {wait}s ({e})...", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise

    return None


def main():
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_AI_API_KEY not set in .env")
        print("Get a free key at https://aistudio.google.com/apikey")
        exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )
    )

    print("=" * 60)
    print("VLM Grid Classification (Gemini)")
    print("=" * 60)

    png_files = sorted(LABEL_DIR.glob('*_*km_*_*.png'))
    print(f"Found {len(png_files)} images to process")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Rate limit delay: {DELAY_BETWEEN_CALLS}s between calls\n")

    success = 0
    skipped = 0
    failed = 0

    for i, png_path in enumerate(png_files):
        site, buffer_km, year, month, period = parse_filename(png_path.name)
        if site is None:
            print(f"  Skip: can't parse {png_path.name}")
            skipped += 1
            continue

        mask_path = MASK_DIR / f"{png_path.stem}_vlm_mask.png"
        json_path = VLM_JSON_DIR / f"{png_path.stem}_vlm.json"

        if mask_path.exists() and json_path.exists():
            print(f"  {png_path.name}: already exists")
            skipped += 1
            continue

        img = Image.open(png_path)
        w, h = img.size

        print(f"  [{i+1}/{len(png_files)}] {png_path.name} ({w}x{h}px) ...", end=" ", flush=True)

        try:
            grid = classify_image(model, png_path, site, year, month, period, buffer_km=buffer_km)

            # Save raw JSON response
            with open(json_path, 'w') as f:
                json.dump({"grid": grid, "site": site, "year": year,
                           "month": month, "period": period}, f, indent=2)

            # Convert to full-resolution mask
            mask = grid_to_mask(grid, h, w)
            Image.fromarray(mask).save(str(mask_path))

            # Summary
            unique, counts = np.unique(mask, return_counts=True)
            class_names = {0: 'bg', 1: 'agri', 2: 'forest', 3: 'water',
                           4: 'urban', 5: 'solar', 6: 'bare'}
            summary = ", ".join(
                f"{class_names.get(v, '?')}:{c}" for v, c in zip(unique, counts)
            )
            print(f"OK ({summary})")
            success += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        # Rate limit
        if i < len(png_files) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    print(f"\n{'=' * 60}")
    print(f"Done: {success} generated, {skipped} skipped, {failed} failed")
    print(f"Masks: {MASK_DIR}")
    print(f"JSON responses: {VLM_JSON_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
