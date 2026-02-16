"""Generate pixel-accurate solar masks from confirmed GRW polygon matches.

For each site with confirmed GRW polygons, rasterizes the polygon geometries
onto the corresponding GeoTIFF grid to produce a mask where solar pixels = 5
and all other pixels = 0.

Pre-construction images get all-zero masks (no solar before construction).
Post-construction images get rasterized GRW polygons as class 5.

Inputs:
    data/grw/confirmed_matches.json  — from the review app (or site_matches.json as fallback)
    data/raw_images/*.tif            — GeoTIFFs for affine transform + dimensions

Outputs:
    data/training_dataset/masks/{stem}_grw_mask.png — single-channel mask (0 or 5)

Usage:
    python scripts/generate_grw_masks.py
"""

import json
import re
import numpy as np
import rasterio
from rasterio.features import rasterize
from PIL import Image
from pathlib import Path
from shapely.geometry import shape

PROJECT_DIR = Path("/Users/anushreechaudhuri/Documents/Projects/solar-landuse")
GRW_DIR = PROJECT_DIR / "data" / "grw"
RAW_DIR = PROJECT_DIR / "data" / "raw_images"
MASK_DIR = PROJECT_DIR / "data" / "training_dataset" / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)

SOLAR_CLASS = 5

# Color map for visualizations
CLASS_COLORS = {
    0: [0, 0, 0],        # background
    5: [128, 0, 128],    # solar_panels - purple
}


def parse_filename(name):
    """Parse site, buffer_km, year, month, period from filename."""
    match = re.match(r'(.+?)_(\d+)km_(\d{4})_(\d{2})_(pre|post)', name)
    if match:
        return (match.group(1), int(match.group(2)),
                int(match.group(3)), int(match.group(4)), match.group(5))
    return None, None, None, None, None


def load_confirmed_matches():
    """Load confirmed GRW matches (prefer confirmed_matches.json, fall back to site_matches.json)."""
    confirmed_path = GRW_DIR / "confirmed_matches.json"
    matches_path = GRW_DIR / "site_matches.json"

    if confirmed_path.exists():
        print(f"Loading confirmed matches from {confirmed_path}")
        with open(confirmed_path) as f:
            return json.load(f), "confirmed"
    elif matches_path.exists():
        print(f"No confirmed_matches.json found, using all matches from {matches_path}")
        with open(matches_path) as f:
            raw = json.load(f)
        # Filter to only matched sites
        return {k: v for k, v in raw.items() if v.get("match_status") == "matched"}, "all"
    else:
        print("ERROR: No GRW match files found. Run query_grw.py first.")
        return {}, None


def rasterize_polygons(geojson_polygons, tif_path):
    """Rasterize GeoJSON polygons onto the GeoTIFF grid as class 5 (solar)."""
    with rasterio.open(tif_path) as src:
        transform = src.transform
        height, width = src.shape

    # Convert GeoJSON dicts to shapely, then to (geometry, value) pairs
    shapes = []
    for poly_geojson in geojson_polygons:
        geom = shape(poly_geojson)
        if not geom.is_empty:
            shapes.append((geom, SOLAR_CLASS))

    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def create_colored_mask(mask):
    """Create RGB visualization from GRW mask."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    return colored


def main():
    print("=" * 60)
    print("Generate GRW Solar Masks")
    print("=" * 60)

    matches, source = load_confirmed_matches()
    if not matches:
        return

    print(f"  {len(matches)} sites with GRW polygons ({source})\n")

    # Find all GeoTIFFs
    tif_files = sorted(RAW_DIR.glob("*_*km_*_*.tif"))
    print(f"Found {len(tif_files)} GeoTIFFs in {RAW_DIR}\n")

    success = 0
    skipped = 0

    for tif_path in tif_files:
        stem = tif_path.stem
        site, buffer_km, year, month, period = parse_filename(stem)
        if site is None:
            continue

        out_path = MASK_DIR / f"{stem}_grw_mask.png"
        colored_path = MASK_DIR / f"{stem}_grw_mask_colored.png"

        if site not in matches:
            # No GRW data for this site — skip (VLM fallback handled by merge)
            continue

        site_data = matches[site]
        polygons = site_data.get("polygons", [])

        if period == "pre":
            # Pre-construction: all-zero mask
            with rasterio.open(tif_path) as src:
                h, w = src.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            Image.fromarray(mask).save(str(out_path))
            colored = create_colored_mask(mask)
            Image.fromarray(colored).save(str(colored_path))
            print(f"  {stem}: pre-construction → all zeros ({h}x{w})")
            success += 1
            continue

        if not polygons:
            print(f"  {stem}: matched but no polygons, skipping")
            skipped += 1
            continue

        # Extract GeoJSON geometries from the match data
        # Handles both formats:
        #   - GRW query output: {"geojson": {...}, "wkt": "...", "area_m2": ...}
        #   - Hand-drawn export: {"type": "Polygon", "coordinates": [...]}
        poly_geojsons = []
        for p in polygons:
            if "geojson" in p:
                poly_geojsons.append(p["geojson"])
            elif "type" in p and "coordinates" in p:
                poly_geojsons.append(p)

        if not poly_geojsons:
            print(f"  {stem}: no geojson in polygon data, skipping")
            skipped += 1
            continue

        # Rasterize
        mask = rasterize_polygons(poly_geojsons, tif_path)
        solar_pixels = np.sum(mask == SOLAR_CLASS)
        total_pixels = mask.size
        pct = 100.0 * solar_pixels / total_pixels if total_pixels > 0 else 0

        Image.fromarray(mask).save(str(out_path))
        colored = create_colored_mask(mask)
        Image.fromarray(colored).save(str(colored_path))

        print(f"  {stem}: {solar_pixels} solar pixels ({pct:.1f}%)")
        success += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success} masks generated, {skipped} skipped")
    print(f"GRW masks: {MASK_DIR}/*_grw_mask.png")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
