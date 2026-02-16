"""Generate Dynamic World baseline land cover masks from Google Earth Engine.

Queries GOOGLE/DYNAMICWORLD/V1 for each site's coordinates and time period,
remaps to our 7-class schema, exports as single-channel mask PNGs matching
image dimensions.

Usage:
    python scripts/generate_dynamic_world_masks.py
"""

import ee
import rasterio
import rasterio.io
import numpy as np
from PIL import Image
from pathlib import Path
import math
import re
import requests
import zipfile
import io

# Initialize Earth Engine
ee.Initialize(project="bangladesh-solar")

PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
LABEL_DIR = PROJECT_DIR / 'data' / 'for_labeling'
MASK_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'masks'
MASK_DIR.mkdir(parents=True, exist_ok=True)

# Site coordinates (from download_all_sites.py)
SITES = {
    "teesta": {"lat": 25.628342, "lon": 89.541082},
    "feni": {"lat": 22.787567, "lon": 91.367187},
    "manikganj": {"lat": 23.780834, "lon": 89.824775},
    "moulvibazar": {"lat": 24.493896, "lon": 91.633043},
    "pabna": {"lat": 23.961375, "lon": 89.159720},
    "mymensingh": {"lat": 24.702233, "lon": 90.461730},
    "tetulia": {"lat": 26.482817, "lon": 88.410139},
    "mongla": {"lat": 22.574239, "lon": 89.570388},
    "sirajganj68": {"lat": 24.403976, "lon": 89.738849},
    "teknaf": {"lat": 20.981669, "lon": 92.256021},
    "sirajganj6": {"lat": 24.386137, "lon": 89.748970},
    "kaptai": {"lat": 22.491471, "lon": 92.226588},
    "sharishabari": {"lat": 24.772287, "lon": 89.842629},
    "barishal": {"lat": 22.657015, "lon": 90.339194},
    "lalmonirhat": {"lat": 25.997873, "lon": 89.154467},
}

BUFFER_KM_DEFAULT = 1  # fallback; actual buffer read from filename

# Dynamic World class -> our class mapping
# DW: 0=water, 1=trees, 2=grass, 3=flooded_veg, 4=crops, 5=shrub, 6=built, 7=bare, 8=snow
# Ours: 0=background, 1=agriculture, 2=forest, 3=water, 4=urban, 5=solar_panels, 6=bare_land
DW_REMAP_FROM = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DW_REMAP_TO = [3, 2, 1, 1, 1, 2, 4, 6, 0]


def parse_filename(name):
    """Parse site, buffer_km, year, month, period from filename like 'manikganj_5km_2017_02_pre.png'"""
    match = re.match(r'(.+?)_(\d+)km_(\d{4})_(\d{2})_(pre|post)\.png', name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4)), match.group(5)
    return None, None, None, None, None


def make_region(lat, lon, buffer_km):
    """Create EE rectangle geometry from center point and buffer."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
    dlat = buffer_km / km_per_deg_lat
    dlon = buffer_km / km_per_deg_lon
    return ee.Geometry.Rectangle([
        lon - dlon, lat - dlat,
        lon + dlon, lat + dlat
    ])


def download_ee_image(image, region, scale=10):
    """Download a single-band EE image as a numpy array."""
    url = image.getDownloadURL({
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': region.getInfo()['coordinates'],
        'format': 'GEO_TIFF',
    })

    resp = requests.get(url)
    resp.raise_for_status()

    content = resp.content

    # EE may return a zip or raw GeoTIFF depending on version
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_name = [n for n in zf.namelist() if n.endswith('.tif')][0]
            content = zf.read(tif_name)
    except zipfile.BadZipFile:
        pass  # Already a raw GeoTIFF

    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as src:
            return src.read(1)


def get_dw_mask(lat, lon, year, month, target_h, target_w, buffer_km=BUFFER_KM_DEFAULT):
    """Query Dynamic World and return remapped mask at target resolution."""
    region = make_region(lat, lon, buffer_km)

    # Date range: target month +/- 2 months for composite
    from datetime import date, timedelta
    center = date(year, month, 15)
    start = (center - timedelta(days=60)).strftime('%Y-%m-%d')
    end = (center + timedelta(days=60)).strftime('%Y-%m-%d')

    # Query Dynamic World
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterBounds(region) \
        .filterDate(start, end) \
        .select('label')

    count = dw.size().getInfo()
    if count == 0:
        return None

    # Mode (most common class per pixel)
    mode_img = dw.reduce(ee.Reducer.mode())

    # Remap to our schema
    remapped = mode_img.remap(
        DW_REMAP_FROM, DW_REMAP_TO,
        defaultValue=0, bandName='label_mode'
    )

    # Download at 10m resolution
    data = download_ee_image(remapped, region, scale=10)

    # Resize to target dimensions using nearest-neighbor
    mask_img = Image.fromarray(data.astype(np.uint8))
    mask_resized = mask_img.resize((target_w, target_h), Image.NEAREST)

    return np.array(mask_resized)


def main():
    print("=" * 60)
    print("Generate Dynamic World Baseline Masks")
    print("=" * 60)

    # Find all PNG images (1km and 5km)
    png_files = sorted(LABEL_DIR.glob('*_*km_*_*.png'))
    print(f"Found {len(png_files)} images to process\n")

    success = 0
    skipped = 0
    failed = 0

    for png_path in png_files:
        site, buffer_km, year, month, period = parse_filename(png_path.name)
        if site is None:
            print(f"  Skip: can't parse {png_path.name}")
            skipped += 1
            continue

        if site not in SITES:
            print(f"  Skip: no coordinates for {site}")
            skipped += 1
            continue

        out_path = MASK_DIR / f"{png_path.stem}_dw_mask.png"
        if out_path.exists():
            print(f"  {png_path.name}: already exists")
            skipped += 1
            continue

        # Get image dimensions
        img = Image.open(png_path)
        w, h = img.size

        coords = SITES[site]
        print(f"  {png_path.name} ({w}x{h}px) ...", end=" ", flush=True)

        try:
            mask = get_dw_mask(coords["lat"], coords["lon"], year, month, h, w, buffer_km=buffer_km)
            if mask is not None:
                Image.fromarray(mask.astype(np.uint8)).save(str(out_path))
                unique, counts = np.unique(mask, return_counts=True)
                class_names = {0: 'bg', 1: 'agri', 2: 'forest', 3: 'water',
                               4: 'urban', 5: 'solar', 6: 'bare'}
                summary = ", ".join(
                    f"{class_names.get(v, '?')}:{c}" for v, c in zip(unique, counts)
                )
                print(f"OK ({summary})")
                success += 1
            else:
                print("no DW data")
                failed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success} generated, {skipped} skipped, {failed} failed")
    print(f"Masks saved to: {MASK_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
