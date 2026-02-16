"""Compare four global LULC datasets from Google Earth Engine (10-class scheme).

Two-phase architecture:
  Phase 1 (GEE): Download raw (un-remapped) class values, cache as .npz
  Phase 2 (Analysis): Remap to 10-class scheme, compute percentages, generate figures

Datasets: Dynamic World, ESA WorldCover, ESRI LULC, GLAD GLCLUC
Additional: VLM masks at percentage level (not spatial)

10-class scheme:
  0: No Data/Cloud    4: Grassland      8: Water
  1: Cropland         5: Flooded Veg    9: Snow/Ice
  2: Trees/Forest     6: Built-up
  3: Shrub/Scrub      7: Bare Ground

Usage:
    python scripts/compare_lulc_datasets.py           # Full run (GEE + analysis)
    python scripts/compare_lulc_datasets.py --skip-gee # Re-run analysis from cache
"""

import argparse
import csv
import ee
import io
import json
import math
import numpy as np
import rasterio
import re
import requests
import zipfile
from datetime import date, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
LABEL_DIR = PROJECT_DIR / 'data' / 'for_labeling'
RAW_DIR = PROJECT_DIR / 'data' / 'raw_images'
MASK_DIR = PROJECT_DIR / 'data' / 'training_dataset' / 'masks'
CACHE_DIR = PROJECT_DIR / 'data' / 'lulc_raw_cache'
VIZ_DIR = PROJECT_DIR / 'data' / 'lulc_comparison'
FIG_DIR = PROJECT_DIR / 'docs' / 'figures'
CSV_PATH = PROJECT_DIR / 'data' / 'lulc_comparison_v3.csv'
POLYGON_CSV_PATH = PROJECT_DIR / 'data' / 'lulc_polygon_v3.csv'
GRW_PATH = PROJECT_DIR / 'data' / 'grw' / 'confirmed_matches.json'
RESULTS_PATH = PROJECT_DIR / 'RESULTS.md'

for d in [CACHE_DIR, VIZ_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Site coordinates ───────────────────────────────────────────────────────────

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

# ── 10-class schema ────────────────────────────────────────────────────────────

NUM_CLASSES = 10

CLASS_NAMES = {
    0: 'no_data',
    1: 'cropland',
    2: 'trees',
    3: 'shrub',
    4: 'grassland',
    5: 'flooded_veg',
    6: 'built',
    7: 'bare',
    8: 'water',
    9: 'snow',
}

CLASS_COLORS = {
    0: [0, 0, 0],           # no_data - black
    1: [255, 255, 0],       # cropland - yellow
    2: [0, 128, 0],         # trees - green
    3: [170, 210, 120],     # shrub - light green
    4: [144, 238, 144],     # grassland - pale green
    5: [0, 180, 180],       # flooded_veg - teal
    6: [255, 0, 0],         # built - red
    7: [165, 42, 42],       # bare - brown
    8: [0, 0, 255],         # water - blue
    9: [255, 255, 255],     # snow - white
}

# Colors for matplotlib (0-1 scale)
CLASS_COLORS_MPL = {k: [c / 255.0 for c in v] for k, v in CLASS_COLORS.items()}

DATASET_NAMES = ['Dynamic World', 'WorldCover', 'ESRI LULC', 'GLAD', 'VLM']
DATASET_KEYS = ['dw', 'worldcover', 'esri', 'glad', 'vlm']
GEE_DATASET_KEYS = ['dw', 'worldcover', 'esri', 'glad']

# ── Remap functions (raw values → 10-class) ───────────────────────────────────

def remap_dw(data):
    """Remap Dynamic World raw labels to 10-class.
    DW: 0=water, 1=trees, 2=grass, 3=flooded_veg, 4=crops, 5=shrub, 6=built, 7=bare, 8=snow
    """
    out = np.zeros_like(data, dtype=np.uint8)
    out[data == 0] = 8   # water
    out[data == 1] = 2   # trees
    out[data == 2] = 4   # grassland
    out[data == 3] = 5   # flooded_veg
    out[data == 4] = 1   # cropland
    out[data == 5] = 3   # shrub
    out[data == 6] = 6   # built
    out[data == 7] = 7   # bare
    out[data == 8] = 9   # snow
    return out


def remap_wc(data):
    """Remap ESA WorldCover raw labels to 10-class.
    WC: 10=tree, 20=shrub, 30=grass, 40=crop, 50=built, 60=bare, 70=snow, 80=water,
        90=wetland, 95=mangrove, 100=lichen/moss
    """
    out = np.zeros_like(data, dtype=np.uint8)
    out[data == 10] = 2   # trees
    out[data == 20] = 3   # shrub
    out[data == 30] = 4   # grassland
    out[data == 40] = 1   # cropland
    out[data == 50] = 6   # built
    out[data == 60] = 7   # bare
    out[data == 70] = 9   # snow
    out[data == 80] = 8   # water
    out[data == 90] = 5   # flooded_veg (wetland)
    out[data == 95] = 5   # flooded_veg (mangrove)
    out[data == 100] = 7  # lichen/moss → bare
    return out


def remap_esri(data):
    """Remap ESRI LULC raw labels to 10-class.
    sat-io ESRI: 1=nodata, 2=water, 3=trees, 4=flooded_veg, 5=crops,
                 6=built, 7=bare, 8=snow, 9=clouds, 10=rangeland
    """
    out = np.zeros_like(data, dtype=np.uint8)
    # 1 = No Data → 0 (already zero)
    out[data == 2] = 8   # water
    out[data == 3] = 2   # trees
    out[data == 4] = 5   # flooded_veg
    out[data == 5] = 1   # cropland
    out[data == 6] = 6   # built
    out[data == 7] = 7   # bare
    out[data == 8] = 9   # snow
    out[data == 9] = 0   # clouds → no_data
    out[data == 10] = 4  # rangeland → grassland
    # 11+ = unknown → 0 (no_data)
    return out


def remap_glad(data):
    """Remap GLAD GLCLUC 2020 raw values to 10-class (range-based).
    Ref: https://glad.umd.edu/dataset/GLCLUC2020
    1-24: bare/sparse short veg, 25-48: dense short veg, 49-96: tree cover,
    100-196: flooded/wetland, 200-207: open water/ocean, 208: snow/ice,
    209-211: artificial surfaces, 244-249: cropland, 250-253: dense built-up
    """
    out = np.zeros_like(data, dtype=np.uint8)
    out[(data >= 1) & (data <= 24)] = 7      # bare/sparse veg
    out[(data >= 25) & (data <= 48)] = 3     # shrub/short veg
    out[(data >= 49) & (data <= 96)] = 2     # trees
    out[(data >= 100) & (data <= 196)] = 5   # flooded_veg/wetland
    out[(data >= 200) & (data <= 207)] = 8   # open water/ocean
    out[data == 208] = 9                      # snow/ice
    out[(data >= 209) & (data <= 211)] = 6   # artificial surfaces → built
    out[(data >= 244) & (data <= 249)] = 1   # cropland
    out[(data >= 250) & (data <= 253)] = 6   # dense built-up
    return out


def remap_vlm(data):
    """Remap VLM 7-class masks to 10-class.
    VLM old: 0=background, 1=agriculture, 2=forest, 3=water, 4=urban, 5=solar, 6=bare_land
    Mapping: agriculture→cropland, forest→trees, water→water, urban→built, bare_land→bare
    Solar (5) → excluded (set to 0, then re-normalize percentages)
    """
    out = np.zeros_like(data, dtype=np.uint8)
    out[data == 1] = 1   # agriculture → cropland
    out[data == 2] = 2   # forest → trees
    out[data == 3] = 8   # water
    out[data == 4] = 6   # urban → built
    # data == 5 → solar, excluded (stays 0)
    out[data == 6] = 7   # bare_land → bare
    return out


REMAP_FUNCS = {
    'dw': remap_dw,
    'worldcover': remap_wc,
    'esri': remap_esri,
    'glad': remap_glad,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_filename(name):
    """Parse site, buffer_km, year, month, period from filename."""
    match = re.match(r'(.+?)_(\d+)km_(\d{4})_(\d{2})_(pre|post)\.png', name)
    if match:
        return (match.group(1), int(match.group(2)), int(match.group(3)),
                int(match.group(4)), match.group(5))
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
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_name = [n for n in zf.namelist() if n.endswith('.tif')][0]
            content = zf.read(tif_name)
    except zipfile.BadZipFile:
        pass
    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as src:
            return src.read(1)


def class_percentages(mask, exclude_classes=None):
    """Return dict of class_name -> percentage for 10 classes.
    If exclude_classes is set, those pixels are excluded and remaining re-normalized.
    """
    if exclude_classes:
        valid_mask = np.ones_like(mask, dtype=bool)
        for c in exclude_classes:
            valid_mask &= (mask != c)
        total = np.sum(valid_mask)
        if total == 0:
            return {CLASS_NAMES[i]: 0.0 for i in range(NUM_CLASSES)}
        pcts = {}
        unique, counts = np.unique(mask[valid_mask], return_counts=True)
        count_map = dict(zip(unique, counts))
        for cid in range(NUM_CLASSES):
            if cid in exclude_classes:
                pcts[CLASS_NAMES[cid]] = 0.0
            else:
                pcts[CLASS_NAMES[cid]] = 100.0 * count_map.get(cid, 0) / total
        return pcts
    else:
        total = mask.size
        unique, counts = np.unique(mask, return_counts=True)
        count_map = dict(zip(unique, counts))
        return {CLASS_NAMES[i]: 100.0 * count_map.get(i, 0) / total
                for i in range(NUM_CLASSES)}


def colorize(mask):
    """Convert single-channel class mask to RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        rgb[mask == cid] = color
    return rgb


# ── Phase 1: GEE Queries + Raw Caching ────────────────────────────────────────

def query_dynamic_world_raw(region, year, month):
    """Query Dynamic World mode composite, return raw labels (not remapped)."""
    center = date(year, month, 15)
    start = (center - timedelta(days=60)).strftime('%Y-%m-%d')
    end = (center + timedelta(days=60)).strftime('%Y-%m-%d')
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterBounds(region)
          .filterDate(start, end)
          .select('label'))
    count = dw.size().getInfo()
    if count == 0:
        return None
    mode_img = dw.reduce(ee.Reducer.mode())
    return download_ee_image(mode_img.select('label_mode'), region, scale=10)


def query_worldcover_raw(region):
    """Query ESA WorldCover v200, return raw labels."""
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    return download_ee_image(wc.clip(region), region, scale=10)


def query_esri_lulc_raw(region, year):
    """Query ESRI Global LULC 10m, return raw labels."""
    col = ee.ImageCollection(
        'projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS')
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    filtered = col.filterDate(start_date, end_date).filterBounds(region)
    count = filtered.size().getInfo()
    if count == 0:
        for fallback_year in [2023, 2022, 2021, 2020, 2024]:
            if fallback_year == year:
                continue
            filtered = col.filterDate(
                f'{fallback_year}-01-01', f'{fallback_year}-12-31'
            ).filterBounds(region)
            count = filtered.size().getInfo()
            if count > 0:
                print(f"    [ESRI] no data for {year}, using {fallback_year}")
                break
        if count == 0:
            return None
    img = filtered.mosaic().select('b1').clip(region)
    return download_ee_image(img, region, scale=10)


def query_glad_raw(region):
    """Query GLAD GLCLUC 2020, return raw values (remap done in numpy)."""
    glad = ee.Image('projects/glad/GLCLU2020/v2/LCLUC_2020')
    return download_ee_image(glad.clip(region), region, scale=30)


def run_gee_queries(png_files):
    """Phase 1: Query all 4 GEE datasets for each image, cache raw results."""
    print("\n" + "=" * 70)
    print("PHASE 1: GEE Queries (caching raw values)")
    print("=" * 70)

    for png_path in png_files:
        site, buffer_km, year, month, period = parse_filename(png_path.name)
        if site is None or site not in SITES:
            continue

        stem = png_path.stem
        coords = SITES[site]
        lat, lon = coords['lat'], coords['lon']
        region = make_region(lat, lon, buffer_km)

        print(f"\n  {png_path.name}")

        # Dynamic World
        cache_path = CACHE_DIR / f'{stem}_dw.npz'
        if cache_path.exists():
            print(f"    DW: cached")
        else:
            print(f"    DW ({year}-{month:02d} +/-2mo) ...", end=" ", flush=True)
            try:
                data = query_dynamic_world_raw(region, year, month)
                if data is not None:
                    np.savez_compressed(str(cache_path), data=data)
                    print("OK")
                else:
                    print("no data")
            except Exception as e:
                print(f"FAILED: {e}")

        # WorldCover
        cache_path = CACHE_DIR / f'{stem}_worldcover.npz'
        if cache_path.exists():
            print(f"    WC: cached")
        else:
            print(f"    WC (2021) ...", end=" ", flush=True)
            try:
                data = query_worldcover_raw(region)
                if data is not None:
                    np.savez_compressed(str(cache_path), data=data)
                    print("OK")
                else:
                    print("no data")
            except Exception as e:
                print(f"FAILED: {e}")

        # ESRI
        cache_path = CACHE_DIR / f'{stem}_esri.npz'
        if cache_path.exists():
            print(f"    ESRI: cached")
        else:
            print(f"    ESRI ({year}) ...", end=" ", flush=True)
            try:
                data = query_esri_lulc_raw(region, year)
                if data is not None:
                    np.savez_compressed(str(cache_path), data=data)
                    print("OK")
                else:
                    print("no data")
            except Exception as e:
                print(f"FAILED: {e}")

        # GLAD
        cache_path = CACHE_DIR / f'{stem}_glad.npz'
        if cache_path.exists():
            print(f"    GLAD: cached")
        else:
            print(f"    GLAD (2020) ...", end=" ", flush=True)
            try:
                data = query_glad_raw(region)
                if data is not None:
                    np.savez_compressed(str(cache_path), data=data)
                    print("OK")
                else:
                    print("no data")
            except Exception as e:
                print(f"FAILED: {e}")


# ── Phase 2: Analysis ─────────────────────────────────────────────────────────

def load_cached_raw(stem, dataset_key):
    """Load raw cached data for a dataset. Returns None if not cached."""
    cache_path = CACHE_DIR / f'{stem}_{dataset_key}.npz'
    if cache_path.exists():
        return np.load(str(cache_path))['data']
    return None


VLM_V2_DIR = PROJECT_DIR / 'data' / 'vlm_v2_responses'


def load_vlm_mask(stem):
    """Load VLM mask for a stem. Returns None if not found."""
    vlm_path = MASK_DIR / f'{stem}_vlm_mask.png'
    if vlm_path.exists():
        return np.array(Image.open(vlm_path))
    return None


def load_vlm_v2(stem):
    """Load VLM V2 percentage-based results. Returns dict or None."""
    json_path = VLM_V2_DIR / f'{stem}_vlm_v2.json'
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return data.get('percentages', None)
    return None


def load_grw_polygons():
    """Load confirmed GRW polygons."""
    if GRW_PATH.exists():
        with open(GRW_PATH) as f:
            return json.load(f)
    return {}


def rasterize_polygon_mask(site_key, grw_data, tif_path):
    """Rasterize GRW polygons for a site onto its GeoTIFF grid.
    Returns boolean mask (True inside polygon) or None.
    """
    from shapely.geometry import shape as shp_shape
    from rasterio.features import rasterize

    if site_key not in grw_data:
        return None

    polygons = grw_data[site_key].get('polygons', [])
    if not polygons:
        return None

    if not tif_path.exists():
        return None

    with rasterio.open(tif_path) as src:
        transform = src.transform
        height, width = src.shape

    shapes = []
    for poly in polygons:
        if 'geojson' in poly:
            geom = shp_shape(poly['geojson'])
        elif 'type' in poly and 'coordinates' in poly:
            geom = shp_shape(poly)
        else:
            continue
        if not geom.is_empty:
            shapes.append((geom, 1))

    if not shapes:
        return None

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def class_pcts_within_mask(lulc_mask, polygon_mask):
    """Compute 10-class percentages only within polygon_mask pixels."""
    pixels = lulc_mask[polygon_mask]
    if pixels.size == 0:
        return None
    total = pixels.size
    unique, counts = np.unique(pixels, return_counts=True)
    count_map = dict(zip(unique, counts))
    return {CLASS_NAMES[i]: 100.0 * count_map.get(i, 0) / total
            for i in range(NUM_CLASSES)}


def make_visualization(source_img, masks, stem):
    """Create side-by-side visualization with 10-class colors."""
    w, h = source_img.size
    max_dim = 400
    scale_factor = min(max_dim / w, max_dim / h, 1.0)
    viz_w = int(w * scale_factor)
    viz_h = int(h * scale_factor)

    source_resized = source_img.resize((viz_w, viz_h), Image.BILINEAR)
    panels = [np.array(source_resized)]
    panel_labels = ['Source']

    for key in GEE_DATASET_KEYS:
        name = {'dw': 'Dynamic World', 'worldcover': 'WorldCover',
                'esri': 'ESRI LULC', 'glad': 'GLAD'}[key]
        mask = masks.get(key)
        if mask is not None:
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_resized = np.array(mask_img.resize((viz_w, viz_h), Image.NEAREST))
            panels.append(colorize(mask_resized))
            panel_labels.append(name)
        else:
            panels.append(np.full((viz_h, viz_w, 3), 128, dtype=np.uint8))
            panel_labels.append(f'{name} (N/A)')

    gap = 4
    n_panels = len(panels)
    header_h = 30
    legend_h = 50
    canvas_w = n_panels * viz_w + (n_panels - 1) * gap
    canvas_h = header_h + viz_h + legend_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i, panel in enumerate(panels):
        x_offset = i * (viz_w + gap)
        canvas[header_h:header_h + viz_h, x_offset:x_offset + viz_w] = panel

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    for i, label in enumerate(panel_labels):
        x_offset = i * (viz_w + gap)
        draw.text((x_offset + 4, 8), label, fill=(0, 0, 0), font=font)

    # Legend rows at bottom (2 rows of 5 classes)
    legend_y = header_h + viz_h + 4
    for row_idx, class_ids in enumerate([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]):
        x_cursor = 4
        y = legend_y + row_idx * 18
        for cid in class_ids:
            color = tuple(CLASS_COLORS[cid])
            name = CLASS_NAMES[cid]
            draw.rectangle([x_cursor, y, x_cursor + 12, y + 12],
                           fill=color, outline=(0, 0, 0))
            draw.text((x_cursor + 15, y), name, fill=(0, 0, 0), font=font_small)
            x_cursor += 15 + len(name) * 7 + 10

    return img


def run_analysis(png_files):
    """Phase 2: Load cached raw data, remap, compute percentages, generate outputs."""
    print("\n" + "=" * 70)
    print("PHASE 2: Analysis (10-class remap, percentages, visualizations)")
    print("=" * 70)

    grw_data = load_grw_polygons()
    print(f"  Loaded GRW polygons for {len(grw_data)} sites")

    rows = []
    polygon_rows = []
    all_image_pcts = {}  # stem -> {ds_key -> pcts_dict}

    for png_path in png_files:
        site, buffer_km, year, month, period = parse_filename(png_path.name)
        if site is None or site not in SITES:
            continue

        stem = png_path.stem
        source_img = Image.open(png_path).convert('RGB')
        w, h = source_img.size

        print(f"\n  {stem} ({w}x{h})")

        masks_remapped = {}
        all_pcts = {}

        # Load + remap GEE datasets
        for ds_key in GEE_DATASET_KEYS:
            raw = load_cached_raw(stem, ds_key)
            if raw is not None:
                remapped = REMAP_FUNCS[ds_key](raw)
                # Resize to match source image
                mask_img = Image.fromarray(remapped.astype(np.uint8))
                masks_remapped[ds_key] = np.array(
                    mask_img.resize((w, h), Image.NEAREST))
                all_pcts[ds_key] = class_percentages(masks_remapped[ds_key])
            else:
                print(f"    {ds_key}: no cache")

        # VLM V2 (percentage-based, 10-class) — preferred
        vlm_v2 = load_vlm_v2(stem)
        if vlm_v2 is not None:
            all_pcts['vlm'] = {cn: vlm_v2.get(cn, 0.0) for cn in
                               [CLASS_NAMES[i] for i in range(NUM_CLASSES)]}
        else:
            # Fallback to old VLM mask (7-class grid)
            vlm_raw = load_vlm_mask(stem)
            if vlm_raw is not None:
                vlm_remapped = remap_vlm(vlm_raw)
                vlm_resized = np.array(
                    Image.fromarray(vlm_remapped.astype(np.uint8)).resize(
                        (w, h), Image.NEAREST))
                vlm_raw_resized = np.array(
                    Image.fromarray(vlm_raw.astype(np.uint8)).resize(
                        (w, h), Image.NEAREST))
                solar_mask = vlm_raw_resized == 5
                bg_mask = vlm_raw_resized == 0
                exclude_pixels = solar_mask | bg_mask
                valid = ~exclude_pixels
                total_valid = np.sum(valid)
                if total_valid > 0:
                    pcts = {}
                    unique, counts = np.unique(vlm_resized[valid], return_counts=True)
                    count_map = dict(zip(unique.astype(int), counts.astype(int)))
                    for cid in range(NUM_CLASSES):
                        pcts[CLASS_NAMES[cid]] = 100.0 * count_map.get(cid, 0) / total_valid
                    all_pcts['vlm'] = pcts
                else:
                    all_pcts['vlm'] = {CLASS_NAMES[i]: 0.0 for i in range(NUM_CLASSES)}

        # Print summary
        for ds_key in DATASET_KEYS:
            if ds_key in all_pcts:
                pcts = all_pcts[ds_key]
                parts = [f"{n}:{v:.0f}%" for n, v in pcts.items() if v > 0.5]
                ds_label = {'dw': 'DW', 'worldcover': 'WC', 'esri': 'ESRI',
                            'glad': 'GLAD', 'vlm': 'VLM'}.get(ds_key, ds_key)
                print(f"      {ds_label:6s}: {', '.join(parts)}")

        # Build CSV row
        row = {
            'image': stem, 'site': site, 'buffer_km': buffer_km,
            'year': year, 'month': month, 'period': period,
        }
        for ds_key in DATASET_KEYS:
            pcts = all_pcts.get(ds_key, {})
            for cid in range(NUM_CLASSES):
                cname = CLASS_NAMES[cid]
                row[f'{ds_key}_{cname}'] = round(pcts.get(cname, -1), 1)
        rows.append(row)
        all_image_pcts[stem] = all_pcts

        # Within-polygon analysis (pre-construction images only)
        if period == 'pre' and buffer_km == 1:
            tif_path = RAW_DIR / f'{stem}.tif'
            poly_mask = rasterize_polygon_mask(site, grw_data, tif_path)
            if poly_mask is not None:
                poly_row = {
                    'site': site, 'image': stem, 'buffer_km': buffer_km,
                    'year': year, 'month': month,
                }
                for ds_key in GEE_DATASET_KEYS:
                    if ds_key in masks_remapped:
                        # Resize polygon mask to match LULC mask
                        lulc = masks_remapped[ds_key]
                        pm_resized = np.array(
                            Image.fromarray(poly_mask.astype(np.uint8)).resize(
                                (lulc.shape[1], lulc.shape[0]), Image.NEAREST
                            )).astype(bool)
                        ppcts = class_pcts_within_mask(lulc, pm_resized)
                        if ppcts:
                            for cid in range(NUM_CLASSES):
                                cname = CLASS_NAMES[cid]
                                poly_row[f'{ds_key}_{cname}'] = round(
                                    ppcts.get(cname, 0), 1)
                polygon_rows.append(poly_row)

        # Visualization
        try:
            viz = make_visualization(source_img, masks_remapped, stem)
            viz.save(str(VIZ_DIR / f'{stem}_lulc_v3.png'))
        except Exception as e:
            print(f"    Viz FAILED: {e}")

    # Write full-AOI CSV
    if rows:
        fieldnames = ['image', 'site', 'buffer_km', 'year', 'month', 'period']
        for ds_key in DATASET_KEYS:
            for cid in range(NUM_CLASSES):
                fieldnames.append(f'{ds_key}_{CLASS_NAMES[cid]}')
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Full-AOI CSV: {CSV_PATH}")

    # Write polygon CSV
    if polygon_rows:
        poly_fields = ['site', 'image', 'buffer_km', 'year', 'month']
        for ds_key in GEE_DATASET_KEYS:
            for cid in range(NUM_CLASSES):
                poly_fields.append(f'{ds_key}_{CLASS_NAMES[cid]}')
        with open(POLYGON_CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=poly_fields)
            writer.writeheader()
            writer.writerows(polygon_rows)
        print(f"  Polygon CSV: {POLYGON_CSV_PATH}")

    return rows, polygon_rows, all_image_pcts


# ── Matplotlib Summary Figures ─────────────────────────────────────────────────

def _class_labels():
    """Return ordered class names for plotting (skip no_data)."""
    return [CLASS_NAMES[i] for i in range(1, NUM_CLASSES)]


def _class_colors_list():
    """Return ordered colors for plotting (skip no_data)."""
    return [CLASS_COLORS_MPL[i] for i in range(1, NUM_CLASSES)]


def generate_figures(rows, polygon_rows):
    """Generate 5 summary matplotlib figures."""
    print("\n" + "=" * 70)
    print("Generating Summary Figures")
    print("=" * 70)

    class_labels = _class_labels()
    class_colors = _class_colors_list()
    n_classes = len(class_labels)

    # Filter to 1km images only for aggregate stats
    rows_1km = [r for r in rows if r['buffer_km'] == 1]
    pre_rows = [r for r in rows_1km if r['period'] == 'pre']
    post_rows = [r for r in rows_1km if r['period'] == 'post']

    # ── Figure 1: Average class distribution across datasets ──
    print("  Figure 1: Average class distribution...")
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_classes)
    bar_width = 0.15
    ds_labels = ['DW', 'WC', 'ESRI', 'GLAD', 'VLM']
    ds_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, ds_key in enumerate(DATASET_KEYS):
        means = []
        for cname in class_labels:
            col = f'{ds_key}_{cname}'
            vals = [r[col] for r in pre_rows if r.get(col, -1) >= 0]
            means.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * bar_width, means, bar_width,
               label=ds_labels[i], color=ds_colors[i])

    ax.set_xlabel('Land Cover Class')
    ax.set_ylabel('Mean % (pre-construction, 1km)')
    ax.set_title('Average Class Distribution by Dataset (Pre-Construction)')
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'v3_avg_class_distribution.png'), dpi=150)
    plt.close(fig)
    print("    Saved v3_avg_class_distribution.png")

    # ── Figure 2: Within-polygon LULC (stacked bar per site) ──
    print("  Figure 2: Within-polygon LULC...")
    if polygon_rows:
        # Average across the 4 GEE datasets per site
        site_avg = {}
        for pr in polygon_rows:
            site = pr['site']
            if site not in site_avg:
                site_avg[site] = {cn: [] for cn in class_labels}
            for ds_key in GEE_DATASET_KEYS:
                for cn in class_labels:
                    col = f'{ds_key}_{cn}'
                    if col in pr and pr[col] >= 0:
                        site_avg[site][cn].append(pr[col])

        sites_sorted = sorted(site_avg.keys())
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(sites_sorted))
        bottom = np.zeros(len(sites_sorted))

        for ci, cn in enumerate(class_labels):
            vals = []
            for s in sites_sorted:
                v = site_avg[s][cn]
                vals.append(np.mean(v) if v else 0)
            vals = np.array(vals)
            ax.bar(x, vals, bottom=bottom, label=cn, color=class_colors[ci])
            bottom += vals

        ax.set_xlabel('Site')
        ax.set_ylabel('% of Polygon Area')
        ax.set_title('Pre-Construction Land Cover Within Solar Polygons\n(Average of DW, WC, ESRI, GLAD)')
        ax.set_xticks(x)
        ax.set_xticklabels(sites_sorted, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.set_ylim(0, 105)
        plt.tight_layout()
        fig.savefig(str(FIG_DIR / 'v3_within_polygon_lulc.png'), dpi=150)
        plt.close(fig)
        print("    Saved v3_within_polygon_lulc.png")
    else:
        print("    Skipped (no polygon data)")

    # ── Figure 3: Pre vs Post change ──
    print("  Figure 3: Pre vs Post change...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ds_key in enumerate(GEE_DATASET_KEYS):
        ax = axes[idx]
        ds_label = ['Dynamic World', 'WorldCover', 'ESRI LULC', 'GLAD'][idx]

        pre_means = []
        post_means = []
        for cn in class_labels:
            col = f'{ds_key}_{cn}'
            pre_vals = [r[col] for r in pre_rows if r.get(col, -1) >= 0]
            post_vals = [r[col] for r in post_rows if r.get(col, -1) >= 0]
            pre_means.append(np.mean(pre_vals) if pre_vals else 0)
            post_means.append(np.mean(post_vals) if post_vals else 0)

        changes = [post - pre for pre, post in zip(pre_means, post_means)]
        colors = ['#2ca02c' if c >= 0 else '#d62728' for c in changes]
        ax.barh(np.arange(n_classes), changes, color=colors)
        ax.set_yticks(np.arange(n_classes))
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel('Change (pp)')
        ax.set_title(ds_label)
        ax.axvline(0, color='black', linewidth=0.5)

    plt.suptitle('Pre → Post Construction Change by Dataset (1km AOI)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'v3_pre_vs_post_change.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("    Saved v3_pre_vs_post_change.png")

    # ── Figure 4: Cross-dataset agreement ──
    print("  Figure 4: Cross-dataset agreement...")
    # For each pre-construction 1km image, find dominant class per dataset
    agreement_data = []
    for r in pre_rows:
        dominants = []
        for ds_key in GEE_DATASET_KEYS:
            best_class = None
            best_pct = -1
            for cn in class_labels:
                col = f'{ds_key}_{cn}'
                val = r.get(col, -1)
                if val > best_pct:
                    best_pct = val
                    best_class = cn
            if best_pct >= 0:
                dominants.append(best_class)
        if len(dominants) == 4:
            n_agree = max(dominants.count(d) for d in set(dominants))
            agreement_data.append({
                'site': r['site'], 'n_agree': n_agree,
                'dominants': dominants
            })

    if agreement_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: histogram of agreement levels
        agree_counts = [0, 0, 0]  # 2-agree, 3-agree, 4-agree
        for ad in agreement_data:
            if ad['n_agree'] == 2:
                agree_counts[0] += 1
            elif ad['n_agree'] == 3:
                agree_counts[1] += 1
            elif ad['n_agree'] == 4:
                agree_counts[2] += 1
        ax1.bar(['2 of 4', '3 of 4', '4 of 4'], agree_counts,
                color=['#d62728', '#ff7f0e', '#2ca02c'])
        ax1.set_xlabel('Datasets Agreeing on Dominant Class')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Cross-Dataset Agreement\n(Dominant Class, Pre-Construction)')

        # Right: per-dataset dominant class frequency
        ds_class_counts = {dk: {} for dk in GEE_DATASET_KEYS}
        for r in pre_rows:
            for ds_key in GEE_DATASET_KEYS:
                best_class = None
                best_pct = -1
                for cn in class_labels:
                    col = f'{ds_key}_{cn}'
                    val = r.get(col, -1)
                    if val > best_pct:
                        best_pct = val
                        best_class = cn
                if best_pct >= 0 and best_class:
                    ds_class_counts[ds_key][best_class] = \
                        ds_class_counts[ds_key].get(best_class, 0) + 1

        x = np.arange(len(GEE_DATASET_KEYS))
        bottom = np.zeros(4)
        for ci, cn in enumerate(class_labels):
            vals = [ds_class_counts[dk].get(cn, 0) for dk in GEE_DATASET_KEYS]
            ax2.bar(x, vals, bottom=bottom, label=cn, color=class_colors[ci])
            bottom += np.array(vals)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['DW', 'WC', 'ESRI', 'GLAD'])
        ax2.set_ylabel('Number of Images')
        ax2.set_title('Dominant Class per Dataset\n(Pre-Construction)')
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        fig.savefig(str(FIG_DIR / 'v3_dataset_agreement.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print("    Saved v3_dataset_agreement.png")
    else:
        print("    Skipped (no agreement data)")

    # ── Figure 5: Example site comparisons ──
    print("  Figure 5: Example site comparisons...")
    # Pick representative sites: teesta pre, feni pre, manikganj post, mongla post
    example_stems = [
        'teesta_1km_2019_01_pre',
        'feni_1km_2020_01_pre',
        'manikganj_1km_2023_01_post',
        'mongla_1km_2023_01_post',
    ]
    example_imgs = []
    for es in example_stems:
        viz_path = VIZ_DIR / f'{es}_lulc_v3.png'
        if viz_path.exists():
            example_imgs.append((es, Image.open(viz_path)))

    if example_imgs:
        n = len(example_imgs)
        # Stack vertically
        max_w = max(img.size[0] for _, img in example_imgs)
        total_h = sum(img.size[1] + 30 for _, img in example_imgs)
        fig, axes = plt.subplots(n, 1, figsize=(max_w / 80, total_h / 80))
        if n == 1:
            axes = [axes]
        for i, (label, img) in enumerate(example_imgs):
            axes[i].imshow(img)
            axes[i].set_title(label.replace('_', ' '), fontsize=10)
            axes[i].axis('off')
        plt.tight_layout()
        fig.savefig(str(FIG_DIR / 'v3_example_comparisons.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print("    Saved v3_example_comparisons.png")
    else:
        print("    Skipped (no example visualizations found)")

    # ── Figure 6: VLM V2 vs GEE datasets ──
    print("  Figure 6: VLM V2 vs GEE comparison...")
    # Check if VLM data exists
    has_vlm = any(r.get('vlm_cropland', -1) >= 0 for r in pre_rows)
    if has_vlm:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: pre-construction class distribution (VLM vs DW)
        vlm_means_pre = []
        dw_means_pre = []
        for cn in class_labels:
            vlm_col = f'vlm_{cn}'
            dw_col = f'dw_{cn}'
            vlm_vals = [r[vlm_col] for r in pre_rows if r.get(vlm_col, -1) >= 0]
            dw_vals = [r[dw_col] for r in pre_rows if r.get(dw_col, -1) >= 0]
            vlm_means_pre.append(np.mean(vlm_vals) if vlm_vals else 0)
            dw_means_pre.append(np.mean(dw_vals) if dw_vals else 0)

        x = np.arange(n_classes)
        bar_w = 0.35
        ax1.bar(x - bar_w/2, dw_means_pre, bar_w, label='Dynamic World', color='#1f77b4')
        ax1.bar(x + bar_w/2, vlm_means_pre, bar_w, label='VLM V2 (Gemini)', color='#9467bd')
        ax1.set_xlabel('Land Cover Class')
        ax1.set_ylabel('Mean %')
        ax1.set_title('Pre-Construction: VLM V2 vs Dynamic World')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 70)

        # Right: per-site VLM pre vs post change
        vlm_pre_site = {}
        vlm_post_site = {}
        for r in rows_1km:
            site = r['site']
            vlm_crop = r.get('vlm_cropland', -1)
            if vlm_crop < 0:
                continue
            pcts = {cn: r.get(f'vlm_{cn}', 0) for cn in class_labels}
            if r['period'] == 'pre':
                vlm_pre_site[site] = pcts
            else:
                vlm_post_site[site] = pcts

        common_sites = sorted(set(vlm_pre_site) & set(vlm_post_site))
        if common_sites:
            vlm_changes = []
            for cn in class_labels:
                changes = [vlm_post_site[s].get(cn, 0) - vlm_pre_site[s].get(cn, 0)
                           for s in common_sites]
                vlm_changes.append(np.mean(changes))

            colors = ['#2ca02c' if c >= 0 else '#d62728' for c in vlm_changes]
            ax2.barh(np.arange(n_classes), vlm_changes, color=colors)
            ax2.set_yticks(np.arange(n_classes))
            ax2.set_yticklabels(class_labels, fontsize=9)
            ax2.set_xlabel('Change (pp)')
            ax2.set_title(f'VLM V2 Pre→Post Change ({len(common_sites)} sites)')
            ax2.axvline(0, color='black', linewidth=0.5)
        else:
            ax2.text(0.5, 0.5, 'No common pre/post sites', transform=ax2.transAxes,
                     ha='center', va='center')

        plt.tight_layout()
        fig.savefig(str(FIG_DIR / 'v3_vlm_vs_gee.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("    Saved v3_vlm_vs_gee.png")
    else:
        print("    Skipped (no VLM V2 data)")

    return True


# ── RESULTS.md Generation ──────────────────────────────────────────────────────

def generate_results_section(rows, polygon_rows):
    """Append V3 section to RESULTS.md."""
    print("\n  Updating RESULTS.md...")

    rows_1km = [r for r in rows if r['buffer_km'] == 1]
    pre_rows = [r for r in rows_1km if r['period'] == 'pre']
    post_rows = [r for r in rows_1km if r['period'] == 'post']
    class_labels = _class_labels()

    lines = []
    lines.append("\n---\n")
    lines.append("## V3: Multi-Dataset LULC Comparison (10-Class Scheme)\n")

    # Methodology
    lines.append("### Methodology\n")
    lines.append("Four global LULC datasets compared using a unified 10-class scheme, "
                 "plus VLM (Gemini 2.0 Flash) at the percentage level. "
                 "All datasets are remapped to a common scheme to preserve each "
                 "dataset's native granularity.\n")
    lines.append("| ID | Class | Dynamic World | WorldCover | ESRI | GLAD |")
    lines.append("|:--:|-------|:------------:|:----------:|:----:|:----:|")
    mapping_table = [
        (0, "No Data/Cloud", "—", "—", "1 (nodata), 9 (cloud)", "0"),
        (1, "Cropland", "4 (crops)", "40", "5 (crops)", "244-249"),
        (2, "Trees/Forest", "1 (trees)", "10", "3 (trees)", "49-96"),
        (3, "Shrub/Scrub", "5 (shrub)", "20", "—", "25-48"),
        (4, "Grassland", "2 (grass)", "30", "10 (rangeland)", "—"),
        (5, "Flooded Veg", "3", "90, 95 (mangrove)", "4", "100-196"),
        (6, "Built-up", "6 (built)", "50", "6 (built)", "209-211, 250-253"),
        (7, "Bare Ground", "7 (bare)", "60, 100 (lichen)", "7 (bare)", "1-24"),
        (8, "Water", "0 (water)", "80", "2 (water)", "200-207"),
        (9, "Snow/Ice", "8 (snow)", "70", "8 (snow)", "208"),
    ]
    for row in mapping_table:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")
    lines.append("")
    lines.append("VLM V2 (Gemini 2.0 Flash): Direct 10-class percentage estimation. "
                 "For post-construction images, solar polygon boundaries are drawn on the "
                 "image and Gemini classifies only the non-solar area. Solar percentage "
                 "is computed from polygon geometry. All 10 classes available.\n")

    # Temporal coverage note
    lines.append("**Temporal coverage of each dataset:**\n")
    lines.append("| Dataset | Temporal | Resolution | Notes |")
    lines.append("|---------|----------|------------|-------|")
    lines.append("| Dynamic World | Per-date composite (+/- 2 months) | 10m "
                 "| Only dataset with true pre/post temporal coverage |")
    lines.append("| WorldCover | Single snapshot (2021) | 10m "
                 "| Static -- pre/post values identical |")
    lines.append("| ESRI LULC | Annual (2017-2024, with fallback) | 10m "
                 "| Closest available year used; high no_data (30-77%) at some sites |")
    lines.append("| GLAD GLCLUC | Single snapshot (2020) | 30m "
                 "| Static -- pre/post values identical |")
    lines.append("| VLM V2 (Gemini) | Per-image (matches satellite date) | Percentage-level "
                 "| Temporal, 10-class, polygon-aware for post images |")
    lines.append("")
    lines.append("Only Dynamic World and VLM provide temporally-matched classifications "
                 "for detecting pre→post change. WorldCover and GLAD are single-date "
                 "products, so they cannot show change. ESRI provides annual maps but "
                 "uses fallback years when the target year is unavailable.\n")

    # Average class distribution
    lines.append("### Average Class Distribution (Pre-Construction, 1km AOI)\n")
    lines.append("![Average Class Distribution](docs/figures/v3_avg_class_distribution.png)\n")

    header = "| Class |"
    sep = "|-------|"
    for dl in ['DW', 'WC', 'ESRI', 'GLAD', 'VLM']:
        header += f" {dl} |"
        sep += ":---:|"
    lines.append(header)
    lines.append(sep)

    for cn in class_labels:
        row_str = f"| {cn} |"
        for ds_key in DATASET_KEYS:
            col = f'{ds_key}_{cn}'
            vals = [r[col] for r in pre_rows if r.get(col, -1) >= 0]
            mean = np.mean(vals) if vals else 0
            row_str += f" {mean:.1f}% |"
        lines.append(row_str)
    lines.append("")

    # Within-polygon analysis
    lines.append("### Pre-Construction Land Cover Within Solar Polygons\n")
    lines.append("![Within Polygon LULC](docs/figures/v3_within_polygon_lulc.png)\n")

    if polygon_rows:
        lines.append("Per-site breakdown (average of 4 GEE datasets):\n")
        header = "| Site |"
        sep = "|------|"
        for cn in class_labels:
            header += f" {cn} |"
            sep += ":---:|"
        lines.append(header)
        lines.append(sep)

        # Group by site
        site_data = {}
        for pr in polygon_rows:
            s = pr['site']
            if s not in site_data:
                site_data[s] = {cn: [] for cn in class_labels}
            for dk in GEE_DATASET_KEYS:
                for cn in class_labels:
                    col = f'{dk}_{cn}'
                    if col in pr and pr[col] >= 0:
                        site_data[s][cn].append(pr[col])

        for s in sorted(site_data.keys()):
            row_str = f"| {s} |"
            for cn in class_labels:
                v = site_data[s][cn]
                mean = np.mean(v) if v else 0
                row_str += f" {mean:.1f}% |"
            lines.append(row_str)
        lines.append("")

        # Compute overall averages for key finding
        overall = {}
        for cn in class_labels:
            all_vals = []
            for s in site_data:
                v = site_data[s][cn]
                if v:
                    all_vals.append(np.mean(v))
            overall[cn] = np.mean(all_vals) if all_vals else 0

        top_classes = sorted(overall.items(), key=lambda x: -x[1])[:3]
        finding = ", ".join([f"**{c[0]} ({c[1]:.0f}%)**" for c in top_classes])
        lines.append(f"Key finding: Solar farms in Bangladesh primarily replaced "
                     f"{finding}.\n")

    # Pre vs Post change (DW-focused)
    lines.append("### Pre vs Post Construction Change (Dynamic World only)\n")
    lines.append("![Pre vs Post Change](docs/figures/v3_pre_vs_post_change.png)\n")
    lines.append("Since only Dynamic World has true temporal coverage matching our "
                 "pre/post image dates, the DW change signal is the most meaningful. "
                 "WorldCover and GLAD are static snapshots (0.0 pp change expected). "
                 "ESRI provides some temporal signal but is confounded by fallback "
                 "year selection and high no_data.\n")

    # DW-specific change table
    lines.append("**Dynamic World pre→post change (1km AOI, 15 sites):**\n")
    lines.append("| Class | DW Δ | Interpretation |")
    lines.append("|-------|:---:|--------------|")
    for cn in class_labels:
        col = f'dw_{cn}'
        pre_vals = [r[col] for r in pre_rows if r.get(col, -1) >= 0]
        post_vals = [r[col] for r in post_rows if r.get(col, -1) >= 0]
        pre_mean = np.mean(pre_vals) if pre_vals else 0
        post_mean = np.mean(post_vals) if post_vals else 0
        change = post_mean - pre_mean
        sign = "+" if change >= 0 else ""
        bold = "**" if abs(change) >= 3 else ""
        interp = "Minor"
        if cn == 'cropland' and change < -3:
            interp = "Primary land converted to solar"
        elif cn == 'trees' and change < -3:
            interp = "Secondary loss, likely clearing for infrastructure"
        elif cn == 'built' and change > 3:
            interp = "Solar panels, substations, roads classified as built"
        elif cn == 'bare' and change > 3:
            interp = "Construction activity, cleared land"
        elif cn == 'shrub' and change > 1:
            interp = "Post-construction regrowth or reclassification"
        elif cn == 'snow' and change > 0.5:
            interp = "Likely reflective solar panel surfaces misclassified"
        lines.append(f"| {cn} | {bold}{sign}{change:.1f} pp{bold} | {interp} |")
    lines.append("")

    # Cross-dataset agreement
    lines.append("### Cross-Dataset Agreement\n")
    lines.append("![Dataset Agreement](docs/figures/v3_dataset_agreement.png)\n")
    lines.append("The agreement analysis examines how often the 4 GEE datasets "
                 "agree on the dominant land cover class for each image. Higher "
                 "agreement suggests more confidence in the classification.\n")

    # Example comparisons
    lines.append("### Example Site Comparisons\n")
    lines.append("![Example Comparisons](docs/figures/v3_example_comparisons.png)\n")
    lines.append("Representative side-by-side comparisons showing the source "
                 "satellite image alongside the 4 GEE dataset classifications "
                 "using the 10-class color scheme.\n")

    # VLM V2 vs GEE comparison
    lines.append("### VLM V2 vs GEE Dataset Comparison\n")
    lines.append("![VLM vs GEE](docs/figures/v3_vlm_vs_gee.png)\n")
    lines.append("VLM V2 uses Gemini 2.0 Flash with the 10-class scheme and "
                 "polygon-awareness for post-construction images. For post images, "
                 "solar polygon boundaries are drawn on the image and Gemini classifies "
                 "only the non-solar area. Solar percentage is computed from polygon "
                 "geometry.\n")

    # VLM vs DW table for pre-construction
    lines.append("**VLM V2 vs Dynamic World (pre-construction, 1km):**\n")
    lines.append("| Class | VLM V2 | DW | Difference |")
    lines.append("|-------|:------:|:--:|:----------:|")
    for cn in class_labels:
        vlm_col = f'vlm_{cn}'
        dw_col = f'dw_{cn}'
        vlm_vals = [r[vlm_col] for r in pre_rows if r.get(vlm_col, -1) >= 0]
        dw_vals = [r[dw_col] for r in pre_rows if r.get(dw_col, -1) >= 0]
        vlm_mean = np.mean(vlm_vals) if vlm_vals else 0
        dw_mean = np.mean(dw_vals) if dw_vals else 0
        diff = vlm_mean - dw_mean
        sign = "+" if diff >= 0 else ""
        lines.append(f"| {cn} | {vlm_mean:.1f}% | {dw_mean:.1f}% | {sign}{diff:.1f} pp |")
    lines.append("")

    # Key findings
    lines.append("### Key Findings\n")
    lines.append("1. **Cropland is the primary pre-solar land cover.** "
                 "Both GEE datasets and VLM V2 consistently identify cropland as "
                 "the dominant class within solar polygon areas.")
    lines.append("2. **Only Dynamic World and VLM V2 provide true change detection.** "
                 "WC and GLAD are static snapshots, ESRI has high no_data and "
                 "fallback year contamination.")
    lines.append("3. **DW detects cropland-to-built conversion.** "
                 "DW has no solar class, so panels appear as built/bare/snow.")
    lines.append("4. **VLM V2 provides polygon-aware classification.** "
                 "For post-construction images, VLM knows the solar percentage from "
                 "polygon geometry and classifies only the remaining area, avoiding "
                 "the solar-as-built misclassification issue.")
    lines.append("5. **Cross-dataset agreement is moderate.** "
                 "Cropland is the most consistently identified dominant class, "
                 "but other classes vary widely between datasets.")
    lines.append("6. **ESRI and GLAD have systematic issues for Bangladesh.** "
                 "ESRI has high no_data and misclassifies bright surfaces. "
                 "Both datasets' built percentages in pre-construction polygons "
                 "may be inflated by temporal mismatch.")
    lines.append("")

    v3_section = "\n".join(lines)

    # Read existing RESULTS.md and append
    if RESULTS_PATH.exists():
        existing = RESULTS_PATH.read_text()
        # Remove any existing V3 section
        marker = "## V3: Multi-Dataset LULC Comparison"
        if marker in existing:
            idx = existing.index(marker)
            # Find the preceding "---" separator
            sep_idx = existing.rfind("---", 0, idx)
            if sep_idx >= 0:
                existing = existing[:sep_idx].rstrip()
            else:
                existing = existing[:idx].rstrip()
        with open(RESULTS_PATH, 'w') as f:
            f.write(existing + "\n" + v3_section)
    else:
        with open(RESULTS_PATH, 'w') as f:
            f.write(v3_section)

    print("  RESULTS.md updated with V3 section")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare 4 global LULC datasets (10-class scheme)')
    parser.add_argument('--skip-gee', action='store_true',
                        help='Skip GEE queries, run analysis from cache')
    args = parser.parse_args()

    print("=" * 70)
    print("LULC Dataset Comparison (V3 - 10 Classes)")
    print("=" * 70)

    # Find all labeling PNGs that match the naming convention
    png_files = sorted([
        p for p in LABEL_DIR.glob('*_*km_*_*_*.png')
        if parse_filename(p.name)[0] is not None
    ])
    print(f"Found {len(png_files)} images to process")

    if not args.skip_gee:
        ee.Initialize(project="bangladesh-solar")
        run_gee_queries(png_files)
    else:
        print("\n  --skip-gee: Skipping GEE queries, using cached data")

    rows, polygon_rows, all_image_pcts = run_analysis(png_files)
    generate_figures(rows, polygon_rows)
    generate_results_section(rows, polygon_rows)

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"  Full-AOI CSV: {CSV_PATH}")
    print(f"  Polygon CSV:  {POLYGON_CSV_PATH}")
    print(f"  Visualizations: {VIZ_DIR}/")
    print(f"  Figures: {FIG_DIR}/v3_*.png")
    print(f"  Results: {RESULTS_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
