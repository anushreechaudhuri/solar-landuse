"""
Collect annual DW compositions and Sentinel-2 imagery for within-site event study.

For each treatment site (3,676) across 10 years (2016-2025), collects:
1. Dynamic World annual mode composition (9 classes + NDVI)
2. Sentinel-2 RGB composite thumbnail (for VLM classification)

Buffer is polygon-proportional: max(polygon_radius, 500m), capped at 5000m.
This ensures the polygon occupies a consistent fraction of each image.

Output:
  data/annual_panel.csv — one row per site × year (up to 36,760 rows)
  data/annual_cache/*.json — cached GEE results
  data/s2_images/{site_id}_{year}.png — S2 RGB thumbnails for VLM

Usage:
    python scripts/collect_annual_data.py --workers 8
    python scripts/collect_annual_data.py --workers 8 --skip-images  # DW only
    python scripts/collect_annual_data.py --workers 8 --images-only  # S2 images only
    python scripts/collect_annual_data.py --country india --workers 8
"""
import argparse
import csv
import json
import math
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from io import BytesIO

import ee
import numpy as np
import requests
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
UNIFIED_DB = DATA_DIR / "unified_solar_db.json"
OUTPUT_CSV = DATA_DIR / "annual_panel.csv"
CACHE_DIR = DATA_DIR / "annual_cache"
IMAGE_DIR = DATA_DIR / "s2_images"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2016, 2026))  # 2016-2025

DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation",
              "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]

# Rate limiting
_gee_lock = threading.Lock()
_gee_call_count = 0
_gee_start_time = time.time()


def gee_throttle():
    """Simple rate limiter: max ~10 calls/sec to avoid quota errors."""
    global _gee_call_count, _gee_start_time
    with _gee_lock:
        _gee_call_count += 1
        elapsed = time.time() - _gee_start_time
        if _gee_call_count > 10 and elapsed < 1.0:
            time.sleep(1.0 - elapsed)
            _gee_call_count = 0
            _gee_start_time = time.time()
        elif elapsed >= 1.0:
            _gee_call_count = 0
            _gee_start_time = time.time()


def compute_buffer(polygon_geojson):
    """Compute polygon-proportional buffer.

    Buffer = max(polygon_radius, 500m), capped at 5000m.
    This ensures the polygon occupies ~5-25% of the image area consistently.
    """
    if not polygon_geojson or not polygon_geojson.get("coordinates"):
        return 1000  # default 1km

    try:
        from shapely.geometry import shape
        s = shape(polygon_geojson)
        # Approximate area in m² (rough, using degree-to-km at ~25°N)
        deg_to_m = 111320 * math.cos(math.radians(25))
        area_m2 = s.area * deg_to_m ** 2
        radius_m = math.sqrt(area_m2 / math.pi)
        buffer_m = max(min(radius_m, 5000), 500)
        return buffer_m
    except Exception:
        return 1000


def make_geometry(lat, lon, polygon_geojson, buffer_m):
    """Create analysis geometry: polygon buffered, or circle."""
    if polygon_geojson and polygon_geojson.get("coordinates"):
        try:
            poly = ee.Geometry.Polygon(polygon_geojson["coordinates"])
            return poly.buffer(buffer_m)
        except Exception:
            pass
    return ee.Geometry.Point([lon, lat]).buffer(buffer_m)


def query_dw_annual(geometry, year):
    """Query DW annual mode composite within geometry."""
    gee_throttle()
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(geometry)
          .filterDate(start, end)
          .select("label"))

    count = dw.size().getInfo()
    if count == 0:
        return None

    mode_img = dw.reduce(ee.Reducer.mode()).select("label_mode")
    histogram = mode_img.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        scale=10,
        maxPixels=1e7,
    ).getInfo()

    hist = histogram.get("label_mode", {})
    if not hist:
        return None

    total = sum(hist.values())
    result = {f"dw_{cn}_pct": 100.0 * hist.get(str(i), 0) / total
              for i, cn in enumerate(DW_CLASSES)}

    # Also get NDVI
    try:
        gee_throttle()
        ndvi_col = (ee.ImageCollection("MODIS/061/MOD13Q1")
                    .filterDate(start, end)
                    .select("NDVI"))
        if ndvi_col.size().getInfo() > 0:
            ndvi_img = ndvi_col.median().multiply(0.0001)
            ndvi_stats = ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=250,
                maxPixels=1e7,
            ).getInfo()
            result["ndvi_mean"] = ndvi_stats.get("NDVI")
    except Exception:
        pass

    result["dw_n_scenes"] = count
    return result


def download_s2_thumbnail(lat, lon, polygon_geojson, buffer_m, year, site_id):
    """Download Sentinel-2 RGB composite thumbnail from GEE."""
    out_path = IMAGE_DIR / f"{site_id}_{year}.png"
    if out_path.exists():
        return str(out_path)

    gee_throttle()

    # Create bounding box from polygon + buffer
    geom = make_geometry(lat, lon, polygon_geojson, buffer_m)

    start = f"{year}-01-01"
    end = f"{year}-12-31"

    try:
        # Sentinel-2 surface reflectance, cloud-masked
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(geom)
              .filterDate(start, end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))

        count = s2.size().getInfo()
        if count == 0:
            # Fall back to less strict cloud filter
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(geom)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
            count = s2.size().getInfo()
            if count == 0:
                return None

        # Cloud masking using SCL band
        def mask_clouds(img):
            scl = img.select("SCL")
            mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            return img.updateMask(mask)

        composite = s2.map(mask_clouds).median().select(["B4", "B3", "B2"])

        # Visualization parameters
        vis = {"min": 0, "max": 3000, "bands": ["B4", "B3", "B2"]}

        # Get thumbnail URL (512px max dimension)
        bounds = geom.bounds().getInfo()["coordinates"][0]
        min_lon = min(c[0] for c in bounds)
        max_lon = max(c[0] for c in bounds)
        min_lat = min(c[1] for c in bounds)
        max_lat = max(c[1] for c in bounds)

        region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        url = composite.getThumbURL({
            "region": region.getInfo()["coordinates"],
            "dimensions": 512,
            "format": "png",
            **vis,
        })

        # Download
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return str(out_path)

    except Exception as e:
        pass

    return None


def load_sites(country_filter=None):
    """Load treatment sites from unified DB."""
    with open(UNIFIED_DB) as f:
        db = json.load(f)

    sites = []
    for e in db:
        if (e["treatment_group"] == "operational"
                and e["confidence"] in ("very_high", "high")
                and e.get("centroid_lat") and e.get("centroid_lon")):
            if country_filter and e["country"].lower() != country_filter.lower():
                continue
            polygon = None
            if e.get("grw", {}).get("polygon"):
                polygon = e["grw"]["polygon"]
            elif e.get("tzsam", {}).get("polygon"):
                polygon = e["tzsam"]["polygon"]
            sites.append({
                "site_id": e["site_id"],
                "country": e["country"],
                "lat": e["centroid_lat"],
                "lon": e["centroid_lon"],
                "capacity_mw": e.get("best_capacity_mw"),
                "construction_year": e.get("best_construction_year"),
                "confidence": e["confidence"],
                "polygon": polygon,
            })

    print(f"Loaded {len(sites)} treatment sites")
    if country_filter:
        print(f"  Filtered to: {country_filter}")
    return sites


def process_site_year(site, year, skip_images=False, images_only=False):
    """Process one site × year: query DW + optionally download S2 image."""
    site_id = site["site_id"]
    cache_path = CACHE_DIR / f"{site_id}_{year}.json"

    result = {
        "site_id": site_id,
        "country": site["country"],
        "year": year,
        "lat": site["lat"],
        "lon": site["lon"],
        "capacity_mw": site["capacity_mw"],
        "construction_year": site["construction_year"],
        "confidence": site["confidence"],
    }

    buffer_m = compute_buffer(site["polygon"])
    result["buffer_m"] = buffer_m

    # Event time relative to construction
    if site["construction_year"]:
        result["event_time"] = year - int(site["construction_year"])
    else:
        result["event_time"] = None

    # Check cache for DW data
    if not images_only:
        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            result.update(cached)
        else:
            try:
                geom = make_geometry(
                    site["lat"], site["lon"], site["polygon"], buffer_m)
                dw = query_dw_annual(geom, year)
                if dw:
                    result.update(dw)
                    # Cache
                    with open(cache_path, "w") as f:
                        json.dump(dw, f)
            except Exception as e:
                result["error"] = str(e)

    # Download S2 image
    if not skip_images:
        img_path = IMAGE_DIR / f"{site_id}_{year}.png"
        if not img_path.exists():
            try:
                download_s2_thumbnail(
                    site["lat"], site["lon"], site["polygon"],
                    buffer_m, year, site_id)
            except Exception:
                pass
        result["s2_image"] = str(img_path) if img_path.exists() else None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect annual DW + S2 imagery for event study")
    parser.add_argument("--country", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-images", action="store_true",
                        help="Only collect DW data, skip S2 image download")
    parser.add_argument("--images-only", action="store_true",
                        help="Only download S2 images, skip DW queries")
    parser.add_argument("--max-sites", type=int, default=None,
                        help="Limit number of sites (for testing)")
    args = parser.parse_args()

    # Initialize GEE
    try:
        ee.Initialize(project="bangladesh-solar")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="bangladesh-solar")

    sites = load_sites(args.country)
    if args.max_sites:
        sites = sites[:args.max_sites]

    n_years = len(YEARS)
    total_tasks = len(sites) * n_years
    print(f"\nTotal tasks: {len(sites)} sites × {n_years} years = {total_tasks:,}")
    print(f"Workers: {args.workers}")
    if args.skip_images:
        print("Mode: DW data only (skipping S2 images)")
    elif args.images_only:
        print("Mode: S2 images only (skipping DW queries)")

    # Check how many are already cached
    n_cached = sum(1 for s in sites for y in YEARS
                   if (CACHE_DIR / f"{s['site_id']}_{y}.json").exists())
    n_images = sum(1 for s in sites for y in YEARS
                   if (IMAGE_DIR / f"{s['site_id']}_{y}.png").exists())
    print(f"Already cached: {n_cached:,}/{total_tasks:,} DW results, "
          f"{n_images:,}/{total_tasks:,} S2 images")

    # Build task list
    tasks = []
    for site in sites:
        for year in YEARS:
            tasks.append((site, year))

    # Process in parallel
    all_results = []
    completed = 0
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_site_year, site, year,
                skip_images=args.skip_images,
                images_only=args.images_only
            ): (site["site_id"], year)
            for site, year in tasks
        }

        for future in as_completed(futures):
            site_id, year = futures[future]
            completed += 1
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  Error {site_id}/{year}: {e}")

            if completed % 500 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"  [{completed:,}/{total_tasks:,}] "
                      f"{rate:.1f}/sec, ETA {eta/60:.0f} min, "
                      f"{errors} errors")

    # Save to CSV
    if all_results and not args.images_only:
        # Determine all columns
        all_cols = set()
        for r in all_results:
            all_cols.update(r.keys())
        all_cols = sorted(all_cols)

        # Sort by site_id, year
        all_results.sort(key=lambda r: (r["site_id"], r.get("year", 0)))

        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols,
                                     extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        print(f"\nSaved {len(all_results):,} rows to {OUTPUT_CSV}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hrs)")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
