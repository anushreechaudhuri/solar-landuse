"""
Collect multi-temporal panel data for all solar sites (treatment + control).

For each site at 4 time points (baseline, pre-construction, post-construction,
current), queries:
- Dynamic World: LULC class percentages within 1km
- VIIRS NTL: mean nighttime light radiance within 1km
- Sentinel-1: mean VV/VH backscatter within 1km
- MODIS NDVI/EVI: vegetation indices within 1km (250m, 16-day)
- MODIS LST: land surface temperature within 1km (1km, 8-day)
- WorldPop: population density within 1km (100m, annual, 2000-2020)
- Google Open Buildings Temporal: building metrics within 1km (2.5m, 2016-2023)
- Global Solar Atlas: GHI at centroid (static, queried once)

Output: data/temporal_panel.csv — one row per site × time point.
Cache: data/temporal_cache/*.json per site per time point per source.

Usage:
    python scripts/collect_temporal_data.py --country bangladesh
    python scripts/collect_temporal_data.py --country bangladesh --skip-gee
    python scripts/collect_temporal_data.py --workers 8  # Parallel (8 threads)
    python scripts/collect_temporal_data.py --only-new   # Only query new sources
    python scripts/collect_temporal_data.py  # Full South Asia (slow!)
"""
import argparse
import csv
import json
import math
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import ee
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
UNIFIED_DB = DATA_DIR / "unified_solar_db.json"
COMPARISON_SITES = DATA_DIR / "comparison_sites.json"
OUTPUT_CSV = DATA_DIR / "temporal_panel.csv"
CACHE_DIR = DATA_DIR / "temporal_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Time points
BASELINE_YEAR = 2016
CURRENT_YEAR = 2025

# DW class names (index = DW raw label)
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation",
              "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]


def make_circle(lat, lon, radius_km):
    """Create EE circle geometry."""
    return ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)


def make_geometry(lat, lon, polygon_geojson=None, buffer_m=100):
    """Create analysis geometry: use polygon (with buffer) if available, else 1km circle.

    Args:
        lat, lon: Site centroid
        polygon_geojson: GeoJSON polygon dict (type + coordinates), or None
        buffer_m: Buffer around polygon in meters (default 100m for edge effects)
    """
    if polygon_geojson and polygon_geojson.get("coordinates"):
        try:
            poly = ee.Geometry.Polygon(polygon_geojson["coordinates"])
            if buffer_m > 0:
                return poly.buffer(buffer_m)
            return poly
        except Exception:
            pass
    return ee.Geometry.Point([lon, lat]).buffer(1000)


# ── GEE query functions ─────────────────────────────────────────────────────

def query_dw(lat, lon, year, radius_km=1, geometry=None):
    """Query Dynamic World mode composite, return class percentages."""
    circle = geometry or make_circle(lat, lon, radius_km)
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(circle)
          .filterDate(start, end)
          .select("label"))

    count = dw.size().getInfo()
    if count == 0:
        return None

    mode_img = dw.reduce(ee.Reducer.mode()).select("label_mode")
    histogram = mode_img.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=circle,
        scale=10,
        maxPixels=1e7,
    ).getInfo()

    hist = histogram.get("label_mode", {})
    if not hist:
        return None

    total = sum(hist.values())
    return {cn: 100.0 * hist.get(str(i), 0) / total
            for i, cn in enumerate(DW_CLASSES)}


def query_viirs(lat, lon, year, radius_km=1, geometry=None):
    """Query VIIRS nighttime lights mean radiance within 1km.
    Uses annual median of monthly composites, filtering for quality."""
    circle = geometry or make_circle(lat, lon, radius_km)
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterDate(start, end)
             .select(["avg_rad", "cf_cvg"]))

    count = viirs.size().getInfo()
    if count == 0:
        return None

    # Filter for quality: at least 3 cloud-free observations
    def mask_quality(img):
        return img.updateMask(img.select("cf_cvg").gte(3))

    viirs_clean = viirs.map(mask_quality)

    # Take median across the year
    median_img = viirs_clean.select("avg_rad").median()

    stats = median_img.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=circle,
        scale=463,
        maxPixels=1e7,
    ).getInfo()

    return {
        "avg_rad_mean": stats.get("avg_rad_mean"),
        "avg_rad_stddev": stats.get("avg_rad_stdDev"),
    }


def query_sar(lat, lon, year, radius_km=1, geometry=None):
    """Query Sentinel-1 mean VV/VH backscatter within 1km.
    Uses 6-month composite centered on middle of year, IW mode, descending."""
    circle = geometry or make_circle(lat, lon, radius_km)
    # Use 6-month window centered on July
    start = f"{year}-04-01"
    end = f"{year}-10-01"

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(circle)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))

    count = s1.size().getInfo()
    if count == 0:
        return None

    median_img = s1.median()

    stats = median_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=circle,
        scale=10,
        maxPixels=1e7,
    ).getInfo()

    return {
        "vv_mean_db": stats.get("VV"),
        "vh_mean_db": stats.get("VH"),
        "n_scenes": count,
    }


def query_solar_atlas(lat, lon):
    """Query Global Solar Atlas GHI at point (static, only need once)."""
    point = ee.Geometry.Point([lon, lat])
    ghi = ee.Image("projects/sat-io/open-datasets/global_solar_atlas/ghi_LTAy_AvgDailyTotals")
    val = ghi.sample(point, scale=250).first().getInfo()
    if val and val.get("properties"):
        return {"ghi_kwh_m2_day": val["properties"].get("b1")}
    return None


def query_modis_ndvi(lat, lon, year, radius_km=1, geometry=None):
    """Query MODIS MOD13Q1 NDVI and EVI within 1km (250m, 16-day).
    Returns annual mean NDVI and EVI (scaled to 0-1)."""
    circle = geometry or make_circle(lat, lon, radius_km)
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(start, end)
           .select(["NDVI", "EVI", "SummaryQA"]))

    count = col.size().getInfo()
    if count == 0:
        return None

    # Mask poor quality (SummaryQA: 0=good, 1=marginal, 2/3=bad)
    def mask_qa(img):
        qa = img.select("SummaryQA")
        return img.updateMask(qa.lte(1))

    clean = col.map(mask_qa)

    # Scale: NDVI and EVI have scale factor 0.0001
    mean_img = clean.select(["NDVI", "EVI"]).mean().multiply(0.0001)

    stats = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=circle,
        scale=250,
        maxPixels=1e7,
    ).getInfo()

    return {
        "ndvi_mean": stats.get("NDVI"),
        "evi_mean": stats.get("EVI"),
    }


def query_modis_lst(lat, lon, year, radius_km=1, geometry=None):
    """Query MODIS MOD11A2 land surface temperature within 1km (1km, 8-day).
    Returns annual mean day/night LST in Celsius."""
    circle = geometry or make_circle(lat, lon, radius_km)
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    col = (ee.ImageCollection("MODIS/061/MOD11A2")
           .filterDate(start, end)
           .select(["LST_Day_1km", "LST_Night_1km", "QC_Day"]))

    count = col.size().getInfo()
    if count == 0:
        return None

    # Scale factor 0.02, convert K to C
    mean_img = col.select(["LST_Day_1km", "LST_Night_1km"]).mean()
    mean_celsius = mean_img.multiply(0.02).subtract(273.15)

    stats = mean_celsius.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=circle,
        scale=1000,
        maxPixels=1e7,
    ).getInfo()

    return {
        "lst_day_c": stats.get("LST_Day_1km"),
        "lst_night_c": stats.get("LST_Night_1km"),
    }


def query_worldpop(lat, lon, year, radius_km=1, geometry=None):
    """Query WorldPop population density within 1km (100m, annual, 2000-2020).
    Returns sum and mean population within buffer."""
    circle = geometry or make_circle(lat, lon, radius_km)

    # WorldPop only available 2000-2020; clamp to available range
    query_year = min(year, 2020)

    img = (ee.ImageCollection("WorldPop/GP/100m/pop")
           .filterDate(f"{query_year}-01-01", f"{query_year}-12-31")
           .mosaic())

    stats = img.reduceRegion(
        reducer=ee.Reducer.sum().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=circle,
        scale=100,
        maxPixels=1e7,
    ).getInfo()

    return {
        "pop_sum": stats.get("population_sum"),
        "pop_mean": stats.get("population_mean"),
        "pop_year_actual": query_year,
    }


def query_open_buildings(lat, lon, year, radius_km=1, geometry=None):
    """Query Google Open Buildings 2.5D Temporal within 1km (2.5m, 2016-2023).
    Returns building presence fraction, mean height, and fractional count."""
    circle = geometry or make_circle(lat, lon, radius_km)

    # Available 2016-2023; clamp to range
    query_year = max(min(year, 2023), 2016)

    try:
        col = (ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
               .filterDate(f"{query_year}-01-01", f"{query_year}-12-31")
               .filterBounds(circle))

        count = col.size().getInfo()
        if count == 0:
            return None

        img = col.mosaic()

        # building_presence: probability 0-1
        # building_height: meters
        # building_fractional_count: count per pixel
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=circle,
            scale=4,
            maxPixels=5e7,
        ).getInfo()

        return {
            "bldg_presence": stats.get("building_presence"),
            "bldg_height_m": stats.get("building_height"),
            "bldg_frac_count": stats.get("building_fractional_count"),
            "bldg_year_actual": query_year,
        }
    except Exception:
        return None


# ── Time point computation ───────────────────────────────────────────────────

def compute_time_points(construction_year):
    """Compute 4 time points for a site based on construction year.
    Returns dict: time_point_name -> year.
    """
    points = {"baseline": BASELINE_YEAR, "current": CURRENT_YEAR}

    if construction_year and construction_year > BASELINE_YEAR:
        # Pre: 1 year before construction (or baseline if too early)
        pre_year = max(construction_year - 1, BASELINE_YEAR + 1)
        points["pre_construction"] = pre_year

        # Post: 1 year after construction (or current if too recent)
        post_year = min(construction_year + 1, CURRENT_YEAR)
        points["post_construction"] = post_year
    else:
        # No construction year: use 2019 and 2022 as reasonable defaults
        points["pre_construction"] = 2019
        points["post_construction"] = 2022

    return points


# ── Main collection loop ────────────────────────────────────────────────────

def load_sites(country_filter=None):
    """Load treatment + control sites from unified DB and comparison sites."""
    with open(UNIFIED_DB) as f:
        db = json.load(f)

    sites = []

    # Treatment: only high/very_high confidence operational
    for e in db:
        if (e["treatment_group"] == "operational"
                and e["confidence"] in ("very_high", "high")
                and e.get("centroid_lat") and e.get("centroid_lon")):
            if country_filter and e["country"].lower() != country_filter.lower():
                continue
            # Pick best polygon: prefer GRW (higher res), fall back to TZ-SAM
            polygon = None
            if e.get("grw", {}).get("polygon"):
                polygon = e["grw"]["polygon"]
            elif e.get("tzsam", {}).get("polygon"):
                polygon = e["tzsam"]["polygon"]
            sites.append({
                "site_id": e["site_id"],
                "country": e["country"],
                "group": "treatment",
                "lat": e["centroid_lat"],
                "lon": e["centroid_lon"],
                "capacity_mw": e.get("best_capacity_mw"),
                "construction_year": e.get("best_construction_year"),
                "confidence": e["confidence"],
                "project_name": e.get("gem", {}).get("project_name", ""),
                "polygon": polygon,
            })

    # Control: proposed sites from comparison_sites.json (if available)
    if COMPARISON_SITES.exists():
        with open(COMPARISON_SITES) as f:
            comp = json.load(f)
        for s in comp.get("comparison_sites", []):
            if country_filter and s["country"].lower() != country_filter.lower():
                continue
            sites.append({
                "site_id": s["site_id"],
                "country": s["country"],
                "group": "control",
                "lat": s["centroid_lat"],
                "lon": s["centroid_lon"],
                "capacity_mw": s.get("capacity_mw"),
                "construction_year": None,  # Not built
                "confidence": "proposed",
                "project_name": s.get("project_name", ""),
            })
    else:
        # Fallback: load proposed directly from unified DB
        for e in db:
            if (e["treatment_group"] == "proposed"
                    and e.get("centroid_lat") and e.get("centroid_lon")):
                if country_filter and e["country"].lower() != country_filter.lower():
                    continue
                sites.append({
                    "site_id": e["site_id"],
                    "country": e["country"],
                    "group": "control",
                    "lat": e["centroid_lat"],
                    "lon": e["centroid_lon"],
                    "capacity_mw": e.get("best_capacity_mw"),
                    "construction_year": None,
                    "confidence": "proposed",
                    "project_name": e.get("gem", {}).get("project_name", ""),
                })

    return sites


def collect_site_data(site, skip_gee=False, only_new=False, use_polygons=False):
    """Collect all temporal data for one site. Returns list of row dicts.
    If only_new=True, loads DW/VIIRS/SAR from cache, only queries new sources.
    If use_polygons=True, uses polygon geometry instead of 1km circle."""
    site_id = site["site_id"]
    lat, lon = site["lat"], site["lon"]
    construction_year = site.get("construction_year")
    polygon = site.get("polygon") if use_polygons else None

    time_points = compute_time_points(construction_year)

    # Query Solar Atlas once (static)
    sa_cache = CACHE_DIR / f"{site_id}_solar_atlas.json"
    if sa_cache.exists():
        with open(sa_cache) as f:
            solar_atlas = json.load(f)
    elif not skip_gee:
        try:
            solar_atlas = query_solar_atlas(lat, lon) or {}
        except Exception as e:
            solar_atlas = {"error": str(e)}
        with open(sa_cache, "w") as f:
            json.dump(solar_atlas, f)
    else:
        solar_atlas = {}

    # Create analysis geometry (polygon or circle)
    geom = None
    if polygon and not skip_gee:
        try:
            geom = make_geometry(lat, lon, polygon)
        except Exception:
            geom = None

    rows = []
    for tp_name, year in time_points.items():
        row = {
            "site_id": site_id,
            "country": site["country"],
            "group": site["group"],
            "confidence": site.get("confidence", ""),
            "capacity_mw": site.get("capacity_mw"),
            "construction_year": construction_year,
            "project_name": site.get("project_name", ""),
            "time_point": tp_name,
            "year": year,
            "lat": lat,
            "lon": lon,
            "ghi_kwh_m2_day": solar_atlas.get("ghi_kwh_m2_day"),
            "uses_polygon": polygon is not None and geom is not None,
        }

        if skip_gee:
            # Load from cache only
            for source in ["dw", "viirs", "sar", "ndvi", "lst", "worldpop",
                           "buildings"]:
                cache_path = CACHE_DIR / f"{site_id}_{tp_name}_{source}.json"
                if cache_path.exists():
                    with open(cache_path) as f:
                        data = json.load(f)
                    _merge_source_data(row, source, data)
            rows.append(row)
            continue

        # ── Original sources (skip if only_new, load from cache) ──
        if only_new:
            for source in ["dw", "viirs", "sar"]:
                cache_path = CACHE_DIR / f"{site_id}_{tp_name}_{source}.json"
                if cache_path.exists():
                    with open(cache_path) as f:
                        data = json.load(f)
                    _merge_source_data(row, source, data)
        else:
            # Dynamic World
            dw_cache = CACHE_DIR / f"{site_id}_{tp_name}_dw.json"
            if dw_cache.exists():
                with open(dw_cache) as f:
                    dw_data = json.load(f)
            else:
                try:
                    dw_data = query_dw(lat, lon, year, geometry=geom) or {}
                except Exception as e:
                    dw_data = {"error": str(e)}
                with open(dw_cache, "w") as f:
                    json.dump(dw_data, f)
            _merge_source_data(row, "dw", dw_data)

            # VIIRS
            viirs_cache = CACHE_DIR / f"{site_id}_{tp_name}_viirs.json"
            if viirs_cache.exists():
                with open(viirs_cache) as f:
                    viirs_data = json.load(f)
            else:
                try:
                    viirs_data = query_viirs(lat, lon, year, geometry=geom) or {}
                except Exception as e:
                    viirs_data = {"error": str(e)}
                with open(viirs_cache, "w") as f:
                    json.dump(viirs_data, f)
            _merge_source_data(row, "viirs", viirs_data)

            # Sentinel-1
            sar_cache = CACHE_DIR / f"{site_id}_{tp_name}_sar.json"
            if sar_cache.exists():
                with open(sar_cache) as f:
                    sar_data = json.load(f)
            else:
                try:
                    sar_data = query_sar(lat, lon, year, geometry=geom) or {}
                except Exception as e:
                    sar_data = {"error": str(e)}
                with open(sar_cache, "w") as f:
                    json.dump(sar_data, f)
            _merge_source_data(row, "sar", sar_data)

        # MODIS NDVI/EVI
        ndvi_cache = CACHE_DIR / f"{site_id}_{tp_name}_ndvi.json"
        if ndvi_cache.exists():
            with open(ndvi_cache) as f:
                ndvi_data = json.load(f)
        else:
            try:
                ndvi_data = query_modis_ndvi(lat, lon, year, geometry=geom) or {}
            except Exception as e:
                ndvi_data = {"error": str(e)}
            with open(ndvi_cache, "w") as f:
                json.dump(ndvi_data, f)
        _merge_source_data(row, "ndvi", ndvi_data)

        # MODIS LST
        lst_cache = CACHE_DIR / f"{site_id}_{tp_name}_lst.json"
        if lst_cache.exists():
            with open(lst_cache) as f:
                lst_data = json.load(f)
        else:
            try:
                lst_data = query_modis_lst(lat, lon, year, geometry=geom) or {}
            except Exception as e:
                lst_data = {"error": str(e)}
            with open(lst_cache, "w") as f:
                json.dump(lst_data, f)
        _merge_source_data(row, "lst", lst_data)

        # WorldPop population
        pop_cache = CACHE_DIR / f"{site_id}_{tp_name}_worldpop.json"
        if pop_cache.exists():
            with open(pop_cache) as f:
                pop_data = json.load(f)
        else:
            try:
                pop_data = query_worldpop(lat, lon, year, geometry=geom) or {}
            except Exception as e:
                pop_data = {"error": str(e)}
            with open(pop_cache, "w") as f:
                json.dump(pop_data, f)
        _merge_source_data(row, "worldpop", pop_data)

        # Google Open Buildings Temporal
        bldg_cache = CACHE_DIR / f"{site_id}_{tp_name}_buildings.json"
        if bldg_cache.exists():
            with open(bldg_cache) as f:
                bldg_data = json.load(f)
        else:
            try:
                bldg_data = query_open_buildings(lat, lon, year, geometry=geom) or {}
            except Exception as e:
                bldg_data = {"error": str(e)}
            with open(bldg_cache, "w") as f:
                json.dump(bldg_data, f)
        _merge_source_data(row, "buildings", bldg_data)

        rows.append(row)

    return rows


def _merge_source_data(row, source, data):
    """Merge source data dict into row with prefixed keys."""
    if data is None or "error" in data:
        return

    if source == "dw":
        for cn in DW_CLASSES:
            row[f"dw_{cn}_pct"] = data.get(cn)
    elif source == "viirs":
        row["viirs_avg_rad"] = data.get("avg_rad_mean")
        row["viirs_avg_rad_sd"] = data.get("avg_rad_stddev")
    elif source == "sar":
        row["sar_vv_db"] = data.get("vv_mean_db")
        row["sar_vh_db"] = data.get("vh_mean_db")
        row["sar_n_scenes"] = data.get("n_scenes")
    elif source == "ndvi":
        row["ndvi_mean"] = data.get("ndvi_mean")
        row["evi_mean"] = data.get("evi_mean")
    elif source == "lst":
        row["lst_day_c"] = data.get("lst_day_c")
        row["lst_night_c"] = data.get("lst_night_c")
    elif source == "worldpop":
        row["pop_sum"] = data.get("pop_sum")
        row["pop_mean"] = data.get("pop_mean")
        row["pop_year_actual"] = data.get("pop_year_actual")
    elif source == "buildings":
        row["bldg_presence"] = data.get("bldg_presence")
        row["bldg_height_m"] = data.get("bldg_height_m")
        row["bldg_frac_count"] = data.get("bldg_frac_count")
        row["bldg_year_actual"] = data.get("bldg_year_actual")


def _process_site(site, skip_gee, only_new=False, use_polygons=False):
    """Worker function: process one site and return (site_id, rows)."""
    try:
        rows = collect_site_data(site, skip_gee=skip_gee, only_new=only_new,
                                 use_polygons=use_polygons)
        return site["site_id"], rows, None
    except Exception as e:
        return site["site_id"], [], str(e)


def collect_all(country_filter=None, skip_gee=False, workers=1, only_new=False,
                use_polygons=False):
    sites = load_sites(country_filter)
    n_treat = sum(1 for s in sites if s['group'] == 'treatment')
    n_ctrl = sum(1 for s in sites if s['group'] == 'control')
    n_poly = sum(1 for s in sites if s.get('polygon'))
    print(f"Loaded {len(sites)} sites ({n_treat} treatment, {n_ctrl} control)")
    if use_polygons:
        print(f"Polygon mode: {n_poly}/{len(sites)} sites have polygon geometries")

    if not skip_gee:
        print("Initializing GEE...")
        ee.Initialize(project="bangladesh-solar")

    start_time = time.time()
    all_rows = []
    completed = 0
    errors = 0
    lock = threading.Lock()

    if only_new:
        print("Mode: only querying new sources (NDVI, LST, WorldPop, Buildings)")

    if workers > 1:
        print(f"Using {workers} parallel workers")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_site, site, skip_gee, only_new,
                                use_polygons): site
                for site in sites
            }

            for future in as_completed(futures):
                site = futures[future]
                site_id, rows, error = future.result()
                with lock:
                    completed += 1
                    if error:
                        errors += 1
                        print(f"  [{completed}/{len(sites)}] {site_id}: ERROR {error}")
                    else:
                        all_rows.extend(rows)
                        if completed % 50 == 0 or completed == len(sites):
                            elapsed = time.time() - start_time
                            rate = completed / elapsed * 60 if elapsed > 0 else 0
                            print(f"  [{completed}/{len(sites)}] "
                                  f"{rate:.1f} sites/min, "
                                  f"{elapsed/60:.1f}m elapsed, "
                                  f"{errors} errors")
    else:
        for i, site in enumerate(sites):
            name = site.get("project_name") or site["site_id"]
            n_tp = len(compute_time_points(site.get("construction_year")))

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                print(f"\n[{i+1}/{len(sites)}] "
                      f"({rate:.1f} sites/min, {elapsed/60:.1f}m elapsed)")

            print(f"  {name} ({site['group']}, {n_tp} time points)...",
                  end=" ", flush=True)

            rows = collect_site_data(site, skip_gee=skip_gee, only_new=only_new,
                                    use_polygons=use_polygons)
            all_rows.extend(rows)
            print(f"{len(rows)} rows")

    elapsed = time.time() - start_time
    print(f"\nCollection complete: {len(all_rows)} rows in {elapsed/60:.1f} minutes")
    if errors:
        print(f"  {errors} sites had errors")

    _write_panel(all_rows, sites)


def _write_panel(all_rows, sites):
    """Write collected rows to CSV with summary stats."""
    if not all_rows:
        print("No data collected!")
        return

    fieldnames = sorted(all_rows[0].keys())
    # Ensure key columns come first
    priority = ["site_id", "country", "group", "confidence", "project_name",
                "time_point", "year", "lat", "lon", "capacity_mw",
                "construction_year", "ghi_kwh_m2_day"]
    ordered = [f for f in priority if f in fieldnames]
    ordered.extend(f for f in fieldnames if f not in ordered)

    print(f"\nWriting {len(all_rows)} rows to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    size_kb = OUTPUT_CSV.stat().st_size / 1024
    print(f"Saved ({size_kb:.1f} KB)")

    # Summary
    from collections import Counter
    groups = Counter(r["group"] for r in all_rows)
    tps = Counter(r["time_point"] for r in all_rows)
    countries = Counter(r["country"] for r in all_rows
                        if r["time_point"] == "baseline")
    print(f"\nSummary:")
    print(f"  Rows: {len(all_rows)}")
    print(f"  Sites: {len(sites)}")
    print(f"  By group: {dict(groups)}")
    print(f"  By time point: {dict(tps)}")
    print(f"  By country: {dict(countries)}")

    # Data completeness
    sources = [
        ("DW", "dw_crops_pct"), ("VIIRS", "viirs_avg_rad"),
        ("SAR", "sar_vv_db"), ("NDVI", "ndvi_mean"),
        ("LST", "lst_day_c"), ("WorldPop", "pop_sum"),
        ("Buildings", "bldg_presence"),
    ]
    for name, col in sources:
        filled = sum(1 for r in all_rows if r.get(col) is not None)
        print(f"  {name} data: {filled}/{len(all_rows)} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Collect multi-temporal panel data for solar sites")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter to single country (e.g. 'bangladesh')")
    parser.add_argument("--skip-gee", action="store_true",
                        help="Re-analyze from cache without querying GEE")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default 1, try 8 for speed)")
    parser.add_argument("--only-new", action="store_true",
                        help="Only query new sources (NDVI, LST, WorldPop, Buildings); "
                             "load DW/VIIRS/SAR from cache")
    parser.add_argument("--use-polygons", action="store_true",
                        help="Use actual GRW/TZ-SAM polygon boundaries instead of "
                             "fixed 1km circles (reduces signal dilution)")
    args = parser.parse_args()

    collect_all(country_filter=args.country, skip_gee=args.skip_gee,
                workers=args.workers, only_new=args.only_new,
                use_polygons=args.use_polygons)


if __name__ == "__main__":
    main()
