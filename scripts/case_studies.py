"""Detailed annual case studies for 4 Bangladesh solar sites (2016-2026).

Collects annual data from 7 GEE sources, downloads Planet basemap images,
runs VLM classification, and generates publication-quality figures for each site.

Usage:
    python scripts/case_studies.py --collect          # Collect GEE data
    python scripts/case_studies.py --download-images  # Download Planet images
    python scripts/case_studies.py --classify          # Run VLM on images
    python scripts/case_studies.py --figures           # Generate all figures
    python scripts/case_studies.py --all              # Do everything
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import ee
import numpy as np
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (
    apply_style, save_fig, FULL_WIDTH, HALF_WIDTH, DPI,
    LULC_COLORS, CLASS_LABELS, CLASS_ORDER,
)

DATA_DIR = Path(__file__).parent.parent / "data"
UNIFIED_DB = DATA_DIR / "unified_solar_db.json"
CACHE_DIR = DATA_DIR / "case_study_cache"
IMG_DIR = DATA_DIR / "case_study_images"
VLM_DIR = DATA_DIR / "case_study_vlm"
FIG_DIR = Path(__file__).parent.parent / "docs" / "figures" / "case_studies"
for d in [CACHE_DIR, IMG_DIR, VLM_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2016, 2027))  # 2016-2026

# ── Site definitions ─────────────────────────────────────────────────────────

SITES = [
    {
        "name": "Teesta (Beximco) 200 MW",
        "short_name": "teesta",
        "lat": 25.629209,
        "lon": 89.544870,
        "capacity_mw": 200,
        "construction_year": 2023,
        "cod_date": "2023-01-08",
        "developer": "Beximco Power Co. Ltd.",
        "offtaker": "BPDB",
        "location": "Sundarganj, Gaibandha",
        "db_ids": ["BA_0098", "BA_0109", "BA_0108"],  # polygon entries
        "gem_id": "BA_0026",
        "issues": "Violent and illegal land acquisition, loss of farmer livelihoods",
    },
    {
        "name": "Feni (Sonagazi EGCB) 75 MW",
        "short_name": "feni",
        "lat": 22.787567,
        "lon": 91.367187,
        "capacity_mw": 75,
        "construction_year": 2024,
        "cod_date": "2024-04-01",
        "developer": "EGCB Ltd.",
        "offtaker": "EGCB",
        "funding": "World Bank",
        "location": "Sonagazi, Feni",
        "db_ids": ["BA_0088", "BA_0023"],
        "issues": "Illegal land acquisition on three-crop land, farmer protests",
    },
    {
        "name": "Manikganj (Spectra) 35 MW",
        "short_name": "manikganj",
        "lat": 23.780834,
        "lon": 89.824775,
        "capacity_mw": 35,
        "construction_year": 2021,
        "cod_date": "2021-03-12",
        "developer": "Spectra Engineers Ltd. & Shunfeng Investment Ltd.",
        "offtaker": "BPDB",
        "location": "Shibalaya Upazila, Manikganj",
        "db_ids": ["BA_0048"],
        "issues": "Illegal land acquisition on three-crop land, threats, low compensation, river erosion",
    },
    {
        "name": "Moulvibazar 10 MW",
        "short_name": "moulvibazar",
        "lat": 24.493896,
        "lon": 91.633043,
        "capacity_mw": 10,
        "construction_year": 2025,
        "cod_date": "2025-10-01",
        "developer": "Moulvibazar Solar Power Ltd.",
        "offtaker": None,
        "location": "Moulvibazar, Sylhet",
        "db_ids": ["BA_0055"],
        "issues": "Forced land acquisition in haor wetland, ecological impacts, farmer protests",
    },
]

# DW class names (index = DW raw label)
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation",
              "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]

# Map DW to our scheme
DW_TO_OURS = {
    "water": "water", "trees": "trees", "grass": "grassland",
    "flooded_vegetation": "flooded_veg", "crops": "cropland",
    "shrub_and_scrub": "shrub", "built": "built", "bare": "bare",
    "snow_and_ice": "snow",
}
OURS_TO_DW = {v: k for k, v in DW_TO_OURS.items()}


# ── GEE Query Functions ──────────────────────────────────────────────────────

def make_geometry(site, buffer_km=1):
    """Create analysis geometry from polygon or circle."""
    # Try to load polygon from unified DB
    polygon = get_polygon(site)
    if polygon:
        try:
            poly = ee.Geometry.Polygon(polygon["coordinates"])
            return poly.buffer(100)  # 100m buffer
        except Exception:
            pass
    return ee.Geometry.Point([site["lon"], site["lat"]]).buffer(buffer_km * 1000)


def get_polygon(site):
    """Get best polygon for a site from unified DB."""
    try:
        with open(UNIFIED_DB) as f:
            db = json.load(f)
        for db_id in site.get("db_ids", []):
            for entry in db:
                if entry["site_id"] == db_id:
                    if entry.get("grw", {}).get("polygon", {}).get("coordinates"):
                        return entry["grw"]["polygon"]
                    if entry.get("tzsam", {}).get("polygon", {}).get("coordinates"):
                        return entry["tzsam"]["polygon"]
    except Exception:
        pass
    return None


def query_dw(geometry, year):
    """Query Dynamic World mode composite."""
    start, end = f"{year}-01-01", f"{year}-12-31"
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(geometry).filterDate(start, end).select("label"))
    count = dw.size().getInfo()
    if count == 0:
        return None
    mode_img = dw.reduce(ee.Reducer.mode()).select("label_mode")
    hist = mode_img.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry, scale=10, maxPixels=1e7,
    ).getInfo().get("label_mode", {})
    if not hist:
        return None
    total = sum(hist.values())
    return {cn: round(100.0 * hist.get(str(i), 0) / total, 2)
            for i, cn in enumerate(DW_CLASSES)}


def query_viirs(geometry, year):
    """Query VIIRS nighttime lights."""
    start, end = f"{year}-01-01", f"{year}-12-31"
    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterDate(start, end).select(["avg_rad", "cf_cvg"]))
    count = viirs.size().getInfo()
    if count == 0:
        return None
    def mask_quality(img):
        return img.updateMask(img.select("cf_cvg").gte(3))
    median = viirs.map(mask_quality).select("avg_rad").median()
    stats = median.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry,
        scale=463, maxPixels=1e7,
    ).getInfo()
    return {"avg_rad_mean": stats.get("avg_rad_mean")}


def query_sar(geometry, year):
    """Query Sentinel-1 VV/VH."""
    start, end = f"{year}-04-01", f"{year}-10-01"
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))
    count = s1.size().getInfo()
    if count == 0:
        return None
    median = s1.median()
    stats = median.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry,
        scale=10, maxPixels=1e7,
    ).getInfo()
    return {
        "vv_mean_db": stats.get("VV"),
        "vh_mean_db": stats.get("VH"),
        "n_scenes": count,
    }


def query_ndvi(geometry, year):
    """Query MODIS NDVI/EVI."""
    start, end = f"{year}-01-01", f"{year}-12-31"
    modis = (ee.ImageCollection("MODIS/061/MOD13Q1")
             .filterDate(start, end).select(["NDVI", "EVI"]))
    count = modis.size().getInfo()
    if count == 0:
        return None
    mean_img = modis.mean()
    stats = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry,
        scale=250, maxPixels=1e7,
    ).getInfo()
    ndvi = stats.get("NDVI")
    evi = stats.get("EVI")
    return {
        "ndvi_mean": round(ndvi * 0.0001, 4) if ndvi else None,
        "evi_mean": round(evi * 0.0001, 4) if evi else None,
    }


def query_lst(geometry, year):
    """Query MODIS LST day/night."""
    start, end = f"{year}-01-01", f"{year}-12-31"
    modis = (ee.ImageCollection("MODIS/061/MOD11A2")
             .filterDate(start, end)
             .select(["LST_Day_1km", "LST_Night_1km"]))
    count = modis.size().getInfo()
    if count == 0:
        return None
    mean_img = modis.mean()
    stats = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geometry,
        scale=1000, maxPixels=1e7,
    ).getInfo()
    day = stats.get("LST_Day_1km")
    night = stats.get("LST_Night_1km")
    return {
        "lst_day_c": round(day * 0.02 - 273.15, 2) if day else None,
        "lst_night_c": round(night * 0.02 - 273.15, 2) if night else None,
    }


def query_worldpop(geometry, year):
    """Query WorldPop population density."""
    actual_year = min(year, 2020)
    wp = (ee.ImageCollection("WorldPop/GP/100m/pop")
          .filterDate(f"{actual_year}-01-01", f"{actual_year}-12-31")
          .filterBounds(geometry)
          .first())
    stats = wp.select("population").reduceRegion(
        reducer=ee.Reducer.sum().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=geometry, scale=100, maxPixels=1e7,
    ).getInfo()
    return {
        "pop_sum": stats.get("population_sum"),
        "pop_mean": stats.get("population_mean"),
        "pop_year_actual": actual_year,
    }


def query_buildings(geometry, year):
    """Query Google Open Buildings Temporal."""
    actual_year = min(max(year, 2016), 2023)
    try:
        coll = ee.ImageCollection(
            "GOOGLE/Research/open-buildings-temporal/v1"
        ).filterDate(f"{actual_year}-01-01", f"{actual_year}-12-31")
        count = coll.size().getInfo()
        if count == 0:
            return None
        img = coll.mosaic()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry,
            scale=4, maxPixels=1e8,
        ).getInfo()
        return {
            "bldg_presence": stats.get("building_presence"),
            "bldg_height_m": stats.get("building_height"),
            "bldg_frac_count": stats.get("building_fractional_count"),
            "bldg_year_actual": actual_year,
        }
    except Exception:
        return None


# ── Data Collection ──────────────────────────────────────────────────────────

SOURCES = {
    "dw": query_dw,
    "viirs": query_viirs,
    "sar": query_sar,
    "ndvi": query_ndvi,
    "lst": query_lst,
    "worldpop": query_worldpop,
    "buildings": query_buildings,
}


def collect_site_data(site):
    """Collect all annual data for a site."""
    name = site["short_name"]
    geom = make_geometry(site)
    results = {}

    for year in YEARS:
        year_data = {"year": year}
        for source_name, query_fn in SOURCES.items():
            cache_path = CACHE_DIR / f"{name}_{year}_{source_name}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    data = json.load(f)
            else:
                print(f"    {source_name} {year}...", end=" ", flush=True)
                try:
                    data = query_fn(geom, year)
                    with open(cache_path, "w") as f:
                        json.dump(data, f, indent=2)
                    print("OK", end="  ")
                except Exception as e:
                    print(f"ERR ({e})", end="  ")
                    data = None
                time.sleep(0.5)

            if data:
                year_data[source_name] = data
        results[year] = year_data
        if not cache_path.exists():
            print()

    return results


def collect_all(sites):
    """Collect data for all sites."""
    ee.Initialize(project="bangladesh-solar")
    all_data = {}
    for site in sites:
        print(f"\n{'='*60}")
        print(f"Collecting: {site['name']}")
        print(f"{'='*60}")
        all_data[site["short_name"]] = collect_site_data(site)
    return all_data


# ── Planet Image Download ────────────────────────────────────────────────────

def download_planet_images(sites):
    """Download Planet basemap images for all sites and years."""
    from download_planet_basemaps import (
        find_mosaic_id, get_quads, download_quad,
        mosaic_and_clip, make_bbox,
    )
    from PIL import Image
    import rasterio

    for site in sites:
        name = site["short_name"]
        print(f"\n--- {site['name']} ---")
        bbox = make_bbox(site["lat"], site["lon"], buffer_km=2)

        for year in YEARS:
            png_path = IMG_DIR / f"{name}_{year}.png"
            if png_path.exists():
                print(f"  {year}: exists")
                continue

            # Try months: Jan, Jun, then nearby
            mosaic_id = None
            for month in [1, 6, 3, 9]:
                mid, _ = find_mosaic_id(f"{year}_{month:02d}")
                if mid:
                    mosaic_id = mid
                    break
            if not mosaic_id:
                print(f"  {year}: no mosaic")
                continue

            quads = get_quads(mosaic_id, bbox)
            if not quads:
                print(f"  {year}: no quads")
                continue

            # Download quads
            quad_paths = []
            for qi, quad in enumerate(quads):
                qpath = IMG_DIR / f"{name}_{year}_q{qi}.tif"
                download_quad(quad, str(qpath))
                quad_paths.append(str(qpath))

            # Mosaic and clip
            tif_path = IMG_DIR / f"{name}_{year}.tif"
            mosaic_and_clip(quad_paths, bbox, str(tif_path))

            # Convert to PNG
            with rasterio.open(tif_path) as src:
                rgb = src.read([1, 2, 3]).astype(float)
                for b in range(3):
                    valid = rgb[b][rgb[b] > 0]
                    if len(valid) > 0:
                        p2 = np.percentile(valid, 2)
                        p98 = np.percentile(valid, 98)
                        if p98 > p2:
                            rgb[b] = ((rgb[b] - p2) / (p98 - p2) * 255).clip(0, 255)
                rgb = rgb.astype("uint8")
                img = Image.fromarray(rgb.transpose(1, 2, 0))
                img.save(png_path)

            # Cleanup
            for qp in quad_paths:
                Path(qp).unlink(missing_ok=True)
            tif_path.unlink(missing_ok=True)
            print(f"  {year}: downloaded")
            time.sleep(1)


# ── VLM Classification ───────────────────────────────────────────────────────

VLM_PROMPT = """You are analyzing a satellite image of a solar energy project site in Bangladesh.

The image covers approximately 4km x 4km centered on coordinates ({lat:.4f}, {lon:.4f}).
This is the {site_name} site, year {year}. Construction year: {construction_year}.

Estimate the percentage breakdown of land cover classes in this image:
- cropland, trees, shrub, grassland, flooded_veg (wetland/haor), built (buildings/roads/structures), bare (bare ground/sand), water, snow, solar (solar panels)

Also note: Is solar panel installation visible? How much of the image area?

Return JSON:
{{
  "land_cover": {{"cropland": %, "trees": %, "shrub": %, "grassland": %, "flooded_veg": %, "built": %, "bare": %, "water": %, "snow": %, "solar": %}},
  "solar_visible": "yes" or "no",
  "solar_area_pct": 0-100,
  "description": "brief description"
}}"""


def classify_images(sites):
    """Run Gemini VLM on all site images."""
    import google.generativeai as genai
    from PIL import Image

    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("GOOGLE_AI_API_KEY not set!")
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    for site in sites:
        name = site["short_name"]
        print(f"\n--- {site['name']} ---")

        for year in YEARS:
            cache_path = VLM_DIR / f"{name}_{year}_vlm.json"
            if cache_path.exists():
                print(f"  {year}: cached")
                continue

            img_path = IMG_DIR / f"{name}_{year}.png"
            if not img_path.exists():
                print(f"  {year}: no image")
                continue

            prompt = VLM_PROMPT.format(
                lat=site["lat"], lon=site["lon"],
                site_name=site["name"], year=year,
                construction_year=site["construction_year"],
            )
            img = Image.open(img_path)

            for attempt in range(3):
                try:
                    response = model.generate_content(
                        [prompt, img],
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=0.1,
                        ),
                    )
                    result = json.loads(response.text)
                    result["year"] = year
                    result["site"] = name
                    with open(cache_path, "w") as f:
                        json.dump(result, f, indent=2)
                    solar = result.get("solar_visible", "?")
                    solar_pct = result.get("solar_area_pct", 0)
                    print(f"  {year}: solar={solar} ({solar_pct}%)")
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(5)
                    else:
                        print(f"  {year}: FAILED ({e})")
            time.sleep(2)


# ── Figure Generation ────────────────────────────────────────────────────────

def load_all_data(sites):
    """Load all cached data for all sites."""
    data = {}
    for site in sites:
        name = site["short_name"]
        site_data = {}
        for year in YEARS:
            year_data = {"year": year}
            for source_name in SOURCES:
                cache_path = CACHE_DIR / f"{name}_{year}_{source_name}.json"
                if cache_path.exists():
                    with open(cache_path) as f:
                        year_data[source_name] = json.load(f)
            # VLM
            vlm_path = VLM_DIR / f"{name}_{year}_vlm.json"
            if vlm_path.exists():
                with open(vlm_path) as f:
                    year_data["vlm"] = json.load(f)
            site_data[year] = year_data
        data[name] = site_data
    return data


def fig_image_grid(sites):
    """Create satellite image timeline grid for each site."""
    import matplotlib.pyplot as plt
    from PIL import Image

    apply_style()

    for site in sites:
        name = site["short_name"]
        available = []
        for year in YEARS:
            img_path = IMG_DIR / f"{name}_{year}.png"
            if img_path.exists():
                available.append(year)

        if len(available) < 3:
            print(f"  {name}: only {len(available)} images, skipping grid")
            continue

        ncols = min(6, len(available))
        nrows = math.ceil(len(available) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, nrows * 1.3))
        if nrows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()

        for i, year in enumerate(available):
            img = Image.open(IMG_DIR / f"{name}_{year}.png")
            axes[i].imshow(img)
            axes[i].set_title(str(year), fontsize=8,
                              fontweight='bold' if year == site["construction_year"] else 'normal',
                              color='red' if year == site["construction_year"] else 'black')
            axes[i].axis("off")

        for i in range(len(available), len(axes)):
            axes[i].axis("off")

        fig.suptitle(f"{site['name']} — Satellite Image Timeline",
                     fontsize=10, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, FIG_DIR / f"{name}_image_grid.png")
        plt.close(fig)
        print(f"  Saved: {name}_image_grid.png")


def fig_lulc_timeseries(sites, data):
    """Create LULC stacked area charts for each site (DW + VLM side by side)."""
    import matplotlib.pyplot as plt

    apply_style()
    classes = ["cropland", "trees", "shrub", "grassland", "flooded_veg",
               "built", "bare", "water"]

    for site in sites:
        name = site["short_name"]
        site_data = data[name]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0),
                                        sharey=True)

        # DW panel
        dw_years = []
        dw_values = {c: [] for c in classes}
        for year in YEARS:
            dw = site_data.get(year, {}).get("dw")
            if dw:
                dw_years.append(year)
                for c in classes:
                    dw_key = OURS_TO_DW.get(c, c)
                    dw_values[c].append(dw.get(dw_key, 0) or 0)

        if dw_years:
            bottom = np.zeros(len(dw_years))
            for c in classes:
                vals = np.array(dw_values[c])
                ax1.fill_between(dw_years, bottom, bottom + vals,
                                 color=LULC_COLORS[c], alpha=0.85,
                                 label=CLASS_LABELS[c])
                bottom += vals
            ax1.axvline(site["construction_year"], color='red', ls='--',
                        lw=1.5, alpha=0.8)
            ax1.set_title("Dynamic World", fontsize=9)
            ax1.set_ylabel("Coverage (%)")
            ax1.set_ylim(0, 105)

        # VLM panel
        vlm_years = []
        vlm_values = {c: [] for c in classes + ["solar"]}
        for year in YEARS:
            vlm = site_data.get(year, {}).get("vlm", {}).get("land_cover")
            if vlm:
                vlm_years.append(year)
                for c in classes + ["solar"]:
                    val = vlm.get(c, 0)
                    try:
                        vlm_values[c].append(float(val) if val else 0)
                    except (TypeError, ValueError):
                        vlm_values[c].append(0)

        if vlm_years:
            bottom = np.zeros(len(vlm_years))
            for c in classes:
                vals = np.array(vlm_values[c])
                ax2.fill_between(vlm_years, bottom, bottom + vals,
                                 color=LULC_COLORS[c], alpha=0.85)
                bottom += vals
            # Solar on top
            solar_vals = np.array(vlm_values["solar"])
            ax2.fill_between(vlm_years, bottom, bottom + solar_vals,
                             color='#FF6B35', alpha=0.9, label='Solar')
            ax2.axvline(site["construction_year"], color='red', ls='--',
                        lw=1.5, alpha=0.8)
            ax2.set_title("VLM (Gemini 2.0 Flash)", fontsize=9)
            ax2.set_ylim(0, 105)

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=LULC_COLORS[c],
                   label=CLASS_LABELS[c]) for c in classes]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='#FF6B35',
                       label='Solar'))
        fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=7,
                   bbox_to_anchor=(0.5, -0.08))

        fig.suptitle(f"{site['name']} — LULC Composition (2016–2026)",
                     fontsize=10, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, FIG_DIR / f"{name}_lulc_timeseries.png")
        plt.close(fig)
        print(f"  Saved: {name}_lulc_timeseries.png")


def fig_proxy_timeseries(sites, data):
    """Create multi-panel time series for all proxies (4 sites overlaid)."""
    import matplotlib.pyplot as plt

    apply_style()

    site_colors = {
        "teesta": "#332288",
        "feni": "#44AA99",
        "manikganj": "#CC6677",
        "moulvibazar": "#DDCC77",
    }

    proxies = [
        ("VIIRS Nighttime Lights", "viirs", "avg_rad_mean", "nW/sr/cm²"),
        ("SAR VV Backscatter", "sar", "vv_mean_db", "dB"),
        ("SAR VH Backscatter", "sar", "vh_mean_db", "dB"),
        ("NDVI", "ndvi", "ndvi_mean", ""),
        ("LST Day", "lst", "lst_day_c", "°C"),
        ("LST Night", "lst", "lst_night_c", "°C"),
        ("Population (WorldPop)", "worldpop", "pop_sum", "people"),
        ("Building Presence", "buildings", "bldg_presence", "fraction"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(FULL_WIDTH, 8), sharex=True)
    axes = axes.flatten()

    for pi, (title, source, key, unit) in enumerate(proxies):
        ax = axes[pi]
        for site in sites:
            name = site["short_name"]
            years_vals = []
            for year in YEARS:
                src_data = data[name].get(year, {}).get(source)
                if src_data and key in src_data and src_data[key] is not None:
                    years_vals.append((year, src_data[key]))
            if years_vals:
                yrs, vals = zip(*years_vals)
                ax.plot(yrs, vals, '-o', markersize=3,
                        color=site_colors[name], label=site["name"].split(" (")[0],
                        linewidth=1.5)
                # Mark construction year
                ax.axvline(site["construction_year"], color=site_colors[name],
                           ls=':', lw=0.8, alpha=0.5)

        ax.set_title(title, fontsize=9)
        if unit:
            ax.set_ylabel(unit, fontsize=8)
        if pi == 1:
            ax.legend(fontsize=6, loc='best')

    axes[-2].set_xlabel("Year")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Environmental Proxy Time Series — 4 Bangladesh Solar Sites",
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "proxy_timeseries.png")
    plt.close(fig)
    print("  Saved: proxy_timeseries.png")


def fig_pre_post_comparison(sites, data):
    """Bar chart comparing pre vs post construction for key metrics."""
    import matplotlib.pyplot as plt

    apply_style()

    metrics = [
        ("Cropland (%)", "dw", "crops"),
        ("Trees (%)", "dw", "trees"),
        ("Built (%)", "dw", "built"),
        ("Bare (%)", "dw", "bare"),
        ("NTL (nW)", "viirs", "avg_rad_mean"),
        ("SAR VH (dB)", "sar", "vh_mean_db"),
        ("NDVI", "ndvi", "ndvi_mean"),
        ("LST Night (°C)", "lst", "lst_night_c"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(FULL_WIDTH, 4))
    axes = axes.flatten()

    for mi, (label, source, key) in enumerate(metrics):
        ax = axes[mi]
        pre_vals = []
        post_vals = []
        site_labels = []

        for site in sites:
            name = site["short_name"]
            cy = site["construction_year"]

            # Pre: average of years before construction
            pre = []
            post = []
            for year in YEARS:
                src = data[name].get(year, {}).get(source)
                if src and key in src and src[key] is not None:
                    val = float(src[key])
                    if year < cy:
                        pre.append(val)
                    elif year > cy:
                        post.append(val)

            if pre and post:
                pre_vals.append(np.mean(pre))
                post_vals.append(np.mean(post))
                site_labels.append(name[:4].title())

        if not pre_vals:
            ax.set_visible(False)
            continue

        x = np.arange(len(site_labels))
        w = 0.35
        ax.bar(x - w/2, pre_vals, w, color='#88CCEE', label='Pre')
        ax.bar(x + w/2, post_vals, w, color='#CC6677', label='Post')
        ax.set_xticks(x)
        ax.set_xticklabels(site_labels, fontsize=7)
        ax.set_title(label, fontsize=8)
        if mi == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Pre- vs Post-Construction Comparison",
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "pre_post_comparison.png")
    plt.close(fig)
    print("  Saved: pre_post_comparison.png")


def generate_all_figures(sites):
    """Generate all publication figures."""
    data = load_all_data(sites)

    print("\nGenerating figures...")
    fig_image_grid(sites)
    fig_lulc_timeseries(sites, data)
    fig_proxy_timeseries(sites, data)
    fig_pre_post_comparison(sites, data)
    print("\nAll figures saved to docs/figures/case_studies/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Case study analysis for 4 BD solar sites")
    parser.add_argument("--collect", action="store_true", help="Collect GEE data")
    parser.add_argument("--download-images", action="store_true", help="Download Planet images")
    parser.add_argument("--classify", action="store_true", help="Run VLM classification")
    parser.add_argument("--figures", action="store_true", help="Generate figures")
    parser.add_argument("--all", action="store_true", help="Do everything")
    args = parser.parse_args()

    if args.all:
        args.collect = args.download_images = args.classify = args.figures = True

    if not any([args.collect, args.download_images, args.classify, args.figures]):
        parser.print_help()
        return

    if args.collect:
        collect_all(SITES)

    if args.download_images:
        download_planet_images(SITES)

    if args.classify:
        classify_images(SITES)

    if args.figures:
        generate_all_figures(SITES)


if __name__ == "__main__":
    main()
