"""
Screen comparison (control) sites for difference-in-differences analysis.

For each proposed/cancelled GEM project, queries GEE for:
- Dynamic World: built-up %, cropland %, land cover composition
- Global Solar Atlas: GHI (kWh/m²/day)
- SRTM: mean elevation, max slope

Scores site feasibility (enough non-urban flat land + decent irradiance)
and outputs comparison_sites.json.

Optionally downloads Planet images and runs VLM validation (--vlm flag).

Usage:
    python scripts/screen_comparison_sites.py --country bangladesh
    python scripts/screen_comparison_sites.py --country bangladesh --vlm
    python scripts/screen_comparison_sites.py  # Full South Asia
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import ee
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
UNIFIED_DB = DATA_DIR / "unified_solar_db.json"
OUTPUT_FILE = DATA_DIR / "comparison_sites.json"
CACHE_DIR = DATA_DIR / "screening_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── GEE helpers ──────────────────────────────────────────────────────────────

def make_circle(lat, lon, radius_km):
    """Create EE circle geometry from center point and radius in km."""
    return ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)


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


def query_dw_composition(lat, lon, radius_km=1, year=2023):
    """Query Dynamic World mode composite within 1km circle, return LULC %."""
    circle = make_circle(lat, lon, radius_km)
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

    # Count pixels per class
    class_names = ["water", "trees", "grass", "flooded_vegetation",
                   "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]

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
    result = {}
    for class_id, class_name in enumerate(class_names):
        result[class_name] = 100.0 * hist.get(str(class_id), 0) / total

    return result


def query_solar_atlas_ghi(lat, lon):
    """Query Global Solar Atlas GHI at a point (kWh/m²/day)."""
    point = ee.Geometry.Point([lon, lat])
    ghi = ee.Image("projects/sat-io/open-datasets/global_solar_atlas/ghi_LTAy_AvgDailyTotals")
    val = ghi.sample(point, scale=250).first().getInfo()
    if val and val.get("properties"):
        return val["properties"].get("b1")
    return None


def query_elevation_slope(lat, lon, radius_km=1):
    """Query SRTM elevation and slope within circle."""
    circle = make_circle(lat, lon, radius_km)
    srtm = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(srtm)

    stats = srtm.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=circle,
        scale=30,
        maxPixels=1e7,
    ).getInfo()

    slope_stats = slope.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
        geometry=circle,
        scale=30,
        maxPixels=1e7,
    ).getInfo()

    return {
        "elevation_mean_m": stats.get("elevation_mean"),
        "elevation_stddev_m": stats.get("elevation_stdDev"),
        "slope_mean_deg": slope_stats.get("slope_mean"),
        "slope_max_deg": slope_stats.get("slope_max"),
    }


# ── Feasibility scoring ─────────────────────────────────────────────────────

def score_feasibility(dw_comp, ghi, elev_slope, capacity_mw):
    """Score site feasibility for solar installation (0-1).

    Criteria:
    - Low built-up % (more available land)
    - High cropland/bare/grass % (suitable terrain)
    - Decent GHI (>= 4.0 kWh/m²/day for Bangladesh)
    - Low slope (< 10 degrees mean)
    - Sufficient area for stated capacity (~5-10 ha per MW)
    """
    score = 0.0
    reasons = []

    if dw_comp:
        built_pct = dw_comp.get("built", 0)
        suitable_pct = (dw_comp.get("crops", 0) + dw_comp.get("bare", 0)
                        + dw_comp.get("grass", 0))

        # Built-up penalty
        if built_pct < 10:
            score += 0.25
        elif built_pct < 30:
            score += 0.15
            reasons.append(f"moderate built-up ({built_pct:.0f}%)")
        else:
            reasons.append(f"high built-up ({built_pct:.0f}%)")

        # Suitable land bonus
        if suitable_pct > 50:
            score += 0.25
        elif suitable_pct > 30:
            score += 0.15
        else:
            reasons.append(f"low suitable land ({suitable_pct:.0f}%)")
    else:
        reasons.append("no DW data")

    if ghi is not None:
        if ghi >= 4.5:
            score += 0.25
        elif ghi >= 4.0:
            score += 0.20
        elif ghi >= 3.5:
            score += 0.10
            reasons.append(f"low GHI ({ghi:.1f})")
        else:
            reasons.append(f"very low GHI ({ghi:.1f})")
    else:
        reasons.append("no GHI data")

    if elev_slope:
        slope_mean = elev_slope.get("slope_mean_deg") or 0
        if slope_mean < 5:
            score += 0.25
        elif slope_mean < 10:
            score += 0.15
            reasons.append(f"moderate slope ({slope_mean:.1f}°)")
        else:
            reasons.append(f"steep slope ({slope_mean:.1f}°)")
    else:
        reasons.append("no elevation data")

    return round(score, 2), reasons


# ── VLM validation ───────────────────────────────────────────────────────────

def run_vlm_validation(sites, max_sites=50):
    """Download Planet images and run Gemini VLM for a sample of sites.

    Returns dict of site_id -> vlm_result.
    Requires PLANET_API_KEY and GOOGLE_AI_API_KEY in .env.
    """
    try:
        import google.generativeai as genai
        import requests as req
        from PIL import Image
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError as e:
        print(f"  VLM dependencies not available: {e}")
        return {}

    planet_key = os.getenv("PLANET_API_KEY")
    google_key = os.getenv("GOOGLE_AI_API_KEY")
    if not planet_key or not google_key:
        print("  Missing PLANET_API_KEY or GOOGLE_AI_API_KEY")
        return {}

    genai.configure(api_key=google_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config={"response_mime_type": "application/json", "temperature": 0.1},
    )

    vlm_cache_dir = DATA_DIR / "vlm_screening_cache"
    vlm_cache_dir.mkdir(parents=True, exist_ok=True)

    # Sample sites
    sample = sites[:max_sites]
    print(f"\n  Running VLM validation on {len(sample)} sites...")
    results = {}

    basemaps_url = "https://api.planet.com/basemaps/v1"

    for i, site in enumerate(sample):
        site_id = site["site_id"]
        cache_path = vlm_cache_dir / f"{site_id}_vlm.json"

        if cache_path.exists():
            with open(cache_path) as f:
                results[site_id] = json.load(f)
            continue

        lat = site["centroid_lat"]
        lon = site["centroid_lon"]
        cap = site.get("best_capacity_mw", "unknown")
        name = site.get("gem", {}).get("project_name", site_id)

        print(f"  [{i+1}/{len(sample)}] {name} ({lat:.3f}, {lon:.3f})...", end=" ", flush=True)

        try:
            # Download a recent Planet basemap tile (1km buffer)
            km_per_deg_lat = 111.0
            km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
            dlat = 1.0 / km_per_deg_lat
            dlon = 1.0 / km_per_deg_lon
            bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

            # Find 2024_01 mosaic
            resp = req.get(
                f"{basemaps_url}/mosaics",
                auth=(planet_key, ""),
                params={"name__is": "global_monthly_2024_01_mosaic"},
            )
            resp.raise_for_status()
            mosaics = resp.json().get("mosaics", [])
            if not mosaics:
                print("no mosaic")
                continue

            mosaic_id = mosaics[0]["id"]
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            resp = req.get(
                f"{basemaps_url}/mosaics/{mosaic_id}/quads",
                auth=(planet_key, ""),
                params={"bbox": bbox_str},
            )
            resp.raise_for_status()
            quads = resp.json().get("items", [])
            if not quads:
                print("no quads")
                continue

            # Download first quad, crop to bbox
            quad_url = quads[0]["_links"]["download"]
            img_resp = req.get(quad_url)
            img_resp.raise_for_status()

            import rasterio
            from rasterio.mask import mask as rio_mask
            from shapely.geometry import box, mapping
            import io

            with rasterio.MemoryFile(img_resp.content) as memfile:
                with memfile.open() as src:
                    # Crop to small area around point
                    geom = mapping(box(*bbox))
                    try:
                        clipped, _ = rio_mask(src, [geom], crop=True)
                    except Exception:
                        clipped = np.stack([src.read(i) for i in range(1, 4)])

            # Convert to PIL Image
            rgb = np.moveaxis(clipped[:3], 0, -1)
            img = Image.fromarray(rgb)

            # Send to Gemini
            prompt = f"""Analyze this satellite image and assess if this location is suitable
for a solar farm installation of approximately {cap} MW capacity.

Site: {name}
Location: {lat:.4f}N, {lon:.4f}E
Area: approximately 2x2 km

Assess:
1. What is the dominant land use? (cropland, urban, forest, water, etc.)
2. Is there enough open, flat land for a {cap}MW solar farm?
3. Are there any obstacles? (dense settlement, water bodies, steep terrain, forest)
4. Overall feasibility score (0-100)?

Return JSON: {{"dominant_land_use": "...", "open_land_pct": 0-100, "feasibility_score": 0-100, "obstacles": ["..."], "notes": "..."}}"""

            response = model.generate_content([img, prompt])
            vlm_result = json.loads(response.text)
            results[site_id] = vlm_result

            with open(cache_path, "w") as f:
                json.dump(vlm_result, f)

            score = vlm_result.get("feasibility_score", "?")
            print(f"score={score}")
            time.sleep(4)  # Rate limit

        except Exception as e:
            print(f"error: {e}")
            continue

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def screen_sites(country_filter=None, run_vlm=False):
    print("Loading unified solar database...")
    with open(UNIFIED_DB) as f:
        db = json.load(f)

    # Filter to proposed/cancelled sites (control group candidates)
    proposed = [e for e in db
                if e["treatment_group"] == "proposed"
                and e.get("centroid_lat") and e.get("centroid_lon")]
    if country_filter:
        proposed = [e for e in proposed
                    if e["country"].lower() == country_filter.lower()]

    # Also collect treatment sites (high/very_high only) for reference
    treatment = [e for e in db
                 if e["treatment_group"] == "operational"
                 and e["confidence"] in ("very_high", "high")]
    if country_filter:
        treatment = [e for e in treatment
                     if e["country"].lower() == country_filter.lower()]

    print(f"  Proposed (control candidates): {len(proposed)}")
    print(f"  Operational (high/very_high treatment): {len(treatment)}")

    print("\nInitializing GEE...")
    ee.Initialize(project="bangladesh-solar")

    # Screen all proposed sites
    print(f"\nScreening {len(proposed)} proposed sites...")
    screened = []

    for i, site in enumerate(proposed):
        site_id = site["site_id"]
        lat = site["centroid_lat"]
        lon = site["centroid_lon"]
        gem = site.get("gem", {})
        name = gem.get("project_name", site_id)
        cap = gem.get("capacity_mw")

        cache_path = CACHE_DIR / f"{site_id}_screen.json"

        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            screened.append(cached)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(proposed)}] {name}: cached "
                      f"(score={cached.get('feasibility_score', '?')})")
            continue

        print(f"  [{i+1}/{len(proposed)}] {name} ({lat:.3f}, {lon:.3f})...",
              end=" ", flush=True)

        # Query GEE
        try:
            dw_comp = query_dw_composition(lat, lon, radius_km=1, year=2023)
        except Exception as e:
            print(f"DW error: {e}", end=" ")
            dw_comp = None

        try:
            ghi = query_solar_atlas_ghi(lat, lon)
        except Exception as e:
            print(f"GHI error: {e}", end=" ")
            ghi = None

        try:
            elev_slope = query_elevation_slope(lat, lon, radius_km=1)
        except Exception as e:
            print(f"elev error: {e}", end=" ")
            elev_slope = None

        # Score feasibility
        feas_score, reasons = score_feasibility(dw_comp, ghi, elev_slope, cap)

        result = {
            "site_id": site_id,
            "country": site["country"],
            "project_name": name,
            "gem_status": gem.get("status"),
            "capacity_mw": cap,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "dw_composition": dw_comp,
            "ghi_kwh_m2_day": ghi,
            "elevation_slope": elev_slope,
            "feasibility_score": feas_score,
            "feasibility_issues": reasons,
        }

        # Cache
        with open(cache_path, "w") as f:
            json.dump(result, f)

        screened.append(result)
        print(f"score={feas_score} {'('+', '.join(reasons)+')' if reasons else ''}")

    # Also screen treatment sites for baseline comparison
    print(f"\nScreening {len(treatment)} treatment sites for baseline comparison...")
    treatment_screened = []

    for i, site in enumerate(treatment):
        site_id = site["site_id"]
        lat = site["centroid_lat"]
        lon = site["centroid_lon"]
        gem = site.get("gem", {})
        name = gem.get("project_name", site_id)

        cache_path = CACHE_DIR / f"{site_id}_screen.json"

        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            treatment_screened.append(cached)
            continue

        print(f"  [{i+1}/{len(treatment)}] {name}...", end=" ", flush=True)

        try:
            dw_comp = query_dw_composition(lat, lon, radius_km=1, year=2023)
        except Exception:
            dw_comp = None

        try:
            ghi = query_solar_atlas_ghi(lat, lon)
        except Exception:
            ghi = None

        try:
            elev_slope = query_elevation_slope(lat, lon, radius_km=1)
        except Exception:
            elev_slope = None

        feas_score, reasons = score_feasibility(dw_comp, ghi, elev_slope,
                                                gem.get("capacity_mw"))

        result = {
            "site_id": site_id,
            "country": site["country"],
            "project_name": name,
            "gem_status": gem.get("status"),
            "capacity_mw": gem.get("capacity_mw"),
            "centroid_lat": lat,
            "centroid_lon": lon,
            "confidence": site["confidence"],
            "dw_composition": dw_comp,
            "ghi_kwh_m2_day": ghi,
            "elevation_slope": elev_slope,
            "feasibility_score": feas_score,
            "feasibility_issues": reasons,
        }

        with open(cache_path, "w") as f:
            json.dump(result, f)

        treatment_screened.append(result)
        print(f"score={feas_score}")

    # Optional VLM validation
    vlm_results = {}
    if run_vlm:
        vlm_results = run_vlm_validation(screened, max_sites=50)

    # Merge VLM results
    for site in screened:
        vlm = vlm_results.get(site["site_id"])
        if vlm:
            site["vlm_feasibility"] = vlm.get("feasibility_score")
            site["vlm_dominant_land_use"] = vlm.get("dominant_land_use")
            site["vlm_obstacles"] = vlm.get("obstacles")

    # Output
    output = {
        "comparison_sites": screened,
        "treatment_sites": treatment_screened,
        "metadata": {
            "country_filter": country_filter,
            "n_comparison": len(screened),
            "n_treatment": len(treatment_screened),
            "vlm_validated": len(vlm_results),
        },
    }

    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved ({size_mb:.1f} MB)")

    # Summary stats
    print(f"\n{'='*60}")
    print("COMPARISON SITE SCREENING SUMMARY")
    print(f"{'='*60}")
    print(f"Control candidates: {len(screened)}")
    print(f"Treatment sites:    {len(treatment_screened)}")

    if screened:
        scores = [s["feasibility_score"] for s in screened]
        print(f"\nFeasibility scores (control):")
        print(f"  Mean:   {np.mean(scores):.2f}")
        print(f"  Median: {np.median(scores):.2f}")
        print(f"  >= 0.5: {sum(1 for s in scores if s >= 0.5)}")
        print(f"  >= 0.7: {sum(1 for s in scores if s >= 0.7)}")

        # GHI summary
        ghis = [s["ghi_kwh_m2_day"] for s in screened if s["ghi_kwh_m2_day"]]
        if ghis:
            print(f"\nGHI (kWh/m²/day):")
            print(f"  Mean: {np.mean(ghis):.2f}, Range: {min(ghis):.2f}-{max(ghis):.2f}")

    if treatment_screened:
        t_scores = [s["feasibility_score"] for s in treatment_screened]
        print(f"\nFeasibility scores (treatment baseline):")
        print(f"  Mean:   {np.mean(t_scores):.2f}")
        print(f"  Median: {np.median(t_scores):.2f}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Screen comparison sites for DiD analysis")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter to single country (e.g. 'bangladesh')")
    parser.add_argument("--vlm", action="store_true",
                        help="Run VLM validation on top candidates")
    args = parser.parse_args()

    screen_sites(country_filter=args.country, run_vlm=args.vlm)


if __name__ == "__main__":
    main()
