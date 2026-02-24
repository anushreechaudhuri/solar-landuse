"""VLM validation of comparison (control) sites using Gemini 2.0 Flash.

Downloads Planet basemap images for a sample of comparison sites, then runs
Gemini visual classification to validate the GEE-based screening. Checks:
  1. No visible solar installation (confirms control status)
  2. Land cover matches GEE/DW classification
  3. Site looks plausible for solar development (flat, non-urban)

Usage:
    python scripts/vlm_validate_comparison.py                # Classify only (use existing images)
    python scripts/vlm_validate_comparison.py --download      # Download + classify
    python scripts/vlm_validate_comparison.py --sample 50     # Sample size (default 50)
    python scripts/vlm_validate_comparison.py --country india  # Filter by country
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
COMPARISON_SITES = DATA_DIR / "comparison_sites.json"
VLM_DIR = DATA_DIR / "vlm_validation"
IMG_DIR = DATA_DIR / "vlm_validation" / "images"
VLM_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "cropland", "trees", "shrub", "grassland", "flooded_veg",
    "built", "bare", "water", "snow", "solar",
]

PROMPT_TEMPLATE = """You are analyzing a satellite image of a site in {country} where a solar power plant was proposed but may not have been built.

The image covers approximately 2km x 2km centered on coordinates ({lat:.4f}, {lon:.4f}).

Please analyze this image and answer:

1. **Solar installation visible?** Is there any solar panel array visible in this image? Answer "yes", "no", or "unclear".

2. **Land cover classification**: Estimate the percentage breakdown of land cover classes in this image. Classes: cropland, trees, shrub, grassland, flooded_veg (wetland/mangrove), built (buildings/roads), bare (bare ground/sand), water, snow.

3. **Solar feasibility**: Rate how suitable this site appears for a utility-scale solar farm (0-1 scale). Consider: available flat open land, proximity to infrastructure, terrain.

4. **Description**: Brief 1-2 sentence description of what you see.

Return your answer as JSON:
{{
  "solar_visible": "yes" or "no" or "unclear",
  "land_cover": {{"cropland": %, "trees": %, "shrub": %, "grassland": %, "flooded_veg": %, "built": %, "bare": %, "water": %, "snow": %}},
  "feasibility": 0.0 to 1.0,
  "description": "..."
}}"""


def load_comparison_sites(country_filter=None):
    with open(COMPARISON_SITES) as f:
        comp = json.load(f)
    sites = comp.get("comparison_sites", [])
    if country_filter:
        sites = [s for s in sites
                 if s["country"].lower() == country_filter.lower()]
    return sites


def sample_sites(sites, n=50):
    """Stratified sample by country."""
    from collections import defaultdict
    by_country = defaultdict(list)
    for s in sites:
        by_country[s["country"]].append(s)

    # Proportional allocation
    sampled = []
    for country, country_sites in by_country.items():
        k = max(1, round(n * len(country_sites) / len(sites)))
        k = min(k, len(country_sites))
        sampled.extend(random.sample(country_sites, k))

    # Trim or pad to target
    if len(sampled) > n:
        sampled = random.sample(sampled, n)

    return sampled


def download_planet_image(site, year=2023):
    """Download Planet basemap for a site using existing download infrastructure."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    site_id = site["site_id"]
    lat = site["centroid_lat"]
    lon = site["centroid_lon"]
    img_path = IMG_DIR / f"{site_id}_{year}.png"

    if img_path.exists():
        return img_path

    try:
        from download_planet_basemaps import (
            find_mosaic_by_date, get_quads_for_bbox, download_and_mosaic
        )
        import rasterio
        from rasterio.transform import from_bounds

        # 1km buffer = ~2km x 2km AOI
        dlat = 0.009  # ~1km
        dlon = 0.011
        bbox = [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

        mosaic = find_mosaic_by_date(year, 6)  # June of target year
        if not mosaic:
            mosaic = find_mosaic_by_date(year, 1)
        if not mosaic:
            print(f"    No mosaic for {site_id} in {year}")
            return None

        quads = get_quads_for_bbox(mosaic["id"], bbox)
        if not quads:
            print(f"    No quads for {site_id}")
            return None

        tiff_path = IMG_DIR / f"{site_id}_{year}.tif"
        download_and_mosaic(quads, bbox, str(tiff_path))

        # Convert to PNG
        with rasterio.open(tiff_path) as src:
            rgb = src.read([1, 2, 3])
            # Normalize to 0-255
            rgb = rgb.astype(float)
            for b in range(3):
                p2, p98 = rgb[b][rgb[b] > 0].min(), rgb[b][rgb[b] > 0].max()
                if p98 > p2:
                    rgb[b] = ((rgb[b] - p2) / (p98 - p2) * 255).clip(0, 255)
            rgb = rgb.astype("uint8")
            img = Image.fromarray(rgb.transpose(1, 2, 0))
            img.save(img_path)

        # Clean up tiff
        tiff_path.unlink(missing_ok=True)
        return img_path

    except Exception as e:
        print(f"    Download failed for {site_id}: {e}")
        return None


def classify_site(site, img_path):
    """Run Gemini classification on a site image."""
    site_id = site["site_id"]
    cache_path = VLM_DIR / f"{site_id}_vlm.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("  GOOGLE_AI_API_KEY not set!")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = PROMPT_TEMPLATE.format(
        country=site["country"],
        lat=site["centroid_lat"],
        lon=site["centroid_lon"],
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
            result["site_id"] = site_id
            result["country"] = site["country"]
            result["project_name"] = site.get("project_name", "")

            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"    Classification failed for {site_id}: {e}")
                return None

    return None


def compare_with_gee(site, vlm_result):
    """Compare VLM classification with GEE screening data."""
    if not vlm_result or not vlm_result.get("land_cover"):
        return None

    gee_dw = site.get("dw_composition", {})
    vlm_lc = vlm_result["land_cover"]

    # Map DW classes to our scheme
    dw_map = {
        "crops": "cropland",
        "trees": "trees",
        "shrub_and_scrub": "shrub",
        "grass": "grassland",
        "flooded_vegetation": "flooded_veg",
        "built": "built",
        "bare": "bare",
        "water": "water",
        "snow_and_ice": "snow",
    }

    comparison = {}
    for dw_key, our_key in dw_map.items():
        dw_val = gee_dw.get(dw_key, 0) or 0
        vlm_val = vlm_lc.get(our_key, 0) or 0
        comparison[our_key] = {
            "dw": round(dw_val, 1),
            "vlm": round(vlm_val, 1),
            "diff": round(vlm_val - dw_val, 1),
        }

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="VLM validation of comparison sites")
    parser.add_argument("--download", action="store_true",
                        help="Download Planet images (requires API key)")
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of sites to sample (default 50)")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter by country")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load and sample sites
    all_sites = load_comparison_sites(args.country)
    print(f"Loaded {len(all_sites)} comparison sites")

    sites = sample_sites(all_sites, args.sample)
    from collections import Counter
    country_dist = Counter(s["country"] for s in sites)
    print(f"Sampled {len(sites)} sites: {dict(country_dist)}")

    # Download images if requested
    if args.download:
        print("\nDownloading Planet basemap images...")
        for i, site in enumerate(sites):
            print(f"  [{i+1}/{len(sites)}] {site['site_id']} "
                  f"({site['country']})...", end=" ", flush=True)
            path = download_planet_image(site)
            if path:
                print(f"OK ({path.name})")
            else:
                print("SKIP")
            time.sleep(1)  # Rate limiting

    # Classify with VLM
    print("\nRunning Gemini classification...")
    results = []
    no_image = 0
    for i, site in enumerate(sites):
        site_id = site["site_id"]
        # Check for existing image
        img_path = IMG_DIR / f"{site_id}_2023.png"
        if not img_path.exists():
            # Try other years
            for year in [2024, 2022, 2021]:
                alt = IMG_DIR / f"{site_id}_{year}.png"
                if alt.exists():
                    img_path = alt
                    break

        if not img_path.exists():
            no_image += 1
            continue

        print(f"  [{i+1}/{len(sites)}] {site_id} ({site['country']})...",
              end=" ", flush=True)
        result = classify_site(site, img_path)
        if result:
            result["comparison"] = compare_with_gee(site, result)
            results.append(result)
            solar = result.get("solar_visible", "?")
            feas = result.get("feasibility", "?")
            print(f"solar={solar}, feasibility={feas}")
        else:
            print("FAILED")
        time.sleep(2)  # Rate limiting

    if no_image > 0:
        print(f"\n{no_image} sites skipped (no image). "
              f"Run with --download to fetch images.")

    # Summary
    print(f"\n{'='*60}")
    print(f"VLM VALIDATION SUMMARY ({len(results)} sites)")
    print(f"{'='*60}")

    if not results:
        print("No results to summarize.")
        return

    # Solar visibility
    solar_counts = Counter(r.get("solar_visible", "unknown") for r in results)
    print(f"\nSolar installation visible?")
    for status, count in solar_counts.most_common():
        print(f"  {status}: {count} ({100*count/len(results):.0f}%)")

    # Feasibility
    feas_scores = [r["feasibility"] for r in results
                   if r.get("feasibility") is not None]
    if feas_scores:
        print(f"\nVLM feasibility: mean={sum(feas_scores)/len(feas_scores):.2f}, "
              f"median={sorted(feas_scores)[len(feas_scores)//2]:.2f}")

    # DW vs VLM comparison
    diffs = {cls: [] for cls in CLASS_NAMES[:9]}
    for r in results:
        comp = r.get("comparison", {})
        for cls in CLASS_NAMES[:9]:
            if cls in comp and comp[cls]["diff"] is not None:
                diffs[cls].append(comp[cls]["diff"])

    if any(diffs.values()):
        print(f"\nDW vs VLM land cover (VLM - DW, percentage points):")
        print(f"  {'Class':<15} {'Mean diff':>10} {'Abs mean':>10} {'N':>5}")
        for cls in CLASS_NAMES[:9]:
            if diffs[cls]:
                vals = diffs[cls]
                mean_d = sum(vals) / len(vals)
                abs_mean = sum(abs(v) for v in vals) / len(vals)
                print(f"  {cls:<15} {mean_d:>+10.1f} {abs_mean:>10.1f} {len(vals):>5}")

    # Save full results
    output = {
        "n_sites": len(results),
        "solar_visibility": dict(solar_counts),
        "results": results,
    }
    out_path = VLM_DIR / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
