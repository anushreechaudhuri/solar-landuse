"""Analyze pre-construction land use within solar polygon boundaries.

For all operational sites with polygons (GRW or TZ-SAM), queries Dynamic World
at the earliest available year (baseline ~2016) using the exact polygon geometry
(no buffer). Reports what the land was before it became solar — cropland, forest,
bare ground, etc.

Results are reported for all of South Asia and broken down by country.
Generates a stacked bar chart figure.

Usage:
    python scripts/analyze_polygon_lulc.py                    # Full run (GEE queries)
    python scripts/analyze_polygon_lulc.py --skip-gee         # Re-analyze from cache
    python scripts/analyze_polygon_lulc.py --country india     # Single country
    python scripts/analyze_polygon_lulc.py --workers 8         # Parallel queries
"""
import argparse
import json
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (
    apply_style, save_fig, FULL_WIDTH, DPI,
    LULC_COLORS, CLASS_LABELS, CLASS_ORDER,
)

DATA_DIR = Path(__file__).parent.parent / "data"
UNIFIED_DB = DATA_DIR / "unified_solar_db.json"
CACHE_DIR = DATA_DIR / "polygon_lulc_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = DATA_DIR / "polygon_lulc_results.json"
FIGURE_DIR = Path(__file__).parent.parent / "docs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# DW class names (index = DW raw label)
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation",
              "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]

# Map DW names to our 10-class scheme
DW_TO_OURS = {
    "water": "water",
    "trees": "trees",
    "grass": "grassland",
    "flooded_vegetation": "flooded_veg",
    "crops": "cropland",
    "shrub_and_scrub": "shrub",
    "built": "built",
    "bare": "bare",
    "snow_and_ice": "snow",
}

# Thread-local storage for EE initialization
_ee_init_lock = threading.Lock()
_ee_initialized = False


def ensure_ee():
    global _ee_initialized
    if not _ee_initialized:
        with _ee_init_lock:
            if not _ee_initialized:
                ee.Initialize(project="bangladesh-solar")
                _ee_initialized = True


def load_operational_sites(country_filter=None):
    """Load operational sites with polygons from unified DB."""
    with open(UNIFIED_DB) as f:
        db = json.load(f)

    sites = []
    for entry in db:
        if entry.get("treatment_group") != "operational":
            continue
        if country_filter and entry["country"].lower() != country_filter.lower():
            continue

        # Get polygon (prefer GRW, fall back to TZ-SAM)
        polygon = None
        polygon_source = None
        if entry.get("grw") and entry["grw"].get("polygon"):
            polygon = entry["grw"]["polygon"]
            polygon_source = "grw"
        elif entry.get("tzsam") and entry["tzsam"].get("polygon"):
            polygon = entry["tzsam"]["polygon"]
            polygon_source = "tzsam"

        if not polygon or not polygon.get("coordinates"):
            continue

        construction_year = entry.get("best_construction_year")
        # Baseline year: earliest DW year (2016) or 2 years before construction
        if construction_year and construction_year > 2016:
            baseline_year = min(construction_year - 2, 2020)
            baseline_year = max(baseline_year, 2016)
        else:
            baseline_year = 2016

        sites.append({
            "site_id": entry["site_id"],
            "country": entry["country"],
            "lat": entry["centroid_lat"],
            "lon": entry["centroid_lon"],
            "polygon": polygon,
            "polygon_source": polygon_source,
            "construction_year": construction_year,
            "baseline_year": baseline_year,
            "capacity_mw": entry.get("best_capacity_mw"),
            "confidence": entry.get("confidence", "low"),
        })

    return sites


def query_dw_polygon(site):
    """Query Dynamic World composition within the exact polygon boundary."""
    ensure_ee()

    site_id = site["site_id"]
    year = site["baseline_year"]
    cache_path = CACHE_DIR / f"{site_id}_baseline_{year}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    try:
        poly = ee.Geometry.Polygon(site["polygon"]["coordinates"])

        start = f"{year}-01-01"
        end = f"{year}-12-31"

        dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
              .filterBounds(poly)
              .filterDate(start, end)
              .select("label"))

        count = dw.size().getInfo()
        if count == 0:
            # Try next year
            year2 = year + 1
            dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                  .filterBounds(poly)
                  .filterDate(f"{year2}-01-01", f"{year2}-12-31")
                  .select("label"))
            count = dw.size().getInfo()
            if count == 0:
                return None

        mode_img = dw.reduce(ee.Reducer.mode()).select("label_mode")
        histogram = mode_img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=poly,
            scale=10,
            maxPixels=1e7,
        ).getInfo()

        hist = histogram.get("label_mode", {})
        if not hist:
            return None

        total = sum(hist.values())
        result = {
            "site_id": site_id,
            "country": site["country"],
            "year": year,
            "n_scenes": count,
            "n_pixels": total,
        }
        for i, cn in enumerate(DW_CLASSES):
            result[cn] = round(100.0 * hist.get(str(i), 0) / total, 2)

        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        print(f"    Error for {site_id}: {e}")
        return None


def process_site(site, idx, total):
    """Process a single site — wrapper for threading."""
    site_id = site["site_id"]
    cache_path = CACHE_DIR / f"{site_id}_baseline_{site['baseline_year']}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    print(f"  [{idx+1}/{total}] {site_id} ({site['country']}, "
          f"yr={site['baseline_year']})...", end=" ", flush=True)
    result = query_dw_polygon(site)
    if result:
        print(f"OK ({result['n_pixels']} px)")
    else:
        print("SKIP")
    return result


def create_figure(results, output_path):
    """Create stacked bar chart of pre-construction LULC by country."""
    import matplotlib.pyplot as plt

    apply_style()

    # Aggregate by country
    country_data = defaultdict(list)
    for r in results:
        country_data[r["country"]].append(r)

    # Sort countries by number of sites (descending)
    countries_sorted = sorted(country_data.keys(),
                              key=lambda c: len(country_data[c]),
                              reverse=True)

    # Add "All" as first bar
    labels = ["All S. Asia"] + countries_sorted
    n_bars = len(labels)

    # Our LULC classes (skip no_data and snow)
    classes = ["cropland", "trees", "shrub", "grassland", "flooded_veg",
               "built", "bare", "water"]

    # Compute mean percentages
    means = {}
    counts = {}
    for label in labels:
        if label == "All S. Asia":
            subset = results
        else:
            subset = country_data[label]
        counts[label] = len(subset)
        means[label] = {}
        for cls in classes:
            dw_key = {v: k for k, v in DW_TO_OURS.items()}.get(cls, cls)
            vals = [r.get(dw_key, 0) or 0 for r in subset]
            means[label][cls] = np.mean(vals) if vals else 0

    # Create figure
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.5))

    x = np.arange(n_bars)
    bottoms = np.zeros(n_bars)

    for cls in classes:
        heights = [means[label][cls] for label in labels]
        color = LULC_COLORS[cls]
        ax.bar(x, heights, bottom=bottoms, color=color,
               label=CLASS_LABELS[cls], edgecolor='white', linewidth=0.3)
        bottoms += heights

    # Labels
    ax.set_ylabel("Land cover (%)")
    ax.set_title("Pre-construction land use within solar polygon boundaries")
    ax.set_xticks(x)
    xlabels = []
    for label in labels:
        n = counts[label]
        xlabels.append(f"{label}\n(n={n:,})")
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylim(0, 105)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=LULC_COLORS[c],
               label=CLASS_LABELS[c]) for c in classes]
    ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2,
              framealpha=0.9)

    # Add percentage annotations for top 2 classes per bar
    for i, label in enumerate(labels):
        sorted_cls = sorted(classes, key=lambda c: means[label][c], reverse=True)
        cumulative = 0
        for cls in classes:
            pct = means[label][cls]
            if cls in sorted_cls[:2] and pct >= 5:
                mid_y = cumulative + pct / 2
                ax.text(i, mid_y, f"{pct:.0f}%", ha='center', va='center',
                        fontsize=6.5, fontweight='bold', color='white')
            cumulative += pct

    fig.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)
    print(f"\nFigure saved: {output_path}")


def print_summary(results):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print(f"PRE-CONSTRUCTION LAND USE WITHIN SOLAR POLYGONS ({len(results)} sites)")
    print(f"{'='*70}")

    classes = ["cropland", "trees", "shrub", "grassland", "flooded_veg",
               "built", "bare", "water"]

    # Aggregate by country
    country_data = defaultdict(list)
    for r in results:
        country_data[r["country"]].append(r)

    # Overall
    print(f"\n--- All South Asia (n={len(results)}) ---")
    print(f"  {'Class':<20} {'Mean %':>8} {'Median %':>10} {'Std %':>8}")
    for cls in classes:
        dw_key = {v: k for k, v in DW_TO_OURS.items()}.get(cls, cls)
        vals = [r.get(dw_key, 0) or 0 for r in results]
        print(f"  {CLASS_LABELS[cls]:<20} {np.mean(vals):>8.1f} "
              f"{np.median(vals):>10.1f} {np.std(vals):>8.1f}")

    # By country
    for country in sorted(country_data.keys(),
                          key=lambda c: len(country_data[c]), reverse=True):
        subset = country_data[country]
        print(f"\n--- {country} (n={len(subset)}) ---")
        print(f"  {'Class':<20} {'Mean %':>8} {'Median %':>10}")
        for cls in classes:
            dw_key = {v: k for k, v in DW_TO_OURS.items()}.get(cls, cls)
            vals = [r.get(dw_key, 0) or 0 for r in subset]
            print(f"  {CLASS_LABELS[cls]:<20} {np.mean(vals):>8.1f} "
                  f"{np.median(vals):>10.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pre-construction land use within solar polygons")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter by country")
    parser.add_argument("--skip-gee", action="store_true",
                        help="Only use cached results (no GEE queries)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel GEE query threads")
    args = parser.parse_args()

    # Load sites
    sites = load_operational_sites(args.country)
    print(f"Loaded {len(sites)} operational sites with polygons")
    from collections import Counter
    country_dist = Counter(s["country"] for s in sites)
    print(f"By country: {dict(country_dist)}")

    if args.skip_gee:
        # Load from cache only
        results = []
        for site in sites:
            cache_path = CACHE_DIR / f"{site['site_id']}_baseline_{site['baseline_year']}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    results.append(json.load(f))
        print(f"Loaded {len(results)} cached results")
    else:
        # Initialize GEE
        ee.Initialize(project="bangladesh-solar")

        # Query in parallel
        print(f"\nQuerying Dynamic World (baseline year per site)...")
        results = []
        total = len(sites)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_site, site, i, total): site
                for i, site in enumerate(sites)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        print(f"\nSuccessfully queried {len(results)}/{total} sites")

    if not results:
        print("No results to analyze.")
        return

    # Print summary
    print_summary(results)

    # Create figure
    fig_path = FIGURE_DIR / "did_fig9_polygon_lulc.png"
    create_figure(results, fig_path)

    # Save full results
    output = {
        "n_sites": len(results),
        "description": "Pre-construction Dynamic World LULC within exact solar polygon boundaries",
        "results": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
