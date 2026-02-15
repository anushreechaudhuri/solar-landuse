"""
Download high-resolution PlanetScope imagery (~3m) via the Planet Data + Orders API.

Usage:
    python scripts/download_planet_images.py                    # Search + download all configured site/periods
    python scripts/download_planet_images.py --search-only      # Search only (no downloads, no quota used)
    python scripts/download_planet_images.py --site manikganj   # Download for one site only
    python scripts/download_planet_images.py --site moulvibazar --period pre  # One site + one period
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

PLANET_API_KEY = os.getenv("PLANET_API_KEY")
DATA_API_URL = "https://api.planet.com/data/v1"
ORDERS_API_URL = "https://api.planet.com/compute/ops/orders/v2"

PROJECT_DIR = Path("/Users/anushreechaudhuri/Documents/Projects/solar-landuse")
OUTPUT_DIR = PROJECT_DIR / "data" / "raw_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test sites ----------------------------------------------------------------
SITES = {
    "manikganj": {
        "name": "Manikganj Spectra Solar",
        "lat": 23.780733,
        "lon": 89.825358,
        "built": "May 2021",
        "periods": {
            "pre": {
                "label": "pre-construction",
                # Dry season 2019-2020, ~1-2 years before construction
                "windows": [
                    ("2019-11-01", "2020-03-31"),
                ],
            },
            "post": {
                "label": "post-construction",
                # Dry season 2022-2023, ~1-2 years after
                "windows": [
                    ("2022-11-01", "2023-03-31"),
                ],
            },
        },
    },
    "moulvibazar": {
        "name": "Moulvibazar 10 MW Solar",
        "lat": 24.493312,
        "lon": 91.633107,
        "built": "Oct 2025",
        "periods": {
            "pre": {
                "label": "pre-construction",
                "windows": [
                    ("2023-11-01", "2024-03-31"),
                ],
            },
            "post": {
                "label": "post-construction",
                # Nov 2025 - Jan 2026 (Planet ~30-day lag, so up to ~mid Jan 2026)
                "windows": [
                    ("2025-11-01", "2026-01-20"),
                ],
            },
        },
    },
}

BUFFER_KM = 5  # 5 km → 10 km × 10 km square AOI
MAX_CLOUD_COVER = 0.15  # 15 % max cloud cover
ITEM_TYPE = "PSScene"
PRODUCT_BUNDLE = "visual"  # RGB visual – good for labeling; switch to analytic_udm2 for NIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _auth():
    return (PLANET_API_KEY, "")


def make_aoi(lat: float, lon: float, buffer_km: float) -> dict:
    """Return a GeoJSON Polygon (square bbox) around *lat/lon* with side = 2*buffer_km."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
    dlat = buffer_km / km_per_deg_lat
    dlon = buffer_km / km_per_deg_lon
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - dlon, lat - dlat],
                [lon + dlon, lat - dlat],
                [lon + dlon, lat + dlat],
                [lon - dlon, lat + dlat],
                [lon - dlon, lat - dlat],
            ]
        ],
    }


# ---------------------------------------------------------------------------
# Step 1 – Search
# ---------------------------------------------------------------------------
def make_center_point(lat: float, lon: float) -> dict:
    """GeoJSON Point for search – ensures scene actually covers the target."""
    return {"type": "Point", "coordinates": [lon, lat]}


def search_scenes(center_geom: dict, start_date: str, end_date: str,
                  max_cloud: float = MAX_CLOUD_COVER) -> list[dict]:
    """Quick-search for PSScene items that *contain* the center point.

    Using a Point geometry guarantees every returned scene covers the actual
    project location, not just the edge of a large AOI box.
    """
    filt = {
        "type": "AndFilter",
        "config": [
            {"type": "GeometryFilter", "field_name": "geometry",
             "config": center_geom},
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {"gte": f"{start_date}T00:00:00Z",
                           "lte": f"{end_date}T23:59:59Z"},
            },
            {"type": "RangeFilter", "field_name": "cloud_cover",
             "config": {"lte": max_cloud}},
            {"type": "PermissionFilter", "config": ["assets:download"]},
        ],
    }

    body = {"item_types": [ITEM_TYPE], "filter": filt}
    resp = requests.post(f"{DATA_API_URL}/quick-search", auth=_auth(), json=body)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    # Sort by cloud cover ascending, then by date descending (freshest clear image)
    def _sort_key(f):
        cc = f["properties"]["cloud_cover"]
        # Parse date string robustly (Python 3.9 compat)
        acq = f["properties"]["acquired"][:19]  # trim tz info
        return (cc, acq)  # lexicographic sort on ISO date string works
    features.sort(key=_sort_key)
    return features


def pick_best_scene(scenes):
    """Return the single clearest scene, or None."""
    return scenes[0] if scenes else None


def print_search_results(scenes: list[dict], label: str):
    """Pretty-print search results."""
    print(f"\n  {label}: {len(scenes)} scenes found")
    for s in scenes[:5]:
        props = s["properties"]
        acq = props["acquired"][:10]
        cc = props["cloud_cover"] * 100
        print(f"    {s['id']}  acquired={acq}  cloud={cc:.1f}%")
    if len(scenes) > 5:
        print(f"    ... and {len(scenes) - 5} more")


# ---------------------------------------------------------------------------
# Step 2 – Order (with clip)
# ---------------------------------------------------------------------------
def place_order(item_id: str, aoi: dict, order_name: str) -> str:
    """Place an Orders API request with clip tool. Returns order ID."""
    order = {
        "name": order_name,
        "source_type": "scenes",
        "products": [
            {
                "item_ids": [item_id],
                "item_type": ITEM_TYPE,
                "product_bundle": PRODUCT_BUNDLE,
            }
        ],
        "tools": [{"clip": {"aoi": aoi}}],
    }

    resp = requests.post(ORDERS_API_URL, auth=_auth(),
                         headers={"Content-Type": "application/json"},
                         json=order)
    resp.raise_for_status()
    result = resp.json()
    order_id = result["id"]
    print(f"    Order placed: {order_id}  (state={result['state']})")
    return order_id


def wait_for_order(order_id: str, timeout: int = 1800, poll: int = 15) -> dict:
    """Poll until the order reaches a terminal state. Returns order dict."""
    url = f"{ORDERS_API_URL}/{order_id}"
    t0 = time.time()
    while time.time() - t0 < timeout:
        resp = requests.get(url, auth=_auth())
        resp.raise_for_status()
        order = resp.json()
        state = order["state"]
        if state in ("success", "partial"):
            return order
        if state in ("failed", "cancelled"):
            hints = order.get("error_hints", [])
            raise RuntimeError(f"Order {order_id} {state}: {hints}")
        # Still running / queued
        elapsed = int(time.time() - t0)
        print(f"    ... waiting ({elapsed}s, state={state})", end="\r")
        time.sleep(poll)
    raise TimeoutError(f"Order {order_id} timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Step 3 – Download
# ---------------------------------------------------------------------------
def download_order(order: dict, output_dir: Path, prefix: str) -> list[Path]:
    """Download all TIFF results from a completed order. Returns file paths."""
    results = []
    for result in order.get("_links", {}).get("results", []):
        name = result["name"]
        # Only download TIFF files (skip manifests, metadata XMLs)
        if not name.endswith(".tif"):
            continue
        loc = result["location"]
        out_path = output_dir / f"{prefix}.tif"
        print(f"    Downloading {name} -> {out_path.name}")
        resp = requests.get(loc, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        size_mb = out_path.stat().st_size / 1e6
        print(f"    Saved: {out_path.name} ({size_mb:.1f} MB)")
        results.append(out_path)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_site(site_key, site, period_filter=None, search_only=False):
    """Run the full search → order → download pipeline for one site."""
    print(f"\n{'='*70}")
    print(f"Site: {site['name']}  ({site['lat']}, {site['lon']})")
    print(f"Built: {site['built']}   Buffer: {BUFFER_KM} km")
    print(f"{'='*70}")

    clip_aoi = make_aoi(site["lat"], site["lon"], BUFFER_KM)
    center_geom = make_center_point(site["lat"], site["lon"])
    aoi_area_km2 = (2 * BUFFER_KM) ** 2
    print(f"AOI area: ~{aoi_area_km2} km2  (clip box for Orders API)")
    print(f"Search: using center point to ensure scene covers target")

    for period_key, period in site["periods"].items():
        if period_filter and period_key != period_filter:
            continue

        print(f"\n--- {period['label']} ---")

        for start, end in period["windows"]:
            window_label = f"{start[:7]}..{end[:7]}"
            print(f"\n  Window: {start} to {end}")

            scenes = search_scenes(center_geom, start, end)
            print_search_results(scenes, window_label)

            best = pick_best_scene(scenes)
            if best is None:
                print("  No usable scenes found for this window.")
                # Retry with relaxed cloud cover
                print("  Retrying with 30% cloud cover threshold...")
                scenes = search_scenes(center_geom, start, end, max_cloud=0.30)
                print_search_results(scenes, f"{window_label} (relaxed)")
                best = pick_best_scene(scenes)
                if best is None:
                    print("  Still no scenes. Skipping this window.")
                    continue

            item_id = best["id"]
            acq_date = best["properties"]["acquired"][:10]
            cc = best["properties"]["cloud_cover"] * 100
            print(f"\n  Selected: {item_id}  date={acq_date}  cloud={cc:.1f}%")

            if search_only:
                print("  (search-only mode, skipping download)")
                continue

            # Build output filename: site_5km_YYYY-MM-DD_pre/post.tif
            prefix = f"{site_key}_5km_{acq_date}_{period_key}"
            out_path = OUTPUT_DIR / f"{prefix}.tif"
            if out_path.exists():
                print(f"  Already downloaded: {out_path.name}, skipping.")
                continue

            # Place order with clip to full AOI
            order_name = f"{site_key}_{period_key}_{acq_date}"
            order_id = place_order(item_id, clip_aoi, order_name)

            # Wait for order
            print("    Waiting for order to complete...")
            completed = wait_for_order(order_id)
            print(f"    Order state: {completed['state']}")

            # Download
            downloaded = download_order(completed, OUTPUT_DIR, prefix)
            if downloaded:
                print(f"  Download complete: {[p.name for p in downloaded]}")
            else:
                print("  Warning: no TIFF files in order results.")


def main():
    global PRODUCT_BUNDLE, MAX_CLOUD_COVER

    parser = argparse.ArgumentParser(description="Download Planet PlanetScope imagery")
    parser.add_argument("--search-only", action="store_true",
                        help="Search for scenes without downloading (no quota used)")
    parser.add_argument("--site", choices=list(SITES.keys()),
                        help="Process only this site")
    parser.add_argument("--period", choices=["pre", "post"],
                        help="Process only pre or post construction period")
    parser.add_argument("--bundle", default=PRODUCT_BUNDLE,
                        help=f"Product bundle (default: {PRODUCT_BUNDLE})")
    parser.add_argument("--max-cloud", type=float, default=MAX_CLOUD_COVER,
                        help=f"Max cloud cover 0-1 (default: {MAX_CLOUD_COVER})")
    args = parser.parse_args()

    PRODUCT_BUNDLE = args.bundle
    MAX_CLOUD_COVER = args.max_cloud

    if not PLANET_API_KEY:
        print("Error: PLANET_API_KEY not set. Add it to .env")
        sys.exit(1)

    # Validate API key
    print("Validating Planet API key...")
    resp = requests.get(f"{DATA_API_URL}", auth=_auth())
    if resp.status_code != 200:
        print(f"Error: API returned {resp.status_code}. Check your API key.")
        sys.exit(1)
    print("API key valid.\n")

    # Estimate quota usage
    sites_to_process = {args.site: SITES[args.site]} if args.site else SITES
    total_windows = sum(
        len(p["windows"])
        for s in sites_to_process.values()
        for pk, p in s["periods"].items()
        if not args.period or pk == args.period
    )
    aoi_area = (2 * BUFFER_KM) ** 2
    est_quota = total_windows * aoi_area
    print(f"Estimated quota usage: ~{est_quota} km2 for {total_windows} scene(s)")
    print(f"Monthly quota: 3000 km2\n")

    if not args.search_only and est_quota > 2000:
        print("Warning: estimated usage exceeds 2/3 of monthly quota.")
        print("Consider using --search-only first, or --site/--period to limit scope.")

    for site_key, site in sites_to_process.items():
        process_site(site_key, site, period_filter=args.period,
                     search_only=args.search_only)

    print(f"\n{'='*70}")
    print("Done.")
    if args.search_only:
        print("Run without --search-only to download imagery.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
