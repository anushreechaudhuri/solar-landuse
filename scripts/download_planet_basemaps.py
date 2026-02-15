"""
Download Planet Global Monthly Basemaps (4.77m resolution, full coverage).

Uses the Basemaps API to download pre-composited monthly mosaics - no Orders API
wait, no black borders, instant download.

Usage:
    python scripts/download_planet_basemaps.py                   # Download all
    python scripts/download_planet_basemaps.py --search-only     # List available mosaics only
    python scripts/download_planet_basemaps.py --site manikganj  # One site
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image
import requests
from dotenv import load_dotenv
from shapely.geometry import box, mapping
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

PLANET_API_KEY = os.getenv("PLANET_API_KEY")
BASEMAPS_URL = "https://api.planet.com/basemaps/v1"

PROJECT_DIR = Path("/Users/anushreechaudhuri/Documents/Projects/solar-landuse")
RAW_DIR = PROJECT_DIR / "data" / "raw_images"
LABEL_DIR = PROJECT_DIR / "data" / "for_labeling"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Sites & time periods -------------------------------------------------------
SITES = {
    "manikganj": {
        "name": "Manikganj Spectra Solar",
        "lat": 23.780733,
        "lon": 89.825358,
        "built": "May 2021",
        "periods": {
            "pre":  ["2019_11", "2020_01"],  # 1-2 years before construction
            "post": ["2022_11", "2023_01"],  # 1-2 years after
        },
    },
    "moulvibazar": {
        "name": "Moulvibazar 10 MW Solar",
        "lat": 24.493312,
        "lon": 91.633107,
        "built": "Oct 2025",
        "periods": {
            "pre":  ["2024_01"],             # ~1 year before
            "post": ["2025_11", "2026_01"],  # right after construction
        },
    },
    "teesta": {
        "name": "Teesta (Gaibandha/Beximco) 200 MW Solar",
        "lat": 25.629209,
        "lon": 89.544870,
        "built": "Jan 2023",
        "periods": {
            "pre":  ["2019_01"],             # ~4 years before
            "post": ["2024_01"],             # ~1 year after
        },
    },
}

BUFFER_KM = 5  # 5 km buffer → 10×10 km AOI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _auth():
    return (PLANET_API_KEY, "")


def make_bbox(lat, lon, buffer_km):
    """Return (west, south, east, north) in EPSG:4326."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
    dlat = buffer_km / km_per_deg_lat
    dlon = buffer_km / km_per_deg_lon
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def find_mosaic_id(year_month):
    """Get the mosaic ID for a given YYYY_MM string."""
    name = f"global_monthly_{year_month}_mosaic"
    resp = requests.get(
        f"{BASEMAPS_URL}/mosaics",
        auth=_auth(),
        params={"name__is": name},
    )
    resp.raise_for_status()
    mosaics = resp.json().get("mosaics", [])
    if not mosaics:
        return None, name
    return mosaics[0]["id"], name


def get_quads(mosaic_id, bbox):
    """Get quad tiles covering the bbox."""
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    resp = requests.get(
        f"{BASEMAPS_URL}/mosaics/{mosaic_id}/quads",
        auth=_auth(),
        params={"bbox": bbox_str},
    )
    resp.raise_for_status()
    return resp.json().get("items", [])


def download_quad(quad, out_path):
    """Download a single quad GeoTIFF."""
    url = quad["_links"]["download"]
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return out_path


def mosaic_and_clip(quad_paths, bbox, out_tif, target_crs="EPSG:4326"):
    """Mosaic quads, reproject to WGS84, and clip to bbox. Save GeoTIFF."""
    # Open all quads
    datasets = [rasterio.open(p) for p in quad_paths]

    # Mosaic
    mosaic_arr, mosaic_transform = merge(datasets)
    mosaic_crs = datasets[0].crs  # EPSG:3857
    mosaic_profile = datasets[0].profile.copy()
    mosaic_profile.update(
        width=mosaic_arr.shape[2],
        height=mosaic_arr.shape[1],
        transform=mosaic_transform,
        count=mosaic_arr.shape[0],
    )
    for ds in datasets:
        ds.close()

    # Write temporary mosaic
    tmp_mosaic = str(out_tif) + ".tmp_mosaic.tif"
    with rasterio.open(tmp_mosaic, "w", **mosaic_profile) as dst:
        dst.write(mosaic_arr)

    # Reproject to WGS84
    tmp_reproj = str(out_tif) + ".tmp_reproj.tif"
    with rasterio.open(tmp_mosaic) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=0.00005  # ~5.5m at equator, close to native 4.77m
        )
        profile = src.profile.copy()
        profile.update(crs=target_crs, transform=transform,
                       width=width, height=height)
        with rasterio.open(tmp_reproj, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

    # Clip to bbox
    clip_geom = mapping(box(*bbox))
    with rasterio.open(tmp_reproj) as src:
        clipped, clip_transform = rio_mask(src, [clip_geom], crop=True)
        profile = src.profile.copy()
        profile.update(
            width=clipped.shape[2],
            height=clipped.shape[1],
            transform=clip_transform,
        )
        with rasterio.open(str(out_tif), "w", **profile) as dst:
            dst.write(clipped)

    # Cleanup temp files
    for tmp in [tmp_mosaic, tmp_reproj]:
        if os.path.exists(tmp):
            os.remove(tmp)

    return out_tif


def tif_to_png(tif_path, png_path):
    """Convert GeoTIFF to PNG for labeling."""
    with rasterio.open(tif_path) as src:
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb)
    img.save(str(png_path), compress_level=0)
    return png_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_site(site_key, site, period_filter=None, search_only=False):
    print(f"\n{'='*70}")
    print(f"Site: {site['name']}  ({site['lat']}, {site['lon']})")
    print(f"Built: {site['built']}   Buffer: {BUFFER_KM} km")
    print(f"{'='*70}")

    bbox = make_bbox(site["lat"], site["lon"], BUFFER_KM)
    print(f"Bbox: {bbox[0]:.4f},{bbox[1]:.4f} to {bbox[2]:.4f},{bbox[3]:.4f}")

    for period_key, months in site["periods"].items():
        if period_filter and period_key != period_filter:
            continue

        print(f"\n--- {period_key} ---")

        for ym in months:
            mosaic_id, mosaic_name = find_mosaic_id(ym)
            if mosaic_id is None:
                print(f"  {ym}: mosaic not found, skipping")
                continue

            quads = get_quads(mosaic_id, bbox)
            print(f"  {ym}: mosaic found, {len(quads)} quads needed")

            if search_only:
                for q in quads:
                    print(f"    quad {q['id']} bbox={q['bbox']}")
                continue

            # Check if already downloaded
            out_name = f"{site_key}_5km_{ym}_{period_key}"
            out_tif = RAW_DIR / f"{out_name}.tif"
            out_png = LABEL_DIR / f"{out_name}.png"

            if out_tif.exists():
                print(f"  Already exists: {out_tif.name}, skipping")
                continue

            # Download quads to temp files
            tmp_dir = RAW_DIR / "tmp_quads"
            tmp_dir.mkdir(exist_ok=True)
            quad_paths = []
            for q in quads:
                qpath = tmp_dir / f"{q['id'].replace('-','_')}.tif"
                if not qpath.exists():
                    print(f"    Downloading quad {q['id']}...")
                    download_quad(q, qpath)
                quad_paths.append(qpath)

            # Mosaic, reproject, clip
            print(f"    Mosaicking and clipping...")
            mosaic_and_clip(quad_paths, bbox, out_tif)

            # Check result
            with rasterio.open(out_tif) as src:
                w_km = (src.bounds.right - src.bounds.left) * 111 * math.cos(math.radians(site["lat"]))
                h_km = (src.bounds.top - src.bounds.bottom) * 111
                nodata = (src.read(1) == 0).sum() / (src.width * src.height) * 100
                mb = os.path.getsize(out_tif) / 1e6
                print(f"    Saved: {out_tif.name} ({src.width}x{src.height}px, {w_km:.1f}x{h_km:.1f}km, {mb:.1f}MB, nodata={nodata:.0f}%)")

            # Convert to PNG
            tif_to_png(out_tif, out_png)
            png_mb = os.path.getsize(out_png) / 1e6
            print(f"    PNG: {out_png.name} ({png_mb:.1f}MB)")

            # Cleanup temp quads
            for qp in quad_paths:
                if qp.exists():
                    os.remove(qp)
            if tmp_dir.exists() and not list(tmp_dir.iterdir()):
                tmp_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(description="Download Planet basemap mosaics")
    parser.add_argument("--search-only", action="store_true",
                        help="List available mosaics without downloading")
    parser.add_argument("--site", choices=sorted(SITES.keys()),
                        help="Process only this site")
    parser.add_argument("--period", choices=["pre", "post"],
                        help="Only pre or post construction")
    args = parser.parse_args()

    if not PLANET_API_KEY:
        print("Error: PLANET_API_KEY not set in .env")
        sys.exit(1)

    print("Planet Global Monthly Basemaps (4.77m resolution)")
    print("Full-coverage pre-composited mosaics, no black borders\n")

    sites = {args.site: SITES[args.site]} if args.site else SITES
    for site_key, site in sites.items():
        process_site(site_key, site, period_filter=args.period,
                     search_only=args.search_only)

    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
