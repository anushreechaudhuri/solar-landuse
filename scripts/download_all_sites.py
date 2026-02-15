"""
Download Planet basemap imagery (4.77m, 1km buffer) for all non-proposed solar projects.

Pre: dry-season month at least 3 years before completion
Post: dry-season month at least 1 year after completion (or latest available)
"""
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

load_dotenv()

PLANET_API_KEY = os.getenv("PLANET_API_KEY")
BASEMAPS_URL = "https://api.planet.com/basemaps/v1"

PROJECT_DIR = Path("/Users/anushreechaudhuri/Documents/Projects/solar-landuse")
RAW_DIR = PROJECT_DIR / "data" / "raw_images"
LABEL_DIR = PROJECT_DIR / "data" / "for_labeling"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

BUFFER_KM = 1  # 1 km buffer → 2×2 km AOI

# All non-proposed solar projects with coordinates and completion dates.
# Pre/post months chosen as dry-season (Jan) at least 3yr before / 1yr after completion.
SITES = {
    "teesta": {
        "name": "Teesta (Gaibandha/Beximco) 200 MW",
        "lat": 25.629209, "lon": 89.544870,
        "completed": "2023-01-08", "mw": 200,
        "pre": ["2019_01"], "post": ["2024_01"],
    },
    "feni": {
        "name": "Feni/Sonagazi 75 MW",
        "lat": 22.787567, "lon": 91.367187,
        "completed": "2024-04-01", "mw": 75,
        "pre": ["2020_01"], "post": ["2026_01"],
    },
    "manikganj": {
        "name": "Manikganj (Spectra) 35 MW",
        "lat": 23.780834, "lon": 89.824775,
        "completed": "2021-03-12", "mw": 35,
        "pre": ["2017_01"], "post": ["2023_01"],
    },
    "moulvibazar": {
        "name": "Moulvibazar 10 MW",
        "lat": 24.493896, "lon": 91.633043,
        "completed": "2025-10-01", "mw": 10,
        "pre": ["2022_01"], "post": ["2026_01"],
    },
    "pabna": {
        "name": "Pabna 100 MW",
        "lat": 23.826372, "lon": 89.606831,
        "completed": "2024-10-23", "mw": 100,
        "pre": ["2021_01"], "post": ["2026_01"],
    },
    "mymensingh": {
        "name": "Mymensingh (HDFC) 50 MW",
        "lat": 24.702233, "lon": 90.461730,
        "completed": "2020-11-04", "mw": 50,
        "pre": ["2017_01"], "post": ["2022_01"],
    },
    "tetulia": {
        "name": "Tetulia/Panchagarh (Sympa) 8 MW",
        "lat": 26.482817, "lon": 88.410139,
        "completed": "2019-05-13", "mw": 8,
        "pre": ["2016_01"], "post": ["2021_01"],
    },
    # Lalmonirhat coordinates missing from paste - add when available
    # "lalmonirhat": {
    #     "name": "Lalmonirhat Rangpur (Intraco) 30 MW",
    #     "lat": ???, "lon": ???,
    #     "completed": "2022-08-28", "mw": 30,
    #     "pre": ["2019_01"], "post": ["2024_01"],
    # },
    "mongla": {
        "name": "Mongla 100 MW",
        "lat": 22.574239, "lon": 89.570388,
        "completed": "2021-12-29", "mw": 100,
        "pre": ["2018_01"], "post": ["2023_01"],
    },
    "sirajganj68": {
        "name": "Sirajganj 68 MW",
        "lat": 24.403976, "lon": 89.738849,
        "completed": "2024-07-14", "mw": 68,
        "pre": ["2021_01"], "post": ["2026_01"],
    },
    "teknaf": {
        "name": "Teknaf (Joules) 20 MW",
        "lat": 20.981669, "lon": 92.256021,
        "completed": "2018-09-15", "mw": 20,
        "pre": ["2016_01"], "post": ["2020_01"],
    },
    "sirajganj6": {
        "name": "Sirajganj 6 MW",
        "lat": 24.386137, "lon": 89.748970,
        "completed": "2021-03-30", "mw": 6,
        "pre": ["2017_01"], "post": ["2023_01"],
    },
    "kaptai": {
        "name": "Kaptai 7.4 MW",
        "lat": 22.491471, "lon": 92.226588,
        "completed": "2019-05-06", "mw": 7.4,
        "pre": ["2016_01"], "post": ["2021_01"],
    },
    "sharishabari": {
        "name": "Sharishabari 3 MW",
        "lat": 24.772287, "lon": 89.842629,
        "completed": "2017-07-14", "mw": 3,
        "pre": ["2016_01"], "post": ["2019_01"],
    },
    "barishal": {
        "name": "Barishal 1 MW",
        "lat": 22.657015, "lon": 90.339194,
        "completed": "2024-06-08", "mw": 1,
        "pre": ["2021_01"], "post": ["2026_01"],
    },
}


def _auth():
    return (PLANET_API_KEY, "")


def make_bbox(lat, lon, buffer_km):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
    dlat = buffer_km / km_per_deg_lat
    dlon = buffer_km / km_per_deg_lon
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def find_mosaic_id(year_month):
    name = f"global_monthly_{year_month}_mosaic"
    resp = requests.get(f"{BASEMAPS_URL}/mosaics", auth=_auth(),
                        params={"name__is": name})
    resp.raise_for_status()
    mosaics = resp.json().get("mosaics", [])
    if not mosaics:
        return None, name
    return mosaics[0]["id"], name


def get_quads(mosaic_id, bbox):
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    resp = requests.get(f"{BASEMAPS_URL}/mosaics/{mosaic_id}/quads",
                        auth=_auth(), params={"bbox": bbox_str})
    resp.raise_for_status()
    return resp.json().get("items", [])


def download_quad(quad, out_path):
    url = quad["_links"]["download"]
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return out_path


def mosaic_and_clip(quad_paths, bbox, out_tif):
    datasets = [rasterio.open(p) for p in quad_paths]
    mosaic_arr, mosaic_transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update(width=mosaic_arr.shape[2], height=mosaic_arr.shape[1],
                   transform=mosaic_transform, count=mosaic_arr.shape[0])
    for ds in datasets:
        ds.close()

    # Write temp mosaic
    tmp_mosaic = str(out_tif) + ".tmp_mosaic.tif"
    with rasterio.open(tmp_mosaic, "w", **profile) as dst:
        dst.write(mosaic_arr)

    # Reproject to WGS84
    tmp_reproj = str(out_tif) + ".tmp_reproj.tif"
    with rasterio.open(tmp_mosaic) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds,
            resolution=0.00005)
        prof = src.profile.copy()
        prof.update(crs="EPSG:4326", transform=transform,
                    width=width, height=height)
        with rasterio.open(tmp_reproj, "w", **prof) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i),
                          destination=rasterio.band(dst, i),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs="EPSG:4326",
                          resampling=Resampling.bilinear)

    # Clip to bbox
    clip_geom = mapping(box(*bbox))
    with rasterio.open(tmp_reproj) as src:
        clipped, clip_transform = rio_mask(src, [clip_geom], crop=True)
        prof = src.profile.copy()
        prof.update(width=clipped.shape[2], height=clipped.shape[1],
                    transform=clip_transform)
        with rasterio.open(str(out_tif), "w", **prof) as dst:
            dst.write(clipped)

    for tmp in [tmp_mosaic, tmp_reproj]:
        if os.path.exists(tmp):
            os.remove(tmp)


def tif_to_png(tif_path, png_path):
    with rasterio.open(tif_path) as src:
        r, g, b = src.read(1), src.read(2), src.read(3)
    img = Image.fromarray(np.stack([r, g, b], axis=-1))
    img.save(str(png_path), compress_level=0)


def process_site(key, site):
    print(f"\n  {site['name']} ({site['lat']:.4f}, {site['lon']:.4f})")
    print(f"  Completed: {site['completed']}  |  {site['mw']} MW  |  Buffer: {BUFFER_KM}km")

    bbox = make_bbox(site["lat"], site["lon"], BUFFER_KM)

    for period_key in ["pre", "post"]:
        months = site[period_key]
        for ym in months:
            out_name = f"{key}_1km_{ym}_{period_key}"
            out_tif = RAW_DIR / f"{out_name}.tif"
            out_png = LABEL_DIR / f"{out_name}.png"

            if out_tif.exists():
                print(f"    {ym} {period_key}: already exists, skipping")
                continue

            mosaic_id, mosaic_name = find_mosaic_id(ym)
            if mosaic_id is None:
                print(f"    {ym} {period_key}: mosaic not found!")
                continue

            quads = get_quads(mosaic_id, bbox)
            if not quads:
                print(f"    {ym} {period_key}: no quads found!")
                continue

            # Download quads to temp
            tmp_dir = RAW_DIR / "tmp_quads"
            tmp_dir.mkdir(exist_ok=True)
            quad_paths = []
            for q in quads:
                qpath = tmp_dir / f"{q['id'].replace('-', '_')}_{ym}.tif"
                if not qpath.exists():
                    download_quad(q, qpath)
                quad_paths.append(qpath)

            # Mosaic, reproject, clip
            mosaic_and_clip(quad_paths, bbox, out_tif)

            # Check result
            with rasterio.open(out_tif) as src:
                w_km = (src.bounds.right - src.bounds.left) * 111 * math.cos(math.radians(site["lat"]))
                h_km = (src.bounds.top - src.bounds.bottom) * 111
                mb = os.path.getsize(out_tif) / 1e6

            # Convert to PNG
            tif_to_png(out_tif, out_png)

            print(f"    {ym} {period_key}: {src.width}x{src.height}px, "
                  f"{w_km:.1f}x{h_km:.1f}km, {mb:.1f}MB")

            # Cleanup temp
            for qp in quad_paths:
                if qp.exists():
                    os.remove(qp)
            if tmp_dir.exists() and not list(tmp_dir.iterdir()):
                tmp_dir.rmdir()


def main():
    if not PLANET_API_KEY:
        print("Error: PLANET_API_KEY not set in .env")
        sys.exit(1)

    print(f"Downloading Planet basemaps for {len(SITES)} solar projects")
    print(f"Resolution: 4.77m  |  Buffer: {BUFFER_KM}km  |  AOI: {2*BUFFER_KM}x{2*BUFFER_KM}km")
    print(f"Images per site: 1 pre + 1 post = 2")
    print(f"Total downloads: ~{len(SITES) * 2}")

    for key, site in SITES.items():
        process_site(key, site)

    print(f"\n{'='*70}")
    print("Done. GeoTIFFs in data/raw_images/, PNGs in data/for_labeling/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
