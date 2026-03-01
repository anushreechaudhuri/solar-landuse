"""
Modal deployment for full-dataset annual DW + S2 + VLM pipeline.

Runs on Modal's serverless infrastructure with parallel workers.
Three stages:
  1. collect_dw: Annual DW compositions for 3,676 sites × 10 years
  2. collect_s2: Sentinel-2 RGB thumbnails for VLM input
  3. run_vlm:    Gemini 2.5 Flash LULC classification

Usage:
    # Run all three stages
    modal run scripts/modal_pipeline.py

    # Run individual stages
    modal run scripts/modal_pipeline.py::collect_dw
    modal run scripts/modal_pipeline.py::collect_s2
    modal run scripts/modal_pipeline.py::run_vlm

    # Test with small sample
    modal run scripts/modal_pipeline.py --max-sites 10

Prerequisites:
    modal secret create gee-credentials \\
      EE_CREDENTIALS="$(cat ~/.config/earthengine/credentials)" \\
      EE_PROJECT="bangladesh-solar"

    modal secret create gemini-api-key \\
      GOOGLE_AI_API_KEY="$(grep GOOGLE_AI_API_KEY .env | cut -d= -f2)"
"""
import modal

app = modal.App("solar-landuse-pipeline")

# Persistent volume for caching results across runs
vol = modal.Volume.from_name("solar-landuse-data", create_if_missing=True)
VOL_PATH = "/data"

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "earthengine-api",
        "google-auth",
        "google-generativeai",
        "numpy",
        "pandas",
        "shapely",
        "Pillow",
        "requests",
    )
)

YEARS = list(range(2016, 2026))

DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation",
              "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]

# Use dry season (Oct 15 - Mar 15) for temporal consistency.
# Avoids monsoon cloud contamination, matches Planet Jan basemaps.
# For year Y, window is Oct 15 of Y-1 to Mar 15 of Y ("winter of year Y").
# Exception: for DW we use Nov 1 - Mar 31 to ensure enough scenes.
DW_SEASON_START_MONTH, DW_SEASON_START_DAY = 11, 1   # Nov 1
DW_SEASON_END_MONTH, DW_SEASON_END_DAY = 3, 31       # Mar 31
S2_SEASON_START_MONTH, S2_SEASON_START_DAY = 11, 1    # Nov 1
S2_SEASON_END_MONTH, S2_SEASON_END_DAY = 3, 31        # Mar 31


def season_dates(year, start_month, start_day, end_month, end_day):
    """Return (start_date, end_date) for the dry season centered on `year`.

    If start_month > end_month, the window crosses the year boundary:
    e.g. Nov 1 of year-1 to Mar 31 of year.
    """
    if start_month > end_month:
        return f"{year - 1}-{start_month:02d}-{start_day:02d}", \
               f"{year}-{end_month:02d}-{end_day:02d}"
    else:
        return f"{year}-{start_month:02d}-{start_day:02d}", \
               f"{year}-{end_month:02d}-{end_day:02d}"


_ee_initialized = False

def init_ee():
    """Initialize Earth Engine from Modal secret.

    Uses the same OAuth2 refresh-token flow as local `ee.Authenticate()`.
    Writes credentials to ~/.config/earthengine/credentials so that
    ee.Initialize() picks them up automatically.
    """
    global _ee_initialized
    if _ee_initialized:
        return

    import ee
    import json
    import os
    from pathlib import Path

    creds_json = os.environ.get("EE_CREDENTIALS", "")
    project = os.environ.get("EE_PROJECT", "bangladesh-solar")

    if creds_json:
        # Write credentials file where ee.Initialize() expects it
        creds_dir = Path.home() / ".config" / "earthengine"
        creds_dir.mkdir(parents=True, exist_ok=True)
        creds_path = creds_dir / "credentials"
        with open(creds_path, "w") as f:
            f.write(creds_json)

    ee.Initialize(project=project)
    _ee_initialized = True


def compute_buffer(polygon_geojson):
    """Polygon-proportional buffer: max(radius, 500m), capped at 5000m."""
    import math
    if not polygon_geojson or not polygon_geojson.get("coordinates"):
        return 1000
    try:
        from shapely.geometry import shape
        s = shape(polygon_geojson)
        deg_to_m = 111320 * math.cos(math.radians(25))
        area_m2 = s.area * deg_to_m ** 2
        radius_m = math.sqrt(area_m2 / math.pi)
        return max(min(radius_m, 5000), 500)
    except Exception:
        return 1000


def load_sites(max_sites=None, country=None):
    """Load treatment sites from unified DB on the volume."""
    import json
    db_path = f"{VOL_PATH}/unified_solar_db.json"
    with open(db_path) as f:
        db = json.load(f)

    sites = []
    for e in db:
        if (e["treatment_group"] == "operational"
                and e["confidence"] in ("very_high", "high")
                and e.get("centroid_lat") and e.get("centroid_lon")):
            if country and e["country"].lower() != country.lower():
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

    if max_sites:
        sites = sites[:max_sites]
    return sites


# ── Stage 1: DW annual compositions ──────────────────────────────────────────

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gee-credentials")],
    volumes={VOL_PATH: vol},
    timeout=300,
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
)
def query_dw_site_year(site: dict, year: int) -> dict:
    """Query DW annual composition + NDVI for one site × year."""
    import ee
    import json
    import numpy as np
    from pathlib import Path

    site_id = site["site_id"]
    cache_path = Path(f"{VOL_PATH}/annual_cache/{site_id}_{year}.json")

    # Check cache
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        result = {
            "site_id": site_id, "country": site["country"],
            "year": year, "cached": True,
        }
        result.update(cached)
        return result

    init_ee()

    buffer_m = compute_buffer(site["polygon"])
    if site["polygon"] and site["polygon"].get("coordinates"):
        try:
            geom = ee.Geometry.Polygon(
                site["polygon"]["coordinates"]).buffer(buffer_m)
        except Exception:
            geom = ee.Geometry.Point(
                [site["lon"], site["lat"]]).buffer(buffer_m)
    else:
        geom = ee.Geometry.Point([site["lon"], site["lat"]]).buffer(buffer_m)

    # Dry season window for temporal consistency (Nov-Mar)
    start, end = season_dates(year, DW_SEASON_START_MONTH, DW_SEASON_START_DAY,
                              DW_SEASON_END_MONTH, DW_SEASON_END_DAY)

    # DW mode composite
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(geom)
          .filterDate(start, end)
          .select("label"))
    count = dw.size().getInfo()

    dw_data = {"season": f"{start}_to_{end}"}
    if count > 0:
        mode_img = dw.reduce(ee.Reducer.mode()).select("label_mode")
        histogram = mode_img.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geom, scale=10, maxPixels=1e7,
        ).getInfo()
        hist = histogram.get("label_mode", {})
        if hist:
            total = sum(hist.values())
            dw_data.update({f"dw_{cn}_pct": 100.0 * hist.get(str(i), 0) / total
                            for i, cn in enumerate(DW_CLASSES)})
        dw_data["dw_n_scenes"] = count

    # NDVI (same season)
    try:
        ndvi_col = (ee.ImageCollection("MODIS/061/MOD13Q1")
                    .filterDate(start, end).select("NDVI"))
        if ndvi_col.size().getInfo() > 0:
            ndvi_img = ndvi_col.median().multiply(0.0001)
            ndvi_stats = ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom, scale=250, maxPixels=1e7,
            ).getInfo()
            dw_data["ndvi_mean"] = ndvi_stats.get("NDVI")
    except Exception:
        pass

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(dw_data, f)
    vol.commit()

    result = {
        "site_id": site_id, "country": site["country"],
        "year": year, "lat": site["lat"], "lon": site["lon"],
        "capacity_mw": site["capacity_mw"],
        "construction_year": site["construction_year"],
        "buffer_m": buffer_m,
        "cached": False,
    }
    if site["construction_year"]:
        result["event_time"] = year - int(site["construction_year"])
    result.update(dw_data)
    return result


# ── Stage 2: Sentinel-2 thumbnails ───────────────────────────────────────────

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gee-credentials")],
    volumes={VOL_PATH: vol},
    timeout=120,
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
)
def download_s2_image(site: dict, year: int) -> str:
    """Download S2 RGB composite thumbnail for one site × year."""
    import ee
    import requests
    from pathlib import Path

    site_id = site["site_id"]
    out_path = Path(f"{VOL_PATH}/s2_images/{site_id}_{year}.png")

    if out_path.exists():
        return f"cached:{site_id}_{year}"

    init_ee()

    buffer_m = compute_buffer(site["polygon"])
    if site["polygon"] and site["polygon"].get("coordinates"):
        try:
            geom = ee.Geometry.Polygon(
                site["polygon"]["coordinates"]).buffer(buffer_m)
        except Exception:
            geom = ee.Geometry.Point(
                [site["lon"], site["lat"]]).buffer(buffer_m)
    else:
        geom = ee.Geometry.Point([site["lon"], site["lat"]]).buffer(buffer_m)

    # Dry season window for consistent imagery
    start, end = season_dates(year, S2_SEASON_START_MONTH, S2_SEASON_START_DAY,
                              S2_SEASON_END_MONTH, S2_SEASON_END_DAY)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(geom)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))

    count = s2.size().getInfo()
    if count == 0:
        # Relax cloud filter
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(geom)
              .filterDate(start, end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
        count = s2.size().getInfo()
        if count == 0:
            # Fall back to full year if dry season has no data
            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(geom)
                  .filterDate(f"{year}-01-01", f"{year}-12-31")
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)))
            count = s2.size().getInfo()
            if count == 0:
                return f"no_data:{site_id}_{year}"

    # Cloud mask using SCL
    def mask_clouds(img):
        scl = img.select("SCL")
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(mask)

    composite = s2.map(mask_clouds).median().select(["B4", "B3", "B2"])

    bounds = geom.bounds().getInfo()["coordinates"][0]
    region = ee.Geometry.Rectangle([
        min(c[0] for c in bounds), min(c[1] for c in bounds),
        max(c[0] for c in bounds), max(c[1] for c in bounds),
    ])

    url = composite.getThumbURL({
        "region": region.getInfo()["coordinates"],
        "dimensions": 512,
        "format": "png",
        "min": 0, "max": 3000,
        "bands": ["B4", "B3", "B2"],
    })

    resp = requests.get(url, timeout=30)
    if resp.status_code == 200 and len(resp.content) > 1000:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(resp.content)
        vol.commit()
        return f"ok:{site_id}_{year}"

    return f"failed:{site_id}_{year}"


# ── Stage 3: Gemini VLM classification ───────────────────────────────────────

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("gemini-api-key")],
    volumes={VOL_PATH: vol},
    timeout=60,
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
)
def classify_vlm(site_id: str, year: int) -> dict:
    """Run Gemini 2.5 Flash LULC classification on one S2 image."""
    import json
    import os
    import time
    from pathlib import Path

    result_path = Path(f"{VOL_PATH}/vlm_results/{site_id}_{year}.json")
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)

    img_path = Path(f"{VOL_PATH}/s2_images/{site_id}_{year}.png")
    if not img_path.exists():
        return {"site_id": site_id, "year": year, "error": "no_image"}

    import google.generativeai as genai
    from PIL import Image

    api_key = os.environ.get("GOOGLE_AI_API_KEY", "")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

    img = Image.open(img_path)

    prompt = """Analyze this satellite image and estimate the percentage of land cover for each class.
The image shows a ~1-5km area around a potential solar energy site in South Asia.

Return a JSON object with these exact keys and percentage values (must sum to 100):
{
  "water": <float>,
  "trees": <float>,
  "grass": <float>,
  "flooded_vegetation": <float>,
  "crops": <float>,
  "shrub_and_scrub": <float>,
  "built": <float>,
  "bare": <float>,
  "snow_and_ice": <float>,
  "solar_panels": <float>
}

Also include:
  "solar_visible": true/false (are solar panels clearly visible?)
  "description": "<brief 1-sentence description of the landscape>"

Return ONLY the JSON object, no other text."""

    try:
        response = model.generate_content(
            [prompt, img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        result_data = json.loads(response.text)
        result_data["site_id"] = site_id
        result_data["year"] = year
        result_data["model"] = "gemini-2.5-flash"
    except Exception as e:
        result_data = {
            "site_id": site_id, "year": year,
            "error": str(e), "model": "gemini-2.5-flash",
        }

    # Cache result
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2)
    vol.commit()

    return result_data


# ── Orchestration ────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={VOL_PATH: vol},
    timeout=7200,
)
def collect_dw(max_sites: int = None, country: str = None):
    """Stage 1: Collect annual DW compositions for all sites."""
    import csv
    import time
    from pathlib import Path

    sites = load_sites(max_sites=max_sites, country=country)
    print(f"Stage 1: DW annual compositions for {len(sites)} sites × {len(YEARS)} years")

    tasks = [(site, year) for site in sites for year in YEARS]
    print(f"Total tasks: {len(tasks):,}")

    # Check cache
    n_cached = sum(1 for s, y in tasks
                   if Path(f"{VOL_PATH}/annual_cache/{s['site_id']}_{y}.json").exists())
    print(f"Already cached: {n_cached:,}/{len(tasks):,}")

    start = time.time()
    results = []

    # Fan out to parallel workers
    for i, result in enumerate(query_dw_site_year.map(
        [t[0] for t in tasks],
        [t[1] for t in tasks],
        order_outputs=False,
    )):
        results.append(result)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(tasks) - i - 1) / rate
            print(f"  [{i+1:,}/{len(tasks):,}] {rate:.1f}/sec, "
                  f"ETA {eta/60:.0f} min")

    # Save CSV
    if results:
        csv_path = Path(f"{VOL_PATH}/annual_panel.csv")
        all_cols = sorted(set(k for r in results for k in r.keys()))
        results.sort(key=lambda r: (r["site_id"], r.get("year", 0)))

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        vol.commit()
        print(f"Saved {len(results):,} rows to {csv_path}")

    elapsed = time.time() - start
    print(f"Stage 1 complete: {elapsed/60:.1f} min")


@app.function(
    image=image,
    volumes={VOL_PATH: vol},
    timeout=7200,
)
def collect_s2(max_sites: int = None, country: str = None):
    """Stage 2: Download S2 RGB thumbnails for all sites."""
    import time
    from pathlib import Path

    sites = load_sites(max_sites=max_sites, country=country)
    print(f"Stage 2: S2 images for {len(sites)} sites × {len(YEARS)} years")

    tasks = [(site, year) for site in sites for year in YEARS]
    n_cached = sum(1 for s, y in tasks
                   if Path(f"{VOL_PATH}/s2_images/{s['site_id']}_{y}.png").exists())
    print(f"Already cached: {n_cached:,}/{len(tasks):,}")

    start = time.time()
    ok = 0
    failed = 0

    for i, status in enumerate(download_s2_image.map(
        [t[0] for t in tasks],
        [t[1] for t in tasks],
        order_outputs=False,
    )):
        if status.startswith("ok") or status.startswith("cached"):
            ok += 1
        else:
            failed += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1:,}/{len(tasks):,}] {rate:.1f}/sec, "
                  f"ok={ok}, failed={failed}")

    elapsed = time.time() - start
    print(f"Stage 2 complete: {elapsed/60:.1f} min, "
          f"ok={ok}, failed={failed}")


@app.function(
    image=image,
    volumes={VOL_PATH: vol},
    timeout=7200,
)
def run_vlm(max_sites: int = None, country: str = None):
    """Stage 3: Run Gemini VLM classification on all S2 images."""
    import time
    from pathlib import Path

    sites = load_sites(max_sites=max_sites, country=country)
    print(f"Stage 3: VLM classification for {len(sites)} sites × {len(YEARS)} years")

    # Only classify images that exist and aren't already classified
    tasks = []
    for site in sites:
        for year in YEARS:
            img_path = Path(f"{VOL_PATH}/s2_images/{site['site_id']}_{year}.png")
            result_path = Path(f"{VOL_PATH}/vlm_results/{site['site_id']}_{year}.json")
            if img_path.exists() and not result_path.exists():
                tasks.append((site["site_id"], year))

    print(f"Images to classify: {len(tasks):,}")

    start = time.time()
    ok = 0
    errors = 0

    for i, result in enumerate(classify_vlm.map(
        [t[0] for t in tasks],
        [t[1] for t in tasks],
        order_outputs=False,
    )):
        if "error" not in result:
            ok += 1
        else:
            errors += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1:,}/{len(tasks):,}] {rate:.1f}/sec, "
                  f"ok={ok}, errors={errors}")

    elapsed = time.time() - start
    print(f"Stage 3 complete: {elapsed/60:.1f} min, "
          f"ok={ok}, errors={errors}")


@app.function(
    image=image,
    volumes={VOL_PATH: vol},
    timeout=300,
)
def download_results():
    """Download results from Modal volume to local filesystem."""
    import shutil
    from pathlib import Path

    # List what's available
    for subdir in ["annual_cache", "s2_images", "vlm_results"]:
        p = Path(f"{VOL_PATH}/{subdir}")
        if p.exists():
            n = sum(1 for _ in p.iterdir())
            print(f"  {subdir}: {n:,} files")

    csv_path = Path(f"{VOL_PATH}/annual_panel.csv")
    if csv_path.exists():
        print(f"  annual_panel.csv: {csv_path.stat().st_size / 1e6:.1f} MB")


# ── CLI entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    max_sites: int = None,
    country: str = None,
    stage: str = "all",
):
    """Run the full pipeline or individual stages.

    Args:
        max_sites: Limit sites for testing
        country: Filter to one country
        stage: "all", "dw", "s2", "vlm", or "status"
    """
    print(f"Solar Land-Use Pipeline")
    print(f"  Stage: {stage}")
    print(f"  Max sites: {max_sites or 'all'}")
    print(f"  Country: {country or 'all'}")

    if stage in ("all", "dw"):
        collect_dw.remote(max_sites=max_sites, country=country)

    if stage in ("all", "s2"):
        collect_s2.remote(max_sites=max_sites, country=country)

    if stage in ("all", "vlm"):
        run_vlm.remote(max_sites=max_sites, country=country)

    if stage == "status":
        download_results.remote()
