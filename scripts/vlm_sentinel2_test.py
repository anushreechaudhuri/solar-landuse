"""Test VLM classification on Sentinel-2 images from GEE.

Downloads S2 RGB composites for case study sites, runs Gemini 2.5 Flash
and 3 Flash, then compares with DW and existing Planet-based VLM results.

Usage:
    python scripts/vlm_sentinel2_test.py                    # Download + classify
    python scripts/vlm_sentinel2_test.py --skip-download    # Classify only (use cached images)
    python scripts/vlm_sentinel2_test.py --skip-classify    # Download only
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (
    apply_style, save_fig, FULL_WIDTH, CLASS_LABELS, LULC_COLORS,
)

DATA_DIR = Path(__file__).parent.parent / "data"
S2_DIR = DATA_DIR / "case_study_s2_images"
S2_DIR.mkdir(parents=True, exist_ok=True)
VLM_DIR = DATA_DIR / "case_study_vlm_s2"
VLM_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(__file__).parent.parent / "docs" / "figures" / "case_studies"
PLANET_VLM_DIR = DATA_DIR / "case_study_vlm"

API_KEY = os.getenv("GOOGLE_AI_API_KEY")

# Case study sites and test years (pre + post construction)
SITES = {
    "teesta": {
        "name": "Beximco Teesta 200 MW",
        "lat": 25.629209, "lon": 89.544870,
        "construction_year": 2023,
        "test_years": [2020, 2024],
    },
    "feni": {
        "name": "Feni Sonagazi 75 MW",
        "lat": 22.787567, "lon": 91.367187,
        "construction_year": 2024,
        "test_years": [2020, 2025],
    },
    "manikganj": {
        "name": "Manikganj Spectra 35 MW",
        "lat": 23.780834, "lon": 89.824775,
        "construction_year": 2021,
        "test_years": [2018, 2023],
    },
    "moulvibazar": {
        "name": "Moulvibazar 10 MW",
        "lat": 24.493312, "lon": 91.633107,
        "construction_year": 2025,
        "test_years": [2022, 2026],
    },
}

CLASSES = ["cropland", "trees", "shrub", "grassland", "flooded_veg",
           "built", "bare", "water", "snow", "solar"]

BUFFER_KM = 2  # match existing case study extent


# ── Sentinel-2 Download ──────────────────────────────────────────────────────

def download_s2_images():
    """Download Sentinel-2 RGB composites from GEE."""
    import ee
    ee.Initialize(project="bangladesh-solar")

    for site_key, site in SITES.items():
        lat, lon = site["lat"], site["lon"]
        region = ee.Geometry.Rectangle([
            lon - BUFFER_KM / 111.32,
            lat - BUFFER_KM / 110.574,
            lon + BUFFER_KM / 111.32,
            lat + BUFFER_KM / 110.574,
        ])

        print(f"\n--- {site['name']} ---")
        for year in site["test_years"]:
            out_path = S2_DIR / f"{site_key}_{year}_s2.png"
            if out_path.exists():
                print(f"  {year}: cached")
                continue

            # Sentinel-2 cloud-masked median composite (dry season: Nov-Mar)
            # Use a wider window for better coverage
            start = f"{year}-01-01"
            end = f"{year}-12-31"

            s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))

            count = s2.size().getInfo()
            if count == 0:
                print(f"  {year}: no S2 data, trying wider cloud threshold...")
                s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(region)
                      .filterDate(start, end)
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)))
                count = s2.size().getInfo()
                if count == 0:
                    print(f"  {year}: still no data, skipping")
                    continue

            print(f"  {year}: {count} scenes, compositing...")

            # Cloud masking using SCL band
            def mask_clouds(img):
                scl = img.select("SCL")
                # Keep clear (4=veg, 5=bare, 6=water, 7=unclass, 11=snow)
                mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
                return img.updateMask(mask)

            composite = s2.map(mask_clouds).median()

            # True color RGB (B4, B3, B2) — scale to 0-255
            rgb = composite.select(["B4", "B3", "B2"])

            # Visualize: clip to [0, 3000] and scale to byte
            vis = rgb.clamp(0, 3000).divide(3000).multiply(255).toByte()

            try:
                url = vis.getDownloadURL({
                    "scale": 10,
                    "crs": "EPSG:4326",
                    "region": region.getInfo()["coordinates"],
                    "format": "GEO_TIFF",
                })
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                content = resp.content

                # Handle zip response
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        tif_name = [n for n in zf.namelist() if n.endswith(".tif")][0]
                        content = zf.read(tif_name)
                except zipfile.BadZipFile:
                    pass

                import rasterio
                with rasterio.MemoryFile(content) as memfile:
                    with memfile.open() as src:
                        r = src.read(1)
                        g = src.read(2)
                        b = src.read(3)

                # Stack to RGB and save as PNG
                rgb_arr = np.stack([r, g, b], axis=-1)
                img = Image.fromarray(rgb_arr, "RGB")
                img.save(out_path)
                print(f"  {year}: saved ({rgb_arr.shape[1]}x{rgb_arr.shape[0]}px)")

            except Exception as e:
                print(f"  {year}: ERROR ({e})")

            time.sleep(1)


# ── VLM Classification ───────────────────────────────────────────────────────

VLM_PROMPT = """Analyze this satellite image (~4km x 4km, {resolution}).
Site: {site_name}, Year: {year}, Bangladesh.

Estimate the percentage breakdown of land cover classes:
cropland, trees, shrub, grassland, flooded_veg, built, bare, water, snow, solar

Return JSON:
{{"land_cover": {{"cropland": %, "trees": %, "shrub": %, "grassland": %, "flooded_veg": %, "built": %, "bare": %, "water": %, "snow": %, "solar": %}},
  "solar_visible": "yes" or "no",
  "solar_area_pct": 0-100,
  "description": "brief description"}}"""


def classify_with_model(img_path, site_name, year, model_name, resolution="10m"):
    """Run VLM classification with a specific Gemini model."""
    img = Image.open(img_path)
    prompt = VLM_PROMPT.format(
        site_name=site_name, year=year, resolution=resolution,
    )

    if model_name == "gemini-2.0-flash":
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            [prompt, img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        result = json.loads(response.text)
    else:
        from google import genai as genai_new
        client = genai_new.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, img],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )
        result = json.loads(response.text)

    pcts = {k: float(v) for k, v in result.get("land_cover", {}).items()}
    return {
        "model": model_name,
        "percentages": pcts,
        "solar_visible": result.get("solar_visible", "unknown"),
        "solar_area_pct": result.get("solar_area_pct", 0),
        "description": result.get("description", ""),
    }


def classify_all():
    """Run both Gemini 2.5 Flash and 3 Flash on all S2 images."""
    models = ["gemini-2.5-flash", "gemini-3-flash-preview"]

    for site_key, site in SITES.items():
        print(f"\n--- {site['name']} ---")
        for year in site["test_years"]:
            img_path = S2_DIR / f"{site_key}_{year}_s2.png"
            if not img_path.exists():
                print(f"  {year}: no S2 image, skipping")
                continue

            cache_path = VLM_DIR / f"{site_key}_{year}_s2_vlm.json"
            if cache_path.exists():
                print(f"  {year}: cached")
                continue

            results = {}
            for model_name in models:
                short = model_name.replace("gemini-", "").replace("-preview", "")
                print(f"  {year} [{short}]: classifying...", end=" ", flush=True)
                try:
                    r = classify_with_model(
                        img_path, site["name"], year, model_name,
                        resolution="10m Sentinel-2",
                    )
                    results[model_name] = r
                    solar = r["percentages"].get("solar", 0)
                    print(f"solar={solar}%")
                except Exception as e:
                    print(f"ERROR: {e}")
                time.sleep(3)

            # Save results
            with open(cache_path, "w") as f:
                json.dump({
                    "site": site_key,
                    "year": year,
                    "image_source": "sentinel-2",
                    "resolution": "10m",
                    "results": results,
                }, f, indent=2)


# ── Comparison ────────────────────────────────────────────────────────────────

def load_dw_pcts(site_key, year):
    """Load DW percentages from cached rasters."""
    npz_path = DATA_DIR / "case_study_dw_rasters" / f"{site_key}_{year}_dw.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    remapped = data["remapped"]
    total = remapped.size
    id_to_class = {
        1: "cropland", 2: "trees", 3: "shrub", 4: "grassland",
        5: "flooded_veg", 6: "built", 7: "bare", 8: "water", 9: "snow",
    }
    pcts = {}
    for cid, cname in id_to_class.items():
        pcts[cname] = round(100.0 * np.sum(remapped == cid) / total, 1)
    pcts["solar"] = 0.0
    return pcts


def load_planet_vlm(site_key, year):
    """Load existing Planet-based VLM results."""
    # Check the batch comparison file first
    batch_path = FIG_DIR / "vlm_model_comparison_batch.json"
    if batch_path.exists():
        with open(batch_path) as f:
            batch = json.load(f)
        key = f"{site_key}_{year}"
        if key in batch:
            entry = batch[key]
            results = {}
            for approach in entry.get("approaches", []):
                model = approach["model"]
                if model in ("gemini-2.5-flash", "gemini-3-flash-preview"):
                    if approach["approach"] == "percentage_json":
                        results[model] = approach["percentages"]
            return results

    # Fall back to per-site JSON
    json_path = FIG_DIR / f"vlm_model_comparison_{site_key}_{year}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        results = {}
        for approach in data.get("approaches", []):
            model = approach["model"]
            if model in ("gemini-2.5-flash", "gemini-3-flash-preview"):
                if approach["approach"] == "percentage_json":
                    results[model] = approach["percentages"]
        return results
    return None


def compare_and_report():
    """Compare S2 VLM results with Planet VLM and DW."""
    import matplotlib.pyplot as plt
    apply_style()

    print("\n" + "=" * 80)
    print("SENTINEL-2 vs PLANET VLM COMPARISON")
    print("=" * 80)

    all_rows = []

    for site_key, site in SITES.items():
        for year in site["test_years"]:
            is_post = year > site["construction_year"]
            phase = "post" if is_post else "pre"

            # Load S2 VLM results
            s2_cache = VLM_DIR / f"{site_key}_{year}_s2_vlm.json"
            if not s2_cache.exists():
                continue
            with open(s2_cache) as f:
                s2_data = json.load(f)

            # Load DW
            dw_pcts = load_dw_pcts(site_key, year)

            # Load Planet VLM
            planet_vlm = load_planet_vlm(site_key, year)

            print(f"\n{site['name']} — {year} ({phase}-construction)")
            print("-" * 60)

            # Header
            header = f"{'Class':<15} {'DW':>6}"
            models = ["gemini-2.5-flash", "gemini-3-flash-preview"]
            for model in models:
                short = model.replace("gemini-", "").replace("-preview", "")
                header += f"  {'S2-'+short:>14}"
                if planet_vlm and model in planet_vlm:
                    header += f"  {'Planet-'+short:>14}"
            print(header)
            print("-" * len(header))

            for cls in CLASSES:
                dw_val = dw_pcts.get(cls, 0) if dw_pcts else 0
                row = f"{cls:<15} {dw_val:>5.1f}%"

                for model in models:
                    # S2 value
                    s2_result = s2_data.get("results", {}).get(model, {})
                    s2_pcts = s2_result.get("percentages", {})
                    s2_val = s2_pcts.get(cls, 0)
                    row += f"  {s2_val:>13.1f}%"

                    # Planet value
                    if planet_vlm and model in planet_vlm:
                        p_val = planet_vlm[model].get(cls, 0)
                        row += f"  {p_val:>13.1f}%"

                print(row)

            # Collect for summary
            for model in models:
                s2_result = s2_data.get("results", {}).get(model, {})
                s2_solar = s2_result.get("percentages", {}).get("solar", 0)
                p_solar = planet_vlm.get(model, {}).get("solar", 0) if planet_vlm else None
                all_rows.append({
                    "site": site_key, "year": year, "phase": phase,
                    "model": model.replace("gemini-", "").replace("-preview", ""),
                    "capacity_mw": int(site["name"].split()[-2]),
                    "s2_solar": s2_solar,
                    "planet_solar": p_solar,
                })

    # Summary table
    print("\n\n" + "=" * 80)
    print("SOLAR DETECTION SUMMARY")
    print("=" * 80)
    print(f"{'Site':<15} {'Year':>5} {'Phase':>5} {'MW':>5} {'Model':<20} "
          f"{'S2 Solar%':>10} {'Planet Solar%':>14} {'Diff':>6}")
    print("-" * 85)
    for r in all_rows:
        p_str = f"{r['planet_solar']:.0f}%" if r["planet_solar"] is not None else "N/A"
        diff = ""
        if r["planet_solar"] is not None:
            diff = f"{r['s2_solar'] - r['planet_solar']:+.0f}"
        print(f"{r['site']:<15} {r['year']:>5} {r['phase']:>5} {r['capacity_mw']:>5} "
              f"{r['model']:<20} {r['s2_solar']:>9.0f}% {p_str:>14} {diff:>6}")

    # Make comparison figure
    _make_comparison_figure(all_rows)


def _make_comparison_figure(rows):
    """Create a figure comparing S2 vs Planet VLM solar detection."""
    import matplotlib.pyplot as plt
    apply_style()

    # Only post-construction rows for solar detection comparison
    post_rows = [r for r in rows if r["phase"] == "post"]
    pre_rows = [r for r in rows if r["phase"] == "pre"]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.5),
                              gridspec_kw={"wspace": 0.35})

    # Left: Post-construction solar detection
    ax = axes[0]
    models = sorted(set(r["model"] for r in post_rows))
    sites = list(dict.fromkeys(r["site"] for r in post_rows))  # preserve order
    x = np.arange(len(sites))
    width = 0.35

    for i, model in enumerate(models):
        s2_vals = []
        planet_vals = []
        for site in sites:
            matching = [r for r in post_rows if r["site"] == site and r["model"] == model]
            if matching:
                s2_vals.append(matching[0]["s2_solar"])
                planet_vals.append(matching[0]["planet_solar"] or 0)
            else:
                s2_vals.append(0)
                planet_vals.append(0)

        offset = (i - 0.5) * width
        bars_s2 = ax.bar(x + offset - width/4, s2_vals, width/2,
                         label=f"S2 — {model}", alpha=0.8,
                         color=['#44AA99', '#DDCC77'][i])
        bars_planet = ax.bar(x + offset + width/4, planet_vals, width/2,
                             label=f"Planet — {model}", alpha=0.8,
                             color=['#44AA99', '#DDCC77'][i],
                             hatch='///', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.title()}\n({[r['capacity_mw'] for r in post_rows if r['site']==s][0]} MW)"
                        for s in sites], fontsize=7)
    ax.set_ylabel("Solar Detection (%)", fontsize=8)
    ax.set_title("Post-Construction: Solar Detection", fontsize=9, fontweight='bold')
    ax.legend(fontsize=5.5, loc='upper right')

    # Right: Pre-construction false positive check
    ax = axes[1]
    for i, model in enumerate(models):
        s2_vals = []
        for site in [r["site"] for r in pre_rows if r["model"] == model]:
            matching = [r for r in pre_rows if r["site"] == site and r["model"] == model]
            if matching:
                s2_vals.append(matching[0]["s2_solar"])
        sites_pre = list(dict.fromkeys(r["site"] for r in pre_rows if r["model"] == model))
        if s2_vals:
            ax.bar(np.arange(len(sites_pre)) + (i - 0.5) * 0.35,
                   s2_vals, 0.3,
                   label=f"S2 — {model}", alpha=0.8,
                   color=['#44AA99', '#DDCC77'][i])

    sites_pre_all = list(dict.fromkeys(r["site"] for r in pre_rows))
    ax.set_xticks(np.arange(len(sites_pre_all)))
    ax.set_xticklabels([s.title() for s in sites_pre_all], fontsize=7)
    ax.set_ylabel("Solar Detection (%)", fontsize=8)
    ax.set_title("Pre-Construction: False Positives", fontsize=9, fontweight='bold')
    ax.legend(fontsize=5.5, loc='upper right')
    ax.set_ylim(0, max(5, ax.get_ylim()[1]))

    fig.suptitle("Sentinel-2 (10m) vs Planet (4.77m) — VLM Solar Detection",
                 fontsize=10, fontweight='bold')

    out_path = FIG_DIR / "vlm_s2_vs_planet_comparison.png"
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test VLM on Sentinel-2 images")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip S2 image download (use cached)")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip VLM classification (use cached)")
    args = parser.parse_args()

    if not args.skip_download:
        print("=== Step 1: Downloading Sentinel-2 composites from GEE ===")
        download_s2_images()

    if not args.skip_classify:
        if not API_KEY:
            print("ERROR: GOOGLE_AI_API_KEY not set")
            return
        print("\n=== Step 2: Running VLM classification ===")
        classify_all()

    print("\n=== Step 3: Comparison ===")
    compare_and_report()


if __name__ == "__main__":
    main()
