# Solar Land Use Change Detection

Analyzes land use change at solar energy project sites in South Asia using satellite imagery, multiple global LULC datasets, and VLM-based classification. Uses Planet Basemaps (4.77m), Google Earth Engine datasets (10m), and Gemini 2.0 Flash for classification.

## Current Scope

- **15 solar project sites** across Bangladesh (utility-scale, 1MW+)
- **Pre/post construction** imagery comparison (2016-2026)
- **5 classification sources**: Dynamic World, ESA WorldCover, ESRI LULC, GLAD GLCLUC, VLM V2 (Gemini)
- **10-class unified scheme**: cropland, trees, shrub, grassland, flooded veg, built, bare, water, snow, no data
- **GRW polygon matching**: confirmed solar footprint polygons from Global Renewables Watch

## Setup

### Environment

```bash
git clone https://github.com/anushreechaudhuri/solar-landuse.git
cd solar-landuse

# System Python 3.9 works; install dependencies
pip3 install -r requirements.txt

# Set up environment variables
cp local.env .env
# Edit .env: PLANET_API_KEY, GOOGLE_AI_API_KEY, AWS credentials

# Authenticate with Google Earth Engine (one-time)
python3 -c "import ee; ee.Authenticate(); ee.Initialize(project='bangladesh-solar')"
```

## Key Scripts

### Data Download

| Script | Purpose |
|--------|---------|
| `scripts/download_planet_basemaps.py` | Download Planet monthly basemap quads (4.77m) for all sites |
| `scripts/download_satellite_images.py` | Download Sentinel-2 imagery via GEE (10m) |

### Classification & Analysis

| Script | Purpose |
|--------|---------|
| `scripts/compare_lulc_datasets.py` | Multi-dataset LULC comparison (main analysis pipeline) |
| `scripts/vlm_classify_v2.py` | VLM classification using Gemini 2.0 Flash (10-class, polygon-aware) |
| `scripts/figure_style.py` | Publication-quality figure styling (Paul Tol colorblind-safe palette) |

### Training

| Script | Purpose |
|--------|---------|
| `scripts/train_segmentation.py` | DINOv2-based segmentation head training |
| `scripts/apply_segmentation.py` | Apply trained model to generate land cover maps |

## Multi-Dataset LULC Comparison (V3)

The main analysis (`scripts/compare_lulc_datasets.py`) runs in two phases:

```bash
# Full run: query GEE + analyze
python3 scripts/compare_lulc_datasets.py

# Re-analyze from cache (no GEE queries)
python3 scripts/compare_lulc_datasets.py --skip-gee
```

**Phase 1**: Queries 4 GEE datasets (Dynamic World, WorldCover, ESRI, GLAD) for each site, caches raw values as `.npz` files.

**Phase 2**: Remaps to unified 10-class scheme, loads VLM V2 percentages, computes within-polygon stats, generates 6 publication figures and RESULTS.md.

### Outputs

- `data/lulc_comparison_v3.csv` — Full-AOI percentages (all datasets x all images)
- `data/lulc_polygon_v3.csv` — Within-polygon percentages (pre-construction only)
- `docs/figures/v3_*.png` — 6 summary figures (300 DPI, colorblind-safe)
- `data/lulc_comparison/` — Per-image side-by-side visualizations
- `RESULTS.md` — Full analysis writeup with tables and figures

## Project Structure

```
solar-landuse/
├── data/
│   ├── raw_images/              # GeoTIFF files (Planet basemaps)
│   ├── for_labeling/            # PNG files for labeling
│   ├── training_dataset/        # DINOv2 training images and masks
│   ├── vlm_v2_responses/        # Cached Gemini classification results
│   ├── lulc_raw_cache/          # Cached GEE dataset values (.npz)
│   ├── lulc_comparison/         # Per-image LULC visualizations
│   ├── grw/                     # Global Renewables Watch polygon matches
│   ├── lulc_comparison_v3.csv   # Full-AOI comparison results
│   └── lulc_polygon_v3.csv     # Within-polygon results
├── scripts/
│   ├── compare_lulc_datasets.py # Main V3 analysis pipeline
│   ├── figure_style.py          # Publication figure styling module
│   ├── vlm_classify_v2.py       # Gemini 2.0 Flash classifier
│   ├── download_planet_basemaps.py
│   ├── train_segmentation.py
│   └── apply_segmentation.py
├── models/                      # Trained model weights
├── docs/figures/                # Publication-quality figures
├── RESULTS.md                   # Analysis results and findings
├── LOG.md                       # Detailed change log
└── requirements.txt
```

## Key Findings

1. **Cropland is the primary pre-solar land cover** across all datasets and VLM
2. **Only Dynamic World and VLM V2** provide true temporal change detection (WC/GLAD are static snapshots)
3. **DW detects cropland-to-built conversion** at solar sites (no solar class, so panels appear as built/bare/snow)
4. **VLM V2 is polygon-aware** for post-construction images, avoiding the solar-as-built misclassification
5. **Cross-dataset agreement is moderate** — cropland is most consistently identified, other classes vary

## Data Sources

- **Planet Basemaps**: 4.77m monthly mosaics (Jan 2016 - Jan 2026) via Basemaps API
- **Dynamic World**: 10m per-date composite via GEE
- **ESA WorldCover**: 10m single snapshot (2021) via GEE
- **ESRI LULC**: 10m annual (2017-2024) via GEE (sat-io)
- **GLAD GLCLUC**: 30m single snapshot (2020) via GEE
- **VLM V2**: Gemini 2.0 Flash percentage-based classification per image
- **GRW**: Global Renewables Watch solar farm polygons

## Environment Variables

Copy `local.env` to `.env` and fill in credentials:

```
PLANET_API_KEY=...
GOOGLE_AI_API_KEY=...
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

The `.env` file is git-ignored. Never commit it.
