# Solar Land Use Change Detection

Analyzes land use change at solar energy project sites in South Asia using satellite imagery, multiple global LULC datasets, and VLM-based classification. Uses Planet Basemaps (4.77m), Google Earth Engine datasets (10m), and Gemini 2.0 Flash for classification.

Includes a **polygon verification web app** for labelers to confirm/edit solar installation boundaries across ~5,000 South Asian projects, browse 628 unmatched GRW solar detections, and merge them with GEM projects.

Also includes a **LULC labeling studio** (`/label`) for hand-labeling satellite images with 10 land use classes using polygon annotation tools, with pre-loaded solar installation boundaries for post-construction images.

## Current Scope

- **5,093 utility-scale solar phases** across 6 South Asian countries (India, Bangladesh, Pakistan, Nepal, Sri Lanka, Bhutan)
- **3,957 GRW solar installation polygons** matched against GSPT project coordinates
- **Match results**: 1,829 high confidence / 1,132 medium / 269 low / 1,863 unmatched
- **Pre/post construction** imagery comparison (2016-2026) for 15 Bangladesh test sites
- **5 classification sources**: Dynamic World, ESA WorldCover, ESRI LULC, GLAD GLCLUC, VLM V2 (Gemini)
- **10-class unified scheme**: cropland, trees, shrub, grassland, flooded veg, built, bare, water, snow, no data

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

## Polygon Verification Web App

A Next.js web app for labelers to verify and edit solar installation polygons matched from GSPT and GRW datasets.

### Features

- **Three-tab interface**: Active Projects, Proposed/Other, and GRW Unmatched (628 satellite-detected solar features with no GEM match)
- **Overview map**: Toggle to see all 5,721 points color-coded (green=matched, orange=GEM-only, purple=GRW-only) with canvas rendering
- **Google Satellite map** with Leaflet polygon overlay and Leaflet.Draw for editing/drawing
- **Search & filter**: by name, ID, capacity, country, confidence level, review status
- **Review actions**: confirm polygon, mark no match, edit polygon, draw new polygon, feasibility assessment
- **Merge workflow**: Link unmatched GRW polygons to GEM projects (or vice versa) with nearby-search, undo support via merge history
- **GRW feature editing**: Add name, capacity, status, notes to unmatched satellite detections
- **Coordinates**: click-to-copy in decimal degrees and DMS format
- **Reviewer tracking**: name persisted in localStorage, append-only review log in Postgres

## LULC Labeling Studio

A Leaflet-based annotation interface at `/label` for hand-labeling satellite images with 10 LULC classes.

### Features

- **51 satellite images** across 15 Bangladesh solar sites (pre/post construction, 1km and 5km buffers)
- **10 LULC classes**: Cropland, Trees, Shrub, Grassland, Flooded Veg, Built, Bare, Water, Snow/Ice, No Data
- **Polygon annotation** with Leaflet.Draw: draw, edit, delete polygons
- **Solar polygon overlay**: Pre-loaded GRW solar installation boundaries shown as dashed red outlines on post-construction images
- **Class assignment**: Select class, draw polygon; click existing polygon to reassign class
- **Keyboard shortcuts**: 1-9, 0 for class selection; Ctrl+S to save
- **Per-annotator persistence**: One annotation set per task per annotator, auto-loaded on revisit
- **Export endpoint**: `GET /api/labeling/export` returns all annotations as JSON for ML training
- **Images served from S3**: 51 PNGs stored in S3, served via presigned URLs (no data files in repo)

### Labeling Setup

```bash
# Upload images to S3 and seed labeling tasks
python3 scripts/seed_labeling.py
# Uploads 51 PNGs to S3, creates labeling_tasks + labeling_annotations tables
# Requires POSTGRES_URL and AWS credentials in .env
```

### Pre-processing Pipeline

Run these scripts in order to generate the matching data:

```bash
# 1. Extract GSPT South Asia projects from Excel tracker
python3 scripts/extract_gspt_south_asia.py
# Output: data/gspt_south_asia.json (5,093 projects)

# 2. Query GRW polygons from Google Earth Engine
python3 scripts/query_grw_south_asia.py
# Output: data/grw_south_asia.geojson (3,957 polygons)

# 3. Match GSPT projects with GRW polygons
python3 scripts/match_gspt_grw.py
# Output: data/projects_merged.json (5,093 matched records)
# Output: data/grw_unmatched.geojson (628 unmatched GRW features)

# 4. Seed Vercel Postgres database (projects + GRW features)
python3 scripts/seed_database.py --skip-s3
# Requires POSTGRES_URL env var
# Creates: projects (5,093), grw_features (628), reviews, merge_history tables
```

### Web App Local Development

```bash
cd webapp
npm install
# Set POSTGRES_URL in .env.local (from Vercel dashboard)
npm run dev
# Open http://localhost:3000
```

### Deployment (Vercel)

1. Create Vercel project from this repo's `webapp/` directory
2. Add Vercel Postgres storage in dashboard
3. Set `POSTGRES_URL` env var (auto-set with Postgres storage)
4. Run `python3 scripts/seed_database.py` with the Postgres URL
5. Deploy via `git push`

## Key Scripts

### Data Download

| Script | Purpose |
|--------|---------|
| `scripts/download_planet_basemaps.py` | Download Planet monthly basemap quads (4.77m) for all sites |
| `scripts/download_satellite_images.py` | Download Sentinel-2 imagery via GEE (10m) |

### Pre-processing (South Asia Scale-Up)

| Script | Purpose |
|--------|---------|
| `scripts/extract_gspt_south_asia.py` | Extract & filter GSPT tracker for South Asia (5,093 phases) |
| `scripts/query_grw_south_asia.py` | Query GRW solar polygons from GEE (3,957 features) |
| `scripts/match_gspt_grw.py` | Spatial matching of GSPT coords to GRW polygons |
| `scripts/seed_database.py` | Seed Vercel Postgres from matched data |
| `scripts/seed_labeling.py` | Upload images to S3 + seed labeling tasks |

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
│   ├── Global-Solar-Power-Tracker-February-2026.xlsx  # GSPT source (gitignored)
│   ├── gspt_south_asia.json       # Extracted South Asia projects (gitignored)
│   ├── grw_south_asia.geojson     # GRW polygons from GEE (gitignored)
│   ├── projects_merged.json       # Matched GSPT+GRW data (gitignored)
│   ├── grw_unmatched.geojson     # Unmatched GRW features (gitignored)
│   ├── grw/                       # Bangladesh GRW polygon matches + HTML tools
│   ├── raw_images/                # GeoTIFF files (Planet basemaps)
│   ├── for_labeling/              # PNG files for labeling
│   ├── training_dataset/          # DINOv2 training images and masks
│   ├── vlm_v2_responses/          # Cached Gemini classification results
│   ├── lulc_raw_cache/            # Cached GEE dataset values (.npz)
│   └── lulc_comparison/           # Per-image LULC visualizations
├── webapp/                        # Next.js web app (polygon verification + LULC labeling)
│   ├── app/                       # App Router pages & API routes
│   │   ├── api/                   # projects, grw-features, merge, nearby, overview, stats, labeling
│   │   └── label/                 # LULC labeling studio page
│   ├── components/                # React components (Map, ProjectList, LabelingApp, etc.)
│   ├── lib/                       # Database queries, TypeScript types, S3 helpers
│   └── package.json
├── scripts/
│   ├── extract_gspt_south_asia.py # GSPT extraction for South Asia
│   ├── query_grw_south_asia.py    # GRW polygon query from GEE
│   ├── match_gspt_grw.py          # GSPT-GRW spatial matching
│   ├── seed_database.py           # Vercel Postgres seeder
│   ├── compare_lulc_datasets.py   # Main V3 analysis pipeline
│   ├── figure_style.py            # Publication figure styling module
│   ├── vlm_classify_v2.py         # Gemini 2.0 Flash classifier
│   ├── download_planet_basemaps.py
│   ├── train_segmentation.py
│   └── apply_segmentation.py
├── models/                        # Trained model weights
├── docs/figures/                  # Publication-quality figures
├── RESULTS.md                     # Analysis results and findings
├── LOG.md                         # Detailed change log
└── requirements.txt
```

## Key Findings

1. **Cropland is the primary pre-solar land cover** across all datasets and VLM
2. **Only Dynamic World and VLM V2** provide true temporal change detection (WC/GLAD are static snapshots)
3. **DW detects cropland-to-built conversion** at solar sites (no solar class, so panels appear as built/bare/snow)
4. **VLM V2 is polygon-aware** for post-construction images, avoiding the solar-as-built misclassification
5. **Cross-dataset agreement is moderate** — cropland is most consistently identified, other classes vary

## Data Sources

- **GSPT**: Global Solar Power Tracker (February 2026) — 5,093 South Asia utility-scale phases
- **GRW**: Global Renewables Watch solar polygons via GEE (3,957 South Asia features)
- **Planet Basemaps**: 4.77m monthly mosaics (Jan 2016 - Jan 2026) via Basemaps API
- **Dynamic World**: 10m per-date composite via GEE
- **ESA WorldCover**: 10m single snapshot (2021) via GEE
- **ESRI LULC**: 10m annual (2017-2024) via GEE (sat-io)
- **GLAD GLCLUC**: 30m single snapshot (2020) via GEE
- **VLM V2**: Gemini 2.0 Flash percentage-based classification per image

## Environment Variables

Copy `local.env` to `.env` and fill in credentials:

```
PLANET_API_KEY=...
GOOGLE_AI_API_KEY=...
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

For the web app, also set `POSTGRES_URL` (from Vercel Postgres dashboard).

The `.env` file is git-ignored. Never commit it.
