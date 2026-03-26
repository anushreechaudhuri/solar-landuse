# Solar Land Use Change Detection

Analyzes land use change at utility-scale solar energy project sites across South Asia using satellite imagery, global LULC datasets, VLM-based classification (Gemini 2.5 Flash), and within-site event study econometrics.

## Current Scope

- **3,676 unified solar sites** across 5 South Asian countries (India, Bangladesh, Pakistan, Nepal, Sri Lanka) from 3-way spatial matching of GEM/GSPT, GRW, and TZ-SAM databases
- **36,760-row annual panel** (3,676 sites × 10 years, 2016-2025) with DW LULC, NDVI, VIIRS NTL, SAR, LST, EVI, WorldPop, Buildings, and VLM classifications
- **28,854 VLM classifications** (Gemini 2.5 Flash) from Sentinel-2 imagery — 75% TP / 11% FP solar detection at >5% threshold
- **Within-site event study**: Two-way FE (site + year), clustered SEs, pre/post construction comparison
- **Key finding**: Solar panels predominantly replace bare/fallow land (+10 pp solar, -7 pp bare ground post-construction); cropland loss in surrounding buffer zone is modest (-0.8 pp in balanced sample)
- **Pipeline runs on Modal** (GEE + Gemini API) for laptop-independent batch processing

## Pipeline Architecture

```
Unified Solar DB (3,676 sites)
         │
         ├─► GEE: Dynamic World annual compositions + NDVI  ──► annual_panel.csv
         ├─► GEE: VIIRS/SAR/LST/EVI/WorldPop/Buildings     ──► eo_annual_panel.csv
         ├─► GEE: Sentinel-2 RGB thumbnails                 ──► s2_images/
         └─► Gemini 2.5 Flash: 10-class LULC + solar        ──► vlm_results/
                                                                     │
                          ┌──────────────────────────────────────────┘
                          ▼
              build_full_panel.py ──► full_panel.csv (36,760 × 50+ columns)
                          │
                          ├─► analyze_vlm_results.py (detection rates, DW comparison)
                          └─► event_study_annual.py (within-site TWFE event study)
```

All stages run on Modal (`scripts/modal_pipeline.py --stage dw|eo|s2|vlm|all`).

## Setup

```bash
git clone https://github.com/anushreechaudhuri/solar-landuse.git
cd solar-landuse
pip3 install -r requirements.txt

cp local.env .env
# Edit .env: PLANET_API_KEY, GOOGLE_AI_API_KEY, AWS credentials

# Authenticate with Google Earth Engine (one-time)
python3 -c "import ee; ee.Authenticate(); ee.Initialize(project='bangladesh-solar')"
```

## Key Scripts

### Modal Pipeline (Full Dataset)

| Script | Purpose |
|--------|---------|
| `scripts/modal_pipeline.py` | Main pipeline: DW, EO, S2, VLM stages on Modal |
| `scripts/build_full_panel.py` | Merge DW + EO + VLM into unified panel, cache intermediates |
| `scripts/analyze_vlm_results.py` | VLM analysis: detection rates, DW cross-validation, figures |
| `scripts/event_study_annual.py` | Within-site TWFE event study (DW + VLM outcomes) |

### Data Collection

| Script | Purpose |
|--------|---------|
| `scripts/download_planet_basemaps.py` | Download Planet monthly basemap quads (4.77m) |
| `scripts/integrate_solar_datasets.py` | 3-way spatial matching (GEM + GRW + TZ-SAM) |
| `scripts/collect_temporal_data.py` | Multi-temporal panel collection from 7 EO datasets |

### Analysis & Classification

| Script | Purpose |
|--------|---------|
| `scripts/compare_lulc_datasets.py` | Multi-dataset LULC comparison (DW, WorldCover, ESRI, GLAD) |
| `scripts/vlm_classify_v2.py` | VLM classification using Gemini (10-class, polygon-aware) |
| `scripts/vlm_model_comparison.py` | VLM model selection (Gemini 2.0 vs 2.5 vs 3 Flash) |
| `scripts/figure_style.py` | Figure styling (Paul Tol colorblind-safe palette) |
| `scripts/build_conflict_data.py` | Scrape + geocode land conflict reports |
| `scripts/match_lcw_conflicts.py` | Match conflict reports to solar site IDs |

### Legacy (Bangladesh Case Studies)

| Script | Purpose |
|--------|---------|
| `scripts/run_did_analysis.py` | DiD regression (superseded by event study) |
| `scripts/train_segmentation.py` | DINOv2-based segmentation head training |

## Polygon Verification Web App

A Next.js web app (`webapp/`) for labelers to verify and edit solar installation polygons.

```bash
cd webapp && npm install && npm run dev
# Requires POSTGRES_URL in .env.local
```

## Key Findings

### Event Study (3,469 sites, 2016-2025)

1. **VLM detects solar at scale**: 75% TP rate, 11% FP rate at >5% threshold across 3,017 sites. Sharp step-change at construction year with clean pre-trends (p=0.430).

2. **Bare-to-solar is the dominant transition**: VLM shows +10.0 pp solar panels and -7.3 pp bare ground post-construction. Solar farms predominantly replace bare/fallow land.

3. **Cropland loss is modest**: -0.82 pp (DW, balanced sample with clean pre-trends, p=0.641). The larger full-sample estimate (-3.95 pp) is inflated by pre-existing land conversion trends.

4. **Built-up increase precedes construction**: +1.48 pp, beginning before recorded construction year — reflects infrastructure development (roads, substations, site preparation).

5. **DW and VLM are complementary**: DW provides multi-temporal spectral LULC; VLM adds solar identification that standard LULC products do not include by design.

### VLM vs Dynamic World Cross-Validation

| Class | Pearson r | VLM mean | DW mean |
|-------|-----------|----------|---------|
| Built-up | 0.836 | 8.7% | 12.3% |
| Trees | 0.683 | 7.1% | 12.8% |
| Water | 0.680 | 2.8% | 1.7% |
| Cropland | 0.593 | 20.3% | 46.3% |
| Bare ground | 0.404 | 37.2% | 8.7% |

Systematic differences reflect definitional divergence (DW uses multi-temporal spectral signatures; VLM classifies single dry-season RGB images), not classification errors.

## Data Sources

### Solar Databases
- **GEM/GSPT**: Global Solar Power Tracker (Feb 2026) — utility-scale phases
- **GRW**: Global Renewables Watch polygons via GEE
- **TZ-SAM**: Transition Zero solar polygons via GEE

### Earth Observation
- **Dynamic World**: 10m LULC composition (2016-2025)
- **Sentinel-2**: 10m RGB imagery for VLM classification
- **VIIRS NTL**: 463m nighttime light radiance
- **Sentinel-1 SAR**: 10m VV/VH backscatter
- **MODIS**: 250m NDVI/EVI, 1km LST
- **WorldPop**: 100m population density
- **Google Open Buildings**: 2.5m building presence/height

### VLM Classification
- **Gemini 2.5 Flash**: 10-class LULC + solar panel detection from S2 thumbnails

## Environment Variables

Copy `local.env` to `.env` and fill in:

```
PLANET_API_KEY=...
GOOGLE_AI_API_KEY=...
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

The `.env` file is git-ignored. Never commit it.
