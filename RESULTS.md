# Solar Land Use Change Detection - Results

Automated land cover classification pipeline for 15 solar project sites across Bangladesh, using satellite imagery (Planet Basemaps, 4.77m resolution) to detect land use changes from solar farm construction.

## Pipeline Overview

```
Planet Basemaps (4.77m) ──► PNG crops (2x2km per site)
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          Google Earth Engine        Gemini 2.0 Flash
          Dynamic World (10m)        VLM Grid Classification
          5-class land cover         7-class, 20x20 grid
                    │                       │
                    └───────────┬───────────┘
                                ▼
                        Merged Masks
                    (DW base + VLM solar)
                                │
                                ▼
                      DINOv2-Large (frozen)
                    + Segmentation Decoder
                      50 epochs, MPS GPU
                                │
                                ▼
                    Land Cover Prediction Maps
                      49 images, 7 classes
```

## Sites

15 solar installations totaling 713 MW across Bangladesh, ranging from 1 MW (Barishal) to 200 MW (Teesta/Gaibandha).

Each site has a **pre-construction** and **post-construction** image at 1km buffer (2x2km AOI), plus select sites at 5km buffer (10x10km AOI).

| Site | Capacity | Pre-construction | Post-construction |
|------|----------|-----------------|-------------------|
| Teesta (Gaibandha) | 200 MW | Jan 2019 | Jan 2024 |
| Pabna | 100 MW | Jan 2021 | Jan 2026 |
| Mongla | 100 MW | Jan 2018 | Jan 2023 |
| Feni | 75 MW | Jan 2020 | Jan 2026 |
| Sirajganj 68MW | 68 MW | Jan 2021 | Jan 2026 |
| Mymensingh | 50 MW | Feb 2017 | Jan 2022 |
| Manikganj (Spectra) | 35 MW | Feb 2017 | Jan 2023 |
| Lalmonirhat | 30 MW | Jan 2019 | Jan 2024 |
| Teknaf | 20 MW | Jan 2016 | Jan 2020 |
| Moulvibazar | 10 MW | Jan 2022 | Jan 2026 |
| Tetulia | 8 MW | Dec 2016 | Jan 2021 |
| Kaptai | 7.4 MW | Jan 2016 | Jan 2021 |
| Sirajganj 6MW | 6 MW | Feb 2017 | Jan 2023 |
| Sharishabari | 3 MW | Jan 2016 | Jan 2019 |
| Barishal | 1 MW | Jan 2021 | Jan 2026 |

## Classification Results

### Aggregate Land Cover Change (1km AOI, 15 sites)

| Class | Pre-construction | Post-construction | Change |
|-------|:----------------:|:-----------------:|:------:|
| Agriculture | 22.0% | 14.1% | **-7.9 pp** |
| Forest | 35.6% | 26.2% | **-9.4 pp** |
| Water | 18.9% | 18.0% | -0.9 pp |
| Urban | 14.3% | 14.7% | +0.3 pp |
| Solar panels | 0.0% | 18.9% | **+18.9 pp** |
| Bare land | 9.1% | 7.8% | -1.3 pp |

Key finding: Solar panels account for ~19% of the 2x2km AOI in post-construction images on average. The land converted to solar comes primarily from **agriculture (-7.9 pp)** and **forest (-9.4 pp)**.

### Per-Site Solar Detection

| Site | MW | Pre solar % | Post solar % | Detected |
|------|---:|:-----------:|:------------:|:--------:|
| Teesta | 200 | 0.0% | **78.0%** | Yes |
| Pabna | 100 | 0.0% | **38.3%** | Yes |
| Sirajganj 68 | 68 | 0.0% | **35.8%** | Yes |
| Mongla | 100 | 0.0% | **31.4%** | Yes |
| Feni | 75 | 0.0% | **24.7%** | Yes |
| Lalmonirhat | 30 | 0.0% | **22.3%** | Yes |
| Teknaf | 20 | 0.0% | **21.4%** | Yes |
| Mymensingh | 50 | 0.0% | **21.0%** | Yes |
| Moulvibazar | 10 | 0.0% | 5.8% | Yes |
| Manikganj | 35 | 0.7% | 5.1% | Yes |
| Tetulia | 8 | 0.0% | 0.0% | No |
| Kaptai | 7.4 | 0.0% | 0.0% | No |
| Sirajganj 6 | 6 | 0.0% | 0.0% | No |
| Sharishabari | 3 | 0.0% | 0.0% | No |
| Barishal | 1 | 0.0% | 0.0% | No |

**Detection rate: 10/15 sites (67%)** with >1% solar in post-construction images.

**False positive rate: 0/15** -- no pre-construction images falsely identified solar panels.

**Detection threshold:** All sites >= 10 MW were detected. All missed sites are < 10 MW. This suggests the 4.77m resolution + 2x2km AOI can reliably detect installations of ~10 MW and above.

### Correlation: Capacity vs Detection

```
Solar % in post-construction image vs. installed capacity:

80% |  *  Teesta (200 MW)
    |
    |
40% |     * Pabna    * Sirajganj68
    |     * Mongla
    |  * Feni  * Lalmonirhat  * Teknaf  * Mymensingh
20% |
    |
    |  * Moulvibazar  * Manikganj
 5% |
    |  x Tetulia  x Kaptai  x Sirajganj6  x Sharishabari  x Barishal
 0% +----+----+----+----+----+----+----+----+----+----
    0   20   40   60   80  100  120  140  160  180  200 MW

    * = detected (>1% solar)    x = missed
```

## Training Details

- **Backbone:** DINOv2-Large (facebook/dinov2-large), frozen, 1024-dim patch features
- **Decoder:** 3-layer Conv2d upsampling head (1024 -> 512 -> 256 -> 128 -> 7 classes)
- **Training data:** 30 auto-labeled masks (15 sites x 2 periods), generated via Dynamic World + Gemini VLM
- **Epochs:** 50
- **Loss:** CrossEntropy, 1.82 -> 0.29 (84% reduction)
- **Device:** Apple MPS (M-series GPU) with CPU fallback
- **Training time:** ~12 minutes
- **Inference:** 49 images in ~43 seconds

```
Training Loss Curve:

1.8 |*
    | *
1.4 |  *
    |   **
1.0 |     ***
    |        ****
0.6 |            *****
    |                 ********
0.3 |                         *************
    +----+----+----+----+----+----+----+----+----+----
    0    5   10   15   20   25   30   35   40   45   50
                        Epoch
```

## Auto-Labeling Pipeline

The training masks were generated without any manual annotation:

1. **Dynamic World (GEE):** Free, 10m land cover from Sentinel-2. Provides baseline for agriculture, forest, water, urban, bare land. Cannot distinguish solar panels.

2. **Gemini VLM:** Each image sent to Gemini 2.0 Flash with structured prompt. Returns 20x20 grid (400 cells) of class predictions. Key for solar panel identification. Free tier (15 RPM), ~4 sec between calls.

3. **Merge:** DW provides spatial detail at 10m. VLM provides solar panel class. Final mask uses DW as base, VLM overrides where it identifies solar (class 5). Pre-construction images never get solar labels.

Total API cost: **$0** (GEE free, Gemini free tier).

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_planet_basemaps.py` | Download Planet monthly basemap quads |
| `scripts/download_all_sites.py` | Batch download all 15 sites, pre+post |
| `scripts/generate_dynamic_world_masks.py` | Generate Dynamic World land cover masks via GEE |
| `scripts/vlm_classify.py` | Gemini VLM grid classification for solar detection |
| `scripts/merge_masks.py` | Merge DW + VLM masks, generate colored visualizations |
| `scripts/train_segmentation.py` | Train DINOv2 segmentation decoder |
| `scripts/apply_segmentation.py` | Generate land cover predictions on all images |

## Limitations and Next Steps

**Current limitations:**
- Small installations (< 10 MW) not detected at this resolution and AOI size
- Auto-labels from VLM are coarse (20x20 grid = ~100m cells) -- fine boundaries are approximate
- Single timestamp per period -- seasonal variation not captured
- No validation against ground truth (no manually annotated masks)

**Potential improvements:**
- Higher resolution imagery (PlanetScope 3m scenes) for small sites
- Manual review of auto-generated masks in Label Studio
- Multi-temporal analysis (monthly time series instead of single pre/post)
- Larger AOI (5km buffer) for regional land use context
- Cross-validation with official SREDA solar farm registry data
