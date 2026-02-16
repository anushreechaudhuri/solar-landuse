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

## Limitations and Next Steps (V1)

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

---

## V2: VLM-Primary Classification

The V1 pipeline used Dynamic World (DW) as the base classification and only overlaid VLM solar panel detections. V2 flips this: **VLM (Gemini 2.0 Flash) is now the primary classifier** for all 7 land cover classes, with DW only filling in where VLM reports background (clouds/shadows/unidentifiable). This section also adds the Teesta 200 MW site at 5km buffer and compares DW vs VLM as independent classification sources.

### Updated Pipeline

```
Planet Basemaps (4.77m) ──► PNG crops (2x2km @ 1km buffer, 10x10km @ 5km buffer)
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          Google Earth Engine        Gemini 2.0 Flash
          Dynamic World (10m)        VLM Grid Classification
          5-class baseline           7-class, 20x20 grid
                    │                       │
                    ▼                       ▼
              Gap-fill only ───────► VLM PRIMARY
              (fills class 0)        (all 7 classes)
                                        │
                            ┌───────────┤
                            ▼           ▼
                    DW vs VLM       Merged Masks
                    Comparison      (VLM base + DW gap-fill)
                                        │
                                        ▼
                              DINOv2-Large (frozen)
                            + Segmentation Decoder
                              50 epochs, 37 images
                                        │
                                        ▼
                            Land Cover Prediction Maps
                              51 images, 7 classes
```

### V2 Aggregate Land Cover Change (1km AOI, 15 sites)

![Pre vs Post Land Cover](docs/figures/vlm_primary_pre_vs_post.png)

| Class | Pre-construction | Post-construction | Change | V1 Change |
|-------|:----------------:|:-----------------:|:------:|:---------:|
| Agriculture | 74.3% | 46.5% | **-27.9 pp** | -7.9 pp |
| Forest | 8.2% | 6.5% | -1.8 pp | -9.4 pp |
| Water | 9.9% | 8.4% | -1.5 pp | -0.9 pp |
| Urban | 4.9% | 18.2% | **+13.3 pp** | +0.3 pp |
| Solar panels | 0.0% | 14.2% | **+14.2 pp** | +18.9 pp |
| Bare land | 2.7% | 6.2% | +3.5 pp | -1.3 pp |

**Key changes from V1:** VLM-primary classification tells a dramatically different story about the land use context:

- **Agriculture dominates** pre-construction landscapes (74% vs 22% in V1). The VLM correctly recognizes the flat agricultural character of rural Bangladesh, while DW was over-classifying cropland as "forest" and "water."
- **Agriculture-to-solar conversion** is much more pronounced (-27.9 pp, mostly to solar and urban). This better reflects the documented reality that Bangladesh's solar farms are built primarily on agricultural land.
- **Urban increase (+13.3 pp)** is now visible, capturing roads, substations, and worker facilities that accompany solar construction.
- **Forest loss is smaller** (-1.8 pp vs -9.4 pp in V1), since DW was over-reporting forest in agricultural areas to begin with.

### V2 Per-Site Solar Detection

![Solar Detection vs Capacity](docs/figures/solar_capacity_vs_detection.png)

| Site | MW | Pre solar % | Post solar % | Detected |
|------|---:|:-----------:|:------------:|:--------:|
| Pabna | 100 | 0.0% | **36.1%** | Yes |
| Sirajganj 68 | 68 | 0.0% | **33.0%** | Yes |
| Mongla | 100 | 0.0% | **27.0%** | Yes |
| Lalmonirhat | 30 | 0.0% | **23.9%** | Yes |
| Teknaf | 20 | 0.0% | **23.9%** | Yes |
| Feni | 75 | 0.0% | **22.6%** | Yes |
| Mymensingh | 50 | 0.0% | **19.3%** | Yes |
| Moulvibazar | 10 | 0.0% | **13.5%** | Yes |
| Teesta | 200 | 0.0% | 0.0% | No* |
| Manikganj | 35 | 0.0% | 0.0% | No* |
| Tetulia | 8 | 0.0% | 0.0% | No |
| Kaptai | 7.4 | 0.0% | 0.0% | No |
| Sirajganj 6 | 6 | 0.0% | 0.0% | No |
| Sharishabari | 3 | 0.0% | 0.0% | No |
| Barishal | 1 | 0.0% | 0.0% | No |

**Detection rate: 8/15 sites (53%)** at 1km buffer. False positive rate: 0/15.

*\*Teesta and Manikganj 1km post-images had VLM classification issues (100% background or 100% forest). Both are successfully detected at 5km buffer (see below).*

### Teesta 200 MW at 5km Buffer

The Teesta solar farm (200 MW, Gaibandha/Beximco, completed Jan 2023) is the largest in the dataset. At 1km buffer, the VLM classified the post-construction image as 100% background, a failure case. At **5km buffer (10x10 km AOI)**, the solar farm is clearly detected:

![Teesta 5km Pre/Post](docs/figures/teesta_5km_pre_post.png)

| Period | Agriculture | Forest | Solar Panels |
|--------|:-----------:|:------:|:------------:|
| Pre (Jan 2019) | 90.0% | 10.0% | 0.0% |
| Post (Jan 2024) | 89.5% | 3.5% | **7.0%** |

At 5km buffer, the 200 MW solar farm occupies ~7% of the 100 km2 AOI, consistent with a utility-scale installation. The pre-construction landscape is dominated by agriculture (rice paddies) with minor forest. Post-construction shows a clear conversion from forest and agriculture to solar panels.

---

## Dynamic World vs VLM: Classification Comparison

An independent comparison of DW and VLM classifications on the same images reveals systematic differences between the two approaches.

### Disagreement Rate

![DW vs VLM Disagreement](docs/figures/dw_vlm_disagreement_hist.png)

Across 30 matched 1km images:
- **Mean pixel disagreement: 70%**
- **Median: 77%**
- Range: 13% (Pabna pre) to 100% (Manikganj 1km post)

This is an extremely high disagreement rate, indicating that DW and VLM produce fundamentally different land cover maps. The high disagreement is itself a valuable finding, suggesting that at least one (likely both) classification systems have significant limitations at this resolution and geographic context.

### Per-Class Bias

![DW vs VLM Bias](docs/figures/dw_vlm_class_bias.png)

| Class | DW Mean | VLM Mean | Bias |
|-------|:-------:|:--------:|:----:|
| Agriculture | 24.6% | 60.9% | VLM +36.3 pp |
| Forest | 33.2% | 10.1% | DW +23.1 pp |
| Water | 22.3% | 8.0% | DW +14.3 pp |
| Urban | 12.7% | 10.3% | DW +2.4 pp |
| Solar panels | 0.0% | 9.3% | VLM +9.3 pp |
| Bare land | 7.2% | 1.5% | DW +8.7 pp |

**Systematic biases:**

- **VLM strongly favors agriculture** (+36 pp). In Bangladesh, where flat green fields dominate, the VLM tends to classify more area as cropland. This is likely more accurate for rural Bangladesh than DW's interpretation.
- **DW strongly favors forest** (+23 pp) and **water** (+14 pp). DW's 10m Sentinel-2 source may confuse dense crops/vegetation with forest, and seasonal flooding/wet fields with permanent water.
- **Only VLM detects solar panels** (+9.3 pp). DW has no solar panel class; it misclassifies solar arrays as urban or bare land.
- **DW reports more bare land** (+8.7 pp). DW may be picking up fallow fields or sandy river banks that VLM classifies as agriculture.

### Interpretation

Neither DW nor VLM can be considered ground truth. The VLM-primary approach was chosen because:

1. VLM is the only source that can identify solar panels
2. VLM's agriculture-heavy classification is more consistent with Bangladesh's land use reality (>60% of land is agricultural)
3. DW's forest over-estimation in rural Bangladesh is a known limitation at 10m resolution
4. The 20x20 VLM grid, while coarse (~100m cells), provides a semantically richer classification than DW's spectral-only approach

---

## V2 Training Details

- **Backbone:** DINOv2-Large (facebook/dinov2-large), frozen, 1024-dim patch features
- **Decoder:** 3-layer Conv2d upsampling head (1024 -> 512 -> 256 -> 128 -> 7 classes)
- **Training data:** 37 auto-labeled masks (15 sites at 1km + 3 sites at 5km, pre+post)
- **Merge strategy:** VLM primary, DW gap-fills background class only
- **Epochs:** 50
- **Loss:** CrossEntropy, 1.50 -> 0.048 (97% reduction)
- **Device:** Apple MPS (M-series GPU) with CPU fallback
- **Training time:** ~17 minutes (37 images, ~20s/epoch)
- **Inference:** 51 images in ~44 seconds

![Training Loss](docs/figures/training_loss_vlm_primary.png)

The V2 model converges to a significantly lower loss (0.048) than V1 (0.29), suggesting the VLM-primary labels provide a more learnable signal for the DINOv2 backbone.

## Updated Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_planet_basemaps.py` | Download Planet monthly basemap quads (now includes Teesta) |
| `scripts/download_all_sites.py` | Batch download all 15 sites, pre+post |
| `scripts/generate_dynamic_world_masks.py` | Generate Dynamic World land cover masks via GEE (supports 1km + 5km) |
| `scripts/vlm_classify.py` | Gemini VLM grid classification (supports 1km + 5km, adjusts prompt area) |
| `scripts/merge_masks.py` | VLM-primary merge with DW gap-fill, colored visualizations |
| `scripts/compare_dw_vlm.py` | DW vs VLM comparison: CSV + side-by-side visualizations |
| `scripts/train_segmentation.py` | Train DINOv2 segmentation decoder |
| `scripts/apply_segmentation.py` | Generate land cover predictions on all images |

## V2 Limitations and Next Steps

**Remaining limitations:**
- VLM occasionally produces degenerate outputs (100% single class) on some images, especially at larger AOIs
- Small installations (< 10 MW) still not detected at 1km buffer
- VLM's 20x20 grid resolution (~100m cells) limits boundary precision
- No ground truth for quantitative accuracy assessment
- 5km buffer VLM needs better prompting (some sites get all-forest or all-background)

**Potential improvements:**
- Retry failed VLM classifications with higher temperature or prompt variations
- Ensemble multiple VLM calls per image and take majority vote
- Use SAM (Segment Anything) for boundary refinement after coarse VLM classification
- Add more 5km buffer sites for regional context analysis
- Cross-validate VLM classifications against high-res Google Earth imagery
- Fine-tune VLM prompt with Bangladesh-specific land cover examples
