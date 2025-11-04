# PNG Images for Label Studio

All PNG files are converted from GeoTIFF files in `data/raw_images/` for use in Label Studio.

## Image Specifications

### 1km Buffer Images (10m resolution)
- Dimensions: 217×201 pixels
- File size: ~128 KB each
- Files: `mongla_1km_2019.png`, `mongla_1km_2023.png`
- Use: Detailed feature labeling (solar panels, buildings, roads)

### 5km Buffer Images (10m resolution)
- Dimensions: 1079×1003 pixels
- File size: ~3.1 MB each
- Files: `mongla_5km_2019.png`, `mongla_5km_2023.png`
- Use: Regional land use patterns

### 10km Buffer Images (20m resolution)
- Dimensions: ~2000×2000 pixels
- File size: ~2-2.4 MB each
- Files: `mongla_2014.png` through `mongla_2023.png`
- Use: Broader regional context

## Conversion Details

All PNGs converted from GeoTIFF using `scripts/convert_to_png.py`:
- Lossless conversion (no compression)
- Full resolution preserved by default
- Percentile clipping (2-98%) for contrast enhancement
- RGB bands only (bands 1-3 from GeoTIFF)

## Label Studio Setup

Import these images into Label Studio:
1. Create new project: "Bangladesh Solar Land Use"
2. Choose template: "Semantic Segmentation with Polygons"
3. Import from: `data/for_labeling/`
4. Suggested labels: Agriculture, Forest, Water, Urban, Solar_Panels, Bare_Land

Start with 1km images for detailed labeling - they're smaller and faster to load.
