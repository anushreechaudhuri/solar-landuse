# Training Dataset

This directory contains labeled images and masks for training the segmentation model.

## Directory Structure

```
training_dataset/
├── images/          # GeoTIFF files (input images)
├── masks/           # PNG masks (pixel values = class IDs)
└── classes.json     # Class name to ID mapping
```

## File Naming Convention

- Images: `{site}_{buffer}_{year}.tif` (e.g., `mongla_5km_2019.tif`)
- Masks: `{site}_{buffer}_{year}_mask.png` (e.g., `mongla_5km_2019_mask.png`)

## Mask Format

Masks should be PNG files with pixel values corresponding to class IDs:
- 0: background
- 1: agriculture
- 2: forest
- 3: water
- 4: urban
- 5: solar_panels
- 6: bare_land

Single channel PNG files (grayscale) are expected. RGB masks will be converted to grayscale by taking the first channel.

## Creating Masks from Label Studio

Export masks from Label Studio as PNG files with class IDs as pixel values. Ensure filenames match the image naming convention (image stem + `_mask.png`).

