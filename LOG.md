# Change Log

Detailed record of all changes made to the project.

## October 2025

### 2025-10-26: Initial Project Setup
- Created project structure with data directories (raw_images, for_labeling, labels, processed/masks)
- Created scripts directory with download and conversion scripts
- Set up conda environment `solar-landuse` with Python 3.10
- Installed dependencies: earthengine-api, geemap, rasterio, pillow, numpy, label-studio

### 2025-10-26: Google Earth Engine Integration
- Created `scripts/download_satellite_images.py` - downloads Sentinel-2 and Landsat 8 imagery
- Configured for Mongla site: 22.573611°N, 89.569444°E
- Initial download: 10km buffer images at 20m resolution (2014-2023, some years missing)
- Updated to use `bangladesh-solar` project for Earth Engine initialization

### 2025-10-26: High-Resolution Downloads
- Modified download script to support 1km and 5km buffer sizes
- Changed resolution to 10m for Sentinel-2 (from 20m)
- Updated file naming: `mongla_{buffer}_{year}.tif`
- Successfully downloaded 1km and 5km images for 2019 and 2023

### 2025-10-26: PNG Conversion
- Created `scripts/convert_to_png.py` - converts GeoTIFF to PNG for Label Studio
- Lossless conversion with no compression
- Full resolution preserved by default
- Added percentile clipping for contrast enhancement

### 2025-10-26: AWS S3 Integration
- Created `scripts/s3_utils.py` - S3 utilities for file storage
- Created `scripts/sync_to_s3.py` - syncs local files to S3
- Added `--upload-s3` flag to download script
- All existing files uploaded to `s3://anuc-satellite-analysis/data/`
- Created `local.env` template for AWS credentials
- Configured `.gitignore` to exclude `.env` and large files

### 2025-10-26: Documentation Updates
- Consolidated all documentation into README.md
- Removed emojis and promotional language
- Added change log section to README
- Created `data/raw_images/DATA_SOURCES.md` - technical documentation for each file
- Created `data/for_labeling/README.md` - image specifications

### 2025-10-27: GitHub Repository Setup
- Initialized git repository
- Created `.gitignore` to exclude large files and credentials
- Pushed to GitHub: https://github.com/anushreechaudhuri/solar-landuse
- Repository structure: code and documentation only, data in S3

### 2025-11-04: Training Pipeline
- Created `scripts/train_segmentation.py` - trains DINOv3-based segmentation model
- Created `scripts/apply_segmentation.py` - applies trained model to generate maps
- Added torch, torchvision, transformers, tqdm to requirements.txt
- Created `data/training_dataset/` structure with `classes.json`
- Model uses frozen DINOv3 backbone (facebook/dinov2-vitl16-pretrain-sat493m)
- Only segmentation decoder head is trained
- Updated README with training section

### 2025-11-05: Project Cleanup
- Deleted `scripts/download_planetary_nicfi.py` (not using Planetary Computer)
- Deleted `data/planetary_computer/` directory
- Deleted `notebooks/` directory (empty, not used)
- Kept `models/` directory (used for trained weights)
- Created `LOG.md` for detailed change tracking
- Created `memory.md` for future chat reference (gitignored)
- Added `data/for_labeling/LABEL_STUDIO_IMPORT.md` for import instructions

