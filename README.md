# Solar Land Use Change Detection - Bangladesh

Downloads satellite imagery for solar project sites in Bangladesh and converts it for labeling. Uses Google Earth Engine for data download and Label Studio for annotation.

## Setup

### Environment

```bash
# Clone the repository
git clone https://github.com/anushreechaudhuri/solar-landuse.git
cd solar-landuse

# Create conda environment
conda create -n solar-landuse python=3.10 -y
conda activate solar-landuse

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp local.env .env
# Edit .env and add your AWS credentials and Google Earth Engine project name

# Authenticate with Google Earth Engine (one-time)
python
>>> import ee
>>> ee.Authenticate()
>>> ee.Initialize(project="bangladesh-solar")
```

## Usage

### Download Satellite Imagery

```bash
# Download specific years with multiple buffer sizes
python scripts/download_satellite_images.py --years 2019 2023

# Upload to S3 after download
python scripts/download_satellite_images.py --years 2019 2023 --upload-s3
```

The script downloads at 10m resolution for Sentinel-2 data. If downloads fail due to size limits, it falls back to 20m resolution.

### Convert to PNG

```bash
# Convert GeoTIFF to PNG
python scripts/convert_to_png.py
# Choose resolution option when prompted (option 1 for full resolution)
```

### Sync to S3

```bash
# Upload local files to S3
python scripts/sync_to_s3.py
```

### Label Studio

```bash
label-studio start
# Access at http://localhost:8080
```

Import images from `data/for_labeling/` and create a semantic segmentation project.

## Project Structure

```
solar-landuse/
├── data/
│   ├── raw_images/           # GeoTIFF files
│   ├── for_labeling/         # PNG files for Label Studio
│   ├── labels/               # Exported annotations
│   └── processed/masks/      # Segmentation masks
├── scripts/
│   ├── download_satellite_images.py  # Download imagery
│   ├── convert_to_png.py            # Convert to PNG
│   ├── s3_utils.py                   # AWS S3 utilities
│   └── sync_to_s3.py                 # Sync to S3
├── local.env                 # Template for .env (copy to .env)
├── .env                      # Your credentials (not in git)
└── requirements.txt
```

## Current State

### Site: Mongla Solar Project, Bangladesh

Coordinates: 22°34'25.0"N 89°34'10.4"E

### Available Data

**1km buffer images (10m resolution, 217×201 pixels)**:
- `mongla_1km_2019.tif` / `.png` - Pre-development
- `mongla_1km_2023.tif` / `.png` - Post-development

**5km buffer images (10m resolution, 1079×1003 pixels)**:
- `mongla_5km_2019.tif` / `.png` - Pre-development
- `mongla_5km_2023.tif` / `.png` - Post-development

**10km buffer images (20m resolution, ~2000×2000 pixels)**:
- `mongla_2014.tif` / `.png` through `mongla_2023.tif` / `.png`
- Years: 2014, 2016-2017, 2019-2023 (some years missing due to data availability)

### Files

- 13 GeoTIFF files in `data/raw_images/`
- 14 PNG files in `data/for_labeling/`
- Documentation: `data/raw_images/DATA_SOURCES.md` (describes each file's data source, bands, resolution, etc.)

All files are backed up to S3 at `s3://anuc-satellite-analysis/data/`

## Environment Variables

Copy `local.env` to `.env` and fill in your credentials:

```bash
AWS_DEFAULT_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your-key-here"
AWS_SECRET_ACCESS_KEY="your-secret-here"
```

The `.env` file is git-ignored. Never commit it.

## S3 Integration

Files are stored in S3 for backup and team sharing. Download from S3:

```bash
aws s3 sync s3://anuc-satellite-analysis/data/ ./data/
```

Upload new files:

```bash
python scripts/sync_to_s3.py
```

## Common Issues

**Earth Engine not initialized**
```bash
python
>>> import ee
>>> ee.Authenticate()
>>> ee.Initialize(project="bangladesh-solar")
```

**No images found for year**
Normal if data not available for that year/location. Try different years.

**Large files won't upload**
Scripts skip existing files. Use `--upload-s3` flag or `sync_to_s3.py` for S3 backup.

## Data Sources

- Sentinel-2 MSI: 10m resolution RGB+NIR bands
  - Dataset: `COPERNICUS/S2_SR_HARMONIZED`
  - Years: 2015-2024
- Landsat 8 OLI: 30m resolution (fallback for earlier years)
  - Dataset: `LANDSAT/LC08/C02/T1_L2`
  - Years: 2013-2024

## Notes

- Large files (`.tif`, `.png`) are git-ignored. Download from S3 instead.
- Label Studio needs local files - keep them even with S3 sync.
- Start with 1km images for detailed labeling, they're smaller and easier to work with.
