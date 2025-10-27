# Solar Land Use Change Detection - Bangladesh

This project downloads high-resolution multi-year satellite imagery for solar project sites in Bangladesh and prepares them for land use change detection labeling using semantic segmentation.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd solar-landuse

# Create and activate conda environment
conda create -n solar-landuse python=3.10 -y
conda activate solar-landuse

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (one-time setup)
python
>>> import ee
>>> ee.Authenticate()
>>> ee.Initialize(project="bangladesh-solar")
```

### 2. Download High-Resolution Satellite Imagery

```bash
# Download 1km and 5km buffer images for specific years
python scripts/download_satellite_images.py --years 2019 2023

# With automatic S3 backup
python scripts/download_satellite_images.py --years 2019 2023 --upload-s3
```

### 3. Convert to PNG for Labeling

```bash
# Convert to high-quality PNG (lossless)
python scripts/convert_to_png.py
# Press Enter for full resolution (recommended)
```

### 4. Start Labeling in Label Studio

```bash
label-studio start
# Access at http://localhost:8080
```

## 📁 Project Structure

```
solar-landuse/
├── data/
│   ├── raw_images/           # GeoTIFF files (synced to S3)
│   ├── for_labeling/         # PNG files for Label Studio
│   ├── labels/               # Exported annotations
│   └── processed/masks/      # Segmentation masks
├── scripts/
│   ├── download_satellite_images.py  # Download imagery
│   ├── convert_to_png.py             # Convert to PNG
│   ├── s3_utils.py                    # AWS S3 utilities
│   └── sync_to_s3.py                  # Sync to S3
├── models/                    # Model weights
├── notebooks/                 # Analysis notebooks
└── results/                   # Outputs
```

## 🌍 Current Dataset

### Test Site: Mongla Solar Project, Bangladesh

- **Coordinates**: 22°34'25.0"N 89°34'10.4"E
- **Buffer Sizes**: 1km, 5km, 10km
- **Resolution**: 10m per pixel (Sentinel-2 native)
- **Years**: 2014-2024 (with some gaps)

### Available Images

**High-Resolution (10m/pixel)**:
- 1km buffer: 217×201 pixels (~128 KB PNG)
- 5km buffer: 1079×1003 pixels (~3.1 MB PNG)

**Standard Resolution (20m/pixel)**:
- 10km buffer: ~2000×2000 pixels (~2.4 MB PNG)

## 📊 Data Download

### Download Scripts

#### Download Satellite Imagery

```bash
# Download specific years with multiple buffer sizes
python scripts/download_satellite_images.py --years 2019 2023

# Upload to S3 after download
python scripts/download_satellite_images.py --upload-s3

# All options
python scripts/download_satellite_images.py \
  --years 2019 2020 2023 \
  --upload-s3
```

**Features**:
- ✅ High-resolution (10m/pixel for Sentinel-2)
- ✅ Multiple buffer sizes (1km, 5km, 10km)
- ✅ Automatic S3 backup option
- ✅ Skips existing files
- ✅ Fallback to 20m if 10m fails

#### Convert to PNG

```bash
# Convert GeoTIFF to PNG with full resolution
python scripts/convert_to_png.py

# Or limit to specific size
python scripts/convert_to_png.py  # Then choose option 2, 3, or 4
```

## ☁️ AWS S3 Integration

All imagery is automatically backed up to S3 for team collaboration and version control.

### Upload to S3

```bash
# Sync all local files to S3
python scripts/sync_to_s3.py
```

### Download from S3

```bash
# Using AWS CLI
aws s3 sync s3://anuc-satellite-analysis/data/ ./data/

# Or manually download specific files
aws s3 cp s3://anuc-satellite-analysis/data/raw_images/mongla_1km_2019.tif ./data/raw_images/
```

**S3 Bucket**: `s3://anuc-satellite-analysis/data/`

## 🎨 Label Studio Setup

### Create Project

1. Open http://localhost:8080
2. Create new project: "Bangladesh Solar Land Use"
3. Choose: "Semantic Segmentation with Polygons"

### Import Images

4. Import from: `data/for_labeling/`
5. Start with 1km images for detailed labeling

### Suggested Labels

- Agriculture
- Forest  
- Water
- Urban
- Solar_Panels
- Bare_Land

## 🔧 Common Issues

### "Earth Engine not initialized"
```bash
python
>>> import ee
>>> ee.Authenticate()
>>> ee.Initialize(project="bangladesh-solar")
```

### "No images found for year XXXX"
- Normal if data not available for that year/location
- Try different years or adjust cloud filter

### Large files won't upload
- Use `--upload-s3` for automatic AWS S3 backup
- Or use `python scripts/sync_to_s3.py`

## 📚 Data Sources

### Satellite Data

- **Sentinel-2 MSI**: 10m resolution RGB+NIR bands
  - Dataset: `COPERNICUS/S2_SR_HARMONIZED`
  - Years: 2015-2024
- **Landsat 8 OLI**: 30m resolution
  - Dataset: `LANDSAT/LC08/C02/T1_L2`
  - Years: 2013-2024 (fallback for earlier years)

### Reference Documentation

- [Google Earth Engine](https://earthengine.google.com/)
- [Sentinel-2 Dataset](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Label Studio](https://labelstud.io/guide/)
- [DINOv3](https://ai.meta.com/dinov3/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Download data from S3: `aws s3 sync s3://anuc-satellite-analysis/data/ ./data/`
4. Make your changes
5. Sync data back: `python scripts/sync_to_s3.py`
6. Submit a pull request

**Note**: Large files (`.tif`, `.png`) are git-ignored. They are stored in S3 and should be downloaded separately.

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- Google Earth Engine for satellite data
- ESA Copernicus for Sentinel-2 imagery
- Label Studio for annotation tools
