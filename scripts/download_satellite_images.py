"""
Quick download for testing - 2019 and 2023 only, 1km and 5km buffers
Supports --upload-s3 flag to upload files to AWS S3 after download
"""
import argparse
import ee
import geemap
from pathlib import Path
import rasterio

# Parse arguments
parser = argparse.ArgumentParser(description='Download satellite imagery')
parser.add_argument('--upload-s3', action='store_true', 
                   help='Upload files to S3 after download')
args = parser.parse_args()

# Initialize Earth Engine
ee.Initialize(project="bangladesh-solar")
print("✓ Earth Engine initialized\n")

PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
OUTPUT_DIR = PROJECT_DIR / 'data' / 'raw_images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_NAME = 'mongla'
LON = 89.569444
LAT = 22.573611

# Download config
years = [2019, 2023]
buffers = [
    {'name': '1km', 'meters': 1000},
    {'name': '5km', 'meters': 5000}
]

def download(year, buffer_config):
    buffer_name = buffer_config['name']
    buffer_m = buffer_config['meters']
    
    print(f"\n{'='*60}")
    print(f"Year: {year}, Buffer: {buffer_name}")
    print(f"{'='*60}")
    
    output_path = OUTPUT_DIR / f'{SITE_NAME}_{buffer_name}_{year}.tif'
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1e6
        print(f"Already exists ({size_mb:.1f} MB), skipping...")
        return True
    
    # Create buffer and collection
    point = ee.Geometry.Point([LON, LAT])
    buffer = point.buffer(buffer_m)
    
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(buffer) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
    
    bands = ['B4', 'B3', 'B2', 'B8']
    
    count = collection.size().getInfo()
    print(f"Found {count} images")
    
    if count == 0:
        print("⚠️  No images found")
        return False
    
    composite = collection.median().select(bands)
    
    # Try 10m first, fallback to 20m if too large
    for scale in [10, 20]:
        try:
            print(f"Attempting {scale}m resolution...")
            geemap.ee_export_image(
                composite,
                filename=str(output_path),
                scale=scale,
                region=buffer,
                file_per_band=False,
                crs='EPSG:4326'
            )
            
            if output_path.exists() and output_path.stat().st_size > 1000:
                size_mb = output_path.stat().st_size / 1e6
                
                with rasterio.open(output_path) as src:
                    width, height = src.width, src.height
                    print(f"✓ SUCCESS: {scale}m ({width}x{height}px, {size_mb:.1f}MB)")
                
                return True
        except Exception as e:
            print(f"✗ Failed at {scale}m: {str(e)[:100]}")
            if scale == 20:  # Last attempt
                return False
    return False

# Download
successful = 0
failed = 0

for year in years:
    for buffer in buffers:
        if download(year, buffer):
            successful += 1
        else:
            failed += 1

print(f"\n{'='*60}")
print(f"SUMMARY: {successful} successful, {failed} failed")
print(f"{'='*60}")

# Upload to S3 if requested
if args.upload_s3:
    print(f"\nUploading to S3...")
    try:
        from s3_utils import S3Storage
        import sys
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        s3 = S3Storage()
        
        # Upload successful downloads
        for year in years:
            for buffer in buffers:
                output_path = OUTPUT_DIR / f'{SITE_NAME}_{buffer["name"]}_{year}.tif'
                if output_path.exists():
                    s3_key = f"data/raw_images/{output_path.name}"
                    s3.upload_file(str(output_path), s3_key)
        
        print("✓ Upload complete!")
    except Exception as e:
        print(f"✗ S3 upload failed: {e}")
        print("Files saved locally only.")


