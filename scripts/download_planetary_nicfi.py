"""
Download Planet NICFI imagery from Microsoft Planetary Computer
Resolution: 4.77m, Monthly composites, 2015-present, Tropics only
No API key needed - completely open access

Docs: 
- Dataset: https://planetarycomputer.microsoft.com/dataset/planet-nicfi-analytic
- STAC API: https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
"""

import planetary_computer
import pystac_client
import rasterio
from rasterio.windows import from_bounds
from rasterio.plot import reshape_as_image
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import argparse
from dotenv import load_dotenv
import os

# Load environment variables for S3
load_dotenv()

# Config
SITE = {
    'name': 'mongla',
    'lon': 89.569444,
    'lat': 22.573611
}

BUFFER_KM = 5
PREFERRED_MONTHS = [1, 11, 12]  # Jan, Nov, Dec (dry season for Bangladesh)

PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
OUTPUT_DIR = PROJECT_DIR / 'data' / 'planetary_computer'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_bbox(lon, lat, buffer_km):
    """Calculate bounding box from center point and buffer"""
    # Approximate degrees per km
    lat_deg_per_km = 1 / 111.32
    lon_deg_per_km = 1 / (111.32 * np.cos(np.radians(lat)))
    
    return [
        lon - (buffer_km * lon_deg_per_km),  # min_lon
        lat - (buffer_km * lat_deg_per_km),  # min_lat
        lon + (buffer_km * lon_deg_per_km),  # max_lon
        lat + (buffer_km * lat_deg_per_km)   # max_lat
    ]

def download_nicfi_image(site_name, lon, lat, buffer_km, year, preferred_months, output_dir, upload_s3=False):
    """
    Download single Planet NICFI image for specific year
    
    Returns: (success: bool, month: str, tif_path: Path, png_path: Path)
    """
    
    print(f"\n{'='*60}")
    print(f"Downloading: {site_name} - {buffer_km}km - {year}")
    print(f"{'='*60}")
    
    bbox = get_bbox(lon, lat, buffer_km)
    print(f"BBox: {bbox}")
    
    # Connect to Planetary Computer STAC catalog
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    except Exception as e:
        print(f"Error connecting to catalog: {e}")
        return False, None, None, None
    
    # Search for Planet NICFI mosaics
    try:
        search = catalog.search(
            collections=["planet-nicfi-analytic"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
        )
        
        items = list(search.items())
    except Exception as e:
        print(f"Error searching catalog: {e}")
        print("Note: Planet NICFI may require registration with Planet Labs")
        return False, None, None, None
    
    if not items:
        print(f"No images found for {year}")
        return False, None, None, None
    
    print(f"Found {len(items)} monthly mosaics")
    
    # Sort by date
    items_sorted = sorted(items, key=lambda x: x.datetime)
    
    # Select best month (prefer dry season)
    selected_item = None
    selected_month = None
    
    for month in preferred_months:
        for item in items_sorted:
            if item.datetime.month == month:
                selected_item = item
                selected_month = item.datetime.strftime('%Y-%m')
                break
        if selected_item:
            break
    
    # Fallback to middle of year
    if not selected_item:
        selected_item = items_sorted[len(items_sorted)//2]
        selected_month = selected_item.datetime.strftime('%Y-%m')
    
    print(f"Selected mosaic: {selected_month}")
    print(f"Item ID: {selected_item.id}")
    
    # Get the data asset (RGB + NIR bands)
    asset = selected_item.assets["data"]
    
    print(f"Downloading from: {asset.href}")
    
    try:
        with rasterio.open(asset.href) as src:
            print(f"Source CRS: {src.crs}")
            print(f"Source resolution: {src.res[0]:.2f}m")
            print(f"Source dimensions: {src.width}x{src.height}")
            print(f"Bands: {src.count} (should be 4: R,G,B,NIR)")
            
            # Read windowed data (crop to bbox)
            window = from_bounds(*bbox, src.transform)
            data = src.read(window=window)
            
            window_transform = src.window_transform(window)
            
            print(f"Cropped dimensions: {data.shape[2]}x{data.shape[1]} pixels")
            
            # Save as GeoTIFF
            output_tif = output_dir / f'{site_name}_{buffer_km}km_{selected_month}.tif'
            
            with rasterio.open(
                output_tif,
                'w',
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                crs=src.crs,
                transform=window_transform,
                compress='lzw'
            ) as dst:
                dst.write(data)
            
            tif_size_mb = output_tif.stat().st_size / 1e6
            print(f"✓ GeoTIFF: {output_tif.name} ({tif_size_mb:.1f} MB)")
            
            # Convert to PNG for Label Studio
            output_png = output_dir / f'{site_name}_{buffer_km}km_{selected_month}.png'
            
            # RGB only (first 3 bands)
            rgb = data[:3, :, :].transpose(1, 2, 0)
            
            # Normalize using percentiles (removes outliers)
            p2, p98 = np.percentile(rgb, (2, 98))
            rgb_clipped = np.clip(rgb, p2, p98)
            rgb_normalized = ((rgb_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
            
            # Save PNG (uncompressed for quality)
            img = Image.fromarray(rgb_normalized)
            img.save(output_png, compress_level=0)
            
            png_size_mb = output_png.stat().st_size / 1e6
            print(f"✓ PNG: {output_png.name} ({png_size_mb:.1f} MB, {img.width}x{img.height}px)")
            
            # Upload to S3 if requested
            if upload_s3:
                try:
                    from s3_utils import S3Storage
                    s3 = S3Storage()
                    
                    # Upload TIF
                    s3_tif_key = f"data/planetary_computer/{output_tif.name}"
                    s3.upload_file(str(output_tif), s3_tif_key)
                    
                    # Upload PNG
                    s3_png_key = f"data/planetary_computer/{output_png.name}"
                    s3.upload_file(str(output_png), s3_png_key)
                    
                    print(f"✓ Uploaded to S3")
                except Exception as e:
                    print(f"Warning: S3 upload failed: {e}")
            
            return True, selected_month, output_tif, output_png
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def find_year_before_construction(start_year=2014, max_year=2020):
    """Try years from start_year upward until successful"""
    for year in range(start_year, max_year + 1):
        print(f"\nTrying year {year} (pre-construction)...")
        success, month, tif_path, png_path = download_nicfi_image(
            SITE['name'],
            SITE['lon'],
            SITE['lat'],
            BUFFER_KM,
            year,
            PREFERRED_MONTHS,
            OUTPUT_DIR,
            upload_s3=False  # Will upload after success
        )
        if success:
            return year, month, tif_path, png_path
    return None, None, None, None

def find_year_latest(start_year=2025, min_year=2020):
    """Try years from start_year downward until successful"""
    for year in range(start_year, min_year - 1, -1):
        print(f"\nTrying year {year} (latest)...")
        success, month, tif_path, png_path = download_nicfi_image(
            SITE['name'],
            SITE['lon'],
            SITE['lat'],
            BUFFER_KM,
            year,
            PREFERRED_MONTHS,
            OUTPUT_DIR,
            upload_s3=False  # Will upload after success
        )
        if success:
            return year, month, tif_path, png_path
    return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description='Download Planet NICFI imagery')
    parser.add_argument('--upload-s3', action='store_true', 
                       help='Upload files to S3 after download')
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# Planet NICFI Download via Microsoft Planetary Computer")
    print(f"# Resolution: 4.77m | Coverage: Tropics | Free & Open")
    print(f"{'#'*60}\n")
    
    print(f"Site: {SITE['name']}")
    print(f"Coordinates: {SITE['lat']:.6f}°N, {SITE['lon']:.6f}°E")
    print(f"Buffer: {BUFFER_KM}km")
    print(f"Preferred months: {PREFERRED_MONTHS} (dry season)\n")
    
    # Find pre-construction year (starting from 2014)
    print(f"\n{'='*60}")
    print(f"Finding pre-construction year (starting from 2014)...")
    print(f"{'='*60}")
    
    pre_year, pre_month, pre_tif, pre_png = find_year_before_construction(2014, 2020)
    
    # Find latest year (starting from 2025)
    print(f"\n{'='*60}")
    print(f"Finding latest year (starting from 2025)...")
    print(f"{'='*60}")
    
    latest_year, latest_month, latest_tif, latest_png = find_year_latest(2025, 2020)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    if pre_year:
        print(f"✓ Pre-construction: {pre_month} ({pre_tif.name if pre_tif else 'N/A'})")
    else:
        print(f"✗ Pre-construction: No data found (tried 2014-2020)")
    
    if latest_year:
        print(f"✓ Latest year: {latest_month} ({latest_tif.name if latest_tif else 'N/A'})")
    else:
        print(f"✗ Latest year: No data found (tried 2025-2020)")
    
    # Upload to S3 if requested and we have successful downloads
    if args.upload_s3:
        print(f"\nUploading to S3...")
        try:
            from s3_utils import S3Storage
            s3 = S3Storage()
            
            if pre_tif and pre_tif.exists():
                s3_tif_key = f"data/planetary_computer/{pre_tif.name}"
                s3.upload_file(str(pre_tif), s3_tif_key)
                print(f"✓ Uploaded {pre_tif.name}")
            
            if pre_png and pre_png.exists():
                s3_png_key = f"data/planetary_computer/{pre_png.name}"
                s3.upload_file(str(pre_png), s3_png_key)
                print(f"✓ Uploaded {pre_png.name}")
            
            if latest_tif and latest_tif.exists():
                s3_tif_key = f"data/planetary_computer/{latest_tif.name}"
                s3.upload_file(str(latest_tif), s3_tif_key)
                print(f"✓ Uploaded {latest_tif.name}")
            
            if latest_png and latest_png.exists():
                s3_png_key = f"data/planetary_computer/{latest_png.name}"
                s3.upload_file(str(latest_png), s3_png_key)
                print(f"✓ Uploaded {latest_png.name}")
            
            print(f"✓ S3 upload complete")
        except Exception as e:
            print(f"✗ S3 upload failed: {e}")
    
    print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")
    
    if pre_png or latest_png:
        print(f"\nPNG files ready for Label Studio import:")
        if pre_png:
            print(f"  - {pre_png.name}")
        if latest_png:
            print(f"  - {latest_png.name}")

if __name__ == '__main__':
    main()

