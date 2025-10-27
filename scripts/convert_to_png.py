"""
Convert GeoTIFF satellite images to HIGH QUALITY PNG format for labeling in Label Studio
Preserves maximum resolution without downsampling
"""

import rasterio
from rasterio.plot import reshape_as_image
from PIL import Image
import numpy as np
from pathlib import Path

# Project directory
PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
INPUT_DIR = PROJECT_DIR / 'data' / 'raw_images'
OUTPUT_DIR = PROJECT_DIR / 'data' / 'for_labeling'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def geotiff_to_png(tiff_path, output_path, max_dimension=None):
    """
    Convert GeoTIFF to RGB PNG for labeling - MAXIMUM QUALITY
    
    Args:
        tiff_path: Path to input GeoTIFF file
        output_path: Path to output PNG file
        max_dimension: Maximum width/height in pixels (None = no limit, preserve full resolution)
    """
    print(f"\nConverting: {Path(tiff_path).name}")
    
    try:
        with rasterio.open(tiff_path) as src:
            print(f"  Input dimensions: {src.width} x {src.height} pixels")
            print(f"  Input bands: {src.count}")
            print(f"  Input resolution: {src.res[0]:.2f} units per pixel")
            
            # Read first 3 bands (RGB) - preserve full resolution
            image = src.read([1, 2, 3])
            
            # Reshape to (height, width, bands)
            image = reshape_as_image(image)
            
            print(f"  Original shape: {image.shape}")
            
            # Handle potential issues with data range
            # Use percentile clipping for better visualization (removes extreme outliers)
            p2, p98 = np.percentile(image, (2, 98))
            image = np.clip(image, p2, p98)
            
            # Normalize to 0-255 for PNG
            if image.max() > image.min():
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                print(f"  ⚠️  Warning: No variation in image values")
                image = image.astype(np.uint8)
            
            # Create PIL Image
            img_pil = Image.fromarray(image)
            
            # Optionally resize if dimensions are too large (but preserve aspect ratio)
            if max_dimension and (img_pil.width > max_dimension or img_pil.height > max_dimension):
                print(f"  Resizing from {img_pil.width}x{img_pil.height} to fit {max_dimension}px...")
                img_pil.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                print(f"  New size: {img_pil.width}x{img_pil.height}")
            else:
                print(f"  Preserving full resolution: {img_pil.width}x{img_pil.height}")
            
            # Save as PNG with NO compression (maximum quality)
            # compress_level=0 means no compression, faster write, larger file
            # For Label Studio, we want maximum quality
            img_pil.save(output_path, 
                        format='PNG',
                        compress_level=0,  # No compression = fastest, largest file, no quality loss
                        optimize=False)    # Don't optimize = preserve all data
            
            size_mb = Path(output_path).stat().st_size / 1e6
            print(f"  ✓ Saved: {Path(output_path).name} ({size_mb:.1f} MB)")
            print(f"  Output dimensions: {img_pil.width} x {img_pil.height}")
            return True
            
    except Exception as e:
        print(f"  ✗ Error converting {Path(tiff_path).name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main conversion function"""
    
    print(f"\n{'#'*60}")
    print(f"# Convert GeoTIFF to HIGH QUALITY PNG for Label Studio")
    print(f"{'#'*60}\n")
    
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Find all TIF files
    tiff_files = sorted(INPUT_DIR.glob('*.tif'))
    
    if not tiff_files:
        print(f"\n✗ No .tif files found in {INPUT_DIR}")
        print(f"Run download_satellite_images_highres.py first")
        return
    
    print(f"\nFound {len(tiff_files)} GeoTIFF files\n")
    
    # Ask user about resolution limits (with non-interactive default)
    print("Resolution options:")
    print("  1. Full resolution (no limit) - Recommended for detailed labeling")
    print("  2. Limit to 5000px - Good balance of quality and file size")
    print("  3. Limit to 3000px - Smaller files, still high quality")
    print("  4. Limit to 2000px - Fastest to load in Label Studio")
    
    # Non-interactive default for automation
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-4) or press Enter for option 1 (full resolution): ").strip() or '1'
    
    max_dimension = None
    if choice == '2':
        max_dimension = 5000
        print(f"\nWill limit images to {max_dimension}px maximum dimension")
    elif choice == '3':
        max_dimension = 3000
        print(f"\nWill limit images to {max_dimension}px maximum dimension")
    elif choice == '4':
        max_dimension = 2000
        print(f"\nWill limit images to {max_dimension}px maximum dimension")
    else:
        max_dimension = None
        print("\nUsing full resolution (no downsampling)")
    
    print("\nStarting conversion...\n")
    
    successful = 0
    failed = 0
    
    for tiff_path in tiff_files:
        # Create output filename (keep same name, change extension)
        output_filename = tiff_path.stem + '.png'
        output_path = OUTPUT_DIR / output_filename
        
        # Skip if already exists
        if output_path.exists():
            size_mb = Path(output_path).stat().st_size / 1e6
            print(f"\n{output_filename}: Already exists ({size_mb:.1f} MB), skipping...")
            successful += 1
            continue
        
        # Convert
        if geotiff_to_png(tiff_path, output_path, max_dimension=max_dimension):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(tiff_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nPNG files saved to: {OUTPUT_DIR.absolute()}")
    
    # Show some statistics
    png_files = list(OUTPUT_DIR.glob('*.png'))
    if png_files:
        total_size = sum(f.stat().st_size for f in png_files) / 1e6
        avg_size = total_size / len(png_files)
        print(f"\nTotal PNG size: {total_size:.1f} MB")
        print(f"Average PNG size: {avg_size:.1f} MB")
    
    print(f"\nNext step: Import these PNG files into Label Studio for labeling")
    print(f"\nTIP: Start with the 1km buffer images for easier labeling of small features")

if __name__ == '__main__':
    main()

