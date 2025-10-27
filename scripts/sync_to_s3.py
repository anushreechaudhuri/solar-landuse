"""
Upload local satellite imagery files to S3 bucket
"""
import argparse
import sys
from pathlib import Path
import logging
from s3_utils import S3Storage

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Sync satellite imagery to S3')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be uploaded without actually uploading')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# Sync Satellite Imagery to S3")
    print(f"{'#'*60}\n")
    
    PROJECT_DIR = Path('/Users/anushreechaudhuri/Documents/Projects/solar-landuse')
    
    try:
        s3 = S3Storage()
    except Exception as e:
        print(f"✗ Could not connect to S3: {e}")
        sys.exit(1)
    
    # Sync raw_images
    print("Syncing data/raw_images/...")
    if args.dry_run:
        print("(DRY RUN - no files uploaded)")
    else:
        uploaded, failed = s3.sync_directory_to_s3(
            PROJECT_DIR / 'data' / 'raw_images',
            s3_prefix='data/raw_images/'
        )
        print(f"  Uploaded: {uploaded}, Failed: {failed}")
    
    # Sync PNG files for labeling
    print("\nSyncing data/for_labeling/...")
    if args.dry_run:
        print("(DRY RUN - no files uploaded)")
    else:
        uploaded, failed = s3.sync_directory_to_s3(
            PROJECT_DIR / 'data' / 'for_labeling',
            s3_prefix='data/for_labeling/'
        )
        print(f"  Uploaded: {uploaded}, Failed: {failed}")
    
    print(f"\n✓ Sync complete!")
    print(f"S3 bucket: {s3.bucket_name}")
    print(f"Files available at: s3://{s3.bucket_name}/data/")

if __name__ == '__main__':
    main()

