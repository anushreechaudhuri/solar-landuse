"""
Sync all project data to/from S3 bucket.

Backs up all local data files to s3://anuc-satellite-analysis/data/.
Large cache directories are archived as .tar.gz before upload for efficiency.

Usage:
    python scripts/sync_to_s3.py              # Upload all data to S3
    python scripts/sync_to_s3.py --dry-run    # Preview what would upload
    python scripts/sync_to_s3.py --restore    # Download all data from S3
    python scripts/sync_to_s3.py --status     # Show what's in S3 vs local
"""
import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_utils import S3Storage

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Directories to archive as tar.gz (many small files)
ARCHIVE_DIRS = [
    "temporal_cache",
    "screening_cache",
    "lulc_raw_cache",
    "vlm_v2_responses",
    "training_dataset",
]

# Directories to sync file-by-file (fewer, larger files)
SYNC_DIRS = [
    ("did_results", "data/did_results/"),
    ("did_results_bangladesh", "data/did_results_bangladesh/"),
    ("lulc_comparison", "data/lulc_comparison/"),
    ("raw_images", "data/raw_images/"),
    ("for_labeling", "data/for_labeling/"),
    ("grw", "data/grw/"),
]

# Individual files to sync
SYNC_FILES = [
    "temporal_panel.csv",
    "temporal_panel_bangladesh.csv",
    "unified_solar_db.json",
    "gspt_south_asia.json",
    "grw_south_asia.geojson",
    "grw_unmatched.geojson",
    "tzsam_south_asia.geojson",
    "projects_merged.json",
    "comparison_sites.json",
    "comparison_sites_bangladesh.json",
    "lulc_comparison.csv",
    "lulc_comparison_v3.csv",
    "lulc_polygon_v3.csv",
    "Global-Solar-Power-Tracker-February-2026.xlsx",
]


def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def archive_and_upload(s3, dirname, dry_run=False):
    """Tar.gz a directory and upload the archive to S3."""
    local_dir = DATA_DIR / dirname
    if not local_dir.exists():
        return 0

    file_count = sum(1 for f in local_dir.rglob('*') if f.is_file())
    if file_count == 0:
        return 0

    s3_key = f"data/archives/{dirname}.tar.gz"

    # Check if archive needs updating: compare file count marker
    marker_key = f"data/archives/{dirname}.count"
    existing_count_raw = None
    try:
        resp = s3.s3_client.get_object(Bucket=s3.bucket_name, Key=marker_key)
        existing_count_raw = resp['Body'].read().decode().strip()
    except Exception:
        pass

    if existing_count_raw == str(file_count):
        print(f"  {dirname}/: {file_count} files (archive up to date, skipping)")
        return 0

    if dry_run:
        dir_size = sum(f.stat().st_size for f in local_dir.rglob('*')
                       if f.is_file())
        print(f"  {dirname}/: {file_count} files ({format_size(dir_size)}) "
              f"-> would archive + upload")
        return 1

    # Create tar.gz in temp directory
    print(f"  {dirname}/: archiving {file_count} files...", end=" ",
          flush=True)
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with tarfile.open(tmp_path, 'w:gz') as tar:
            tar.add(str(local_dir), arcname=dirname)

        archive_size = os.path.getsize(tmp_path)
        print(f"{format_size(archive_size)} archive, uploading...", end=" ",
              flush=True)

        if s3.upload_file(tmp_path, s3_key):
            # Update file count marker
            s3.s3_client.put_object(
                Bucket=s3.bucket_name, Key=marker_key,
                Body=str(file_count).encode())
            print("done")
            return 1
        else:
            print("FAILED")
            return 0
    finally:
        os.unlink(tmp_path)


def restore_archive(s3, dirname):
    """Download and extract a tar.gz archive from S3."""
    s3_key = f"data/archives/{dirname}.tar.gz"
    if not s3.file_exists(s3_key):
        print(f"  {dirname}/: no archive in S3")
        return False

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        print(f"  {dirname}/: downloading archive...", end=" ", flush=True)
        if not s3.download_file(s3_key, tmp_path):
            print("FAILED")
            return False

        archive_size = os.path.getsize(tmp_path)
        print(f"{format_size(archive_size)}, extracting...", end=" ",
              flush=True)

        with tarfile.open(tmp_path, 'r:gz') as tar:
            tar.extractall(path=str(DATA_DIR))

        file_count = sum(1 for f in (DATA_DIR / dirname).rglob('*')
                         if f.is_file())
        print(f"done ({file_count} files)")
        return True
    finally:
        os.unlink(tmp_path)


def sync_upload(s3, dry_run=False):
    """Upload all local data to S3."""
    total_uploaded = 0
    total_skipped = 0
    start = time.time()

    # 1. Individual files
    print("Syncing individual files...")
    for filename in SYNC_FILES:
        local_path = DATA_DIR / filename
        if not local_path.exists():
            continue
        s3_key = f"data/{filename}"
        local_size = local_path.stat().st_size
        s3_size = s3.get_object_size(s3_key)

        if s3_size == local_size:
            total_skipped += 1
            continue

        if dry_run:
            print(f"  [upload] {filename} ({format_size(local_size)})")
            total_uploaded += 1
        else:
            print(f"  Uploading {filename} ({format_size(local_size)})...",
                  end=" ", flush=True)
            if s3.upload_file(str(local_path), s3_key):
                print("done")
                total_uploaded += 1
            else:
                print("FAILED")

    # 2. Archive large cache directories
    print("\nArchiving cache directories...")
    for dirname in ARCHIVE_DIRS:
        result = archive_and_upload(s3, dirname, dry_run=dry_run)
        total_uploaded += result

    # 3. Sync smaller directories file-by-file
    print("\nSyncing directories...")
    for dirname, s3_prefix in SYNC_DIRS:
        local_dir = DATA_DIR / dirname
        if not local_dir.exists():
            continue
        file_count = sum(1 for f in local_dir.rglob('*') if f.is_file())
        if file_count == 0:
            continue

        if dry_run:
            s3_files = s3.list_files(prefix=s3_prefix)
            local_files = [f for f in local_dir.rglob('*') if f.is_file()]
            to_upload = sum(1 for f in local_files
                           if f"{s3_prefix}{f.relative_to(local_dir)}".replace('\\', '/')
                           not in s3_files
                           or s3_files.get(f"{s3_prefix}{f.relative_to(local_dir)}".replace('\\', '/'), -1)
                           != f.stat().st_size)
            print(f"  {dirname}/: {to_upload} to upload, "
                  f"{file_count - to_upload} up to date")
            total_uploaded += to_upload
            total_skipped += file_count - to_upload
        else:
            up, skip, fail = s3.sync_directory_to_s3(
                local_dir, s3_prefix=s3_prefix, skip_existing=True)
            print(f"  {dirname}/: uploaded {up}, skipped {skip}"
                  + (f", failed {fail}" if fail else ""))
            total_uploaded += up
            total_skipped += skip

    elapsed = time.time() - start
    action = "Would upload" if dry_run else "Uploaded"
    print(f"\n{'='*50}")
    print(f"{action}: {total_uploaded} items")
    print(f"Skipped (already in S3): {total_skipped}")
    print(f"Time: {elapsed:.1f}s")
    print(f"S3 bucket: s3://{s3.bucket_name}/data/")


def sync_restore(s3, dry_run=False):
    """Download all data from S3 to local."""
    start = time.time()

    # 1. Individual files
    print("Restoring individual files...")
    for filename in SYNC_FILES:
        s3_key = f"data/{filename}"
        local_path = DATA_DIR / filename
        s3_size = s3.get_object_size(s3_key)
        if s3_size is None:
            continue
        if local_path.exists() and local_path.stat().st_size == s3_size:
            continue
        if dry_run:
            print(f"  [download] {filename} ({format_size(s3_size)})")
        else:
            print(f"  Downloading {filename}...", end=" ", flush=True)
            s3.download_file(s3_key, str(local_path))
            print("done")

    # 2. Restore archived directories
    print("\nRestoring archived directories...")
    for dirname in ARCHIVE_DIRS:
        local_dir = DATA_DIR / dirname
        if local_dir.exists() and any(local_dir.rglob('*')):
            # Check if archive is newer (different file count)
            marker_key = f"data/archives/{dirname}.count"
            try:
                resp = s3.s3_client.get_object(
                    Bucket=s3.bucket_name, Key=marker_key)
                s3_count = int(resp['Body'].read().decode().strip())
                local_count = sum(1 for f in local_dir.rglob('*')
                                  if f.is_file())
                if local_count >= s3_count:
                    print(f"  {dirname}/: up to date "
                          f"(local={local_count}, s3={s3_count})")
                    continue
            except Exception:
                pass

        if dry_run:
            print(f"  {dirname}/: would restore from archive")
        else:
            restore_archive(s3, dirname)

    # 3. Restore file-synced directories
    print("\nRestoring synced directories...")
    for dirname, s3_prefix in SYNC_DIRS:
        local_dir = DATA_DIR / dirname
        if dry_run:
            s3_files = s3.list_files(prefix=s3_prefix)
            print(f"  {dirname}/: {len(s3_files)} files in S3")
        else:
            down, skip, fail = s3.sync_s3_to_directory(
                s3_prefix=s3_prefix, local_dir=str(local_dir),
                skip_existing=True)
            print(f"  {dirname}/: downloaded {down}, skipped {skip}")

    elapsed = time.time() - start
    print(f"\nRestore complete ({elapsed:.1f}s)")


def show_status(s3):
    """Compare local vs S3 state."""
    print("S3 sync status\n")

    print("Individual files:")
    for filename in SYNC_FILES:
        local_path = DATA_DIR / filename
        s3_key = f"data/{filename}"
        local_size = local_path.stat().st_size if local_path.exists() else None
        s3_size = s3.get_object_size(s3_key)

        if local_size and s3_size:
            status = "synced" if local_size == s3_size else \
                f"STALE (local={format_size(local_size)}, s3={format_size(s3_size)})"
        elif local_size:
            status = f"NOT IN S3 ({format_size(local_size)})"
        elif s3_size:
            status = f"NOT LOCAL ({format_size(s3_size)})"
        else:
            status = "missing"
        print(f"  {filename}: {status}")

    print("\nArchived directories:")
    for dirname in ARCHIVE_DIRS:
        local_dir = DATA_DIR / dirname
        local_count = sum(1 for f in local_dir.rglob('*') if f.is_file()) \
            if local_dir.exists() else 0
        s3_exists = s3.file_exists(f"data/archives/{dirname}.tar.gz")

        marker_key = f"data/archives/{dirname}.count"
        s3_count = None
        try:
            resp = s3.s3_client.get_object(
                Bucket=s3.bucket_name, Key=marker_key)
            s3_count = int(resp['Body'].read().decode().strip())
        except Exception:
            pass

        if s3_exists and s3_count:
            if local_count == s3_count:
                status = f"synced ({local_count} files)"
            else:
                status = f"STALE (local={local_count}, s3={s3_count})"
        elif local_count > 0 and not s3_exists:
            status = f"NOT IN S3 ({local_count} files)"
        elif s3_exists:
            status = f"NOT LOCAL (s3={s3_count} files)"
        else:
            status = "empty"
        print(f"  {dirname}/: {status}")

    print("\nSynced directories:")
    for dirname, s3_prefix in SYNC_DIRS:
        local_dir = DATA_DIR / dirname
        local_count = sum(1 for f in local_dir.rglob('*') if f.is_file()) \
            if local_dir.exists() else 0
        s3_files = s3.list_files(prefix=s3_prefix)
        s3_count = len(s3_files)

        if local_count == s3_count:
            status = f"synced ({local_count} files)" if local_count else "empty"
        else:
            status = f"DIFF (local={local_count}, s3={s3_count})"
        print(f"  {dirname}/: {status}")


def main():
    parser = argparse.ArgumentParser(
        description='Sync project data to/from S3')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without uploading/downloading')
    parser.add_argument('--restore', action='store_true',
                        help='Download data from S3 to local')
    parser.add_argument('--status', action='store_true',
                        help='Show sync status (local vs S3)')
    args = parser.parse_args()

    try:
        s3 = S3Storage()
    except Exception as e:
        print(f"Could not connect to S3: {e}")
        sys.exit(1)

    if args.status:
        show_status(s3)
    elif args.restore:
        sync_restore(s3, dry_run=args.dry_run)
    else:
        sync_upload(s3, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
