"""
AWS S3 utilities for storing and retrieving project data.
"""
import boto3
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class S3Storage:
    """Handle S3 operations for project data storage."""

    def __init__(self, bucket_name='anuc-satellite-analysis'):
        self.bucket_name = bucket_name
        self.s3_client = None
        self._connect()

    def _connect(self):
        """Initialize S3 client with credentials from .env"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Connected to S3 bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to connect to S3: {e}")
            raise

    def upload_file(self, local_path, s3_key):
        """Upload a file to S3."""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            return True
        except Exception as e:
            logger.error(f"Failed to upload {s3_key}: {e}")
            return False

    def download_file(self, s3_key, local_path):
        """Download a file from S3."""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False

    def file_exists(self, s3_key):
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception:
            return False

    def get_object_size(self, s3_key):
        """Get file size in S3 (bytes), or None if not found."""
        try:
            resp = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return resp['ContentLength']
        except Exception:
            return None

    def list_files(self, prefix=''):
        """List all files in S3 with given prefix (handles pagination)."""
        files = {}
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    files[obj['Key']] = obj['Size']
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
        return files

    def sync_directory_to_s3(self, local_dir, s3_prefix='', skip_existing=True):
        """Upload directory to S3, optionally skipping files that match size.

        Returns (uploaded, skipped, failed) counts.
        """
        local_path = Path(local_dir)
        if not local_path.exists():
            logger.warning(f"Directory does not exist: {local_dir}")
            return 0, 0, 0

        # Get existing S3 file sizes for incremental sync
        s3_files = {}
        if skip_existing:
            s3_files = self.list_files(prefix=s3_prefix)

        uploaded = 0
        skipped = 0
        failed = 0

        local_files = [f for f in local_path.rglob('*') if f.is_file()]

        for file_path in local_files:
            rel_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}{rel_path}".replace('\\', '/')

            # Skip if same size exists in S3
            if skip_existing and s3_key in s3_files:
                local_size = file_path.stat().st_size
                if s3_files[s3_key] == local_size:
                    skipped += 1
                    continue

            if self.upload_file(str(file_path), s3_key):
                uploaded += 1
            else:
                failed += 1

        return uploaded, skipped, failed

    def sync_s3_to_directory(self, s3_prefix='', local_dir='.', skip_existing=True):
        """Download all files from S3 prefix to local directory.

        Returns (downloaded, skipped, failed) counts.
        """
        files = self.list_files(prefix=s3_prefix)

        downloaded = 0
        skipped = 0
        failed = 0

        for s3_key, s3_size in files.items():
            rel_path = s3_key[len(s3_prefix):] if s3_prefix else s3_key
            if not rel_path:
                continue
            local_path = Path(local_dir) / rel_path

            # Skip if local file has same size
            if skip_existing and local_path.exists():
                if local_path.stat().st_size == s3_size:
                    skipped += 1
                    continue

            if self.download_file(s3_key, str(local_path)):
                downloaded += 1
            else:
                failed += 1

        return downloaded, skipped, failed
