"""
AWS S3 utilities for storing and retrieving satellite imagery
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
    """Handle S3 operations for satellite imagery storage"""
    
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
            logger.info(f"✓ Connected to S3 bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to S3: {e}")
            raise
    
    def upload_file(self, local_path, s3_key):
        """
        Upload a file to S3
        
        Args:
            local_path: Local file path
            s3_key: S3 key (path in bucket)
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"✓ Uploaded: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to upload {s3_key}: {e}")
            return False
    
    def download_file(self, s3_key, local_path):
        """
        Download a file from S3
        
        Args:
            s3_key: S3 key (path in bucket)
            local_path: Local destination path
        """
        try:
            # Create parent directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"✓ Downloaded: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to download {s3_key}: {e}")
            return False
    
    def file_exists(self, s3_key):
        """Check if a file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False
    
    def list_files(self, prefix=''):
        """List files in S3 with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"✗ Failed to list files: {e}")
            return []
    
    def sync_directory_to_s3(self, local_dir, s3_prefix=''):
        """
        Upload all files in a directory to S3
        
        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix to prepend to files
        """
        local_path = Path(local_dir)
        if not local_path.exists():
            logger.error(f"Directory does not exist: {local_dir}")
            return
        
        uploaded = 0
        failed = 0
        
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Get relative path from local_dir
                rel_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}{rel_path}".replace('\\', '/')
                
                if self.upload_file(str(file_path), s3_key):
                    uploaded += 1
                else:
                    failed += 1
        
        logger.info(f"Upload summary: {uploaded} uploaded, {failed} failed")
        return uploaded, failed
    
    def sync_s3_to_directory(self, s3_prefix='', local_dir='.'):
        """
        Download all files from S3 to a directory
        
        Args:
            s3_prefix: S3 prefix to download from
            local_dir: Local destination directory
        """
        files = self.list_files(prefix=s3_prefix)
        
        downloaded = 0
        failed = 0
        
        for s3_key in files:
            # Get relative path by removing prefix
            rel_path = s3_key[len(s3_prefix):] if s3_prefix else s3_key
            local_path = Path(local_dir) / rel_path
            
            if self.download_file(s3_key, str(local_path)):
                downloaded += 1
            else:
                failed += 1
        
        logger.info(f"Download summary: {downloaded} downloaded, {failed} failed")
        return downloaded, failed

