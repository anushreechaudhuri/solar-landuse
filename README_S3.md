# AWS S3 Integration for Satellite Imagery

This project stores satellite imagery in AWS S3 for easy sharing and backup while keeping files locally for Label Studio.

## Quick Start

### Upload Existing Files to S3

All current files are already in S3:

```bash
# View what's in S3
aws s3 ls s3://anuc-satellite-analysis/data/raw_images/
aws s3 ls s3://anuc-satellite-analysis/data/for_labeling/

# Upload/re-sync local files to S3
conda activate solar-landuse
python scripts/sync_to_s3.py
```

### Download Files from S3

```bash
# Download all files from S3 to local
aws s3 sync s3://anuc-satellite-analysis/data/ ./data/ --exact-timestamps

# Or use the Python utility (coming soon)
python scripts/sync_from_s3.py
```

## Configuration

### AWS Credentials

Credentials are stored in `.env` file (not committed to git):

```bash
AWS_DEFAULT_REGION="us-east-1"
AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
```

**⚠️ Important**: This `.env` file should be in `.gitignore` and never committed to GitHub.

### S3 Bucket Structure

```
s3://anuc-satellite-analysis/
├── data/
│   ├── raw_images/          # GeoTIFF files
│   │   ├── mongla_1km_2019.tif
│   │   ├── mongla_1km_2023.tif
│   │   ├── mongla_5km_2019.tif
│   │   ├── mongla_5km_2023.tif
│   │   ├── DATA_SOURCES.md
│   │   └── ...
│   └── for_labeling/         # PNG files for Label Studio
│       ├── mongla_1km_2019.png
│       ├── mongla_1km_2023.png
│       └── ...
```

## Using S3 with Scripts

### Download Script with S3 Upload

```bash
# Download and automatically upload to S3
conda activate solar-landuse
python scripts/quick_download.py --upload-s3

# Download only (no S3)
python scripts/quick_download.py
```

### Sync Scripts

```bash
# Upload local files to S3
python scripts/sync_to_s3.py

# Download files from S3 (keeps structure)
# Coming soon
```

## For Team Members

### Downloading the Dataset

If you're cloning this repository, download the satellite imagery from S3:

```bash
# Using AWS CLI
aws configure  # Set up your credentials
aws s3 sync s3://anuc-satellite-analysis/data/ ./data/

# Using Python (recommended)
conda activate solar-landuse
python scripts/download_from_s3.py
```

### Uploading New Downloads

When you download new images, upload them to S3:

```bash
python scripts/sync_to_s3.py
```

## File Storage Strategy

- **Local**: Files are kept locally for fast access in Label Studio
- **S3**: Files are backed up to S3 for:
  - Team collaboration (shared dataset)
  - Version control (avoid pushing large files to GitHub)
  - Backup and disaster recovery
  - CI/CD pipelines (can download dataset for automated analysis)

## Benefits

1. **No Large Files in GitHub**: Repository stays lightweight
2. **Easy Collaboration**: Team members can download current dataset
3. **Backup**: Automatic backup of imagery data
4. **Flexible**: Can work entirely local or fetch from S3 when needed

## Workflow

### Typical User Workflow

1. **Clone repository**: `git clone ...`
2. **Download imagery**: `python scripts/download_from_s3.py`
3. **Work with Label Studio**: Uses local files
4. **Push code**: Only code changes go to GitHub
5. **Upload new imagery**: `python scripts/sync_to_s3.py`

### New Image Downloads

1. Run download script: `python scripts/quick_download.py --upload-s3`
2. Files are downloaded locally AND uploaded to S3 automatically
3. Team members can fetch latest with `python scripts/sync_to_s3.py`

## Troubleshooting

### "Could not connect to S3"

Check your `.env` file exists and has correct credentials:
```bash
cat .env
```

### "Access Denied" Error

Verify your AWS credentials have access to `anuc-satellite-analysis` bucket:
```bash
aws s3 ls s3://anuc-satellite-analysis/
```

### Large Files Not Uploading

S3 has a 5GB limit per file. Current files (10MB each) are well under this limit.

## Security Notes

- AWS credentials in `.env` are git-ignored
- S3 bucket has proper IAM permissions
- Team members should use their own AWS credentials
- Never commit `.env` file to Git

## Next Steps

- [ ] Create `download_from_s3.py` script
- [ ] Add automatic S3 sync to conversion script
- [ ] Set up CloudFront CDN for faster downloads
- [ ] Add version tagging to datasets

