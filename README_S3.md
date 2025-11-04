# AWS S3 Integration

Satellite imagery is stored in S3 for backup and team sharing. Files are kept locally for Label Studio.

## Upload to S3

```bash
python scripts/sync_to_s3.py
```

This uploads all files in `data/raw_images/` and `data/for_labeling/` to S3.

## Download from S3

```bash
# Using AWS CLI
aws s3 sync s3://anuc-satellite-analysis/data/ ./data/

# Or download specific files
aws s3 cp s3://anuc-satellite-analysis/data/raw_images/mongla_1km_2019.tif ./data/raw_images/
```

## Configuration

Set up credentials in `.env` file:

```bash
AWS_DEFAULT_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your-key-here"
AWS_SECRET_ACCESS_KEY="your-secret-here"
```

Copy `local.env` to `.env` and fill in your credentials. The `.env` file is git-ignored and should never be committed.

## S3 Bucket Structure

```
s3://anuc-satellite-analysis/
├── data/
│   ├── raw_images/          # GeoTIFF files
│   └── for_labeling/         # PNG files for Label Studio
```

## Using with Download Scripts

Download script can automatically upload to S3:

```bash
python scripts/download_satellite_images.py --upload-s3
```

## Troubleshooting

**Could not connect to S3**
Check your `.env` file exists and has correct credentials.

**Access Denied**
Verify your AWS credentials have access to `anuc-satellite-analysis` bucket.

**Large files not uploading**
S3 has a 5GB limit per file. Current files are under 10MB each.
