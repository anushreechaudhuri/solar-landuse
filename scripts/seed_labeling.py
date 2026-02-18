"""
Seed the labeling tasks: upload images to S3 and create labeling_tasks in Postgres.

Usage:
    python3 scripts/seed_labeling.py                # Upload images + seed DB
    python3 scripts/seed_labeling.py --db-only      # Only seed DB (images already on S3)
    python3 scripts/seed_labeling.py --upload-only   # Only upload images to S3
"""
import json
import math
import os
import re
import sys
from pathlib import Path

try:
    import psycopg2
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

try:
    import rasterio
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rasterio"])
    import rasterio

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
LABEL_DIR = DATA_DIR / "for_labeling"
RAW_DIR = DATA_DIR / "raw_images"
CONFIRMED_MATCHES = DATA_DIR / "grw" / "confirmed_matches.json"

S3_BUCKET = "anuc-satellite-analysis"
S3_PREFIX = "solar-labeling/"

# All 15 Bangladesh confirmed solar sites
SITES = {
    "teesta": {"name": "Teesta 200 MW", "lat": 25.628342, "lon": 89.541082},
    "feni": {"name": "Feni 75 MW", "lat": 22.787567, "lon": 91.367187},
    "manikganj": {"name": "Manikganj 35 MW", "lat": 23.780834, "lon": 89.824775},
    "moulvibazar": {"name": "Moulvibazar 10 MW", "lat": 24.493896, "lon": 91.633043},
    "pabna": {"name": "Pabna 64 MW", "lat": 23.961375, "lon": 89.159720},
    "mymensingh": {"name": "Mymensingh 50 MW", "lat": 24.702233, "lon": 90.461730},
    "tetulia": {"name": "Tetulia 8 MW", "lat": 26.482817, "lon": 88.410139},
    "mongla": {"name": "Mongla 100 MW", "lat": 22.574239, "lon": 89.570388},
    "sirajganj68": {"name": "Sirajganj 68 MW", "lat": 24.403976, "lon": 89.738849},
    "teknaf": {"name": "Teknaf 20 MW", "lat": 20.981669, "lon": 92.256021},
    "sirajganj6": {"name": "Sirajganj 6 MW", "lat": 24.386137, "lon": 89.748970},
    "kaptai": {"name": "Kaptai 7.4 MW", "lat": 22.491471, "lon": 92.226588},
    "sharishabari": {"name": "Sharishabari 3 MW", "lat": 24.772287, "lon": 89.842629},
    "barishal": {"name": "Barishal 1 MW", "lat": 22.657015, "lon": 90.339194},
    "lalmonirhat": {"name": "Lalmonirhat 30 MW", "lat": 25.997873, "lon": 89.154467},
}

# Standard filename pattern: {site}_{buffer}km_{year}_{month}_{period}.png
STANDARD_RE = re.compile(r'^(.+?)_(\d+)km_(\d{4})_(\d{2})_(pre|post)\.png$')
# Legacy patterns: {site}_{buffer}km_{year}.png or {site}_{year}.png
LEGACY_BUFFER_RE = re.compile(r'^(.+?)_(\d+)km_(\d{4})\.png$')
LEGACY_SIMPLE_RE = re.compile(r'^(.+?)_(\d{4})\.png$')


def make_bbox(lat, lon, buffer_km):
    """Return (west, south, east, north) in EPSG:4326."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat))
    dlat = buffer_km / km_per_deg_lat
    dlon = buffer_km / km_per_deg_lon
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def get_tif_bounds(png_filename):
    """Read actual geographic bounds from the corresponding GeoTIFF.

    Returns (west, south, east, north) or None if no TIF found.
    """
    tif_name = png_filename.replace(".png", ".tif")
    tif_path = RAW_DIR / tif_name
    if not tif_path.exists():
        return None
    with rasterio.open(str(tif_path)) as src:
        b = src.bounds
        return (b.left, b.bottom, b.right, b.top)


def geo_to_pixel(lon, lat, bbox, img_width, img_height):
    """Convert geographic coordinates to pixel coordinates."""
    west, south, east, north = bbox
    px = (lon - west) / (east - west) * img_width
    py = (north - lat) / (north - south) * img_height
    return [round(px, 1), round(py, 1)]


def parse_filename(filename):
    """Parse image filename into metadata dict. Returns None if unparseable."""
    m = STANDARD_RE.match(filename)
    if m:
        return {
            "site_name": m.group(1),
            "buffer_km": int(m.group(2)),
            "year": int(m.group(3)),
            "month": int(m.group(4)),
            "period": m.group(5),
        }

    m = LEGACY_BUFFER_RE.match(filename)
    if m:
        return {
            "site_name": m.group(1),
            "buffer_km": int(m.group(2)),
            "year": int(m.group(3)),
            "month": None,
            "period": "unknown",
        }

    m = LEGACY_SIMPLE_RE.match(filename)
    if m:
        return {
            "site_name": m.group(1),
            "buffer_km": 2,  # assume ~2km for legacy without buffer
            "year": int(m.group(2)),
            "month": None,
            "period": "unknown",
        }

    return None


def load_solar_polygons():
    """Load confirmed solar polygons for post-construction images."""
    if not CONFIRMED_MATCHES.exists():
        print(f"  Warning: {CONFIRMED_MATCHES} not found, no solar polygons will be loaded")
        return {}
    with open(CONFIRMED_MATCHES) as f:
        data = json.load(f)
    return data  # {site_key: {name, polygons: [{type, coordinates}], ...}}


def convert_polygon_to_pixels(polygon_coords, bbox, img_width, img_height):
    """Convert a GeoJSON polygon coordinate ring to pixel coordinates."""
    return [geo_to_pixel(lon, lat, bbox, img_width, img_height)
            for lon, lat in polygon_coords]


def upload_images(s3_client):
    """Upload all labeling images to S3."""
    uploaded = 0
    skipped = 0

    for f in sorted(LABEL_DIR.iterdir()):
        if not f.suffix == '.png':
            continue
        s3_key = S3_PREFIX + f.name
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            print(f"  Already exists: {s3_key}")
            skipped += 1
        except Exception:
            print(f"  Uploading: {f.name} → {s3_key}")
            s3_client.upload_file(
                str(f), S3_BUCKET, s3_key,
                ExtraArgs={"ContentType": "image/png"}
            )
            uploaded += 1

    print(f"Upload complete: {uploaded} new, {skipped} already existed")
    return uploaded


def seed_tasks(conn, solar_polygons):
    """Parse images and insert labeling tasks into database."""
    with conn.cursor() as cur:
        # Create tables if not exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS labeling_tasks (
                id SERIAL PRIMARY KEY,
                site_name TEXT NOT NULL,
                site_display_name TEXT,
                buffer_km INT NOT NULL,
                year INT NOT NULL,
                month INT,
                period TEXT NOT NULL,
                image_filename TEXT NOT NULL UNIQUE,
                s3_key TEXT NOT NULL,
                image_width INT,
                image_height INT,
                bbox_west REAL,
                bbox_south REAL,
                bbox_east REAL,
                bbox_north REAL,
                solar_polygon_pixels JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS labeling_annotations (
                id SERIAL PRIMARY KEY,
                task_id INT REFERENCES labeling_tasks(id) ON DELETE CASCADE,
                annotator TEXT NOT NULL,
                regions JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_labeling_tasks_site ON labeling_tasks(site_name);
            CREATE INDEX IF NOT EXISTS idx_labeling_annotations_task ON labeling_annotations(task_id);
        """)

        # Clear existing labeling tasks (fresh seed)
        cur.execute("DELETE FROM labeling_annotations")
        cur.execute("DELETE FROM labeling_tasks")

    conn.commit()

    inserted = 0
    skipped = 0

    with conn.cursor() as cur:
        for f in sorted(LABEL_DIR.iterdir()):
            if not f.suffix == '.png':
                continue

            meta = parse_filename(f.name)
            if meta is None:
                print(f"  Skipping unparseable: {f.name}")
                skipped += 1
                continue

            site_key = meta["site_name"]
            site_info = SITES.get(site_key)
            if site_info is None:
                print(f"  Skipping unknown site: {site_key} ({f.name})")
                skipped += 1
                continue

            # Get image dimensions
            img = Image.open(str(f))
            img_width, img_height = img.size

            # Get geographic bounding box: prefer actual TIF bounds over recomputed
            bbox = get_tif_bounds(f.name)
            if bbox is not None:
                pass  # Using actual GeoTIFF bounds
            else:
                bbox = make_bbox(site_info["lat"], site_info["lon"], meta["buffer_km"])
                print(f"  Warning: no TIF for {f.name}, using computed bbox")

            # Convert solar polygon to pixel coords for post images
            solar_px = None
            if meta["period"] == "post" and site_key in solar_polygons:
                site_polys = solar_polygons[site_key].get("polygons", [])
                pixel_polys = []
                for poly in site_polys:
                    coords = poly.get("coordinates", [])
                    if coords:
                        pixel_ring = convert_polygon_to_pixels(
                            coords[0], bbox, img_width, img_height
                        )
                        pixel_polys.append(pixel_ring)
                if pixel_polys:
                    solar_px = json.dumps(pixel_polys)

            s3_key = S3_PREFIX + f.name

            cur.execute(
                """
                INSERT INTO labeling_tasks
                    (site_name, site_display_name, buffer_km, year, month, period,
                     image_filename, s3_key, image_width, image_height,
                     bbox_west, bbox_south, bbox_east, bbox_north,
                     solar_polygon_pixels)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (image_filename) DO UPDATE SET
                    solar_polygon_pixels = EXCLUDED.solar_polygon_pixels,
                    image_width = EXCLUDED.image_width,
                    image_height = EXCLUDED.image_height
                """,
                (
                    site_key,
                    site_info["name"],
                    meta["buffer_km"],
                    meta["year"],
                    meta["month"],
                    meta["period"],
                    f.name,
                    s3_key,
                    img_width,
                    img_height,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    solar_px,
                ),
            )
            inserted += 1

    conn.commit()
    print(f"Seeded {inserted} labeling tasks, skipped {skipped}")


def main():
    import argparse
    import boto3

    parser = argparse.ArgumentParser(description="Seed labeling tasks")
    parser.add_argument("--db-only", action="store_true", help="Only seed DB")
    parser.add_argument("--upload-only", action="store_true", help="Only upload to S3")
    args = parser.parse_args()

    # Load solar polygons
    solar_polygons = load_solar_polygons()
    print(f"Loaded solar polygons for {len(solar_polygons)} sites")

    # Upload images to S3
    if not args.db_only:
        print("\n--- Uploading images to S3 ---")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        upload_images(s3_client)

    # Seed database
    if not args.upload_only:
        print("\n--- Seeding labeling tasks in database ---")
        url = os.getenv("POSTGRES_URL")
        if not url:
            print("Error: POSTGRES_URL not set")
            sys.exit(1)
        conn = psycopg2.connect(url)
        try:
            seed_tasks(conn, solar_polygons)
        finally:
            conn.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
