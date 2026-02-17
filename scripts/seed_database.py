"""
Seed the Vercel Postgres database from projects_merged.json.

Generates and executes SQL INSERT statements, and optionally
uploads the merged data to S3.
"""
import json
import os
import sys
from pathlib import Path

try:
    import psycopg2
except ImportError:
    print("Installing psycopg2-binary...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
MERGED_FILE = DATA_DIR / "projects_merged.json"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    gem_location_id TEXT,
    project_name TEXT NOT NULL,
    phase_name TEXT,
    country TEXT NOT NULL,
    state_province TEXT,
    capacity_mw REAL NOT NULL,
    capacity_rating TEXT,
    status TEXT NOT NULL,
    start_year INTEGER,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    location_accuracy TEXT,
    owner TEXT,
    operator TEXT,
    other_ids TEXT,
    wiki_url TEXT,
    grw_polygons JSONB,
    merged_polygon JSONB,
    match_confidence TEXT,
    match_distance_km REAL,
    grw_construction_date TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
    id SERIAL PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    reviewer_name TEXT NOT NULL,
    action TEXT NOT NULL,
    polygon JSONB,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reviews_project ON reviews(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_projects_capacity ON projects(capacity_mw DESC);
CREATE INDEX IF NOT EXISTS idx_projects_country ON projects(country);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
"""


def get_connection():
    """Connect to Vercel Postgres."""
    url = os.getenv("POSTGRES_URL")
    if not url:
        print("Error: POSTGRES_URL not set in environment")
        print("Set it in .env or run: npx vercel env pull .env.local")
        sys.exit(1)
    return psycopg2.connect(url)


def create_schema(conn):
    """Create tables and indexes."""
    print("Creating schema...")
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    print("Schema created.")


def seed_projects(conn):
    """Insert projects from merged JSON."""
    print(f"Loading {MERGED_FILE}...")
    with open(MERGED_FILE) as f:
        projects = json.load(f)
    print(f"  {len(projects)} projects to insert")

    inserted = 0
    skipped = 0

    with conn.cursor() as cur:
        # Clear existing data
        cur.execute("DELETE FROM reviews")
        cur.execute("DELETE FROM projects")

        for p in projects:
            gspt = p.get("gspt", {})
            lat = gspt.get("latitude")
            lon = gspt.get("longitude")
            capacity = gspt.get("capacity_mw")

            # Skip projects without required fields
            if lat is None or lon is None or capacity is None:
                skipped += 1
                continue

            cur.execute(
                """
                INSERT INTO projects (
                    id, gem_location_id, project_name, phase_name,
                    country, state_province, capacity_mw, capacity_rating,
                    status, start_year, latitude, longitude, location_accuracy,
                    owner, operator, other_ids, wiki_url,
                    grw_polygons, merged_polygon, match_confidence,
                    match_distance_km, grw_construction_date
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (id) DO UPDATE SET
                    grw_polygons = EXCLUDED.grw_polygons,
                    merged_polygon = EXCLUDED.merged_polygon,
                    match_confidence = EXCLUDED.match_confidence,
                    match_distance_km = EXCLUDED.match_distance_km,
                    grw_construction_date = EXCLUDED.grw_construction_date
                """,
                (
                    p["id"],
                    gspt.get("gem_location_id"),
                    gspt.get("project_name", "Unknown"),
                    gspt.get("phase_name"),
                    gspt.get("country", "Unknown"),
                    gspt.get("state_province"),
                    capacity,
                    gspt.get("capacity_rating"),
                    gspt.get("status", "Unknown"),
                    gspt.get("start_year"),
                    lat,
                    lon,
                    gspt.get("location_accuracy"),
                    gspt.get("owner"),
                    gspt.get("operator"),
                    gspt.get("other_ids"),
                    gspt.get("wiki_url"),
                    json.dumps(p.get("grw_polygons")) if p.get("grw_polygons") else None,
                    json.dumps(p.get("merged_polygon")) if p.get("merged_polygon") else None,
                    p.get("match_confidence"),
                    p.get("match_distance_km"),
                    p.get("grw_construction_date"),
                ),
            )
            inserted += 1

    conn.commit()
    print(f"Inserted {inserted} projects, skipped {skipped}")


def upload_to_s3():
    """Upload merged data to S3."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from s3_utils import S3Storage

        storage = S3Storage()
        storage.upload_file(str(MERGED_FILE), "webapp/projects_merged.json")
        print("Uploaded to S3: webapp/projects_merged.json")
    except Exception as e:
        print(f"S3 upload failed (non-critical): {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Seed Vercel Postgres database")
    parser.add_argument("--schema-only", action="store_true", help="Only create schema, don't insert data")
    parser.add_argument("--skip-s3", action="store_true", help="Skip S3 upload")
    parser.add_argument("--dump-sql", action="store_true", help="Print SQL instead of executing")
    args = parser.parse_args()

    if args.dump_sql:
        print(SCHEMA_SQL)
        print("\n-- Data would be inserted from projects_merged.json")
        return

    conn = get_connection()
    try:
        create_schema(conn)
        if not args.schema_only:
            seed_projects(conn)
    finally:
        conn.close()

    if not args.skip_s3:
        upload_to_s3()

    print("\nDone!")


if __name__ == "__main__":
    main()
