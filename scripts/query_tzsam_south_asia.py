"""
Query TZ-SAM (Transition Zero) solar polygons from Google Earth Engine
for South Asian countries.

TZ-SAM has no 'country' property, so we use spatial bounding boxes to
filter features per country. Downloads all solar installation polygons
and saves as GeoJSON.
"""
import json
import sys
from pathlib import Path

import ee

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "tzsam_south_asia.geojson"

TZSAM_ASSET = "projects/sat-io/open-datasets/TZERO/TZ-SOLAR-2025Q3_ANALYSIS_POLYGONS"

# Bounding boxes [west, south, east, north] for each country
# Generous bounds to avoid missing border features.
# Ordered smallest→largest so dedup tags small countries first (before
# India's giant bbox claims them).
SOUTH_ASIA_BBOXES = [
    ("Bhutan", [88.7, 26.7, 92.1, 28.4]),
    ("Sri Lanka", [79.5, 5.9, 81.9, 9.9]),
    ("Nepal", [80.0, 26.3, 88.2, 30.5]),
    ("Bangladesh", [88.0, 20.5, 92.7, 26.7]),
    ("Pakistan", [60.8, 23.5, 77.8, 37.1]),
    ("India", [68.0, 6.5, 97.5, 37.1]),  # Last: catches all remaining
]

BATCH_SIZE = 5000


def query_country(country_name, bbox):
    """Query TZ-SAM features within a country's bounding box."""
    geom = ee.Geometry.Rectangle(bbox)
    filtered = ee.FeatureCollection(TZSAM_ASSET).filterBounds(geom)

    count = filtered.size().getInfo()
    print(f"  {country_name}: {count} features")

    if count == 0:
        return []

    if count <= BATCH_SIZE:
        fc = filtered.getInfo()
        features = fc["features"]
    else:
        print(f"    Downloading in batches of {BATCH_SIZE}...")
        sorted_fc = filtered.sort("system:index")
        feature_list = sorted_fc.toList(count)
        features = []
        for start in range(0, count, BATCH_SIZE):
            end = min(start + BATCH_SIZE, count)
            print(f"    Batch {start}-{end}...")
            batch = ee.FeatureCollection(feature_list.slice(start, end))
            batch_info = batch.getInfo()
            features.extend(batch_info["features"])
            print(f"    Got {len(batch_info['features'])} (total: {len(features)})")

    # Tag each feature with country name since TZ-SAM doesn't have one
    for f in features:
        f.setdefault("properties", {})["country"] = country_name

    return features


def deduplicate_features(all_features):
    """Remove duplicate features that fall in overlapping bboxes (e.g. India/Nepal border).
    Uses fid as unique key."""
    seen_fids = set()
    unique = []
    for f in all_features:
        fid = f.get("properties", {}).get("fid")
        if fid is not None and fid in seen_fids:
            continue
        if fid is not None:
            seen_fids.add(fid)
        unique.append(f)
    return unique


def main():
    print("Initializing GEE...")
    ee.Initialize(project="bangladesh-solar")

    print(f"Querying TZ-SAM solar dataset: {TZSAM_ASSET}")

    all_features = []
    for country, bbox in SOUTH_ASIA_BBOXES:
        features = query_country(country, bbox)
        all_features.extend(features)

    # Deduplicate (overlapping bboxes at borders)
    before_dedup = len(all_features)
    all_features = deduplicate_features(all_features)
    if before_dedup != len(all_features):
        print(f"\nDeduplicated: {before_dedup} → {len(all_features)} features")

    # Build GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    # Summary
    country_counts = {}
    total_capacity = 0
    for f in all_features:
        props = f.get("properties", {})
        cc = props.get("country", "unknown")
        country_counts[cc] = country_counts.get(cc, 0) + 1
        cap = props.get("capacity_mw", 0) or 0
        total_capacity += cap

    print(f"\nBy country:")
    for cc, n in sorted(country_counts.items(), key=lambda x: -x[1]):
        print(f"  {cc}: {n}")
    print(f"\nTotal capacity: {total_capacity:.1f} MW")
    print(f"Total features: {len(all_features)}")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(geojson, f)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved {len(all_features)} features ({size_mb:.1f} MB)")

    return geojson


if __name__ == "__main__":
    main()
