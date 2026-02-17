"""
Query GRW (Global Renewables Watch) solar polygons from Google Earth Engine
for South Asian countries.

Downloads all solar installation polygons and saves as GeoJSON.
"""
import json
import sys
from pathlib import Path

import ee

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "grw_south_asia.geojson"

# GRW uses full country names
SOUTH_ASIA_COUNTRIES = ["India", "Bangladesh", "Pakistan", "Bhutan", "Nepal", "Sri Lanka"]

GRW_ASSET = "projects/sat-io/open-datasets/GRW/SOLAR_V1"


def query_grw():
    print("Initializing GEE...")
    ee.Initialize(project="bangladesh-solar")

    print(f"Loading GRW solar dataset: {GRW_ASSET}")
    grw = ee.FeatureCollection(GRW_ASSET)

    # Filter by country name
    print(f"Filtering for countries: {SOUTH_ASIA_COUNTRIES}")
    south_asia = grw.filter(ee.Filter.inList("COUNTRY", SOUTH_ASIA_COUNTRIES))

    # Get count
    count = south_asia.size().getInfo()
    print(f"Found {count} solar features in South Asia")

    if count == 0:
        print("No features found. Check country code field name.")
        # Try to get a sample to inspect property names
        sample = grw.limit(1).getInfo()
        if sample["features"]:
            print(f"Available properties: {list(sample['features'][0]['properties'].keys())}")
        return

    # Download in batches to handle large collections
    # GEE has a limit on getInfo() size, so we paginate
    BATCH_SIZE = 5000
    all_features = []

    if count <= BATCH_SIZE:
        print(f"Downloading {count} features in single batch...")
        fc = south_asia.getInfo()
        all_features = fc["features"]
    else:
        print(f"Downloading {count} features in batches of {BATCH_SIZE}...")
        # Sort by a property to ensure consistent pagination
        sorted_fc = south_asia.sort("system:index")
        feature_list = sorted_fc.toList(count)

        for start in range(0, count, BATCH_SIZE):
            end = min(start + BATCH_SIZE, count)
            print(f"  Batch {start}-{end}...")
            batch = ee.FeatureCollection(feature_list.slice(start, end))
            batch_info = batch.getInfo()
            all_features.extend(batch_info["features"])
            print(f"  Got {len(batch_info['features'])} features (total: {len(all_features)})")

    # Build GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    # Summary by country
    country_counts = {}
    total_area = 0
    for f in all_features:
        props = f.get("properties", {})
        cc = props.get("COUNTRY", "unknown")
        country_counts[cc] = country_counts.get(cc, 0) + 1
        area = props.get("area", 0) or 0
        total_area += area

    print(f"\nBy country:")
    for cc, n in sorted(country_counts.items(), key=lambda x: -x[1]):
        print(f"  {cc}: {n}")
    print(f"\nTotal area: {total_area / 1e6:.1f} km²")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(geojson, f)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved {len(all_features)} features ({size_mb:.1f} MB)")

    return geojson


if __name__ == "__main__":
    query_grw()
