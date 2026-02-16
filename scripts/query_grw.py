"""Query Global Renewables Watch (GRW) solar polygons from GEE and match to sites.

Downloads all Bangladesh solar installation polygons from the GRW dataset
(projects/sat-io/open-datasets/GRW/SOLAR_V1), then matches them to our 15 sites
using a 5km search radius. Nearby polygons belonging to the same farm are merged
via morphological close (buffer → union → un-buffer).

Outputs:
    data/grw/bangladesh_solar.geojson  — raw GRW polygons for Bangladesh
    data/grw/site_matches.json         — matched/merged polygons per site

Usage:
    python scripts/query_grw.py
"""

import ee
import json
import math
from pathlib import Path
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union

ee.Initialize(project="bangladesh-solar")

PROJECT_DIR = Path("/Users/anushreechaudhuri/Documents/Projects/solar-landuse")
GRW_DIR = PROJECT_DIR / "data" / "grw"
GRW_DIR.mkdir(parents=True, exist_ok=True)

# All sites (from download_all_sites.py + generate_dynamic_world_masks.py)
SITES = {
    "teesta": {
        "name": "Teesta (Gaibandha/Beximco) 200 MW",
        "lat": 25.628342, "lon": 89.541082,
        "completed": "2023-01-08", "mw": 200,
    },
    "feni": {
        "name": "Feni/Sonagazi 75 MW",
        "lat": 22.787567, "lon": 91.367187,
        "completed": "2024-04-01", "mw": 75,
    },
    "manikganj": {
        "name": "Manikganj (Spectra) 35 MW",
        "lat": 23.780834, "lon": 89.824775,
        "completed": "2021-03-12", "mw": 35,
    },
    "moulvibazar": {
        "name": "Moulvibazar 10 MW",
        "lat": 24.493896, "lon": 91.633043,
        "completed": "2025-10-01", "mw": 10,
    },
    "pabna": {
        "name": "Pabna 64 MW",
        "lat": 23.961375, "lon": 89.159720,
        "completed": "2024-10-23", "mw": 64,
    },
    "mymensingh": {
        "name": "Mymensingh (HDFC) 50 MW",
        "lat": 24.702233, "lon": 90.461730,
        "completed": "2020-11-04", "mw": 50,
    },
    "tetulia": {
        "name": "Tetulia/Panchagarh (Sympa) 8 MW",
        "lat": 26.482817, "lon": 88.410139,
        "completed": "2019-05-13", "mw": 8,
    },
    "lalmonirhat": {
        "name": "Lalmonirhat Rangpur (Intraco) 30 MW",
        "lat": 25.997873, "lon": 89.154467,
        "completed": "2022-08-28", "mw": 30,
    },
    "mongla": {
        "name": "Mongla 100 MW",
        "lat": 22.574239, "lon": 89.570388,
        "completed": "2021-12-29", "mw": 100,
    },
    "sirajganj68": {
        "name": "Sirajganj 68 MW",
        "lat": 24.403976, "lon": 89.738849,
        "completed": "2024-07-14", "mw": 68,
    },
    "teknaf": {
        "name": "Teknaf (Joules) 20 MW",
        "lat": 20.981669, "lon": 92.256021,
        "completed": "2018-09-15", "mw": 20,
    },
    "sirajganj6": {
        "name": "Sirajganj 6 MW",
        "lat": 24.386137, "lon": 89.748970,
        "completed": "2021-03-30", "mw": 6,
    },
    "kaptai": {
        "name": "Kaptai 7.4 MW",
        "lat": 22.491471, "lon": 92.226588,
        "completed": "2019-05-06", "mw": 7.4,
    },
    "sharishabari": {
        "name": "Sharishabari 3 MW",
        "lat": 24.772287, "lon": 89.842629,
        "completed": "2017-07-14", "mw": 3,
    },
    "barishal": {
        "name": "Barishal 1 MW",
        "lat": 22.657015, "lon": 90.339194,
        "completed": "2024-06-08", "mw": 1,
    },
}

SEARCH_RADIUS_KM = 5
MERGE_BUFFER_M = 500  # buffer for morphological close to merge nearby polygons

# Approximate meters-per-degree at Bangladesh latitudes (~23°N)
M_PER_DEG_LAT = 111_000
M_PER_DEG_LON = 111_000 * math.cos(math.radians(23.5))


def query_grw_bangladesh():
    """Download all GRW solar polygons in Bangladesh from GEE."""
    out_path = GRW_DIR / "bangladesh_solar.geojson"

    print("Querying GRW SOLAR_V1 for Bangladesh...")
    fc = ee.FeatureCollection("projects/sat-io/open-datasets/GRW/SOLAR_V1")
    bgd = fc.filter(ee.Filter.eq("COUNTRY", "Bangladesh"))

    count = bgd.size().getInfo()
    print(f"  Found {count} solar polygons in Bangladesh")

    if count == 0:
        print("  WARNING: No polygons found. Saving empty GeoJSON.")
        geojson = {"type": "FeatureCollection", "features": []}
    else:
        geojson = bgd.getInfo()

    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"  Saved to {out_path}")

    return geojson


def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate distance in km between two lat/lon points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def match_polygons_to_sites(geojson):
    """Match GRW polygons to sites within SEARCH_RADIUS_KM, merge nearby polygons."""
    features = geojson.get("features", [])
    if not features:
        print("No GRW features to match.")
        return {}

    # Convert features to shapely geometries with properties
    grw_polys = []
    for feat in features:
        geom = shape(feat["geometry"])
        props = feat.get("properties", {})
        centroid = geom.centroid
        grw_polys.append({
            "geometry": geom,
            "centroid_lat": centroid.y,
            "centroid_lon": centroid.x,
            "area_m2": props.get("area"),
            "construction_year": props.get("construction_year"),
            "construction_quarter": props.get("construction_quarter"),
            "land_cover_2018": props.get("landcover_in_2018"),
        })

    print(f"\nMatching {len(grw_polys)} GRW polygons to {len(SITES)} sites "
          f"(search radius: {SEARCH_RADIUS_KM} km)...\n")

    matches = {}
    for site_key, site in SITES.items():
        site_lat, site_lon = site["lat"], site["lon"]

        # Find all polygons within search radius
        nearby = []
        for poly in grw_polys:
            dist = haversine_km(site_lat, site_lon,
                                poly["centroid_lat"], poly["centroid_lon"])
            if dist <= SEARCH_RADIUS_KM:
                nearby.append({**poly, "distance_km": round(dist, 3)})

        if not nearby:
            matches[site_key] = {
                "name": site["name"],
                "match_status": "no_match",
                "polygons": [],
                "total_area_m2": 0,
            }
            print(f"  {site_key}: no GRW polygons within {SEARCH_RADIUS_KM} km")
            continue

        # Sort by distance
        nearby.sort(key=lambda x: x["distance_km"])

        # Merge nearby polygons via morphological close
        # Buffer in degrees (approximate conversion)
        buf_lat = MERGE_BUFFER_M / M_PER_DEG_LAT
        buf_lon = MERGE_BUFFER_M / M_PER_DEG_LON
        # Use average buffer since we're in geographic coords
        buf_deg = (buf_lat + buf_lon) / 2

        raw_geoms = [p["geometry"] for p in nearby]
        buffered = [g.buffer(buf_deg) for g in raw_geoms]
        merged_buffered = unary_union(buffered)
        # Un-buffer to restore original scale
        merged = merged_buffered.buffer(-buf_deg)

        # Handle MultiPolygon vs Polygon
        if merged.is_empty:
            merged_geoms = []
        elif merged.geom_type == "MultiPolygon":
            merged_geoms = list(merged.geoms)
        else:
            merged_geoms = [merged]

        total_area = sum(p.get("area_m2", 0) or 0 for p in nearby)

        matches[site_key] = {
            "name": site["name"],
            "match_status": "matched",
            "num_raw_polygons": len(nearby),
            "num_merged_polygons": len(merged_geoms),
            "total_area_m2": round(total_area, 1),
            "construction_dates": list(set(
                f"{p['construction_year']}Q{p['construction_quarter']}"
                for p in nearby
                if p.get("construction_year") and p.get("construction_quarter")
            )),
            "min_distance_km": nearby[0]["distance_km"],
            "polygons": [
                {
                    "wkt": g.wkt,
                    "geojson": mapping(g),
                    "area_m2": round(g.area * M_PER_DEG_LAT * M_PER_DEG_LON, 1),
                }
                for g in merged_geoms
            ],
        }

        print(f"  {site_key}: {len(nearby)} raw → {len(merged_geoms)} merged polygon(s), "
              f"total area {total_area:.0f} m², "
              f"nearest {nearby[0]['distance_km']:.1f} km")

    return matches


def main():
    print("=" * 60)
    print("Query GRW Solar Polygons & Match to Sites")
    print("=" * 60)

    # Step 1: Query GRW from GEE
    geojson_path = GRW_DIR / "bangladesh_solar.geojson"
    if geojson_path.exists():
        print(f"Loading cached GRW data from {geojson_path}")
        with open(geojson_path) as f:
            geojson = json.load(f)
        print(f"  {len(geojson.get('features', []))} polygons loaded")
    else:
        geojson = query_grw_bangladesh()

    # Step 2: Match to sites
    matches = match_polygons_to_sites(geojson)

    # Step 3: Save matches
    out_path = GRW_DIR / "site_matches.json"
    with open(out_path, "w") as f:
        json.dump(matches, f, indent=2)
    print(f"\nSaved site matches to {out_path}")

    # Summary
    matched = sum(1 for m in matches.values() if m["match_status"] == "matched")
    total = len(matches)
    print(f"\n{'=' * 60}")
    print(f"Summary: {matched}/{total} sites matched to GRW polygons")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
