"""
Match GSPT projects with GRW polygons.

For each GSPT project with coordinates, find nearby GRW polygons,
cluster them, score match confidence, and output unified records.
"""
import json
import sys
from pathlib import Path

from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
GSPT_FILE = DATA_DIR / "gspt_south_asia.json"
GRW_FILE = DATA_DIR / "grw_south_asia.geojson"
OUTPUT_FILE = DATA_DIR / "projects_merged.json"
UNMATCHED_FILE = DATA_DIR / "grw_unmatched.geojson"

# Search radii in km
EXACT_RADIUS_KM = 5
APPROX_RADIUS_KM = 10
# Buffer for clustering nearby polygons (degrees, ~100m at equator)
CLUSTER_BUFFER_DEG = 0.001


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in km."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def cluster_polygons(polygons):
    """Merge nearby polygons into clusters using buffer-union-unbuffer."""
    if not polygons:
        return []

    shapes = []
    for p in polygons:
        try:
            s = shape(p["geometry"])
            if s.is_valid:
                shapes.append((s, p["properties"]))
        except Exception:
            continue

    if not shapes:
        return []

    # Buffer all polygons, union, then unbuffer to merge nearby ones
    buffered = unary_union([s.buffer(CLUSTER_BUFFER_DEG) for s, _ in shapes])

    # Split back into individual clusters
    if buffered.geom_type == "MultiPolygon":
        cluster_geoms = list(buffered.geoms)
    else:
        cluster_geoms = [buffered]

    # For each cluster, find which original polygons belong to it
    clusters = []
    for cluster_geom in cluster_geoms:
        members = []
        props_list = []
        for s, props in shapes:
            if cluster_geom.intersects(s):
                members.append(s)
                props_list.append(props)

        # Merge the original (unbuffered) polygons
        merged = unary_union(members)
        # Collect properties
        total_area = sum(p.get("area", 0) or 0 for p in props_list)
        # Collect construction years from GRW properties
        years = sorted(set(
            int(p["construction_year"])
            for p in props_list
            if p.get("construction_year") is not None
        ))
        # Format dates as "YYYY-QN" for display
        dates = []
        for p in props_list:
            y = p.get("construction_year")
            q = p.get("construction_quarter")
            if y is not None:
                label = f"{int(y)}Q{int(q)}" if q else str(int(y))
                dates.append(label)
        dates = sorted(set(dates))

        clusters.append({
            "type": "Feature",
            "geometry": mapping(merged),
            "properties": {
                "area_m2": total_area,
                "construction_dates": dates,
                "construction_years": years,
                "num_raw_polygons": len(members),
            },
        })

    return clusters


def score_match(gspt, clusters, distance_km):
    """Score match confidence based on distance and date agreement."""
    if not clusters:
        return "none", None

    # Check date agreement using construction_years
    gspt_year = gspt.get("start_year")
    date_match = False
    if gspt_year:
        for c in clusters:
            for grw_year in c["properties"].get("construction_years", []):
                if abs(grw_year - gspt_year) <= 2:
                    date_match = True

    is_exact = gspt.get("location_accuracy", "").lower() == "exact"

    if is_exact and distance_km < 1 and date_match:
        return "high", distance_km
    elif is_exact and distance_km < 3:
        return "medium", distance_km
    elif distance_km < 5:
        return "medium", distance_km
    else:
        return "low", distance_km


def match_projects():
    print(f"Loading GSPT data from {GSPT_FILE}...")
    with open(GSPT_FILE) as f:
        gspt_projects = json.load(f)
    print(f"  {len(gspt_projects)} projects")

    print(f"Loading GRW data from {GRW_FILE}...")
    with open(GRW_FILE) as f:
        grw_data = json.load(f)
    grw_features = grw_data["features"]
    print(f"  {len(grw_features)} polygons")

    # Build spatial index: precompute centroids for GRW features
    print("Computing GRW centroids...")
    grw_centroids = []
    for feat in grw_features:
        try:
            s = shape(feat["geometry"])
            c = s.centroid
            grw_centroids.append((c.y, c.x))
        except Exception:
            grw_centroids.append((None, None))

    grw_lats = np.array([c[0] if c[0] is not None else 0 for c in grw_centroids])
    grw_lons = np.array([c[1] if c[1] is not None else 0 for c in grw_centroids])
    grw_valid = np.array([c[0] is not None for c in grw_centroids])

    # Match each GSPT project
    print("Matching projects...")
    results = []
    stats = {"high": 0, "medium": 0, "low": 0, "none": 0, "no_coords": 0}
    claimed_indices = set()  # Track which GRW features get matched

    for i, gspt in enumerate(gspt_projects):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i + 1}/{len(gspt_projects)}...")

        lat = gspt.get("latitude")
        lon = gspt.get("longitude")

        # Use GEM phase ID as primary key, fall back to location ID
        project_id = gspt.get("gem_phase_id") or gspt.get("gem_location_id") or f"gspt_{i}"

        record = {
            "id": project_id,
            "gspt": gspt,
            "grw_polygons": [],
            "merged_polygon": None,
            "match_confidence": "none",
            "match_distance_km": None,
            "grw_construction_date": None,
            "needs_review": True,
        }

        if lat is None or lon is None:
            stats["no_coords"] += 1
            results.append(record)
            continue

        # Determine search radius
        is_exact = (gspt.get("location_accuracy", "") or "").lower() == "exact"
        radius_km = EXACT_RADIUS_KM if is_exact else APPROX_RADIUS_KM

        # Quick filter: rough bounding box (1 degree ~ 111 km)
        deg_buffer = radius_km / 111.0 * 1.5  # 1.5x safety margin
        mask = (
            grw_valid
            & (np.abs(grw_lats - lat) < deg_buffer)
            & (np.abs(grw_lons - lon) < deg_buffer)
        )

        nearby_indices = np.where(mask)[0]
        if len(nearby_indices) == 0:
            stats["none"] += 1
            results.append(record)
            continue

        # Precise distance filter
        nearby = []
        nearby_idx_list = []
        min_dist = float("inf")
        for idx in nearby_indices:
            dist = haversine_km(lat, lon, grw_lats[idx], grw_lons[idx])
            if dist <= radius_km:
                nearby.append(grw_features[idx])
                nearby_idx_list.append(int(idx))
                min_dist = min(min_dist, dist)

        if not nearby:
            stats["none"] += 1
            results.append(record)
            continue

        # Cluster nearby polygons
        clusters = cluster_polygons(nearby)

        # Score match
        confidence, dist = score_match(gspt, clusters, min_dist)

        # Merge all clusters into one polygon
        merged = None
        if clusters:
            all_shapes = [shape(c["geometry"]) for c in clusters]
            merged_shape = unary_union(all_shapes)
            merged = {
                "type": "Feature",
                "geometry": mapping(merged_shape),
                "properties": {
                    "total_area_m2": sum(
                        c["properties"].get("area_m2", 0) for c in clusters
                    ),
                    "num_clusters": len(clusters),
                },
            }

        # Collect construction dates
        all_dates = []
        for c in clusters:
            all_dates.extend(c["properties"].get("construction_dates", []))
        all_dates = sorted(set(all_dates))

        record["grw_polygons"] = clusters
        record["merged_polygon"] = merged
        record["match_confidence"] = confidence
        record["match_distance_km"] = round(dist, 3) if dist else None
        record["grw_construction_date"] = ", ".join(all_dates) if all_dates else None
        record["needs_review"] = confidence != "high"

        # Mark these GRW features as claimed
        claimed_indices.update(nearby_idx_list)

        stats[confidence] += 1
        results.append(record)

    # Summary
    print(f"\nMatch results:")
    print(f"  High confidence:   {stats['high']}")
    print(f"  Medium confidence: {stats['medium']}")
    print(f"  Low confidence:    {stats['low']}")
    print(f"  No match:          {stats['none']}")
    print(f"  No coordinates:    {stats['no_coords']}")
    print(f"  Total:             {len(results)}")

    # Save matched results
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved {len(results)} records ({size_mb:.1f} MB)")

    # Save unmatched GRW features
    print(f"\nClaimed {len(claimed_indices)} GRW features across all matches")
    unmatched_features = []
    for idx, feat in enumerate(grw_features):
        if idx not in claimed_indices and grw_centroids[idx][0] is not None:
            props = feat.get("properties", {})
            unmatched_features.append({
                "type": "Feature",
                "geometry": feat["geometry"],
                "properties": {
                    **props,
                    "centroid_lat": grw_centroids[idx][0],
                    "centroid_lon": grw_centroids[idx][1],
                },
            })

    unmatched_geojson = {
        "type": "FeatureCollection",
        "features": unmatched_features,
    }
    print(f"Saving {len(unmatched_features)} unmatched GRW features to {UNMATCHED_FILE}...")
    with open(UNMATCHED_FILE, "w") as f:
        json.dump(unmatched_geojson, f)
    um_size = UNMATCHED_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved ({um_size:.1f} MB)")

    return results


if __name__ == "__main__":
    match_projects()
