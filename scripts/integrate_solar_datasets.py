"""
Integrate three solar detection datasets: GEM/GSPT, GRW, and TZ-SAM.

Performs 3-way spatial matching with R-tree indexing, assigns confidence
tiers based on cross-dataset agreement, and outputs a unified project
database for downstream analysis.

Usage:
    python scripts/integrate_solar_datasets.py                    # Full South Asia
    python scripts/integrate_solar_datasets.py --country bangladesh  # Pilot
    python scripts/integrate_solar_datasets.py --country bangladesh --stats
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

DATA_DIR = Path(__file__).parent.parent / "data"
GSPT_FILE = DATA_DIR / "gspt_south_asia.json"
GRW_FILE = DATA_DIR / "grw_south_asia.geojson"
TZSAM_FILE = DATA_DIR / "tzsam_south_asia.geojson"
OUTPUT_FILE = DATA_DIR / "unified_solar_db.json"

# Matching thresholds
EXACT_RADIUS_KM = 5  # Max distance for exact GEM coords
APPROX_RADIUS_KM = 10  # Max distance for approximate GEM coords
HIGH_CONF_RADIUS_KM = 1  # Distance for high/very_high confidence
MIN_OVERLAP_FRAC = 0.1  # Min intersection/union for polygon overlap


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in km."""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_gspt(country_filter=None):
    """Load GEM/GSPT projects."""
    with open(GSPT_FILE) as f:
        projects = json.load(f)
    if country_filter:
        projects = [p for p in projects
                    if p.get("country", "").lower() == country_filter.lower()]
    print(f"  GEM/GSPT: {len(projects)} projects")
    return projects


def load_polygon_dataset(filepath, country_field, country_filter=None):
    """Load GRW or TZ-SAM geojson, parse geometries, build spatial index."""
    with open(filepath) as f:
        data = json.load(f)

    features = data["features"]
    if country_filter:
        features = [f for f in features
                    if f.get("properties", {}).get(country_field, "").lower()
                    == country_filter.lower()]

    records = []
    geometries = []
    for i, feat in enumerate(features):
        try:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            centroid = geom.centroid
            records.append({
                "idx": i,
                "properties": feat.get("properties", {}),
                "geometry": geom,
                "geojson": feat["geometry"],
                "centroid_lat": centroid.y,
                "centroid_lon": centroid.x,
                "claimed": False,
            })
            geometries.append(geom)
        except Exception as e:
            continue

    # Build STRtree spatial index
    tree = STRtree(geometries) if geometries else None

    print(f"  {filepath.name}: {len(records)} valid polygons")
    return records, tree, geometries


def find_polygon_overlaps(grw_records, grw_geoms, tzsam_records, tzsam_geoms):
    """Find all GRW↔TZ-SAM polygon pairs with significant overlap."""
    if not grw_geoms or not tzsam_geoms:
        return {}, {}

    tzsam_tree = STRtree(tzsam_geoms)

    # grw_idx → [(tzsam_idx, iou), ...]
    grw_to_tzsam = {}
    # tzsam_idx → [(grw_idx, iou), ...]
    tzsam_to_grw = {}

    for gi, grw_rec in enumerate(grw_records):
        grw_geom = grw_rec["geometry"]
        # Query TZ-SAM tree for candidates
        candidates = tzsam_tree.query(grw_geom)
        for ci in candidates:
            tzsam_geom = tzsam_geoms[ci]
            try:
                intersection = grw_geom.intersection(tzsam_geom)
                if intersection.is_empty:
                    continue
                union_area = grw_geom.union(tzsam_geom).area
                if union_area == 0:
                    continue
                iou = intersection.area / union_area
                if iou >= MIN_OVERLAP_FRAC:
                    grw_to_tzsam.setdefault(gi, []).append((ci, iou))
                    tzsam_to_grw.setdefault(ci, []).append((gi, iou))
            except Exception:
                continue

    overlap_count = sum(len(v) for v in grw_to_tzsam.values())
    print(f"  Found {overlap_count} GRW↔TZ-SAM polygon overlaps "
          f"(IoU >= {MIN_OVERLAP_FRAC})")
    return grw_to_tzsam, tzsam_to_grw


def point_to_polygon_km(lat, lon, geom):
    """Distance from point to nearest edge of polygon, in approximate km.
    Uses degree-based distance with latitude correction."""
    pt = Point(lon, lat)
    if geom.contains(pt):
        return 0.0
    # Shapely distance in degrees, convert to km with latitude correction
    dist_deg = geom.exterior.distance(pt) if geom.geom_type == 'Polygon' else geom.distance(pt)
    lat_correction = np.cos(np.radians(lat))
    # Average of lat and lon degree-to-km factors
    km_per_deg = 111.0 * np.sqrt((1 + lat_correction**2) / 2)
    return dist_deg * km_per_deg


def find_nearby_polygons(lat, lon, records, geoms, tree, radius_km):
    """Find polygon records within radius_km of a point.
    Uses point-to-polygon edge distance (not centroid distance) so
    large farms with distant centroids still match nearby GEM coords."""
    if tree is None:
        return []

    # Approximate degree buffer for spatial query (generous)
    deg_buffer = radius_km / 111.0 * 2.0
    query_point = Point(lon, lat)
    query_box = query_point.buffer(deg_buffer)

    candidates = tree.query(query_box)
    results = []
    for ci in candidates:
        rec = records[ci]
        # Use point-to-polygon distance (much better for large installations)
        dist = point_to_polygon_km(lat, lon, rec["geometry"])
        if dist <= radius_km:
            results.append((ci, dist))

    return results


def assign_treatment_group(gem_status, has_polygon):
    """Determine treatment group from GEM status and polygon presence."""
    if has_polygon:
        return "operational"

    if gem_status is None:
        return "unknown"

    status_lower = gem_status.lower()
    operational_statuses = {"operating", "construction"}
    proposed_statuses = {"announced", "pre-construction", "shelved",
                         "shelved - inferred 2 y", "cancelled",
                         "cancelled - inferred 4 y"}

    if status_lower in operational_statuses:
        return "operational"
    elif status_lower in proposed_statuses:
        return "proposed"
    else:
        return "unknown"


def best_construction_year(gem_rec, grw_recs, tzsam_recs):
    """Pick best construction year from available sources."""
    years = []

    if gem_rec and gem_rec.get("start_year"):
        years.append(int(gem_rec["start_year"]))

    for r in grw_recs:
        y = r["properties"].get("construction_year")
        if y is not None:
            years.append(int(y))

    for r in tzsam_recs:
        # TZ-SAM has constructed_before/after dates
        cb = r["properties"].get("constructed_before", "")
        if cb:
            try:
                years.append(int(cb[:4]))
            except (ValueError, IndexError):
                pass

    return min(years) if years else None


def best_capacity(gem_rec, tzsam_recs):
    """Pick best capacity from available sources. Prefer GEM, fall back to TZ-SAM."""
    if gem_rec and gem_rec.get("capacity_mw"):
        return gem_rec["capacity_mw"]

    for r in tzsam_recs:
        cap = r["properties"].get("capacity_mw")
        if cap and cap > 0:
            return round(cap, 2)

    return None


def best_centroid(gem_rec, grw_recs, tzsam_recs):
    """Pick best centroid location. Prefer GEM exact coords, then polygon centroid."""
    if gem_rec and gem_rec.get("latitude") and gem_rec.get("longitude"):
        return gem_rec["latitude"], gem_rec["longitude"]

    # Use largest polygon centroid
    all_recs = grw_recs + tzsam_recs
    if all_recs:
        largest = max(all_recs, key=lambda r: r["geometry"].area)
        return largest["centroid_lat"], largest["centroid_lon"]

    return None, None


def format_polygon_info(rec, source):
    """Format polygon record for output JSON."""
    props = rec["properties"]
    if source == "grw":
        return {
            "fid": props.get("fid"),
            "area_m2": props.get("area"),
            "construction_year": props.get("construction_year"),
            "construction_quarter": props.get("construction_quarter"),
            "polygon": rec["geojson"],
        }
    else:  # tzsam
        return {
            "cluster_id": props.get("cluster_id"),
            "capacity_mw": props.get("capacity_mw"),
            "constructed_before": props.get("constructed_before"),
            "constructed_after": props.get("constructed_after"),
            "polygon": rec["geojson"],
        }


def integrate(country_filter=None, show_stats=False):
    print("Loading datasets...")
    gspt_projects = load_gspt(country_filter)
    grw_records, grw_tree, grw_geoms = load_polygon_dataset(
        GRW_FILE, "COUNTRY", country_filter)
    tzsam_records, tzsam_tree, tzsam_geoms = load_polygon_dataset(
        TZSAM_FILE, "country", country_filter)

    # Step 1: Find GRW↔TZ-SAM polygon overlaps
    print("\nFinding polygon overlaps...")
    grw_to_tzsam, tzsam_to_grw = find_polygon_overlaps(
        grw_records, grw_geoms, tzsam_records, tzsam_geoms)

    # Step 2: Match GEM projects to polygons
    print("\nMatching GEM projects to polygons...")
    unified = []
    country_counters = {}  # country → next ID number

    for i, gem in enumerate(gspt_projects):
        if (i + 1) % 500 == 0:
            print(f"  Processing {i + 1}/{len(gspt_projects)}...")

        lat = gem.get("latitude")
        lon = gem.get("longitude")
        country = gem.get("country", "Unknown")

        sources = ["gem"]
        matched_grw = []
        matched_tzsam = []
        min_grw_dist = float("inf")
        min_tzsam_dist = float("inf")

        if lat is not None and lon is not None:
            # Use location accuracy for search radius
            is_exact = (gem.get("location_accuracy", "") or "").lower() == "exact"
            radius_km = EXACT_RADIUS_KM if is_exact else APPROX_RADIUS_KM

            # Find nearby GRW polygons
            grw_nearby = find_nearby_polygons(
                lat, lon, grw_records, grw_geoms, grw_tree, radius_km)
            for gi, dist in grw_nearby:
                matched_grw.append(grw_records[gi])
                grw_records[gi]["claimed"] = True
                min_grw_dist = min(min_grw_dist, dist)

            # Find nearby TZ-SAM polygons
            tzsam_nearby = find_nearby_polygons(
                lat, lon, tzsam_records, tzsam_geoms, tzsam_tree, radius_km)
            for ti, dist in tzsam_nearby:
                matched_tzsam.append(tzsam_records[ti])
                tzsam_records[ti]["claimed"] = True
                min_tzsam_dist = min(min_tzsam_dist, dist)

        has_grw = len(matched_grw) > 0
        has_tzsam = len(matched_tzsam) > 0
        if has_grw:
            sources.append("grw")
        if has_tzsam:
            sources.append("tzsam")

        # Check if any matched GRW and TZ-SAM polygons overlap each other
        has_polygon_overlap = False
        if has_grw and has_tzsam:
            matched_grw_indices = set()
            for gi, _ in grw_nearby:
                matched_grw_indices.add(gi)
            for gi in matched_grw_indices:
                if gi in grw_to_tzsam:
                    has_polygon_overlap = True
                    break

        # Assign confidence
        min_dist = min(min_grw_dist, min_tzsam_dist)
        if has_polygon_overlap and min_dist <= HIGH_CONF_RADIUS_KM:
            confidence = "very_high"
        elif (has_grw or has_tzsam) and min_dist <= HIGH_CONF_RADIUS_KM:
            confidence = "high"
        elif has_grw or has_tzsam:
            confidence = "medium"
        else:
            confidence = "low"

        # Generate site ID
        cc = country[:2].upper()
        country_counters.setdefault(cc, 0)
        country_counters[cc] += 1
        site_id = f"{cc}_{country_counters[cc]:04d}"

        has_polygon = has_grw or has_tzsam
        gem_status = gem.get("status")

        entry = {
            "site_id": site_id,
            "country": country,
            "confidence": confidence,
            "sources": sources,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "best_capacity_mw": best_capacity(gem, matched_tzsam),
            "best_construction_year": best_construction_year(
                gem, matched_grw, matched_tzsam),
            "treatment_group": assign_treatment_group(gem_status, has_polygon),
            "gem": {
                "gem_phase_id": gem.get("gem_phase_id"),
                "project_name": gem.get("project_name"),
                "status": gem_status,
                "start_year": gem.get("start_year"),
                "capacity_mw": gem.get("capacity_mw"),
                "location_accuracy": gem.get("location_accuracy"),
            },
        }

        if matched_grw:
            # Use closest GRW polygon as primary
            closest_grw = min(matched_grw, key=lambda r: haversine_km(
                lat, lon, r["centroid_lat"], r["centroid_lon"]))
            entry["grw"] = format_polygon_info(closest_grw, "grw")
            entry["grw"]["num_nearby"] = len(matched_grw)

        if matched_tzsam:
            closest_tzsam = min(matched_tzsam, key=lambda r: haversine_km(
                lat, lon, r["centroid_lat"], r["centroid_lon"]))
            entry["tzsam"] = format_polygon_info(closest_tzsam, "tzsam")
            entry["tzsam"]["num_nearby"] = len(matched_tzsam)

        unified.append(entry)

    # Step 3: Process unclaimed GRW↔TZ-SAM overlap pairs
    print("\nProcessing unclaimed polygon pairs...")
    unclaimed_pairs = 0
    for gi, overlaps in grw_to_tzsam.items():
        if grw_records[gi]["claimed"]:
            continue
        for ti, iou in overlaps:
            if tzsam_records[ti]["claimed"]:
                continue

            # Create new entry from polygon pair
            grw_rec = grw_records[gi]
            tzsam_rec = tzsam_records[ti]
            grw_records[gi]["claimed"] = True
            tzsam_records[ti]["claimed"] = True

            centroid_lat, centroid_lon = best_centroid(
                None, [grw_rec], [tzsam_rec])
            country = tzsam_rec["properties"].get(
                "country", grw_rec["properties"].get("COUNTRY", "Unknown"))
            cc = country[:2].upper()
            country_counters.setdefault(cc, 0)
            country_counters[cc] += 1
            site_id = f"{cc}_{country_counters[cc]:04d}"

            entry = {
                "site_id": site_id,
                "country": country,
                "confidence": "medium",
                "sources": ["grw", "tzsam"],
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "best_capacity_mw": best_capacity(None, [tzsam_rec]),
                "best_construction_year": best_construction_year(
                    None, [grw_rec], [tzsam_rec]),
                "treatment_group": "operational",
                "grw": format_polygon_info(grw_rec, "grw"),
                "tzsam": format_polygon_info(tzsam_rec, "tzsam"),
            }
            unified.append(entry)
            unclaimed_pairs += 1

    print(f"  Added {unclaimed_pairs} unclaimed GRW↔TZ-SAM pairs")

    # Step 4: Process remaining unclaimed polygons
    print("Processing remaining unclaimed polygons...")
    unclaimed_grw = 0
    for gi, rec in enumerate(grw_records):
        if rec["claimed"]:
            continue
        country = rec["properties"].get("COUNTRY", "Unknown")
        cc = country[:2].upper()
        country_counters.setdefault(cc, 0)
        country_counters[cc] += 1
        site_id = f"{cc}_{country_counters[cc]:04d}"

        entry = {
            "site_id": site_id,
            "country": country,
            "confidence": "low",
            "sources": ["grw"],
            "centroid_lat": rec["centroid_lat"],
            "centroid_lon": rec["centroid_lon"],
            "best_capacity_mw": None,
            "best_construction_year": best_construction_year(
                None, [rec], []),
            "treatment_group": "operational",
            "grw": format_polygon_info(rec, "grw"),
        }
        unified.append(entry)
        unclaimed_grw += 1

    unclaimed_tzsam = 0
    for ti, rec in enumerate(tzsam_records):
        if rec["claimed"]:
            continue
        country = rec["properties"].get("country", "Unknown")
        cc = country[:2].upper()
        country_counters.setdefault(cc, 0)
        country_counters[cc] += 1
        site_id = f"{cc}_{country_counters[cc]:04d}"

        entry = {
            "site_id": site_id,
            "country": country,
            "confidence": "low",
            "sources": ["tzsam"],
            "centroid_lat": rec["centroid_lat"],
            "centroid_lon": rec["centroid_lon"],
            "best_capacity_mw": best_capacity(None, [rec]),
            "best_construction_year": best_construction_year(
                None, [], [rec]),
            "treatment_group": "operational",
            "tzsam": format_polygon_info(rec, "tzsam"),
        }
        unified.append(entry)
        unclaimed_tzsam += 1

    print(f"  Added {unclaimed_grw} unclaimed GRW-only, "
          f"{unclaimed_tzsam} unclaimed TZ-SAM-only")

    # Save output
    print(f"\nSaving {len(unified)} entries to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(unified, f, indent=2)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved ({size_mb:.1f} MB)")

    if show_stats:
        print_stats(unified)

    return unified


def print_stats(unified):
    """Print summary statistics and source overlap analysis."""
    print("\n" + "=" * 60)
    print("UNIFIED SOLAR DATABASE STATISTICS")
    print("=" * 60)

    # Confidence distribution
    conf_counts = Counter(e["confidence"] for e in unified)
    print(f"\nConfidence distribution:")
    for tier in ["very_high", "high", "medium", "low"]:
        n = conf_counts.get(tier, 0)
        pct = 100 * n / len(unified) if unified else 0
        bar = "#" * int(pct / 2)
        print(f"  {tier:>10}: {n:5d} ({pct:5.1f}%) {bar}")
    print(f"  {'TOTAL':>10}: {len(unified):5d}")

    # Treatment group distribution
    treat_counts = Counter(e["treatment_group"] for e in unified)
    print(f"\nTreatment groups:")
    for group in ["operational", "proposed", "unknown"]:
        n = treat_counts.get(group, 0)
        print(f"  {group:>12}: {n}")

    # Source overlap (Venn diagram counts)
    source_combos = Counter(tuple(sorted(e["sources"])) for e in unified)
    print(f"\nSource overlap:")
    combo_labels = {
        ("gem",): "GEM only",
        ("grw",): "GRW only",
        ("tzsam",): "TZ-SAM only",
        ("gem", "grw"): "GEM ∩ GRW",
        ("gem", "tzsam"): "GEM ∩ TZ-SAM",
        ("grw", "tzsam"): "GRW ∩ TZ-SAM",
        ("gem", "grw", "tzsam"): "GEM ∩ GRW ∩ TZ-SAM",
    }
    for combo in [("gem", "grw", "tzsam"), ("gem", "grw"), ("gem", "tzsam"),
                  ("grw", "tzsam"), ("gem",), ("grw",), ("tzsam",)]:
        n = source_combos.get(combo, 0)
        label = combo_labels.get(combo, str(combo))
        print(f"  {label:>20}: {n}")

    # Country distribution
    country_counts = Counter(e["country"] for e in unified)
    print(f"\nBy country:")
    for cc, n in country_counts.most_common():
        conf_by_country = Counter(
            e["confidence"] for e in unified if e["country"] == cc)
        conf_str = ", ".join(f"{t}={conf_by_country.get(t, 0)}"
                             for t in ["very_high", "high", "medium", "low"]
                             if conf_by_country.get(t, 0) > 0)
        print(f"  {cc:>15}: {n:5d}  ({conf_str})")

    # Capacity summary
    caps = [e["best_capacity_mw"] for e in unified
            if e["best_capacity_mw"] is not None]
    if caps:
        print(f"\nCapacity (MW):")
        print(f"  Known: {len(caps)}/{len(unified)} sites")
        print(f"  Total: {sum(caps):.1f} MW")
        print(f"  Mean:  {np.mean(caps):.1f} MW")
        print(f"  Median: {np.median(caps):.1f} MW")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate GEM, GRW, and TZ-SAM solar datasets")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter to single country (e.g. 'bangladesh')")
    parser.add_argument("--stats", action="store_true",
                        help="Print detailed statistics")
    args = parser.parse_args()

    integrate(country_filter=args.country, show_stats=args.stats)


if __name__ == "__main__":
    main()
