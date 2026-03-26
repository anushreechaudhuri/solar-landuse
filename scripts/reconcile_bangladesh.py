"""
Reconcile user-verified Bangladesh solar projects with GEM entries.

Updates projects_merged.json and optionally the Postgres database with:
- Verified polygon geometries from confirmed_matches.json
- Corrected coordinates (centroid of verified polygons)
- match_confidence = "verified"
- User's common name stored in metadata

Usage:
    python scripts/reconcile_bangladesh.py                  # Update projects_merged.json only
    python scripts/reconcile_bangladesh.py --update-db      # Also update Postgres directly
    python scripts/reconcile_bangladesh.py --dry-run        # Print changes without writing
"""
import json
import math
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_DIR = Path(__file__).parent.parent / "data"
MERGED_FILE = DATA_DIR / "projects_merged.json"
CONFIRMED_FILE = DATA_DIR / "grw" / "confirmed_matches.json"
GEOJSON_FILE = DATA_DIR / "grw" / "bangladesh_solar.geojson"

# ── Reconciliation mapping ─────────────────────────────────────────────
# Maps user project key (from confirmed_matches.json) → GEM phase ID(s)
# with match reasoning
MATCHES = {
    "manikganj": {
        "gem_ids": ["G100000801307"],
        "user_name": "Manikganj (Spectra) 35 MW",
        "reason": "Exact: same name, 35 MW, operating 2021, Spectra/Shunfeng owner, 0.1 km",
    },
    "feni": {
        "gem_ids": ["G100000801623"],
        "user_name": "Feni/Sonagazi 75 MW",
        "reason": "Exact: Sonagazi (EGCB) 75 MW, operating, EGCB owner, 0.5 km",
    },
    "mymensingh": {
        "gem_ids": ["G100000801649"],
        "user_name": "Mymensingh (HDFC) 50 MW",
        "reason": "Exact: Sutiakhali 50 MW, operating 2020, HDFC Sinpower owner, 0.6 km. GEM name 'Sutiakhali' differs from common name",
    },
    "teknaf": {
        "gem_ids": ["G100000801684"],
        "user_name": "Teknaf (Joules) 20 MW",
        "reason": "Exact: Teknaf (Tsel) 20 MW, operating 2018, Joules Power/TSEL, 0.4 km",
    },
    "tetulia": {
        "gem_ids": ["G100001029704"],
        "user_name": "Tetulia/Panchagarh (Sympa) 8 MW",
        "reason": "Close: Tetulia solar project 10.1 MW, operating 2019, 0.1 km. Capacity differs (8 vs 10.1 MW)",
    },
    "kaptai": {
        "gem_ids": ["G100001029703"],
        "user_name": "Kaptai 7.4 MW",
        "reason": "Close: Kaptai Upazila 5.4 MW, operating 2018, 0.0 km. Capacity differs (7.4 vs 5.4 MW)",
    },
    "sharishabari": {
        "gem_ids": ["G100001029702"],
        "user_name": "Sharishabari 3 MW",
        "reason": "Close: 'Mymensingh Division solar project' 2.5 MW, 0.0 km. GEM name is generic/wrong",
    },
    "sirajganj6": {
        "gem_ids": ["G100000833185"],
        "user_name": "Sirajganj 6 MW",
        "reason": "Close: 'Syedabad solar project' 5.7 MW, 0.0 km. GEM name is wrong (should be Sirajganj)",
    },
    "sirajganj68": {
        "gem_ids": ["G100000801593", "G100000801185"],
        "user_name": "Sirajganj 68 MW",
        "reason": "Exact: Sirajganji 68 MW (G100000801593) + duplicate Jamuna 68 MW (G100000801185), both BCRECL, 0.4 km",
    },
    "teesta": {
        "gem_ids": ["G100000801112"],
        "user_name": "Teesta (Gaibandha/Beximco) 200 MW",
        "reason": "Capacity match: Gaibandha Beximco 200 MW, operating 2023, Beximco owner. GEM coords 33 km off (city centroid). Note: G100000801488 Sundarganj 275 MW may be related/duplicate",
    },
    "lalmonirhat": {
        "gem_ids": ["G100000801490"],
        "user_name": "Lalmonirhat Rangpur (Intraco) 30 MW",
        "reason": "Developer+capacity match: Rangpur (Intraco) 30 MW, operating 2022. GEM coords 17 km off",
    },
    "mongla": {
        "gem_ids": ["G100000801342"],
        "user_name": "Mongla 100 MW",
        "reason": "Name+capacity match: Mongla solar farm 100 MW, operating 2021, Energon. GEM coords 21 km off. Note: Moidhara (G100000810865, 134 MW) is 0.5 km away - may be adjacent project",
    },
    "pabna": {
        "gem_ids": ["G100000801420"],
        "user_name": "Pabna 64 MW",
        "reason": "Capacity match: Pabna (BCRECL) 65 MW, operating 2025. GEM coords 22 km off in wrong district",
    },
    # No GEM match for these:
    # "moulvibazar": GEM only has 100 MW announced (G100001021366), not the 10 MW operating project
    # "barishal": Not in GEM at all
}

# Projects without polygons that can still be matched by metadata
METADATA_ONLY_MATCHES = {
    "Barapukuria Dinajpur 50 MW": {
        "gem_ids": ["G100000833175"],
        "reason": "Name+capacity: Barapkuria solar farm 50 MW, pre-construction, Summit Power",
    },
}


def polygon_centroid(polygons):
    """Compute centroid from a list of GeoJSON polygon geometries."""
    all_coords = []
    for poly in polygons:
        ring = poly["coordinates"][0]
        all_coords.extend(ring)
    if not all_coords:
        return None, None
    lat = sum(c[1] for c in all_coords) / len(all_coords)
    lon = sum(c[0] for c in all_coords) / len(all_coords)
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def make_merged_polygon(polygons, user_name, reason):
    """Create a GeoJSON Feature for the merged_polygon field."""
    if len(polygons) == 1:
        geometry = polygons[0]
    else:
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [p["coordinates"] for p in polygons],
        }
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "source": "user_verified",
            "user_common_name": user_name,
            "match_reason": reason,
        },
    }


def run(dry_run=False, update_db=False):
    # Load user data
    print("Loading verified polygons...")
    with open(CONFIRMED_FILE) as f:
        confirmed = json.load(f)

    # Also load the geojson for any additional polygon data
    with open(GEOJSON_FILE) as f:
        geojson = json.load(f)
    geojson_by_centroid = {}
    for feat in geojson["features"]:
        geom = feat["geometry"]
        if geom["type"] == "MultiPolygon":
            coords = [c for poly in geom["coordinates"] for c in poly[0]]
        else:
            coords = geom["coordinates"][0]
        clat = sum(c[1] for c in coords) / len(coords)
        clon = sum(c[0] for c in coords) / len(coords)
        geojson_by_centroid[(round(clat, 3), round(clon, 3))] = feat

    # Load projects_merged.json
    print(f"Loading {MERGED_FILE} (this may take a moment)...")
    with open(MERGED_FILE) as f:
        projects = json.load(f)
    print(f"  {len(projects)} projects loaded")

    # Build lookup by GEM ID
    project_idx = {p["id"]: i for i, p in enumerate(projects)}

    updates = []
    unmatched_user = []

    for key, match_info in MATCHES.items():
        user_data = confirmed.get(key)
        if not user_data:
            print(f"  WARNING: '{key}' not in confirmed_matches.json, skipping")
            continue

        polygons = user_data["polygons"]
        user_name = match_info["user_name"]
        reason = match_info["reason"]
        clat, clon = polygon_centroid(polygons)

        merged_poly = make_merged_polygon(polygons, user_name, reason)

        for gem_id in match_info["gem_ids"]:
            idx = project_idx.get(gem_id)
            if idx is None:
                print(f"  WARNING: GEM ID {gem_id} not found in projects_merged.json")
                continue

            p = projects[idx]
            old_lat = p["gspt"].get("latitude")
            old_lon = p["gspt"].get("longitude")
            dist = haversine_km(clat, clon, old_lat, old_lon) if old_lat and old_lon else None

            updates.append({
                "gem_id": gem_id,
                "idx": idx,
                "gem_name": p["gspt"].get("project_name"),
                "user_name": user_name,
                "old_coords": (old_lat, old_lon),
                "new_coords": (round(clat, 6), round(clon, 6)),
                "coord_shift_km": round(dist, 1) if dist else None,
                "merged_polygon": merged_poly,
                "match_confidence": "verified",
                "reason": reason,
            })

    # Check for unmatched user projects
    for key in confirmed:
        if key not in MATCHES:
            unmatched_user.append((key, confirmed[key]["name"]))

    # Print summary
    print(f"\n{'='*80}")
    print(f"RECONCILIATION SUMMARY")
    print(f"{'='*80}")
    print(f"Matched: {len(updates)} GEM entries ← {len(MATCHES)} user projects")
    print(f"Unmatched user projects: {len(unmatched_user)}")
    for key, name in unmatched_user:
        print(f"  - {name} (no GEM match)")
    print()

    for u in updates:
        shift = f" (shift: {u['coord_shift_km']} km)" if u['coord_shift_km'] and u['coord_shift_km'] > 1 else ""
        print(f"  {u['gem_id']} | {u['gem_name'][:45]:45s} → {u['user_name']}")
        print(f"    coords: ({u['old_coords'][0]}, {u['old_coords'][1]}) → ({u['new_coords'][0]}, {u['new_coords'][1]}){shift}")
        print(f"    reason: {u['reason'][:80]}")
        print()

    if dry_run:
        print("DRY RUN - no changes written")
        return

    # Apply updates to projects_merged.json
    print("Updating projects_merged.json...")
    for u in updates:
        p = projects[u["idx"]]
        p["merged_polygon"] = u["merged_polygon"]
        p["match_confidence"] = "verified"
        p["match_distance_km"] = u["coord_shift_km"]
        # Store user's common name in a metadata field
        if "user_metadata" not in p:
            p["user_metadata"] = {}
        p["user_metadata"]["common_name"] = u["user_name"]
        p["user_metadata"]["match_reason"] = u["reason"]

    print(f"Writing {MERGED_FILE}...")
    with open(MERGED_FILE, "w") as f:
        json.dump(projects, f, separators=(",", ":"))
    print("  Done.")

    # Optionally update database directly
    if update_db:
        update_database(updates)

    print(f"\nReconciliation complete. {len(updates)} projects updated.")
    print("Run `python scripts/seed_database.py` to re-seed the database from updated JSON.")


def update_database(updates):
    """Directly update the Postgres database."""
    try:
        import psycopg2
    except ImportError:
        print("psycopg2 not installed. Install with: pip install psycopg2-binary")
        print("Skipping database update. Re-seed from JSON instead.")
        return

    url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("No POSTGRES_URL or DATABASE_URL set. Skipping database update.")
        print("Re-seed from JSON instead: python scripts/seed_database.py")
        return

    print("\nUpdating Postgres database directly...")
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            for u in updates:
                cur.execute(
                    """
                    UPDATE projects SET
                        merged_polygon = %s,
                        match_confidence = %s,
                        match_distance_km = %s
                    WHERE id = %s
                    """,
                    (
                        json.dumps(u["merged_polygon"]),
                        "verified",
                        u["coord_shift_km"],
                        u["gem_id"],
                    ),
                )
                print(f"  Updated {u['gem_id']} ({u['user_name']})")
        conn.commit()
        print(f"  {len(updates)} rows updated in database.")
    except Exception as e:
        conn.rollback()
        print(f"  Database update failed: {e}")
        print("  Re-seed from JSON instead: python scripts/seed_database.py")
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconcile Bangladesh solar projects")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--update-db", action="store_true", help="Also update Postgres directly")
    args = parser.parse_args()

    run(dry_run=args.dry_run, update_db=args.update_db)
