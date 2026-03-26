"""
Geocode LCW (Land Conflict Watch) conflicts and match them to the unified solar
database, plus integrate Bangladesh conflict data from field research.

Inputs:
  - data/lcw_conflicts.json         (scraped LCW data, optional)
  - data/unified_solar_db.json      (6,705 solar installations)
  - data/Solar Sites with Conflict - Conflict List.csv  (Bangladesh conflicts)

Outputs:
  - data/lcw_geocoded.json          (geocoding cache)
  - data/lcw_matched_conflicts.json (combined matched conflicts)

Usage:
    python scripts/match_lcw_conflicts.py
    python scripts/match_lcw_conflicts.py --skip-geocode   # use cached geocoding
    python scripts/match_lcw_conflicts.py --lcw-only       # skip Bangladesh data
    python scripts/match_lcw_conflicts.py --bd-only        # skip LCW data
"""

import argparse
import csv
import difflib
import json
import math
import os
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
LCW_FILE = DATA_DIR / "lcw_conflicts.json"
SOLAR_DB_FILE = DATA_DIR / "unified_solar_db.json"
BD_CSV_FILE = DATA_DIR / "Solar Sites with Conflict - Conflict List.csv"
GEOCODE_CACHE_FILE = DATA_DIR / "lcw_geocoded.json"
OUTPUT_FILE = DATA_DIR / "lcw_matched_conflicts.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "solar-landuse-research/1.0 (academic research project)"


# ── Haversine distance ────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Geocoding ─────────────────────────────────────────────────────────────

def geocode_location(district, state, country="India"):
    """Geocode a district/state using Nominatim. Returns (lat, lon) or None."""
    import requests

    query = f"{district}, {state}, {country}"
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "countrycodes": "in" if country == "India" else "bd",
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            return lat, lon
    except Exception as e:
        print(f"  [WARN] Geocoding failed for '{query}': {e}")

    return None


def geocode_lcw_conflicts(conflicts, cache_file, skip_geocode=False):
    """Geocode all LCW conflicts. Uses cache if available.

    Returns list of conflicts with lat/lon fields added.
    """
    # Load cache
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached geocoding results from {cache_file.name}")

    if skip_geocode:
        # Apply cached results only
        geocoded = []
        for c in conflicts:
            key = f"{c.get('district', '')}, {c.get('state', '')}"
            if key in cache:
                c["lat"] = cache[key][0]
                c["lon"] = cache[key][1]
                geocoded.append(c)
            elif c.get("lat") and c.get("lon"):
                geocoded.append(c)
            else:
                geocoded.append(c)  # keep it, just without coords
        return geocoded

    # Geocode missing entries
    import requests  # noqa: F811

    new_lookups = 0
    for i, c in enumerate(conflicts):
        district = c.get("district", "")
        state = c.get("state", "")
        key = f"{district}, {state}"

        if key in cache:
            c["lat"] = cache[key][0]
            c["lon"] = cache[key][1]
            continue

        if not district and not state:
            continue

        print(f"  [{i+1}/{len(conflicts)}] Geocoding: {key}")
        result = geocode_location(district, state)
        if result:
            c["lat"] = result[0]
            c["lon"] = result[1]
            cache[key] = list(result)
            new_lookups += 1
        else:
            print(f"    -> No result")

        # Nominatim usage policy: max 1 request/second
        time.sleep(1.1)

    # Save cache
    if new_lookups > 0:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  Saved {len(cache)} geocoding results ({new_lookups} new)")

    return conflicts


# ── Solar DB loading ──────────────────────────────────────────────────────

def load_solar_db(db_file):
    """Load the unified solar DB. Only keep fields needed for matching."""
    print(f"Loading solar DB from {db_file.name}...")
    with open(db_file) as f:
        raw = json.load(f)

    sites = []
    for s in raw:
        entry = {
            "site_id": s["site_id"],
            "country": s.get("country", ""),
            "lat": s.get("centroid_lat"),
            "lon": s.get("centroid_lon"),
            "capacity_mw": s.get("best_capacity_mw"),
            "construction_year": s.get("best_construction_year"),
            "treatment_group": s.get("treatment_group", ""),
            "gem_project_name": "",
        }
        # Extract GEM project name for fuzzy matching
        gem = s.get("gem")
        if gem and isinstance(gem, dict):
            entry["gem_project_name"] = gem.get("project_name", "")
        sites.append(entry)

    print(f"  Loaded {len(sites)} sites ({sum(1 for s in sites if s['country']=='India')} India, "
          f"{sum(1 for s in sites if s['country']=='Bangladesh')} Bangladesh)")
    return sites


# ── Matching logic ────────────────────────────────────────────────────────

def compute_match_score(conflict, site, max_dist_km=10.0):
    """Compute a composite match score between a conflict and a solar DB site.

    Returns (score, distance_km) or (0, None) if no match.
    """
    c_lat = conflict.get("lat")
    c_lon = conflict.get("lon")
    if c_lat is None or c_lon is None:
        return 0.0, None
    if site["lat"] is None or site["lon"] is None:
        return 0.0, None

    dist = haversine_km(c_lat, c_lon, site["lat"], site["lon"])
    if dist > max_dist_km:
        return 0.0, dist

    # Spatial score: linearly decreasing from 1.0 at 0 km to 0.0 at max_dist_km
    spatial_score = max(0.0, 1.0 - dist / max_dist_km)

    # Capacity similarity score (0.3 max)
    capacity_score = 0.0
    c_cap = conflict.get("capacity_mw")
    s_cap = site.get("capacity_mw")
    if c_cap and s_cap and c_cap > 0 and s_cap > 0:
        ratio = min(c_cap, s_cap) / max(c_cap, s_cap)
        if ratio >= 0.5:  # within +/-50%
            capacity_score = 0.3 * ratio

    # Name fuzzy match score (0.3 max)
    name_score = 0.0
    c_name = conflict.get("name", "").lower()
    s_name = site.get("gem_project_name", "").lower()
    if c_name and s_name:
        ratio = difflib.SequenceMatcher(None, c_name, s_name).ratio()
        if ratio > 0.4:
            name_score = 0.3 * ratio

    total = spatial_score + capacity_score + name_score
    return total, dist


def match_conflicts_to_db(conflicts, solar_sites, max_dist_km=10.0, min_score=0.3,
                          country_filter=None):
    """Match a list of conflicts to solar DB sites. Returns enriched conflicts."""
    if country_filter:
        candidates = [s for s in solar_sites if s["country"] == country_filter]
    else:
        candidates = solar_sites

    matched_count = 0
    for conflict in conflicts:
        best_score = 0.0
        best_site = None
        best_dist = None

        for site in candidates:
            score, dist = compute_match_score(conflict, site, max_dist_km=max_dist_km)
            if score > best_score:
                best_score = score
                best_site = site
                best_dist = dist

        if best_score >= min_score and best_site is not None:
            conflict["matched_site_id"] = best_site["site_id"]
            conflict["match_score"] = round(best_score, 4)
            conflict["match_distance_km"] = round(best_dist, 2) if best_dist is not None else None
            matched_count += 1
        else:
            conflict["matched_site_id"] = None
            conflict["match_score"] = 0.0
            conflict["match_distance_km"] = None

    return conflicts, matched_count


# ── LCW data processing ──────────────────────────────────────────────────

def load_lcw_conflicts(lcw_file):
    """Load and normalize LCW conflicts."""
    if not lcw_file.exists():
        print(f"  [WARN] LCW file not found: {lcw_file}")
        print(f"         Run scrape_lcw.py first to generate this file.")
        return []

    with open(lcw_file) as f:
        raw = json.load(f)

    conflicts = []
    for i, entry in enumerate(raw):
        conflict = {
            "conflict_id": f"LCW_{i+1:04d}",
            "source": "lcw",
            "name": entry.get("name", ""),
            "district": entry.get("district", ""),
            "state": entry.get("state", ""),
            "country": "India",
            "lat": entry.get("lat"),
            "lon": entry.get("lon"),
            "land_area_ha": entry.get("land_area_ha"),
            "affected_people": entry.get("affected_people"),
            "capacity_mw": entry.get("capacity_mw"),
            "energy_type": entry.get("energy_type", "").lower() if entry.get("energy_type") else "",
            "status": entry.get("status", ""),
            "conflict_reasons": entry.get("conflict_reasons", ""),
            "evidence_of_controversy": True,  # all LCW entries are conflicts
            "news_links": entry.get("news_links", []),
        }
        conflicts.append(conflict)

    print(f"  Loaded {len(conflicts)} LCW conflicts")
    return conflicts


# ── Bangladesh data processing ────────────────────────────────────────────

def load_bangladesh_conflicts(csv_file):
    """Load Bangladesh conflict data from the manually curated CSV."""
    if not csv_file.exists():
        print(f"  [WARN] Bangladesh CSV not found: {csv_file}")
        return []

    conflicts = []
    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Parse lat/lon from "lat, lon" string
            lat = None
            lon = None
            latlon_str = row.get("lat/lon", "").strip()
            if latlon_str:
                try:
                    parts = latlon_str.split(",")
                    if len(parts) == 2:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                except (ValueError, IndexError):
                    pass

            # Parse capacity
            cap = None
            cap_str = row.get("capacity (float)", "").strip()
            if cap_str:
                try:
                    cap = float(cap_str)
                except ValueError:
                    pass

            # Parse evidence of controversy
            evid = row.get("Evidence of Controversy?", "").strip().upper() == "TRUE"

            # Parse news links
            news_raw = row.get("News Links", "").strip()
            news_links = []
            if news_raw:
                # Links are newline-separated or may be plain text descriptions
                for line in news_raw.split("\n"):
                    line = line.strip()
                    if line:
                        news_links.append(line)

            # Determine status
            present_status = row.get("present_status", "").strip()
            completion_date = row.get("completion_date", "").strip()

            conflict = {
                "conflict_id": f"BD_{i+1:04d}",
                "source": "bangladesh_field",
                "name": row.get("Common Name", "").strip(),
                "district": row.get("detail_District", "").strip(),
                "state": row.get("detail_Division", "").strip(),  # Division = state equiv
                "country": "Bangladesh",
                "lat": lat,
                "lon": lon,
                "land_area_ha": None,
                "affected_people": None,
                "capacity_mw": cap,
                "energy_type": "solar",
                "status": present_status,
                "conflict_reasons": row.get("Conflict Reasons", "").strip(),
                "evidence_of_controversy": evid,
                "completion_date": completion_date,
                "upazilla": row.get("detail_Upazilla", "").strip(),
                "news_links": news_links,
            }
            conflicts.append(conflict)

    print(f"  Loaded {len(conflicts)} Bangladesh entries "
          f"({sum(1 for c in conflicts if c['evidence_of_controversy'])} with controversy evidence)")
    return conflicts


# ── Summary statistics ────────────────────────────────────────────────────

def print_summary(lcw_conflicts, bd_conflicts, all_conflicts):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # LCW stats
    if lcw_conflicts:
        solar_lcw = [c for c in lcw_conflicts if c.get("energy_type") == "solar"]
        wind_lcw = [c for c in lcw_conflicts if c.get("energy_type") == "wind"]
        other_lcw = [c for c in lcw_conflicts
                     if c.get("energy_type") not in ("solar", "wind")]
        geocoded_lcw = [c for c in lcw_conflicts
                        if c.get("lat") is not None and c.get("lon") is not None]
        matched_solar = [c for c in solar_lcw if c.get("matched_site_id")]

        print(f"\nLCW Conflicts (India):")
        print(f"  Total:     {len(lcw_conflicts)}")
        print(f"    Solar:   {len(solar_lcw)}")
        print(f"    Wind:    {len(wind_lcw)}")
        print(f"    Other:   {len(other_lcw)}")
        print(f"  Geocoded:  {len(geocoded_lcw)}/{len(lcw_conflicts)}")
        print(f"  Matched to solar DB: {len(matched_solar)}/{len(solar_lcw)} solar conflicts")
    else:
        print("\nLCW Conflicts: none loaded (run scrape_lcw.py first)")

    # Bangladesh stats
    if bd_conflicts:
        operational_bd = [c for c in bd_conflicts
                          if "completed" in c.get("status", "").lower()
                          or "running" in c.get("status", "").lower()]
        proposed_bd = [c for c in bd_conflicts
                       if "proposed" in c.get("status", "").lower()]
        controversy_bd = [c for c in bd_conflicts if c.get("evidence_of_controversy")]
        matched_bd = [c for c in bd_conflicts if c.get("matched_site_id")]
        matched_operational = [c for c in operational_bd if c.get("matched_site_id")]

        print(f"\nBangladesh Conflicts:")
        print(f"  Total:         {len(bd_conflicts)}")
        print(f"    Operational: {len(operational_bd)}")
        print(f"    Proposed:    {len(proposed_bd)}")
        print(f"  With controversy evidence: {len(controversy_bd)}")
        print(f"  Matched to solar DB: {len(matched_bd)}/{len(bd_conflicts)} total "
              f"({len(matched_operational)}/{len(operational_bd)} operational)")
    else:
        print("\nBangladesh Conflicts: none loaded")

    # Combined stats
    all_matched_ids = set()
    for c in all_conflicts:
        sid = c.get("matched_site_id")
        if sid:
            all_matched_ids.add(sid)

    controversy_matched = set()
    for c in all_conflicts:
        if c.get("evidence_of_controversy") and c.get("matched_site_id"):
            controversy_matched.add(c["matched_site_id"])

    print(f"\nCombined:")
    print(f"  Total conflict entries: {len(all_conflicts)}")
    print(f"  Total unique site_ids with ANY documented conflict: {len(all_matched_ids)}")
    print(f"  Total unique site_ids with CONFIRMED controversy: {len(controversy_matched)}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Geocode LCW conflicts and match to unified solar DB")
    parser.add_argument("--skip-geocode", action="store_true",
                        help="Use cached geocoding results (skip Nominatim API calls)")
    parser.add_argument("--lcw-only", action="store_true",
                        help="Process only LCW (India) conflicts")
    parser.add_argument("--bd-only", action="store_true",
                        help="Process only Bangladesh conflicts")
    parser.add_argument("--max-dist-lcw", type=float, default=10.0,
                        help="Max matching distance for LCW conflicts in km (default: 10)")
    parser.add_argument("--max-dist-bd", type=float, default=5.0,
                        help="Max matching distance for Bangladesh in km (default: 5)")
    args = parser.parse_args()

    if args.lcw_only and args.bd_only:
        print("ERROR: Cannot specify both --lcw-only and --bd-only")
        sys.exit(1)

    # Load solar DB
    solar_sites = load_solar_db(SOLAR_DB_FILE)

    all_conflicts = []

    # ── Process LCW conflicts ─────────────────────────────────────────────
    lcw_conflicts = []
    if not args.bd_only:
        print("\n--- LCW Conflicts (India) ---")
        lcw_conflicts = load_lcw_conflicts(LCW_FILE)

        if lcw_conflicts:
            # Geocode
            print(f"\nGeocoding LCW conflicts...")
            lcw_conflicts = geocode_lcw_conflicts(
                lcw_conflicts, GEOCODE_CACHE_FILE, skip_geocode=args.skip_geocode)

            geocoded = [c for c in lcw_conflicts if c.get("lat") is not None]
            print(f"  Geocoded: {len(geocoded)}/{len(lcw_conflicts)}")

            # Match to solar DB (India only)
            print(f"\nMatching LCW conflicts to solar DB (max {args.max_dist_lcw} km)...")
            lcw_conflicts, n_matched = match_conflicts_to_db(
                lcw_conflicts, solar_sites,
                max_dist_km=args.max_dist_lcw, min_score=0.3,
                country_filter="India")
            print(f"  Matched: {n_matched}/{len(lcw_conflicts)}")

            # Print top matches
            matched = sorted(
                [c for c in lcw_conflicts if c.get("matched_site_id")],
                key=lambda c: c["match_score"], reverse=True)
            if matched:
                print(f"\n  Top matches:")
                for c in matched[:10]:
                    print(f"    {c['name'][:45]:45s} -> {c['matched_site_id']} "
                          f"(score={c['match_score']:.3f}, dist={c['match_distance_km']:.1f} km)")

            all_conflicts.extend(lcw_conflicts)

    # ── Process Bangladesh conflicts ──────────────────────────────────────
    bd_conflicts = []
    if not args.lcw_only:
        print("\n--- Bangladesh Conflicts ---")
        bd_conflicts = load_bangladesh_conflicts(BD_CSV_FILE)

        if bd_conflicts:
            # Bangladesh entries already have lat/lon from CSV; match to solar DB
            has_coords = [c for c in bd_conflicts if c.get("lat") is not None]
            print(f"  With coordinates: {len(has_coords)}/{len(bd_conflicts)}")

            print(f"\nMatching Bangladesh conflicts to solar DB (max {args.max_dist_bd} km)...")
            bd_conflicts, n_matched = match_conflicts_to_db(
                bd_conflicts, solar_sites,
                max_dist_km=args.max_dist_bd, min_score=0.3,
                country_filter="Bangladesh")
            print(f"  Matched: {n_matched}/{len(bd_conflicts)}")

            # Print all Bangladesh matches (small dataset)
            for c in bd_conflicts:
                status = "MATCHED" if c.get("matched_site_id") else "no match"
                controversy = " [CONFLICT]" if c.get("evidence_of_controversy") else ""
                sid = c.get("matched_site_id") or "---"
                dist = c.get("match_distance_km")
                dist_str = f"{dist:.1f} km" if dist is not None else "n/a"
                coords = f"({c['lat']:.3f}, {c['lon']:.3f})" if c.get("lat") else "(no coords)"
                print(f"    {c['name'][:45]:45s} {coords:25s} -> {str(sid):10s} "
                      f"({status}, {dist_str}){controversy}")

            all_conflicts.extend(bd_conflicts)

    # ── Save output ───────────────────────────────────────────────────────
    if all_conflicts:
        # Clean up output: remove internal fields, ensure consistent schema
        output = []
        for c in all_conflicts:
            entry = {
                "conflict_id": c.get("conflict_id"),
                "source": c.get("source"),
                "name": c.get("name"),
                "district": c.get("district"),
                "state": c.get("state"),
                "country": c.get("country"),
                "lat": c.get("lat"),
                "lon": c.get("lon"),
                "land_area_ha": c.get("land_area_ha"),
                "affected_people": c.get("affected_people"),
                "capacity_mw": c.get("capacity_mw"),
                "energy_type": c.get("energy_type"),
                "status": c.get("status"),
                "conflict_reasons": c.get("conflict_reasons"),
                "evidence_of_controversy": c.get("evidence_of_controversy", False),
                "matched_site_id": c.get("matched_site_id"),
                "match_score": c.get("match_score", 0.0),
                "match_distance_km": c.get("match_distance_km"),
                "news_links": c.get("news_links", []),
            }
            output.append(entry)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(output)} conflicts to {OUTPUT_FILE.name}")
    else:
        print("\nNo conflicts to save.")

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(lcw_conflicts, bd_conflicts, all_conflicts)


if __name__ == "__main__":
    main()
