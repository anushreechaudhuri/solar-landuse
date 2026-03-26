#!/usr/bin/env python3
"""
Assemble all Bangladesh solar conflict data sources into a single JSON file
for the webapp at bd-solar-conflict/public/data/sites.json.

Input sources:
  1. data/Solar Sites with Conflict - Conflict List.csv
  2. data/grw/confirmed_matches.json
  3. data/grw/bangladesh_post_construction_lulc.json
  4. data/annual_panel.csv
  5. data/lcw_matched_conflicts.json

Output:
  bd-solar-conflict/bd-solar-conflict/public/data/sites.json
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# Paths (all relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CONFLICT_CSV = DATA_DIR / "Solar Sites with Conflict - Conflict List.csv"
CONFIRMED_MATCHES = DATA_DIR / "grw" / "confirmed_matches.json"
POST_LULC_JSON = DATA_DIR / "grw" / "bangladesh_post_construction_lulc.json"
ANNUAL_PANEL_CSV = DATA_DIR / "annual_panel.csv"
LCW_CONFLICTS_JSON = DATA_DIR / "lcw_matched_conflicts.json"
UNIFIED_SOLAR_DB = DATA_DIR / "unified_solar_db.json"

WEBAPP_PUBLIC = PROJECT_ROOT / "bd-solar-conflict" / "public"
IMAGES_DIR = WEBAPP_PUBLIC / "images"
OUTPUT_FILE = WEBAPP_PUBLIC / "data" / "sites.json"

# ---------------------------------------------------------------------------
# Slug matching: Common Name keyword -> slug
# ---------------------------------------------------------------------------
SLUG_RULES = [
    # Order matters: more specific patterns first
    ("Sirajganj 68", "sirajganj68"),
    ("Sirajganj 6", "sirajganj6"),
    ("Sirajganj 7", "sirajganj6"),
    ("Teesta", "teesta"),
    ("Feni", "feni"),
    ("Sonagazi", "feni"),
    ("Manikganj", "manikganj"),
    ("Spectra", "manikganj"),
    ("Moulvibazar", "moulvibazar"),
    ("Mongla", "mongla"),
    ("Pabna", "pabna"),
    ("Mymensingh", "mymensingh"),
    ("HDFC", "mymensingh"),
    ("Tetulia", "tetulia"),
    ("Sympa", "tetulia"),
    ("Lalmonirhat", "lalmonirhat"),
    ("Intraco", "lalmonirhat"),
    ("Teknaf", "teknaf"),
    ("Joules", "teknaf"),
    ("Kaptai", "kaptai"),
    ("Sharishabari", "sharishabari"),
    ("Engreen", "sharishabari"),
    ("Barishal", "barishal"),
    ("Taltali", "taltali"),
    ("Bargunia", "taltali"),
    ("Korotuya", "korotuya"),
    ("Barapukuria", "barapukuria"),
]


def name_to_slug(common_name: str) -> Optional[str]:
    """Match a Common Name to a slug using keyword rules."""
    for keyword, slug in SLUG_RULES:
        if keyword.lower() in common_name.lower():
            return slug
    return None


# ---------------------------------------------------------------------------
# Conflict tag extraction from free text
# ---------------------------------------------------------------------------
TAG_RULES = [
    (r"illegal|forced|acquisition", "Forced Acquisition"),
    (r"three-crop|cropland", "Three-Crop Land"),
    (r"farmer|livelihood", "Farmer Livelihoods"),
    (r"ecological|haor|wetland|environmental", "Ecological Impact"),
    (r"corruption|corrupt", "Corruption"),
    (r"river|erosion", "River Erosion"),
    (r"compensation", "Inadequate Compensation"),
    (r"protest", "Community Protests"),
]


def extract_conflict_tags(text: str) -> List[str]:
    """Extract conflict tags from free-text conflict reasons."""
    if not text:
        return []
    tags = []
    for pattern, tag in TAG_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# Parse lat/lon from "lat, lon" string
# ---------------------------------------------------------------------------
def parse_latlon(s: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse a 'lat, lon' string into floats. Returns (None, None) on failure."""
    if not s or not s.strip():
        return None, None
    parts = s.strip().split(",")
    if len(parts) != 2:
        return None, None
    try:
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return lat, lon
    except ValueError:
        return None, None


# ---------------------------------------------------------------------------
# Parse capacity from string like "200 MWp" or "10 MWp additional 25-100 MWp proposed"
# ---------------------------------------------------------------------------
def parse_capacity(cap_str: str, cap_float: str) -> Optional[float]:
    """Extract numeric MW capacity. Prefer cap_float if valid."""
    # Try the float column first
    if cap_float and cap_float.strip():
        try:
            return float(cap_float.strip())
        except ValueError:
            pass
    # Fall back to parsing string
    if cap_str and cap_str.strip():
        m = re.match(r"([\d.]+)", cap_str.strip())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Parse news links (newline-separated in CSV cell)
# ---------------------------------------------------------------------------
def parse_news_links(text: str) -> List[str]:
    """Split newline-separated news links and filter non-empty entries."""
    if not text or not text.strip():
        return []
    links = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line and line.startswith("http"):
            links.append(line)
        # Skip non-URL text (hyperlink display text without the actual URL)
    return links


# Manual news link corrections for sites where CSV had display text instead of URLs
NEWS_LINK_FIXES: Dict[str, List[str]] = {
    "feni": ["https://www.observerbd.com/news/295460"],
    "manikganj": ["https://www.observerbd.com/news/166370"],
    "taltali": [
        "https://www.amadershomoy.com/country/article/110189/"
        "%E0%A6%B8%E0%A7%8C%E0%A6%B0-%E0%A6%AC%E0%A6%BF%E0%A6%A6%E0%A7%8D%E0%A6%AF%E0%A7%81%E0%A7%8E-"
        "%E0%A6%95%E0%A7%87%E0%A6%A8%E0%A7%8D%E0%A6%A6%E0%A7%8D%E0%A6%B0%E0%A6%B0-%E0%A7%AC%E2%80%99"
        "%E0%A6%B6p#gsc.tab=0",
    ],
    "mymensingh": [
        "https://www.energytransitionbd.org/infrastructure/sutiakhali-50-mw-hdfc-solar-power-plant",
    ],
}


# ---------------------------------------------------------------------------
# Parse the multi-line CSV carefully
# ---------------------------------------------------------------------------
def read_conflict_csv(path: Path) -> List[dict]:
    """Read the conflict CSV, handling newlines within quoted fields."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


# ---------------------------------------------------------------------------
# Check which images exist for a given slug
# ---------------------------------------------------------------------------
def find_images(slug: str) -> Dict[str, str]:
    """Return dict of image type -> web path for images that exist on disk."""
    images = {}

    # Pre/post comparison image
    pre_post = IMAGES_DIR / "bangladesh_sites" / f"{slug}_pre_post.png"
    if pre_post.exists():
        images["pre_post"] = f"/images/bangladesh_sites/{slug}_pre_post.png"

    # Case study images (only for teesta, feni, manikganj, moulvibazar typically)
    for img_type, filename_suffix in [
        ("image_grid", f"{slug}_image_grid.png"),
        ("lulc_maps", f"{slug}_satellite_lulc_maps.png"),
        ("lulc_timeseries", f"{slug}_lulc_timeseries.png"),
        ("lulc_change_detail", f"{slug}_lulc_change_detail.png"),
    ]:
        img_path = IMAGES_DIR / "case_studies" / filename_suffix
        if img_path.exists():
            images[img_type] = f"/images/case_studies/{filename_suffix}"

    return images


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def main():
    # --- Load all data sources ---
    print("Loading data sources...")

    # 1. Conflict CSV
    csv_rows = read_conflict_csv(CONFLICT_CSV)
    print(f"  CSV rows: {len(csv_rows)}")

    # 2. Confirmed polygon matches
    with open(CONFIRMED_MATCHES, "r") as f:
        confirmed = json.load(f)
    print(f"  Confirmed polygon matches: {len(confirmed)} sites")

    # 3. Post-construction LULC
    with open(POST_LULC_JSON, "r") as f:
        post_lulc_list = json.load(f)
    # Build lookup by key; handle duplicate keys (mongla has 2 polygons)
    # Merge duplicates by averaging or taking the first
    post_lulc_by_key = {}
    for entry in post_lulc_list:
        key = entry["key"]
        if key not in post_lulc_by_key:
            post_lulc_by_key[key] = entry
    print(f"  Post-construction LULC entries: {len(post_lulc_by_key)} unique sites")

    # 4. Annual panel (filter for BA_ sites)
    annual_data = {}  # site_id -> list of yearly records
    with open(ANNUAL_PANEL_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["site_id"]
            if not sid.startswith("BA_"):
                continue
            if sid not in annual_data:
                annual_data[sid] = []
            try:
                annual_data[sid].append({
                    "year": int(row["year"]),
                    "crops": round(float(row["dw_crops_pct"]), 2) if row["dw_crops_pct"] else 0,
                    "trees": round(float(row["dw_trees_pct"]), 2) if row["dw_trees_pct"] else 0,
                    "built": round(float(row["dw_built_pct"]), 2) if row["dw_built_pct"] else 0,
                    "bare": round(float(row["dw_bare_pct"]), 2) if row["dw_bare_pct"] else 0,
                    "water": round(float(row["dw_water_pct"]), 2) if row["dw_water_pct"] else 0,
                    "grass": round(float(row["dw_grass_pct"]), 2) if row["dw_grass_pct"] else 0,
                    "shrub": round(float(row["dw_shrub_and_scrub_pct"]), 2) if row["dw_shrub_and_scrub_pct"] else 0,
                    "flooded_veg": round(float(row["dw_flooded_vegetation_pct"]), 2) if row["dw_flooded_vegetation_pct"] else 0,
                    "ndvi": round(float(row["ndvi_mean"]), 4) if row["ndvi_mean"] else None,
                })
            except (ValueError, KeyError) as e:
                continue
    print(f"  Annual panel: {len(annual_data)} BA_ sites loaded")

    # 5. LCW matched conflicts (bangladesh_field only)
    with open(LCW_CONFLICTS_JSON, "r") as f:
        lcw_all = json.load(f)
    bd_conflicts = [c for c in lcw_all if c.get("source") == "bangladesh_field"]
    # Build lookup by name -> matched_site_id
    lcw_by_name = {}
    for c in bd_conflicts:
        lcw_by_name[c["name"]] = c.get("matched_site_id")
    # Also build lookup by slug (from confirmed_matches name -> lcw name)
    lcw_by_slug = {}
    for c in bd_conflicts:
        slug = name_to_slug(c["name"])
        if slug:
            lcw_by_slug[slug] = c.get("matched_site_id")
    print(f"  LCW Bangladesh conflicts: {len(bd_conflicts)} entries")

    # 6. Unified solar DB (for GEM phase IDs)
    with open(UNIFIED_SOLAR_DB, "r") as f:
        udb = json.load(f)
    udb_gem = {}  # site_id -> gem wiki URL
    for entry in udb:
        sid = entry.get("site_id", "")
        if sid.startswith("BA_") and entry.get("gem"):
            project_name = entry["gem"].get("project_name")
            if project_name:
                wiki_slug = project_name.strip().replace(" ", "_")
                udb_gem[sid] = f"https://www.gem.wiki/{wiki_slug}"
    print(f"  GEM wiki URLs: {len(udb_gem)} Bangladesh sites")

    # --- Assemble output ---
    print("\nAssembling site records...")
    sites = []
    slugs_seen = set()

    for row in csv_rows:
        common_name = row.get("Common Name", "").strip()
        if not common_name:
            continue

        slug = name_to_slug(common_name)
        if not slug:
            print(f"  WARNING: No slug match for '{common_name}'")
            continue

        if slug in slugs_seen:
            print(f"  WARNING: Duplicate slug '{slug}' for '{common_name}', skipping")
            continue
        slugs_seen.add(slug)

        # Parse lat/lon
        lat, lon = parse_latlon(row.get("lat/lon", ""))

        # Parse capacity
        capacity = parse_capacity(
            row.get("capacity (str)", ""),
            row.get("capacity (float)", "")
        )

        # Parse conflict status
        has_conflict = row.get("Evidence of Controversy?", "").strip().upper() == "TRUE"

        # Conflict reasons and tags
        conflict_reasons = row.get("Conflict Reasons", "").strip()
        conflict_tags = extract_conflict_tags(conflict_reasons)

        # News links
        news_links = parse_news_links(row.get("News Links", ""))
        # Apply manual fixes for broken links (CSV had display text not URLs)
        if slug in NEWS_LINK_FIXES:
            # Prepend the corrected links (they go first since they're the primary sources)
            news_links = NEWS_LINK_FIXES[slug] + news_links

        # Google Maps link (from "Link Extracted" column)
        google_maps_link = row.get("Link Extracted", "").strip() or None

        # Status
        status = row.get("present_status", "").strip()

        # Completion date
        completion_date = row.get("completion_date", "").strip() or None

        # District / Upazilla
        district = row.get("detail_District", "").strip()
        upazilla = row.get("detail_Upazilla", "").strip()

        # Developer
        developer = row.get("detail_System_Owner", "").strip()
        # Clean up developer string (remove "Non-individual>" prefix, "Individual>NID: ****")
        if developer:
            # Extract meaningful parts
            parts = []
            for part in developer.split("Individual>"):
                part = part.strip()
                if part.startswith("Non-individual>"):
                    part = part.replace("Non-individual>", "").strip()
                if part and "NID:" not in part and "Not Found" not in part:
                    parts.append(part)
            developer = parts[0] if parts else ""
            # Remove trailing parenthetical IDs like " (42)"
            developer = re.sub(r"\s*\(\d+\)\s*$", "", developer).strip()

        # Financing
        financing = row.get("finance_lmfd", "").strip()

        # --- Polygon from confirmed_matches ---
        polygon = None
        if slug in confirmed:
            polygons = confirmed[slug].get("polygons", [])
            if len(polygons) == 1:
                polygon = polygons[0]
            elif len(polygons) > 1:
                # Use MultiPolygon for sites with multiple polygons
                polygon = {
                    "type": "MultiPolygon",
                    "coordinates": [p["coordinates"] for p in polygons]
                }

        # --- Matched site_id from LCW ---
        matched_site_id = lcw_by_slug.get(slug)

        # --- Post-construction LULC ---
        post_lulc = None
        if slug in post_lulc_by_key:
            entry = post_lulc_by_key[slug]
            post_lulc = {
                "water": round(entry.get("water", 0), 2),
                "trees": round(entry.get("trees", 0), 2),
                "grass": round(entry.get("grass", 0), 2),
                "flooded_veg": round(entry.get("flooded_vegetation", 0), 2),
                "crops": round(entry.get("crops", 0), 2),
                "shrub": round(entry.get("shrub_and_scrub", 0), 2),
                "built": round(entry.get("built", 0), 2),
                "bare": round(entry.get("bare", 0), 2),
                "snow_ice": round(entry.get("snow_and_ice", 0), 2),
            }

        # --- Annual LULC timeseries ---
        annual_lulc = []
        if matched_site_id and matched_site_id in annual_data:
            annual_lulc = sorted(annual_data[matched_site_id], key=lambda x: x["year"])

        # --- Images ---
        images = find_images(slug)

        # --- GEM URL ---
        gem_url = None
        if matched_site_id and matched_site_id in udb_gem:
            gem_url = udb_gem[matched_site_id]

        # --- Build site record ---
        site = {
            "id": slug,
            "name": common_name,
            "capacity_mw": capacity,
            "lat": round(lat, 6) if lat is not None else None,
            "lon": round(lon, 6) if lon is not None else None,
            "district": district,
            "upazilla": upazilla,
            "status": status,
            "completion_date": completion_date,
            "has_conflict": has_conflict,
            "conflict_reasons": conflict_reasons,
            "conflict_tags": conflict_tags,
            "news_links": news_links,
            "google_maps_link": google_maps_link,
            "gem_url": gem_url,
            "developer": developer,
            "financing": financing,
            "polygon": polygon,
            "matched_site_id": matched_site_id,
            "post_lulc": post_lulc,
            "annual_lulc": annual_lulc,
            "images": images if images else None,
        }

        sites.append(site)

    # --- Write output ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sites, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(sites)} sites to {OUTPUT_FILE}")

    # --- Summary stats ---
    print("\n--- Summary ---")
    n_conflict = sum(1 for s in sites if s["has_conflict"])
    n_no_conflict = sum(1 for s in sites if not s["has_conflict"])
    print(f"  Sites with conflict evidence: {n_conflict}")
    print(f"  Sites without conflict evidence: {n_no_conflict}")

    n_completed = sum(1 for s in sites if "Completed" in (s["status"] or ""))
    n_proposed = sum(1 for s in sites if "Proposed" in (s["status"] or ""))
    print(f"  Completed & Running: {n_completed}")
    print(f"  Proposed: {n_proposed}")

    n_with_polygon = sum(1 for s in sites if s["polygon"] is not None)
    n_with_lulc = sum(1 for s in sites if s["annual_lulc"])
    n_with_post_lulc = sum(1 for s in sites if s["post_lulc"] is not None)
    n_with_images = sum(1 for s in sites if s["images"])
    n_with_coords = sum(1 for s in sites if s["lat"] is not None)
    n_with_site_id = sum(1 for s in sites if s["matched_site_id"])
    print(f"  With coordinates: {n_with_coords}")
    print(f"  With polygon: {n_with_polygon}")
    print(f"  With matched_site_id: {n_with_site_id}")
    print(f"  With post-construction LULC: {n_with_post_lulc}")
    print(f"  With annual LULC timeseries: {n_with_lulc}")
    print(f"  With images: {n_with_images}")

    # Tag distribution
    all_tags = {}
    for s in sites:
        for tag in s["conflict_tags"]:
            all_tags[tag] = all_tags.get(tag, 0) + 1
    print(f"\n  Conflict tag distribution:")
    for tag, count in sorted(all_tags.items(), key=lambda x: -x[1]):
        print(f"    {tag}: {count}")

    # Capacity stats
    caps = [s["capacity_mw"] for s in sites if s["capacity_mw"] is not None]
    if caps:
        print(f"\n  Capacity range: {min(caps):.1f} - {max(caps):.1f} MW")
        print(f"  Total capacity: {sum(caps):.1f} MW")

    # Sites without polygon or annual data
    missing_polygon = [s["id"] for s in sites if s["polygon"] is None]
    missing_annual = [s["id"] for s in sites if not s["annual_lulc"]]
    if missing_polygon:
        print(f"\n  Missing polygon: {', '.join(missing_polygon)}")
    if missing_annual:
        print(f"  Missing annual LULC: {', '.join(missing_annual)}")


if __name__ == "__main__":
    main()
