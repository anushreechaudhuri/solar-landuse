"""
Scrape Land Conflict Watch (LCW) renewable energy conflicts from India.

The LCW website (landconflictwatch.org) is JavaScript-rendered (Webflow), so the
main listing page cannot be reliably scraped. Instead, the 45 known renewable energy
conflicts are hardcoded from a manual scrape. Each individual conflict detail page
is then fetched with requests+BeautifulSoup for additional structured data (capacity,
developer, description, etc.).

Usage:
    python scripts/scrape_lcw.py                   # Scrape all conflict detail pages
    python scripts/scrape_lcw.py --dry-run          # List conflicts without scraping
    python scripts/scrape_lcw.py --force             # Re-scrape even if cached
    python scripts/scrape_lcw.py --output data/x.json  # Custom output path
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CACHE_DIR = DATA_DIR / "lcw_cache"
DEFAULT_OUTPUT = DATA_DIR / "lcw_conflicts.json"

BASE_URL = "https://www.landconflictwatch.org"

# ---------------------------------------------------------------------------
# Hardcoded conflict list (from JS-rendered main page, manually extracted)
# Fields: name, url_path, district, state, land_area_ha, affected_people, status
# ---------------------------------------------------------------------------
CONFLICTS = [
    {
        "name": "Charanka Solar Park",
        "url_path": "/conflicts/ten-years-on-maldharis-await-compensation-for-lands-acquired-for-charanka-solar-park-in-gujarat",
        "district": "Patan",
        "state": "Gujarat",
        "land_area_ha": 2179,
        "affected_people": 1500,
        "status": "Ongoing",
    },
    {
        "name": "Alavanthankulam Solar",
        "url_path": "/conflicts/village-residents-protest-acquisition-of-pasture-lands-for-solar-plant-in-tamil-nadu",
        "district": "Tirunelveli",
        "state": "Tamil Nadu",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Pang Solar Park",
        "url_path": "/conflicts/from-pastures-to-power-parks-livelihoods-and-just-energy-transition-concerns-in-the-pang-solar-park-project-ladakh",
        "district": "Samad-Rokchan",
        "state": "Ladakh",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Kanoi Wind Project",
        "url_path": "/conflicts/rajasthan-villages-protest-windmill-project-near-desert-national-park",
        "district": "Jaisalmer",
        "state": "Rajasthan",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Long Kathar Solar",
        "url_path": "/conflicts/controversy-erupts-over-assam-s-1000-mw-solar-power-project-in-karbi-anglong",
        "district": "Karbi Anglong",
        "state": "Assam",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Bodhare & Shivapur Solar",
        "url_path": "/conflicts/farmers-in-maharashtra-s-jalgaon-allege-1100-acres-fraudulently-acquired-by-solar-power-companies",
        "district": "Jalgaon",
        "state": "Maharashtra",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Babra Wind Project",
        "url_path": "/conflicts/wind-energy-infrastructure-agricultural-land-and-farmers-resistance-in-chamardi-village-gujarat",
        "district": "Amreli",
        "state": "Gujarat",
        "land_area_ha": 113,
        "affected_people": 30,
        "status": "Ongoing",
    },
    {
        "name": "Chur Windmill",
        "url_path": "/conflicts/pasture-land-peacocks-and-power-poles-community-resistance-to-windmill-project-at-gujarat-s-jamjodhpur",
        "district": "Jamnagar",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Pavagada Solar Park",
        "url_path": "/conflicts/pavagada-solar-park-set-up-in-karnataka-communities-still-await-jobs-promised-to-them",
        "district": "Tumkur",
        "state": "Karnataka",
        "land_area_ha": 5261,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Rewa Solar Project",
        "url_path": "/conflicts/rewa-ultra-mega-solar-project-fails-to-create-employment-for-locals",
        "district": "Rewa",
        "state": "Madhya Pradesh",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Rewari Solar Park",
        "url_path": "/conflicts/rajasthan-s-rewari-village-opposes-land-transfer-to-adani-group-for-solar-project",
        "district": "Jaisalmer",
        "state": "Rajasthan",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Dawara Solar",
        "url_path": "/conflicts/jaisalmer-s-dawara-villagers-oppose-renewable-project-on-khaterdari-land",
        "district": "Jaisalmer",
        "state": "Rajasthan",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Sakunala Solar Park",
        "url_path": "/conflicts/agitation-against-non-payment-of-compensation-for-land-acquired-for-kurnool-solar-park",
        "district": "Kurnool",
        "state": "Andhra Pradesh",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Nadsal Wind Project",
        "url_path": "/conflicts/windmills-in-sez",
        "district": "Udupi",
        "state": "Karnataka",
        "land_area_ha": 260,
        "affected_people": 2000,
        "status": "Ongoing",
    },
    {
        "name": "NP Kunta Solar Park",
        "url_path": "/conflicts/ananthapuramu-solar-power-park-oustees-demand-higher-compensation-for-their-land",
        "district": "Anantapuramu",
        "state": "Andhra Pradesh",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Attappady Wind Energy",
        "url_path": "/conflicts/windmills-on-tribals-land",
        "district": "Palakkad",
        "state": "Kerala",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Mikir Bamuni Solar",
        "url_path": "/conflicts/farmers-in-assam-resist-land-acquisition-for-solar-plant-beaten-by-police",
        "district": "Nagaon",
        "state": "Assam",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Nedan Solar Park",
        "url_path": "/conflicts/rajasthan-court-cancels-allotment-of-agricultural-land-to-adani-for-solar-park",
        "district": "Jaisalmer",
        "state": "Rajasthan",
        "land_area_ha": None,
        "affected_people": None,
        "status": "In Court",
    },
    {
        "name": "Ugras Solar Plant",
        "url_path": "/conflicts/rajasthan-hc-dismisses-petition-to-cancel-land-allotment-for-solar-park-in-jodhpur",
        "district": "Jodhpur",
        "state": "Rajasthan",
        "land_area_ha": None,
        "affected_people": None,
        "status": "In Court",
    },
    {
        "name": "Sambhar Solar Park",
        "url_path": "/conflicts/rajasthan-high-court-stays-world-s-largest-solar-project-near-sambhar-lake-wetland",
        "district": "Jaipur",
        "state": "Rajasthan",
        "land_area_ha": 6475,
        "affected_people": 2014,
        "status": "In Court",
    },
    {
        "name": "Uttam Nagar Solar",
        "url_path": "/conflicts/farmers-oppose-solar-power-project-on-khatedari-land-in-rajasthan-s-jaisalmer",
        "district": "Jaisalmer",
        "state": "Rajasthan",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Western Ghats Wind Farm",
        "url_path": "/conflicts/wind-farm-in-western-ghats-poses-perennial-flooding-problem-for-locals-in-maharashtra",
        "district": "Pune",
        "state": "Maharashtra",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Doctorwadi & Panzhan Solar",
        "url_path": "/conflicts/farmers-lead-protest-against-tata-power-s-100-mw-solar-power-plant-in-nashik-s-nandgaon-taluka-4f3c2",
        "district": "Nashik",
        "state": "Maharashtra",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Rasipalayam Wind",
        "url_path": "/conflicts/farmers-protest-the-setting-up-of-high-tension-wires-that-will-draw-energy-from-wind-mills-in-tamil-nadu",
        "district": "Tiruppur",
        "state": "Tamil Nadu",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Bhagwanpura Solar",
        "url_path": "/conflicts/nomadic-communities-driven-to-abject-poverty-by-the-welspun-solar-mp-project",
        "district": "Neemuch",
        "state": "Madhya Pradesh",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Dungripali Solar",
        "url_path": "/conflicts/odisha-s-dungripali-village-protests-aditya-birla-solar-power-project",
        "district": "Balangir",
        "state": "Odisha",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Kamuthi Solar Plant",
        "url_path": "/conflicts/adani-s-kamuthi-solar-power-plant-casts-shadow-on-livelihood-of-locals",
        "district": "Ramanathapuram",
        "state": "Tamil Nadu",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Khudigaon Solar",
        "url_path": "/conflicts/assam-government-evicts-villagers-in-dhubri-s-khudigaon-for-construction-of-solar-power-plant",
        "district": "Dhubri",
        "state": "Assam",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Bhimsar Solar Plant",
        "url_path": "/conflicts/adani-solar-plant-threatens-fertile-belt-in-jaisalmer-s-bhimsar-village",
        "district": "Jaisalmer",
        "state": "Rajasthan",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Satara Wind Farm",
        "url_path": "/conflicts/farmers-oppose-suzlon-windmills-in-maharashtra-s-sangli-and-satara-secure-land-lease-for-panchayat",
        "district": "Sangli",
        "state": "Maharashtra",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Doctorwadi Solar",
        "url_path": "/conflicts/farmers-lead-protest-against-tata-power-s-100-mw-solar-power-plant-in-nashik-s-nandgaon-taluka",
        "district": "Nashik",
        "state": "Maharashtra",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Olpad Solar Project",
        "url_path": "/conflicts/locals-oppose-kundiyana-solar-pv-park-project-in-surat",
        "district": "Surat",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Radhanesda Solar Park",
        "url_path": "/conflicts/farmers-oppose-acquisition-of-gauchar-land-for-700-mw-radhanesda-ultra-mega-solar-park",
        "district": "Banaskantha",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Sangnara Wind Farm",
        "url_path": "/conflicts/sangnara-villagers-protest-to-save-forest-gauchar-and-wildlife-from-windmills",
        "district": "Kachchh",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Talaja Wind Project",
        "url_path": "/conflicts/pil-against-windmill-park-near-gir-forest-in-gujarat",
        "district": "Bhavnagar",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "In Court",
    },
    {
        "name": "Sujanpura Solar",
        "url_path": "/conflicts/sujanpura-pastoralists-oppose-modhera-solar-plant-in-gujarat",
        "district": "Mehsana",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Talaja Windmill",
        "url_path": "/conflicts/gujarat-high-court-orders-inquiry-into-windmill-installation-near-school-in-gujarat-s-bhavnagar",
        "district": "Bhavnagar",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "In Court",
    },
    {
        "name": "Dhrangadhra Solar",
        "url_path": "/conflicts/protests-against-solar-power-plant-in-dhrangadhra-s-moti-malvan-village-in-gujarat",
        "district": "Surendranagar",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Gir Solar Project",
        "url_path": "/conflicts/gujarat-puts-solar-power-project-near-gir-sanctuary-on-hold-amid-protest",
        "district": "Gir Somanath",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Mota Jinjuda Solar",
        "url_path": "/conflicts/farmers-protest-solar-power-plant-in-gujarat-s-mota-jinjuda-over-blocked-pathways",
        "district": "Amreli",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Kasaragod Solar Park",
        "url_path": "/conflicts/kasargod-solar-park-s-capacity-reduced-from-200mw-to-50mw-amid-protest",
        "district": "Kasargod",
        "state": "Kerala",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Ukai Floating Solar",
        "url_path": "/conflicts/villagers-oppose-1500-mw-floating-solar-power-project-on-ukai-reservoir-amid-livelihood-loss-fear",
        "district": "Tapi",
        "state": "Gujarat",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Khed & Maval Wind Farm",
        "url_path": "/conflicts/wind-farm-located-in-western-ghats-poses-perennial-flooding-problem-for-locals",
        "district": "Pune",
        "state": "Maharashtra",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Azure Power Solar",
        "url_path": None,  # Case study section, no individual URL
        "district": "Nagaon",
        "state": "Assam",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
    {
        "name": "Jalgaon Solar Projects",
        "url_path": None,  # Case study section, no individual URL
        "district": "Jalgaon",
        "state": "Maharashtra",
        "land_area_ha": None,
        "affected_people": None,
        "status": "Ongoing",
    },
]


# ---------------------------------------------------------------------------
# Energy type classification
# ---------------------------------------------------------------------------
def classify_energy_type(name):
    """Classify a conflict as solar, wind, or other based on project name."""
    name_lower = name.lower()
    if any(kw in name_lower for kw in ["solar", "floating solar"]):
        return "solar"
    elif any(kw in name_lower for kw in ["wind", "windmill"]):
        return "wind"
    else:
        return "other"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def extract_number(text):
    """Extract the first number (int or float) from a string."""
    if not text:
        return None
    match = re.search(r"[\d,]+\.?\d*", text.replace(",", ""))
    if match:
        val = match.group().replace(",", "")
        try:
            return float(val) if "." in val else int(val)
        except ValueError:
            return None
    return None


def extract_capacity_mw(text):
    """Try to extract capacity in MW from text. Handles GW, MW, kW."""
    if not text:
        return None
    text = text.upper()
    # Look for GW
    match = re.search(r"([\d,]+\.?\d*)\s*GW", text)
    if match:
        return float(match.group(1).replace(",", "")) * 1000
    # Look for MW
    match = re.search(r"([\d,]+\.?\d*)\s*MW", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Look for kW
    match = re.search(r"([\d,]+\.?\d*)\s*KW", text)
    if match:
        return float(match.group(1).replace(",", "")) / 1000
    return None


def clean_text(text):
    """Clean whitespace from extracted text."""
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def parse_detail_page(html, url):
    """
    Parse a conflict detail page for structured data.

    LCW pages are Webflow-rendered, so much of the content is in JS-rendered
    components. We extract what we can from static HTML and note limitations.

    Returns a dict of extracted fields.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "capacity_mw": None,
        "developer": None,
        "description": None,
        "conflict_issues": [],
        "land_type": None,
        "legal_status": None,
        "year_started": None,
        "scraped_land_area_ha": None,
        "scraped_affected_people": None,
        "scrape_note": None,
    }

    # --- Full page text for regex extraction ---
    page_text = soup.get_text(separator=" ", strip=True)

    # --- Title / H1 ---
    h1 = soup.find("h1")
    if h1:
        result["page_title"] = clean_text(h1.get_text())

    # --- Meta description ---
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        result["description"] = clean_text(meta_desc["content"])

    # --- OG description fallback ---
    if not result["description"]:
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            result["description"] = clean_text(og_desc["content"])

    # --- Try to find description in body paragraphs ---
    if not result["description"]:
        # Look for the first substantial paragraph
        for p in soup.find_all("p"):
            text = clean_text(p.get_text())
            if text and len(text) > 100:
                result["description"] = text
                break

    # --- Extract capacity from page text ---
    result["capacity_mw"] = extract_capacity_mw(page_text)

    # --- Extract developer/company names ---
    # Common patterns: "by [Company]", "[Company] Ltd", known developers
    known_developers = [
        "Adani", "Tata Power", "Suzlon", "ReNew Power", "Azure Power",
        "Welspun", "Aditya Birla", "NTPC", "GPCL", "KREDL", "SECI",
        "Gujarat Power Corporation", "Hindustan Salts", "Fortum",
        "Avaada", "SoftBank", "ACME Solar", "Rattan India",
    ]
    developers_found = []
    for dev in known_developers:
        if dev.lower() in page_text.lower():
            developers_found.append(dev)
    if developers_found:
        result["developer"] = "; ".join(developers_found)

    # --- Extract conflict issues from text ---
    issue_keywords = {
        "land acquisition": "Land acquisition",
        "compensation": "Compensation dispute",
        "livelihood": "Livelihood loss",
        "employment": "Employment demands",
        "grazing": "Grazing land loss",
        "pasture": "Pasture land loss",
        "displacement": "Displacement",
        "evict": "Eviction",
        "environment": "Environmental concerns",
        "wildlife": "Wildlife/biodiversity",
        "wetland": "Wetland protection",
        "flooding": "Flooding",
        "deforestation": "Deforestation",
        "forest": "Forest land",
        "tribal": "Tribal rights",
        "adivasi": "Tribal rights",
        "protest": "Community protest",
        "police": "Police action",
        "court": "Legal challenge",
        "PIL": "Public interest litigation",
        "fraud": "Fraud allegations",
    }
    page_lower = page_text.lower()
    for keyword, label in issue_keywords.items():
        if keyword.lower() in page_lower:
            if label not in result["conflict_issues"]:
                result["conflict_issues"].append(label)

    # --- Land type ---
    land_types = ["common", "private", "forest", "government", "gauchar",
                  "pasture", "wetland", "agricultural"]
    found_types = []
    for lt in land_types:
        if lt in page_lower:
            found_types.append(lt)
    if found_types:
        result["land_type"] = ", ".join(found_types)

    # --- Legal status ---
    if "high court" in page_lower or "supreme court" in page_lower:
        result["legal_status"] = "In Court"
    elif "PIL" in page_text:
        result["legal_status"] = "PIL Filed"

    # --- Year started ---
    # Look for earliest 4-digit year in a reasonable range (2005-2026)
    # Skip generic years like 2000 which appear in footers/copyrights
    years = [int(y) for y in re.findall(r"\b(20[0-2]\d)\b", page_text) if int(y) >= 2005]
    if years:
        result["year_started"] = min(years)

    # --- Land area from page (hectares or acres) ---
    ha_match = re.search(r"([\d,]+\.?\d*)\s*hectares?", page_text, re.IGNORECASE)
    if ha_match:
        result["scraped_land_area_ha"] = extract_number(ha_match.group(1))
    else:
        acre_match = re.search(r"([\d,]+\.?\d*)\s*acres?", page_text, re.IGNORECASE)
        if acre_match:
            acres = extract_number(acre_match.group(1))
            if acres:
                result["scraped_land_area_ha"] = round(acres * 0.404686, 1)

    # --- Affected people from page ---
    people_match = re.search(
        r"([\d,]+)\s*(households?|families|people|villagers)\s*(affected|displaced|impacted)?",
        page_text, re.IGNORECASE,
    )
    if people_match:
        result["scraped_affected_people"] = extract_number(people_match.group(1))

    # Check if page appears to be mostly JS-rendered (very little text content)
    if len(page_text) < 500:
        result["scrape_note"] = "Page appears JS-rendered; limited data extracted from static HTML"

    return result


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
def scrape_conflict(conflict, cache_dir, force=False, session=None):
    """
    Scrape a single conflict detail page.

    Uses cache to avoid re-fetching. Returns dict of extracted fields.
    """
    if not conflict["url_path"]:
        return {
            "capacity_mw": None,
            "developer": None,
            "description": None,
            "conflict_issues": [],
            "land_type": None,
            "legal_status": None,
            "year_started": None,
            "scrape_note": "No detail page URL (case study section only)",
        }

    url = BASE_URL + conflict["url_path"]
    # Cache filename from URL slug
    slug = conflict["url_path"].rstrip("/").split("/")[-1]
    cache_file = cache_dir / f"{slug}.json"

    # Check cache
    if cache_file.exists() and not force:
        with open(cache_file, "r") as f:
            return json.load(f)

    # Fetch page
    if session is None:
        session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {
            "capacity_mw": None,
            "developer": None,
            "description": None,
            "conflict_issues": [],
            "scrape_note": f"HTTP error: {e}",
        }

    # Parse
    detail = parse_detail_page(resp.text, url)

    # Cache result
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(detail, f, indent=2)

    return detail


def build_output(conflict, detail):
    """Merge hardcoded conflict data with scraped detail into output format."""
    url = (BASE_URL + conflict["url_path"]) if conflict["url_path"] else None

    # Use scraped land area if hardcoded is missing
    land_area_ha = conflict.get("land_area_ha")
    if land_area_ha is None and detail.get("scraped_land_area_ha"):
        land_area_ha = detail["scraped_land_area_ha"]

    # Use scraped affected people if hardcoded is missing
    affected_people = conflict.get("affected_people")
    if affected_people is None and detail.get("scraped_affected_people"):
        affected_people = detail["scraped_affected_people"]

    return {
        "name": conflict["name"],
        "url": url,
        "district": conflict["district"],
        "state": conflict["state"],
        "land_area_ha": land_area_ha,
        "affected_people": affected_people,
        "status": conflict.get("status"),
        "energy_type": classify_energy_type(conflict["name"]),
        "capacity_mw": detail.get("capacity_mw"),
        "developer": detail.get("developer"),
        "description": detail.get("description"),
        "conflict_issues": detail.get("conflict_issues", []),
        "land_type": detail.get("land_type"),
        "legal_status": detail.get("legal_status"),
        "year_started": detail.get("year_started"),
        "scrape_note": detail.get("scrape_note"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scrape Land Conflict Watch renewable energy conflicts (India)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List conflicts without scraping detail pages",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape even if cached results exist",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Classify totals
    solar = sum(1 for c in CONFLICTS if classify_energy_type(c["name"]) == "solar")
    wind = sum(1 for c in CONFLICTS if classify_energy_type(c["name"]) == "wind")
    other = len(CONFLICTS) - solar - wind

    print(f"LCW Renewable Energy Conflicts: {len(CONFLICTS)} total")
    print(f"  Solar: {solar}, Wind: {wind}, Other: {other}")
    print()

    if args.dry_run:
        print(f"{'#':>3}  {'Type':6}  {'State':<20}  {'District':<18}  Name")
        print("-" * 90)
        for i, c in enumerate(CONFLICTS, 1):
            etype = classify_energy_type(c["name"])
            has_url = "Y" if c["url_path"] else "N"
            print(
                f"{i:>3}  {etype:6}  {c['state']:<20}  {c['district']:<18}  "
                f"{c['name']}  [url:{has_url}]"
            )

        # State summary
        print()
        print("By state:")
        from collections import Counter
        state_counts = Counter(c["state"] for c in CONFLICTS)
        for state, count in state_counts.most_common():
            print(f"  {state}: {count}")
        return

    # Scrape detail pages
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    results = []
    scraped = 0
    cached = 0
    errors = 0

    for i, conflict in enumerate(CONFLICTS, 1):
        name = conflict["name"]
        etype = classify_energy_type(name)

        if not conflict["url_path"]:
            print(f"  [{i:>2}/{len(CONFLICTS)}] {name} -- no detail URL (case study)")
            detail = scrape_conflict(conflict, CACHE_DIR, force=args.force, session=session)
            results.append(build_output(conflict, detail))
            continue

        # Check if cached
        slug = conflict["url_path"].rstrip("/").split("/")[-1]
        cache_file = CACHE_DIR / f"{slug}.json"
        is_cached = cache_file.exists() and not args.force

        if is_cached:
            status_str = "cached"
            cached += 1
        else:
            status_str = "scraping..."
            scraped += 1

        print(f"  [{i:>2}/{len(CONFLICTS)}] {etype:6}  {name} -- {status_str}")

        detail = scrape_conflict(conflict, CACHE_DIR, force=args.force, session=session)
        if detail is None:
            detail = {}

        if (detail.get("scrape_note") or "").startswith("HTTP error"):
            errors += 1

        results.append(build_output(conflict, detail))

        # Be polite: sleep between actual requests (not cached)
        if not is_cached and i < len(CONFLICTS):
            time.sleep(1)

    # Write output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print(f"Done. {scraped} scraped, {cached} from cache, {errors} errors.")
    print(f"Output: {output_path}")

    # Summary stats
    with_capacity = sum(1 for r in results if r["capacity_mw"] is not None)
    with_developer = sum(1 for r in results if r["developer"])
    with_desc = sum(1 for r in results if r["description"])
    with_area = sum(1 for r in results if r["land_area_ha"] is not None)
    with_people = sum(1 for r in results if r["affected_people"] is not None)

    print()
    print("Field coverage:")
    print(f"  capacity_mw:    {with_capacity}/{len(results)}")
    print(f"  developer:      {with_developer}/{len(results)}")
    print(f"  description:    {with_desc}/{len(results)}")
    print(f"  land_area_ha:   {with_area}/{len(results)}")
    print(f"  affected_people:{with_people}/{len(results)}")


if __name__ == "__main__":
    main()
