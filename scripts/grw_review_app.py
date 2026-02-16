"""Generate a Leaflet-based review app for confirming GRW polygon matches.

Reads site_matches.json and generates a static HTML file with:
- Satellite basemap tiles
- GRW polygon overlays (purple fill) per site
- Site center markers with search radius circles
- Confirm/Reject buttons per site
- Export to confirmed_matches.json via localStorage

Usage:
    python scripts/grw_review_app.py
"""

import json
import webbrowser
from pathlib import Path

PROJECT_DIR = Path("/Users/anushreechaudhuri/Documents/Projects/solar-landuse")
GRW_DIR = PROJECT_DIR / "data" / "grw"
MATCHES_PATH = GRW_DIR / "site_matches.json"
OUTPUT_HTML = GRW_DIR / "review.html"


def generate_html(matches):
    """Generate static HTML with Leaflet map for reviewing GRW matches."""

    # Build site data for JS (only sites with matches)
    sites_js = []
    for site_key, info in matches.items():
        if info["match_status"] != "matched":
            continue
        polygons_geojson = [p["geojson"] for p in info["polygons"]]
        sites_js.append({
            "key": site_key,
            "name": info["name"],
            "lat": _get_site_coords(site_key)[0],
            "lon": _get_site_coords(site_key)[1],
            "num_raw": info.get("num_raw_polygons", 0),
            "num_merged": info.get("num_merged_polygons", 0),
            "total_area_m2": info.get("total_area_m2", 0),
            "construction_dates": info.get("construction_dates", []),
            "min_distance_km": info.get("min_distance_km", 0),
            "polygons": polygons_geojson,
        })

    # Also include unmatched sites for reference
    unmatched = [
        {"key": k, "name": v["name"],
         "lat": _get_site_coords(k)[0], "lon": _get_site_coords(k)[1]}
        for k, v in matches.items() if v["match_status"] != "matched"
    ]

    # Use string template with placeholders to avoid f-string brace issues
    html = _HTML_TEMPLATE.replace("__SITES_JSON__", json.dumps(sites_js))
    html = html.replace("__UNMATCHED_JSON__", json.dumps(unmatched))

    return html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
    <title>GRW Solar Polygon Review</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; display: flex; height: 100vh; }
        #sidebar { width: 350px; overflow-y: auto; background: #f5f5f5; border-right: 1px solid #ccc; padding: 12px; }
        #map { flex: 1; }
        h2 { margin-bottom: 8px; font-size: 16px; }
        .site-card { background: white; border-radius: 6px; padding: 10px; margin-bottom: 8px;
                      border: 2px solid transparent; cursor: pointer; }
        .site-card:hover { border-color: #6a5acd; }
        .site-card.active { border-color: #6a5acd; background: #f0eeff; }
        .site-card.confirmed { border-left: 4px solid #2ecc71; }
        .site-card.rejected { border-left: 4px solid #e74c3c; }
        .site-card h3 { font-size: 13px; margin-bottom: 4px; }
        .site-card .meta { font-size: 11px; color: #666; }
        .btn-group { margin-top: 6px; display: flex; gap: 4px; }
        .btn { padding: 4px 10px; border: none; border-radius: 4px; cursor: pointer; font-size: 11px; }
        .btn-confirm { background: #2ecc71; color: white; }
        .btn-reject { background: #e74c3c; color: white; }
        .btn-reset { background: #95a5a6; color: white; }
        .btn-export { background: #3498db; color: white; padding: 8px 16px; font-size: 13px;
                       width: 100%; margin-top: 8px; }
        .unmatched { opacity: 0.5; }
        .summary { font-size: 12px; margin-bottom: 12px; padding: 8px; background: #e8e8e8; border-radius: 4px; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h2>GRW Solar Review</h2>
        <div class="summary" id="summary"></div>
        <div id="site-list"></div>
        <button class="btn btn-export" onclick="exportConfirmed()">Export Confirmed Matches</button>
    </div>
    <div id="map"></div>

    <script>
    const SITES = __SITES_JSON__;
    const UNMATCHED = __UNMATCHED_JSON__;
    const SEARCH_RADIUS_KM = 5;

    // State
    let decisions = JSON.parse(localStorage.getItem('grw_decisions') || '{}');

    // Map setup
    const map = L.map('map').setView([23.8, 90.0], 7);
    L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
        maxZoom: 20, subdomains: ['mt0','mt1','mt2','mt3'],
        attribution: 'Google Satellite'
    }).addTo(map);

    // Layer groups per site
    const siteLayers = {};

    function updateSummary() {
        const confirmed = Object.values(decisions).filter(d => d === 'confirmed').length;
        const rejected = Object.values(decisions).filter(d => d === 'rejected').length;
        const pending = SITES.length - confirmed - rejected;
        document.getElementById('summary').innerHTML =
            `<b>${SITES.length}</b> matched sites | ` +
            `<span style="color:#2ecc71">${confirmed} confirmed</span> | ` +
            `<span style="color:#e74c3c">${rejected} rejected</span> | ` +
            `${pending} pending<br>` +
            `<small>${UNMATCHED.length} sites had no GRW match</small>`;
    }

    function setDecision(key, decision) {
        decisions[key] = decision;
        localStorage.setItem('grw_decisions', JSON.stringify(decisions));
        renderSidebar();
        updateSummary();
    }

    function flyToSite(key) {
        const site = SITES.find(s => s.key === key);
        if (site) map.flyTo([site.lat, site.lon], 14);
        document.querySelectorAll('.site-card').forEach(c => c.classList.remove('active'));
        const el = document.getElementById('card-' + key);
        if (el) el.classList.add('active');
    }

    function renderSidebar() {
        const list = document.getElementById('site-list');
        let html = '<h3 style="font-size:12px;margin:8px 0 4px;color:#666;">Matched Sites</h3>';

        SITES.forEach(site => {
            const dec = decisions[site.key] || '';
            const cls = dec ? (dec === 'confirmed' ? 'confirmed' : 'rejected') : '';
            html += `
                <div class="site-card ${cls}" id="card-${site.key}" onclick="flyToSite('${site.key}')">
                    <h3>${site.name}</h3>
                    <div class="meta">
                        ${site.num_raw} raw &rarr; ${site.num_merged} merged |
                        ${(site.total_area_m2/10000).toFixed(1)} ha |
                        ${site.min_distance_km.toFixed(1)} km<br>
                        Dates: ${site.construction_dates.join(', ') || 'unknown'}
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-confirm" onclick="event.stopPropagation(); setDecision('${site.key}','confirmed')">Confirm</button>
                        <button class="btn btn-reject" onclick="event.stopPropagation(); setDecision('${site.key}','rejected')">Reject</button>
                        <button class="btn btn-reset" onclick="event.stopPropagation(); setDecision('${site.key}','')">Reset</button>
                    </div>
                </div>`;
        });

        if (UNMATCHED.length > 0) {
            html += '<h3 style="font-size:12px;margin:12px 0 4px;color:#999;">No GRW Match</h3>';
            UNMATCHED.forEach(site => {
                html += `
                    <div class="site-card unmatched" onclick="map.flyTo([${site.lat},${site.lon}],14)">
                        <h3>${site.name}</h3>
                        <div class="meta">No GRW polygons within ${SEARCH_RADIUS_KM} km</div>
                    </div>`;
            });
        }

        list.innerHTML = html;
    }

    // Add map layers
    SITES.forEach(site => {
        const group = L.layerGroup().addTo(map);
        siteLayers[site.key] = group;

        // Search radius circle
        L.circle([site.lat, site.lon], {
            radius: SEARCH_RADIUS_KM * 1000,
            color: '#888', weight: 1, dashArray: '4', fillOpacity: 0.02
        }).addTo(group);

        // Site center marker
        L.circleMarker([site.lat, site.lon], {
            radius: 5, color: '#ff6600', fillColor: '#ff6600', fillOpacity: 0.8, weight: 1
        }).bindPopup(`<b>${site.key}</b><br>${site.name}`).addTo(group);

        // GRW polygons
        site.polygons.forEach(poly => {
            L.geoJSON(poly, {
                style: { color: '#800080', weight: 2, fillColor: '#800080', fillOpacity: 0.3 }
            }).bindPopup(
                `<b>${site.key}</b><br>` +
                `Area: ${(site.total_area_m2/10000).toFixed(1)} ha<br>` +
                `Dates: ${site.construction_dates.join(', ')}`
            ).addTo(group);
        });
    });

    // Unmatched markers
    UNMATCHED.forEach(site => {
        L.circleMarker([site.lat, site.lon], {
            radius: 4, color: '#999', fillColor: '#999', fillOpacity: 0.5, weight: 1
        }).bindPopup(`<b>${site.key}</b> (no GRW match)`).addTo(map);
    });

    function exportConfirmed() {
        const result = {};
        SITES.forEach(site => {
            if (decisions[site.key] === 'confirmed') {
                result[site.key] = {
                    name: site.name,
                    polygons: site.polygons,
                    total_area_m2: site.total_area_m2,
                    construction_dates: site.construction_dates,
                };
            }
        });

        const blob = new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'confirmed_matches.json';
        a.click();
        URL.revokeObjectURL(url);
    }

    renderSidebar();
    updateSummary();
    </script>
</body>
</html>"""


# Site coordinates (duplicated here to avoid importing from other scripts)
_SITE_COORDS = {
    "teesta": (25.628342, 89.541082),
    "feni": (22.787567, 91.367187),
    "manikganj": (23.780834, 89.824775),
    "moulvibazar": (24.493896, 91.633043),
    "pabna": (23.961375, 89.159720),
    "mymensingh": (24.702233, 90.461730),
    "tetulia": (26.482817, 88.410139),
    "lalmonirhat": (25.997873, 89.154467),
    "mongla": (22.574239, 89.570388),
    "sirajganj68": (24.403976, 89.738849),
    "teknaf": (20.981669, 92.256021),
    "sirajganj6": (24.386137, 89.748970),
    "kaptai": (22.491471, 92.226588),
    "sharishabari": (24.772287, 89.842629),
    "barishal": (22.657015, 90.339194),
}


def _get_site_coords(key):
    return _SITE_COORDS.get(key, (23.8, 90.0))


def main():
    if not MATCHES_PATH.exists():
        print(f"ERROR: {MATCHES_PATH} not found. Run query_grw.py first.")
        return

    with open(MATCHES_PATH) as f:
        matches = json.load(f)

    matched = sum(1 for m in matches.values() if m["match_status"] == "matched")
    print(f"Generating review app for {matched} matched sites...")

    html = generate_html(matches)
    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"Saved to {OUTPUT_HTML}")

    webbrowser.open(f"file://{OUTPUT_HTML}")
    print("Opened in browser. Review each site, then click 'Export Confirmed Matches'.")
    print(f"Save the downloaded file to {GRW_DIR / 'confirmed_matches.json'}")


if __name__ == "__main__":
    main()
