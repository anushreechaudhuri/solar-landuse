"""Generate a Leaflet.Draw app for hand-drawing solar polygon boundaries.

Shows all 15 sites on a satellite map. For sites with confirmed GRW polygons,
those polygons are loaded as editable shapes. For sites without GRW data (or
where GRW polygons were rejected/inaccurate), you can draw new polygons.

Features:
- Satellite basemap (Google)
- All 15 sites listed in sidebar, grouped by status
- Leaflet.Draw toolbar for polygon/rectangle drawing
- Click a site to fly to it and activate drawing for that site
- Edit/delete existing polygons
- Export all polygons as confirmed_matches.json

Usage:
    python scripts/draw_polygons_app.py
"""

import json
import webbrowser
from pathlib import Path

PROJECT_DIR = Path("/Users/anushreechaudhuri/Documents/Projects/solar-landuse")
GRW_DIR = PROJECT_DIR / "data" / "grw"
MATCHES_PATH = GRW_DIR / "site_matches.json"
CONFIRMED_PATH = GRW_DIR / "confirmed_matches.json"
OUTPUT_HTML = GRW_DIR / "draw.html"

# All sites
SITES = {
    "teesta": {"name": "Teesta (Gaibandha/Beximco) 200 MW", "lat": 25.628342, "lon": 89.541082, "mw": 200},
    "feni": {"name": "Feni/Sonagazi 75 MW", "lat": 22.787567, "lon": 91.367187, "mw": 75},
    "manikganj": {"name": "Manikganj (Spectra) 35 MW", "lat": 23.780834, "lon": 89.824775, "mw": 35},
    "moulvibazar": {"name": "Moulvibazar 10 MW", "lat": 24.493896, "lon": 91.633043, "mw": 10},
    "pabna": {"name": "Pabna 64 MW", "lat": 23.961375, "lon": 89.159720, "mw": 64},
    "mymensingh": {"name": "Mymensingh (HDFC) 50 MW", "lat": 24.702233, "lon": 90.461730, "mw": 50},
    "tetulia": {"name": "Tetulia/Panchagarh (Sympa) 8 MW", "lat": 26.482817, "lon": 88.410139, "mw": 8},
    "lalmonirhat": {"name": "Lalmonirhat Rangpur (Intraco) 30 MW", "lat": 25.997873, "lon": 89.154467, "mw": 30},
    "mongla": {"name": "Mongla 100 MW", "lat": 22.574239, "lon": 89.570388, "mw": 100},
    "sirajganj68": {"name": "Sirajganj 68 MW", "lat": 24.403976, "lon": 89.738849, "mw": 68},
    "teknaf": {"name": "Teknaf (Joules) 20 MW", "lat": 20.981669, "lon": 92.256021, "mw": 20},
    "sirajganj6": {"name": "Sirajganj 6 MW", "lat": 24.386137, "lon": 89.748970, "mw": 6},
    "kaptai": {"name": "Kaptai 7.4 MW", "lat": 22.491471, "lon": 92.226588, "mw": 7.4},
    "sharishabari": {"name": "Sharishabari 3 MW", "lat": 24.772287, "lon": 89.842629, "mw": 3},
    "barishal": {"name": "Barishal 1 MW", "lat": 22.657015, "lon": 90.339194, "mw": 1},
}


def load_existing_polygons():
    """Load confirmed or matched GRW polygons as starting data."""
    # Prefer confirmed_matches.json, fall back to site_matches.json
    if CONFIRMED_PATH.exists():
        with open(CONFIRMED_PATH) as f:
            confirmed = json.load(f)
        # confirmed_matches has format: {site_key: {name, polygons: [geojson], ...}}
        result = {}
        for key, data in confirmed.items():
            result[key] = [p for p in data.get("polygons", [])]
        return result

    if MATCHES_PATH.exists():
        with open(MATCHES_PATH) as f:
            matches = json.load(f)
        result = {}
        for key, data in matches.items():
            if data.get("match_status") == "matched":
                result[key] = [p.get("geojson") for p in data.get("polygons", []) if p.get("geojson")]
        return result

    return {}


def main():
    existing = load_existing_polygons()
    print(f"Loaded existing polygons for {len(existing)} sites: {list(existing.keys())}")

    # Build sites list for JS
    sites_js = []
    for key, info in SITES.items():
        sites_js.append({
            "key": key,
            "name": info["name"],
            "lat": info["lat"],
            "lon": info["lon"],
            "mw": info["mw"],
            "has_grw": key in existing,
            "polygons": existing.get(key, []),
        })

    # Sort: sites needing polygons first, then confirmed
    sites_js.sort(key=lambda s: (s["has_grw"], s["key"]))

    html = _HTML_TEMPLATE.replace("__SITES_JSON__", json.dumps(sites_js))

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)
    print(f"Saved to {OUTPUT_HTML}")

    # Serve via local HTTP server so blob downloads work (file:// blocks them)
    import http.server
    import threading
    import os

    os.chdir(str(GRW_DIR))
    port = 8765

    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(('localhost', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}/draw.html"
    webbrowser.open(url)
    print(f"Serving at {url}")
    print("Draw polygons for each site, then click 'Export All Polygons'.")
    print(f"Save the downloaded file to {CONFIRMED_PATH}")
    print("Press Ctrl+C to stop the server when done.")

    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
        print("\nServer stopped.")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
    <title>Solar Polygon Drawing Tool</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; display: flex; height: 100vh; }
        #sidebar { width: 360px; overflow-y: auto; background: #f5f5f5; border-right: 1px solid #ccc; padding: 12px; }
        #map { flex: 1; }
        h2 { margin-bottom: 4px; font-size: 16px; }
        .instructions { font-size: 11px; color: #666; margin-bottom: 10px; line-height: 1.4; }
        .site-card { background: white; border-radius: 6px; padding: 10px; margin-bottom: 6px;
                     border: 2px solid transparent; cursor: pointer; transition: all 0.15s; }
        .site-card:hover { border-color: #6a5acd; }
        .site-card.active { border-color: #6a5acd; background: #f0eeff; }
        .site-card.has-polygons { border-left: 4px solid #2ecc71; }
        .site-card.needs-polygons { border-left: 4px solid #e67e22; }
        .site-card h3 { font-size: 13px; margin-bottom: 2px; }
        .site-card .meta { font-size: 11px; color: #666; }
        .site-card .poly-count { font-size: 11px; font-weight: bold; margin-top: 2px; }
        .poly-count.zero { color: #e67e22; }
        .poly-count.has { color: #2ecc71; }
        .btn-row { display: flex; gap: 6px; margin-top: 8px; }
        .btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .btn-export { background: #3498db; color: white; flex: 1; }
        .btn-clear { background: #e74c3c; color: white; font-size: 11px; padding: 3px 8px; }
        .section-label { font-size: 12px; font-weight: 600; color: #888; margin: 10px 0 4px; }
        .summary { font-size: 12px; margin-bottom: 10px; padding: 8px; background: #e8e8e8; border-radius: 4px; }
        .leaflet-draw-toolbar { z-index: 1001; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h2>Solar Polygon Drawing</h2>
        <div class="instructions">
            Click a site to fly to it. Use the polygon/rectangle tools on the map to draw solar boundaries.
            Drawn shapes are auto-assigned to the active (highlighted) site. Edit or delete shapes freely.
        </div>
        <div class="summary" id="summary"></div>
        <div id="site-list"></div>
        <div class="btn-row">
            <button class="btn btn-export" onclick="exportAll()">Export All Polygons</button>
        </div>
    </div>
    <div id="map"></div>

    <script>
    const SITES = __SITES_JSON__;

    // State: site_key -> [L.Polygon layers]
    let sitePolygons = {};
    let activeSiteKey = null;

    // Restore from localStorage if available
    const saved = JSON.parse(localStorage.getItem('drawn_polygons') || '{}');

    // Initialize map
    const map = L.map('map').setView([23.8, 90.0], 7);
    L.tileLayer('https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
        maxZoom: 20, subdomains: ['mt0','mt1','mt2','mt3'],
        attribution: 'Google Satellite'
    }).addTo(map);

    // Drawing layer
    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    // Draw control
    const drawControl = new L.Control.Draw({
        edit: { featureGroup: drawnItems },
        draw: {
            polygon: { allowIntersection: false, shapeOptions: { color: '#800080', fillOpacity: 0.3 } },
            rectangle: { shapeOptions: { color: '#800080', fillOpacity: 0.3 } },
            polyline: false, circle: false, circlemarker: false, marker: false
        }
    });
    map.addControl(drawControl);

    // Site center markers (always visible)
    SITES.forEach(site => {
        L.circleMarker([site.lat, site.lon], {
            radius: 6, color: '#ff6600', fillColor: '#ff6600', fillOpacity: 0.8, weight: 2
        }).bindTooltip(site.key, { permanent: false }).addTo(map);
    });

    // Initialize polygon state from saved or from GRW data
    SITES.forEach(site => {
        sitePolygons[site.key] = [];

        // Check localStorage first, then fall back to GRW
        const savedPolys = saved[site.key];
        const sourcePolys = savedPolys || site.polygons;

        if (sourcePolys && sourcePolys.length > 0) {
            sourcePolys.forEach(geojson => {
                try {
                    const layer = L.geoJSON(geojson, {
                        style: { color: '#800080', weight: 2, fillColor: '#800080', fillOpacity: 0.3 }
                    });
                    layer.eachLayer(l => {
                        l._siteKey = site.key;
                        drawnItems.addLayer(l);
                        sitePolygons[site.key].push(l);
                    });
                } catch(e) { console.warn('Failed to load polygon for', site.key, e); }
            });
        }
    });

    // Handle new drawings
    map.on(L.Draw.Event.CREATED, function(e) {
        const layer = e.layer;
        if (!activeSiteKey) {
            alert('Please select a site first by clicking it in the sidebar.');
            return;
        }
        layer._siteKey = activeSiteKey;
        layer.setStyle({ color: '#800080', fillColor: '#800080', fillOpacity: 0.3 });
        drawnItems.addLayer(layer);
        sitePolygons[activeSiteKey].push(layer);
        saveState();
        renderSidebar();
    });

    // Handle edits
    map.on(L.Draw.Event.EDITED, function(e) {
        saveState();
    });

    // Handle deletes
    map.on(L.Draw.Event.DELETED, function(e) {
        e.layers.eachLayer(layer => {
            const key = layer._siteKey;
            if (key && sitePolygons[key]) {
                sitePolygons[key] = sitePolygons[key].filter(l => l !== layer);
            }
        });
        saveState();
        renderSidebar();
    });

    function saveState() {
        const state = {};
        Object.keys(sitePolygons).forEach(key => {
            if (sitePolygons[key].length > 0) {
                state[key] = sitePolygons[key].map(l => l.toGeoJSON().geometry);
            }
        });
        localStorage.setItem('drawn_polygons', JSON.stringify(state));
    }

    function setActiveSite(key) {
        activeSiteKey = key;
        const site = SITES.find(s => s.key === key);
        if (site) map.flyTo([site.lat, site.lon], 16);
        renderSidebar();
    }

    function clearSite(key) {
        if (!confirm('Remove all polygons for ' + key + '?')) return;
        sitePolygons[key].forEach(l => drawnItems.removeLayer(l));
        sitePolygons[key] = [];
        saveState();
        renderSidebar();
    }

    function updateSummary() {
        const withPolys = Object.values(sitePolygons).filter(arr => arr.length > 0).length;
        const total = SITES.length;
        document.getElementById('summary').innerHTML =
            `<b>${withPolys}/${total}</b> sites have polygons | ` +
            `<b>${total - withPolys}</b> need drawing`;
    }

    function renderSidebar() {
        const list = document.getElementById('site-list');
        let html = '';

        // Sites needing polygons
        const needsWork = SITES.filter(s => sitePolygons[s.key].length === 0);
        const hasWork = SITES.filter(s => sitePolygons[s.key].length > 0);

        if (needsWork.length > 0) {
            html += '<div class="section-label">Needs Polygons (' + needsWork.length + ')</div>';
            needsWork.forEach(site => { html += siteCardHtml(site); });
        }
        if (hasWork.length > 0) {
            html += '<div class="section-label">Has Polygons (' + hasWork.length + ')</div>';
            hasWork.forEach(site => { html += siteCardHtml(site); });
        }

        list.innerHTML = html;
        updateSummary();
    }

    function siteCardHtml(site) {
        const n = sitePolygons[site.key].length;
        const isActive = activeSiteKey === site.key;
        const cls = [
            'site-card',
            isActive ? 'active' : '',
            n > 0 ? 'has-polygons' : 'needs-polygons'
        ].filter(Boolean).join(' ');
        const countCls = n > 0 ? 'has' : 'zero';

        return `
            <div class="${cls}" onclick="setActiveSite('${site.key}')">
                <h3>${site.name}</h3>
                <div class="meta">${site.lat.toFixed(4)}, ${site.lon.toFixed(4)} | ${site.mw} MW</div>
                <div class="poly-count ${countCls}">
                    ${n > 0 ? n + ' polygon(s)' : 'No polygons yet'}
                    ${n > 0 ? '<button class="btn-clear" onclick="event.stopPropagation(); clearSite(\'' + site.key + '\')">Clear</button>' : ''}
                </div>
            </div>`;
    }

    function exportAll() {
        const result = {};
        SITES.forEach(site => {
            const layers = sitePolygons[site.key];
            if (layers.length > 0) {
                result[site.key] = {
                    name: site.name,
                    polygons: layers.map(l => l.toGeoJSON().geometry),
                    total_area_m2: 0,
                    construction_dates: [],
                };
            }
        });

        const jsonStr = JSON.stringify(result, null, 2);

        // Try blob download first
        try {
            const blob = new Blob([jsonStr], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'confirmed_matches.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch(e) {
            console.warn('Blob download failed, opening in new tab');
        }

        // Always also show in a new window as fallback
        const count = Object.keys(result).length;
        const w = window.open('', '_blank');
        if (w) {
            w.document.write('<pre>' + jsonStr.replace(/</g, '&lt;') + '</pre>');
            w.document.title = 'confirmed_matches.json (' + count + ' sites)';
        }
        alert('Exported ' + count + ' sites.\n\nIf no file downloaded, copy from the new tab that opened.\nSave to: data/grw/confirmed_matches.json');
    }

    renderSidebar();
    </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
