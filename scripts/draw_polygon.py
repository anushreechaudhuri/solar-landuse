"""Interactive polygon drawing tool on DW LULC or satellite images.

Click to add vertices, right-click or press Enter to close the polygon.
Press 'u' to undo last point, 'r' to reset all points.
Outputs coordinates in lon/lat format for confirmed_matches.json.

Usage:
    python scripts/draw_polygon.py --site moulvibazar --year 2026
    python scripts/draw_polygon.py --site moulvibazar --year 2026 --satellite
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
# Use macosx backend on macOS (TkAgg can cause blank windows)
import platform
if platform.system() == 'Darwin':
    matplotlib.use('macosx')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
IMG_DIR = DATA_DIR / "case_study_images"
DW_DIR = DATA_DIR / "case_study_dw_rasters"
GRW_PATH = DATA_DIR / "grw" / "confirmed_matches.json"

# Site definitions (matching case_studies.py)
SITES = {
    "teesta": {"lat": 25.629209, "lon": 89.544870},
    "feni": {"lat": 22.787567, "lon": 91.367187},
    "manikganj": {"lat": 23.780834, "lon": 89.824775},
    "moulvibazar": {"lat": 24.493312, "lon": 91.633107},
}

# DW remap + colorize (matching case_studies.py)
DW_RAW_TO_10CLASS = {0: 8, 1: 2, 2: 4, 3: 5, 4: 1, 5: 3, 6: 6, 7: 7, 8: 9}
LULC_10CLASS_RGB = {
    0: (221, 221, 221), 1: (221, 204, 119), 2: (17, 119, 51),
    3: (153, 153, 51), 4: (68, 170, 153), 5: (51, 34, 136),
    6: (204, 102, 119), 7: (136, 34, 85), 8: (136, 204, 238),
    9: (245, 245, 245),
}


def pixel_to_lonlat(px_x, px_y, img_shape, site_lat, site_lon, buffer_km=2):
    """Convert pixel coordinates to lon/lat."""
    h, w = img_shape[:2]
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.32 * math.cos(math.radians(site_lat))
    lon_min = site_lon - buffer_km / km_per_deg_lon
    lon_max = site_lon + buffer_km / km_per_deg_lon
    lat_min = site_lat - buffer_km / km_per_deg_lat
    lat_max = site_lat + buffer_km / km_per_deg_lat

    lon = lon_min + (px_x / w) * (lon_max - lon_min)
    lat = lat_max - (px_y / h) * (lat_max - lat_min)  # y is inverted
    return round(lon, 6), round(lat, 6)


def main():
    parser = argparse.ArgumentParser(description="Draw polygon on satellite/DW image")
    parser.add_argument("--site", required=True, choices=list(SITES.keys()))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--satellite", action="store_true",
                        help="Use satellite image instead of DW LULC")
    parser.add_argument("--buffer-km", type=float, default=2,
                        help="Buffer in km (must match image extent)")
    args = parser.parse_args()

    site = SITES[args.site]

    # Load image
    if args.satellite:
        img_path = IMG_DIR / f"{args.site}_{args.year}.png"
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            sys.exit(1)
        img = np.array(Image.open(img_path))
        title = f"{args.site} {args.year} (Satellite)"
    else:
        dw_path = DW_DIR / f"{args.site}_{args.year}_dw.npz"
        if not dw_path.exists():
            print(f"DW raster not found: {dw_path}")
            sys.exit(1)
        data = np.load(dw_path)
        remapped = data["remapped"]
        h, w = remapped.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, color in LULC_10CLASS_RGB.items():
            img[remapped == cid] = color
        title = f"{args.site} {args.year} (DW LULC)"

    # Draw existing polygon if any
    existing_coords = []
    if GRW_PATH.exists():
        with open(GRW_PATH) as f:
            grw = json.load(f)
        site_data = grw.get(args.site, {})
        for poly in site_data.get("polygons", []):
            coords = poly.get("coordinates", [[]])[0]
            existing_coords.append(coords)

    # Interactive drawing
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(f"{title}\nClick to add vertices | Right-click/Enter to close | 'u' undo | 'r' reset",
                 fontsize=10)

    # Draw existing polygon in blue
    for coords in existing_coords:
        if len(coords) >= 3:
            km_per_deg_lat = 110.574
            km_per_deg_lon = 111.32 * math.cos(math.radians(site["lat"]))
            lon_min = site["lon"] - args.buffer_km / km_per_deg_lon
            lat_min = site["lat"] - args.buffer_km / km_per_deg_lat
            lon_max = site["lon"] + args.buffer_km / km_per_deg_lon
            lat_max = site["lat"] + args.buffer_km / km_per_deg_lat
            h, w = img.shape[:2]
            xs, ys = [], []
            for lon, lat in coords:
                px_x = (lon - lon_min) / (lon_max - lon_min) * w
                px_y = (1 - (lat - lat_min) / (lat_max - lat_min)) * h
                xs.append(px_x)
                ys.append(px_y)
            ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.5, label="Existing polygon")

    # Collect new points
    points_px = []  # pixel coords
    line, = ax.plot([], [], 'r.-', linewidth=1.5, markersize=8)
    closed_line, = ax.plot([], [], 'r-', linewidth=2)

    def update_line():
        if points_px:
            xs = [p[0] for p in points_px]
            ys = [p[1] for p in points_px]
            line.set_data(xs, ys)
        else:
            line.set_data([], [])
        closed_line.set_data([], [])
        fig.canvas.draw_idle()

    def close_polygon():
        if len(points_px) < 3:
            print("Need at least 3 points to close polygon")
            return
        xs = [p[0] for p in points_px] + [points_px[0][0]]
        ys = [p[1] for p in points_px] + [points_px[0][1]]
        closed_line.set_data(xs, ys)
        fig.canvas.draw_idle()

        # Convert to lon/lat
        coords = []
        for px_x, px_y in points_px:
            lon, lat = pixel_to_lonlat(px_x, px_y, img.shape,
                                       site["lat"], site["lon"], args.buffer_km)
            coords.append([lon, lat])
        # Close the ring
        coords.append(coords[0])

        print("\n" + "=" * 60)
        print(f"POLYGON COORDINATES ({args.site})")
        print("=" * 60)
        print(json.dumps({"type": "Polygon", "coordinates": [coords]}, indent=2))
        print("\nTo update confirmed_matches.json, replace the polygon for "
              f"'{args.site}' with the coordinates above.")
        print(f"\nLon/lat pairs ({len(coords)} vertices):")
        for lon, lat in coords:
            print(f"  [{lon}, {lat}]")

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 3:  # right click
            close_polygon()
            return
        if event.button == 1:  # left click
            points_px.append((event.xdata, event.ydata))
            update_line()
            lon, lat = pixel_to_lonlat(event.xdata, event.ydata, img.shape,
                                       site["lat"], site["lon"], args.buffer_km)
            print(f"  Point {len(points_px)}: pixel=({event.xdata:.0f}, {event.ydata:.0f}) "
                  f"→ lon/lat=({lon}, {lat})")

    def on_key(event):
        if event.key == 'u':  # undo
            if points_px:
                points_px.pop()
                update_line()
                print("  Undone last point")
        elif event.key == 'r':  # reset
            points_px.clear()
            update_line()
            print("  Reset all points")
        elif event.key == 'enter':
            close_polygon()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print(f"\nShowing {title}")
    print(f"Image shape: {img.shape}")
    print("Click to add polygon vertices.")
    print("Right-click or press Enter to close and output coordinates.")
    print("Press 'u' to undo, 'r' to reset.\n")

    plt.tight_layout()
    fig.canvas.draw()
    plt.show()


if __name__ == "__main__":
    main()
