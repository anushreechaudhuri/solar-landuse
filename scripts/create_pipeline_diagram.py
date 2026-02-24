"""Create a data processing/analysis pipeline diagram."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import apply_style, save_fig, FULL_WIDTH, DPI, _TOL_MUTED

FIG_DIR = Path(__file__).parent.parent / "docs" / "figures"


def draw_box(ax, x, y, w, h, text, color, fontsize=7, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor='#555555',
                         linewidth=0.8, alpha=0.85)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, wrap=False)


def draw_arrow(ax, x1, y1, x2, y2, color='#777777'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                linewidth=1.2, shrinkA=2, shrinkB=2))


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 8.0)
    ax.axis('off')

    # Colors
    src = '#88CCEE'    # cyan - data sources
    proc = '#44AA99'   # teal - processing
    out = '#DDCC77'    # sand - outputs
    sel = '#CC6677'    # rose - site selection

    bw, bh = 1.8, 0.65  # box width, height

    # ── Row 1: Data Sources (y=7.0) ──
    ax.text(0.3, 7.7, 'DATA SOURCES', fontsize=8, fontweight='bold',
            color='#555555')

    draw_box(ax, 1.8, 7.0, bw, bh, 'GEM/GSPT\n5,093 projects', src, bold=True)
    draw_box(ax, 4.0, 7.0, bw, bh, 'GRW (GEE)\n3,957 polygons', src, bold=True)
    draw_box(ax, 6.2, 7.0, bw, bh, 'TZ-SAM (GEE)\n5,368 polygons', src,
             bold=True)

    # ── Row 2: Integration (y=5.7) ──
    draw_box(ax, 4.0, 5.7, 3.2, 0.65,
             'Spatial matching + confidence scoring\n'
             'R-tree index, IoU overlap, point-to-polygon',
             proc, fontsize=6.5)
    draw_arrow(ax, 1.8, 6.65, 3.2, 6.05)
    draw_arrow(ax, 4.0, 6.65, 4.0, 6.05)
    draw_arrow(ax, 6.2, 6.65, 4.8, 6.05)

    # Output: unified DB (y=4.5)
    draw_box(ax, 4.0, 4.5, 2.6, 0.65, 'Unified Solar DB\n6,705 entries', out,
             bold=True)
    draw_arrow(ax, 4.0, 5.35, 4.0, 4.85)

    # ── GEE screening side branch (y=4.5, right side) ──
    draw_box(ax, 8.5, 4.5, 2.2, 0.65,
             'GEE screening\nDW + GHI + elevation', proc, fontsize=6.5)
    draw_arrow(ax, 5.32, 4.5, 7.38, 4.5)
    draw_arrow(ax, 8.0, 4.15, 7.2, 3.65)

    # ── Row 3: Site selection (y=3.3) ──
    draw_box(ax, 2.0, 3.3, 2.2, 0.65,
             'Treatment (n=3,676)\nhigh/very_high confidence', sel,
             fontsize=6.5)
    draw_box(ax, 6.2, 3.3, 2.2, 0.65,
             'Control (n=368)\nproposed/cancelled GEM', sel, fontsize=6.5)
    draw_arrow(ax, 3.2, 4.15, 2.0, 3.65)
    draw_arrow(ax, 4.8, 4.15, 6.2, 3.65)

    # ── EO data sources on right ──
    ax.text(9.4, 7.7, 'EO DATA', fontsize=8, fontweight='bold',
            color='#555555')
    gee_sources = [
        'Dynamic World\n(LULC, 10m)',
        'VIIRS NTL\n(radiance, 463m)',
        'Sentinel-1\n(SAR, 10m)',
        'MODIS NDVI/LST\n(250m\u20131km)',
        'WorldPop\n(population, 100m)',
        'Open Buildings\n(2.5m, temporal)',
        'Solar Atlas\n(GHI, 250m)',
    ]
    eo_top = 7.0
    eo_spacing = 0.60
    for i, label in enumerate(gee_sources):
        y = eo_top - i * eo_spacing
        draw_box(ax, 10.7, y, 1.7, 0.45, label, src, fontsize=5.5)

    # Bottom of last EO box
    eo_bottom_y = eo_top - (len(gee_sources) - 1) * eo_spacing  # = 3.4
    eo_arrow_start_y = eo_bottom_y - 0.25  # below the last box

    # Temporal collection box (y=2.0)
    draw_box(ax, 8.8, 2.0, 2.6, 0.65,
             'Temporal data collection\n4 time points \u00d7 4,044 sites', proc,
             fontsize=6.5)
    # Arrow from EO stack to temporal collection (start below Solar Atlas)
    draw_arrow(ax, 10.2, eo_arrow_start_y, 9.4, 2.35)

    # ── Row 4: Panel + DiD (y=0.7) ──
    draw_box(ax, 5.5, 0.7, 2.8, 0.65,
             'Temporal panel\n16,176 rows \u00d7 37 columns', out, fontsize=6.5,
             bold=True)
    draw_arrow(ax, 2.0, 2.95, 4.7, 1.05)
    draw_arrow(ax, 6.2, 2.95, 5.5, 1.05)
    draw_arrow(ax, 8.8, 1.65, 6.92, 0.85)

    # DiD
    draw_box(ax, 10.2, 0.7, 2.4, 0.65,
             'WLS DiD regression\nweighted by confidence', proc, fontsize=6.5)
    draw_arrow(ax, 6.92, 0.7, 8.98, 0.7)

    # ── Step labels on left ──
    steps = [
        (7.0, '1'),
        (5.7, '2'),
        (4.5, '3'),
        (3.3, '3'),
        (2.0, '4'),
        (0.7, '5'),
    ]
    for y, label in steps:
        ax.text(0.15, y, label, fontsize=9, color='#BBBBBB', fontweight='bold',
                va='center', ha='center',
                bbox=dict(boxstyle='circle,pad=0.15', facecolor='#F0F0F0',
                          edgecolor='#CCCCCC', linewidth=0.5))

    # Legend (well below the bottom row)
    legend_items = [
        (src, 'Data source'), (proc, 'Processing'),
        (out, 'Output'), (sel, 'Site selection'),
    ]
    for i, (c, l) in enumerate(legend_items):
        x = 1.5 + i * 2.2
        box = FancyBboxPatch((x, -0.65), 0.3, 0.2,
                             boxstyle="round,pad=0.02",
                             facecolor=c, edgecolor='#555555', linewidth=0.5,
                             alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 0.4, -0.55, l, fontsize=6.5, va='center')

    fig.suptitle('Data processing and analysis pipeline', fontsize=10,
                 fontweight='bold', y=0.98)

    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_pipeline_diagram.png')
    plt.close()
    print("Saved did_pipeline_diagram.png")


if __name__ == "__main__":
    main()
