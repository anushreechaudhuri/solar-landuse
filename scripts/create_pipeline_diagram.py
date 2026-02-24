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
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.4, 7.2)
    ax.axis('off')

    # Colors
    src = '#88CCEE'    # cyan - data sources
    proc = '#44AA99'   # teal - processing
    out = '#DDCC77'    # sand - outputs
    analysis = '#CC6677'  # rose - analysis

    bw, bh = 1.8, 0.65  # box width, height

    # ── Row 1: Data Sources (y=6.2) ──
    ax.text(0.3, 6.9, 'DATA SOURCES', fontsize=8, fontweight='bold',
            color='#555555')

    draw_box(ax, 1.8, 6.2, bw, bh, 'GEM/GSPT\n5,093 projects', src, bold=True)
    draw_box(ax, 4.0, 6.2, bw, bh, 'GRW (GEE)\n3,957 polygons', src, bold=True)
    draw_box(ax, 6.2, 6.2, bw, bh, 'TZ-SAM (GEE)\n5,368 polygons', src,
             bold=True)

    # ── Row 2: Integration (y=4.9) ──
    draw_box(ax, 4.0, 4.9, 3.2, 0.65,
             'Spatial matching + confidence scoring\n'
             'R-tree index, IoU overlap, point-to-polygon',
             proc, fontsize=6.5)
    draw_arrow(ax, 1.8, 5.85, 3.2, 5.25)
    draw_arrow(ax, 4.0, 5.85, 4.0, 5.25)
    draw_arrow(ax, 6.2, 5.85, 4.8, 5.25)

    # Output: unified DB (y=3.7)
    draw_box(ax, 4.0, 3.7, 2.6, 0.65, 'Unified Solar DB\n6,705 entries', out,
             bold=True)
    draw_arrow(ax, 4.0, 4.55, 4.0, 4.05)

    # ── Row 3: Site selection (y=2.5) ──
    draw_box(ax, 2.0, 2.5, 2.2, 0.65,
             'Treatment (n=3,676)\nhigh/very_high confidence', analysis,
             fontsize=6.5)
    draw_box(ax, 6.2, 2.5, 2.2, 0.65,
             'Control (n=368)\nproposed/cancelled GEM', analysis, fontsize=6.5)
    draw_arrow(ax, 3.2, 3.35, 2.0, 2.85)
    draw_arrow(ax, 4.8, 3.35, 6.2, 2.85)

    # ── GEE screening side branch (y=3.7, right side) ──
    draw_box(ax, 8.5, 3.7, 2.2, 0.65,
             'GEE screening\nDW + GHI + elevation', proc, fontsize=6.5)
    draw_arrow(ax, 5.32, 3.7, 7.38, 3.7)
    draw_arrow(ax, 8.0, 3.35, 7.2, 2.85)

    # ── EO data sources on right ──
    ax.text(9.4, 6.9, 'EO DATA', fontsize=8, fontweight='bold',
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
    for i, label in enumerate(gee_sources):
        y = 6.2 - i * 0.58
        draw_box(ax, 10.7, y, 1.7, 0.45, label, src, fontsize=5.5)

    # Temporal collection box (y=1.3)
    draw_box(ax, 8.8, 1.3, 2.6, 0.65,
             'Temporal data collection\n4 time points \u00d7 4,044 sites', proc,
             fontsize=6.5)
    draw_arrow(ax, 10.2, 2.72, 9.4, 1.65)

    # ── Row 4: Panel + DiD (y=0.2) ──
    draw_box(ax, 5.5, 0.2, 2.8, 0.65,
             'Temporal panel\n16,176 rows \u00d7 26 columns', out, fontsize=6.5,
             bold=True)
    draw_arrow(ax, 2.0, 2.15, 4.7, 0.55)
    draw_arrow(ax, 6.2, 2.15, 5.5, 0.55)
    draw_arrow(ax, 8.8, 0.95, 6.92, 0.35)

    # DiD
    draw_box(ax, 10.2, 0.2, 2.4, 0.65,
             'WLS DiD regression\nweighted by confidence', proc, fontsize=6.5)
    draw_arrow(ax, 6.92, 0.2, 8.98, 0.2)

    # Phase labels
    ax.text(0.15, 6.2, 'Phase 1', fontsize=7, color='#999999', rotation=90,
            va='center')
    ax.text(0.15, 4.9, 'P1', fontsize=7, color='#999999', rotation=90,
            va='center')
    ax.text(0.15, 3.7, 'P1', fontsize=7, color='#999999', rotation=90,
            va='center')
    ax.text(0.15, 2.5, 'P2/P3', fontsize=7, color='#999999', rotation=90,
            va='center')
    ax.text(0.15, 0.2, 'P5', fontsize=7, color='#999999', rotation=90,
            va='center')

    # Legend
    legend_items = [
        (src, 'Data source'), (proc, 'Processing'),
        (out, 'Output'), (analysis, 'Site selection'),
    ]
    for i, (c, l) in enumerate(legend_items):
        x = 1.5 + i * 2.2
        box = FancyBboxPatch((x, -0.35), 0.3, 0.2,
                             boxstyle="round,pad=0.02",
                             facecolor=c, edgecolor='#555555', linewidth=0.5,
                             alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 0.4, -0.25, l, fontsize=6.5, va='center')

    fig.suptitle('Data processing and analysis pipeline', fontsize=10,
                 fontweight='bold', y=0.98)

    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_pipeline_diagram.png')
    plt.close()
    print("Saved did_pipeline_diagram.png")


if __name__ == "__main__":
    main()
