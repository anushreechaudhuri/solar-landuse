"""Figure styling for solar land-use analysis.

Single source of truth for colors, fonts, and layout constants.
All figures use the Paul Tol Muted palette (colorblind-safe).

To swap palette: change _TOL_MUTED and LULC_COLORS mapping below.
Everything downstream uses the dicts.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy as np

# ── Page layout constants ─────────────────────────────────────────────────────

FULL_WIDTH = 7.2    # inches — fits A4 with margins
HALF_WIDTH = 3.4    # inches — two-column layout
DPI = 300

# ── Paul Tol Muted palette ────────────────────────────────────────────────────
# Ref: https://personal.sron.nl/~pault/data/colourschemes.pdf

_TOL_MUTED = {
    'indigo':     '#332288',
    'cyan':       '#88CCEE',
    'teal':       '#44AA99',
    'green':      '#117733',
    'olive':      '#999933',
    'sand':       '#DDCC77',
    'rose':       '#CC6677',
    'wine':       '#882255',
    'purple':     '#AA3377',
    'pale_grey':  '#DDDDDD',
}

# ── 10-class LULC color mapping ───────────────────────────────────────────────

CLASS_ORDER = [
    'no_data', 'cropland', 'trees', 'shrub', 'grassland',
    'flooded_veg', 'built', 'bare', 'water', 'snow',
]

LULC_COLORS = {
    'no_data':     _TOL_MUTED['pale_grey'],
    'cropland':    _TOL_MUTED['sand'],
    'trees':       _TOL_MUTED['green'],
    'shrub':       _TOL_MUTED['olive'],
    'grassland':   _TOL_MUTED['teal'],
    'flooded_veg': _TOL_MUTED['indigo'],
    'built':       _TOL_MUTED['rose'],
    'bare':        _TOL_MUTED['wine'],
    'water':       _TOL_MUTED['cyan'],
    'snow':        '#F5F5F5',
}

def _hex_to_rgb(hex_str):
    """Convert '#RRGGBB' to (R, G, B) tuple (0-255)."""
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

LULC_COLORS_RGB = {k: _hex_to_rgb(v) for k, v in LULC_COLORS.items()}

# ── Display labels ────────────────────────────────────────────────────────────

CLASS_LABELS = {
    'no_data':     'No Data',
    'cropland':    'Cropland',
    'trees':       'Trees/Forest',
    'shrub':       'Shrub/Scrub',
    'grassland':   'Grassland',
    'flooded_veg': 'Flooded Veg',
    'built':       'Built-Up',
    'bare':        'Bare Ground',
    'water':       'Water',
    'snow':        'Snow/Ice',
}

# ── Dataset colors (Paul Tol Bright) ─────────────────────────────────────────

DATASET_COLORS = {
    'dw':         '#4477AA',
    'worldcover': '#EE6677',
    'esri':       '#228833',
    'glad':       '#CCBB44',
    'vlm':        '#AA3377',
}

DATASET_LABELS = {
    'dw':         'Dynamic World',
    'worldcover': 'WorldCover',
    'esri':       'ESRI LULC',
    'glad':       'GLAD',
    'vlm':        'VLM V2',
}

# ── Solar overlay styling ────────────────────────────────────────────────────

SOLAR_STYLE = {
    'facecolor': (0.8, 0.8, 0.8, 0.3),
    'edgecolor': '#888888',
    'linewidth': 3,
}

# ── Change colors ─────────────────────────────────────────────────────────────

CHANGE_COLORS = {
    'increase': '#44AA99',   # teal
    'decrease': '#CC6677',   # rose
}

# ── Helper functions ──────────────────────────────────────────────────────────

def apply_style():
    """Set matplotlib rcParams for consistent figure styling."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        # Grid
        'axes.grid': False,
        # Figure
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        # Lines
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


def save_fig(fig, path):
    """Save figure at publication DPI with tight bounding box."""
    fig.savefig(str(path), dpi=DPI, bbox_inches='tight')


def title_case(text):
    """Convert snake_case or lowercase text to Title Case."""
    return text.replace('_', ' ').title()


def get_lulc_color_list(skip_nodata=True):
    """Return list of hex colors in CLASS_ORDER (for bar charts).

    If skip_nodata=True, omits the first entry (no_data).
    """
    start = 1 if skip_nodata else 0
    return [LULC_COLORS[c] for c in CLASS_ORDER[start:]]


def get_class_label_list(skip_nodata=True):
    """Return list of display labels in CLASS_ORDER (for bar charts).

    If skip_nodata=True, omits the first entry (No Data).
    """
    start = 1 if skip_nodata else 0
    return [CLASS_LABELS[c] for c in CLASS_ORDER[start:]]


def make_lulc_legend(skip_nodata=True, **kwargs):
    """Create a list of matplotlib Patch handles for a LULC legend."""
    start = 1 if skip_nodata else 0
    handles = []
    for cname in CLASS_ORDER[start:]:
        handles.append(Patch(
            facecolor=LULC_COLORS[cname],
            edgecolor='#333333' if cname == 'snow' else None,
            label=CLASS_LABELS[cname],
        ))
    return handles


def make_lulc_colormap(skip_nodata=False):
    """Create a ListedColormap for raster display (indexed by class ID)."""
    colors = []
    for cname in CLASS_ORDER:
        r, g, b = LULC_COLORS_RGB[cname]
        colors.append((r / 255.0, g / 255.0, b / 255.0))
    return ListedColormap(colors, name='lulc_10class')
