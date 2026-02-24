"""
Create summary figures for the DiD analysis.

Uses figure_style.py for consistent Paul Tol Muted palette, 300 DPI,
A4-width formatting, no gridlines, no top/right spines.

Figures:
  1. Confidence & source Venn summary (unified DB overview)
  2. LULC composition: stacked bars treatment vs control at 4 time points
  3. DiD coefficient forest plot (improved version)
  4. Parallel trends panel (2x2: built, crops, NTL, SAR)
  5. Site-level change distributions (treatment vs control box plots)
  6. Temporal trajectories (line plots with confidence bands)

Usage:
    python scripts/create_did_figures.py
    python scripts/create_did_figures.py --country bangladesh
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (
    apply_style, save_fig, LULC_COLORS, CLASS_LABELS, CLASS_ORDER,
    CHANGE_COLORS, FULL_WIDTH, HALF_WIDTH, DPI, _TOL_MUTED,
    get_lulc_color_list, get_class_label_list,
)

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PANEL_CSV = DATA_DIR / "temporal_panel.csv"
UNIFIED_DB = DATA_DIR / "unified_solar_db.json"
DID_RESULTS = DATA_DIR / "did_results" / "did_results.json"
COMPARISON_SITES = DATA_DIR / "comparison_sites.json"

# Treatment/control palette
TREAT_COLOR = _TOL_MUTED['rose']      # #CC6677
CONTROL_COLOR = _TOL_MUTED['cyan']    # #88CCEE
TREAT_DARK = _TOL_MUTED['wine']       # #882255
CONTROL_DARK = _TOL_MUTED['indigo']   # #332288

# DW class columns → display mapping
DW_COL_MAP = {
    'dw_crops_pct': ('Cropland', LULC_COLORS['cropland']),
    'dw_trees_pct': ('Trees', LULC_COLORS['trees']),
    'dw_built_pct': ('Built-up', LULC_COLORS['built']),
    'dw_water_pct': ('Water', LULC_COLORS['water']),
    'dw_bare_pct': ('Bare', LULC_COLORS['bare']),
    'dw_grass_pct': ('Grassland', LULC_COLORS['grassland']),
    'dw_flooded_vegetation_pct': ('Flooded', LULC_COLORS['flooded_veg']),
    'dw_shrub_and_scrub_pct': ('Shrub', LULC_COLORS['shrub']),
    'dw_snow_and_ice_pct': ('Snow', LULC_COLORS['snow']),
}

TIME_POINTS = ['baseline', 'pre_construction', 'post_construction', 'current']
TP_LABELS = ['Baseline\n(2016)', 'Pre-\nconstruction', 'Post-\nconstruction', 'Current\n(2025)']


def load_data(country_filter=None):
    df = pd.read_csv(PANEL_CSV)
    if country_filter:
        df = df[df['country'].str.lower() == country_filter.lower()].copy()
        did_path = DATA_DIR / "did_results" / country_filter.lower() / "did_results.json"
    else:
        did_path = DID_RESULTS
    with open(did_path) as f:
        did = json.load(f)
    with open(UNIFIED_DB) as f:
        db = json.load(f)
    return df, did, db


# ── Figure 1: Dataset Integration Summary ────────────────────────────────────

def fig_integration_summary(db, country_filter=None):
    """Source overlap bar chart + confidence tier donut."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))

    if country_filter:
        entries = [e for e in db if e['country'].lower() == country_filter.lower()]
        region_label = country_filter.title()
    else:
        entries = db
        region_label = "South Asia"

    # Panel A: Source overlap
    ax = axes[0]
    from collections import Counter
    combos = Counter(tuple(sorted(e['sources'])) for e in entries)

    labels_map = {
        ('gem', 'grw', 'tzsam'): 'All three',
        ('gem', 'tzsam'): 'GEM + TZ-SAM',
        ('gem', 'grw'): 'GEM + GRW',
        ('grw', 'tzsam'): 'GRW + TZ-SAM',
        ('gem',): 'GEM only',
        ('grw',): 'GRW only',
        ('tzsam',): 'TZ-SAM only',
    }
    combo_order = [('gem', 'grw', 'tzsam'), ('gem', 'tzsam'), ('gem', 'grw'),
                   ('grw', 'tzsam'), ('gem',), ('grw',), ('tzsam',)]

    bar_labels = []
    bar_vals = []
    bar_colors = []
    multi_color = _TOL_MUTED['teal']
    single_color = _TOL_MUTED['pale_grey']

    for combo in combo_order:
        n = combos.get(combo, 0)
        if n > 0:
            bar_labels.append(labels_map[combo])
            bar_vals.append(n)
            bar_colors.append(multi_color if len(combo) > 1 else single_color)

    y_pos = range(len(bar_labels))
    bars = ax.barh(y_pos, bar_vals, color=bar_colors, edgecolor='white',
                   linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels)
    ax.set_xlabel('Number of sites')
    ax.set_title(f'(a) Source overlap ({region_label})', fontsize=10, fontweight='bold')

    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=8)

    ax.invert_yaxis()

    # Panel B: Confidence tiers (only high/very_high for treatment)
    ax = axes[1]
    conf_counts = Counter(e['confidence'] for e in entries)

    tiers = ['very_high', 'high', 'medium', 'low']
    tier_labels = ['Very high', 'High', 'Medium', 'Low']
    tier_colors = [_TOL_MUTED['teal'], _TOL_MUTED['green'],
                   _TOL_MUTED['sand'], _TOL_MUTED['pale_grey']]
    vals = [conf_counts.get(t, 0) for t in tiers]

    wedges, texts, autotexts = ax.pie(
        vals, labels=tier_labels, colors=tier_colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})

    for t in autotexts:
        t.set_fontsize(8)
    for t in texts:
        t.set_fontsize(8)

    ax.set_title(f'(b) Confidence tiers ({region_label})', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_fig1_integration_summary.png')
    plt.close()
    print("  Saved did_fig1_integration_summary.png")


# ── Figure 2: LULC Composition Stacked Bars ─────────────────────────────────

def fig_lulc_stacked(df):
    """Stacked bar chart: LULC composition at 4 time points, treatment vs control."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.5), sharey=True)

    # Classes to show (ordered by typical dominance)
    show_classes = ['dw_built_pct', 'dw_crops_pct', 'dw_trees_pct',
                    'dw_water_pct', 'dw_bare_pct', 'dw_grass_pct',
                    'dw_flooded_vegetation_pct', 'dw_shrub_and_scrub_pct']

    n_treat = df[df['group'] == 'treatment']['site_id'].nunique()
    n_ctrl = df[df['group'] == 'control']['site_id'].nunique()

    for gi, (group, group_label) in enumerate([
        ('treatment', f'Treatment (n={n_treat})'),
        ('control', f'Control (n={n_ctrl})'),
    ]):
        ax = axes[gi]
        sub = df[df['group'] == group]

        bottoms = np.zeros(len(TIME_POINTS))
        for col in show_classes:
            label, color = DW_COL_MAP[col]
            means = [sub[sub.time_point == tp][col].mean() for tp in TIME_POINTS]
            ax.bar(range(len(TIME_POINTS)), means, bottom=bottoms,
                   color=color, label=label, width=0.7, edgecolor='white',
                   linewidth=0.3)
            bottoms += means

        ax.set_xticks(range(len(TIME_POINTS)))
        ax.set_xticklabels(TP_LABELS, fontsize=7)
        ax.set_title(group_label, fontsize=10, fontweight='bold')
        if gi == 0:
            ax.set_ylabel('Land cover (%)')

    # Single legend
    handles = [Patch(facecolor=DW_COL_MAP[c][1], label=DW_COL_MAP[c][0])
               for c in show_classes]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.15, 0.5),
               fontsize=7, frameon=False)

    fig.suptitle('Dynamic World LULC composition over time', fontsize=10,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_fig2_lulc_stacked.png')
    plt.close()
    print("  Saved did_fig2_lulc_stacked.png")


# ── Figure 3: DiD Forest Plot (improved) ─────────────────────────────────────

def fig_did_forest(did):
    """Forest plot of DiD treatment effects with confidence intervals."""
    apply_style()

    results = did['regressions']
    # Reorder: LULC, remote sensing, vegetation, temperature, socioeconomic
    # Note: Population (sum) excluded — CI [-109, -8] compresses x-axis;
    # population density captures the same effect at compatible scale
    order = ['Bare ground (%)', 'Water (%)', 'Built-up (%)', 'Cropland (%)',
             'Trees (%)', 'Grassland (%)',
             'Nighttime light (nW/sr/cm\u00b2)',
             'SAR VV backscatter (dB)', 'SAR VH backscatter (dB)',
             'NDVI', 'EVI',
             'Daytime LST (\u00b0C)', 'Nighttime LST (\u00b0C)',
             'Population density',
             'Building presence', 'Building height (m)', 'Building count']

    ordered = []
    for label in order:
        match = [r for r in results if r['label'] == label]
        if match:
            ordered.append(match[0])

    fig_height = max(3.8, 0.45 * len(ordered))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, fig_height))

    n = len(ordered)
    y_pos = list(range(n))

    for i, r in enumerate(ordered):
        coef = r['treatment_coef']
        ci_lo = r['treatment_ci_low']
        ci_hi = r['treatment_ci_high']
        pval = r['treatment_pval']

        # Color by significance
        if pval < 0.01:
            color = _TOL_MUTED['indigo']
            marker = 'D'
        elif pval < 0.05:
            color = _TOL_MUTED['teal']
            marker = 'D'
        elif pval < 0.1:
            color = _TOL_MUTED['sand']
            marker = 's'
        else:
            color = _TOL_MUTED['pale_grey']
            marker = 'o'

        # CI line
        ax.plot([ci_lo, ci_hi], [i, i], color='#555555', linewidth=1.2,
                solid_capstyle='round', zorder=1)
        # Point estimate
        ax.scatter(coef, i, color=color, marker=marker, s=60, zorder=2,
                   edgecolor='#333333', linewidth=0.5)

        # Annotate coefficient and p-value on the right
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
        ax.text(max(ci_hi + 0.3, 4.0), i,
                f'{coef:+.2f}{sig}  (p={pval:.3f})',
                va='center', fontsize=7, color='#555555')

    # Zero reference line
    ax.axvline(x=0, color='#999999', linewidth=0.8, linestyle='--', zorder=0)

    # Separator lines between sections
    lulc_labels = {'Bare ground (%)', 'Water (%)', 'Built-up (%)', 'Cropland (%)',
                   'Trees (%)', 'Grassland (%)'}
    rs_labels = {'Nighttime light (nW/sr/cm\u00b2)', 'SAR VV backscatter (dB)',
                 'SAR VH backscatter (dB)'}
    lulc_idx = [i for i, r in enumerate(ordered) if r['label'] in lulc_labels]
    rs_idx = [i for i, r in enumerate(ordered) if r['label'] in rs_labels]
    if lulc_idx and rs_idx:
        ax.axhline(y=max(lulc_idx) + 0.5, color='#CCCCCC', linewidth=0.5,
                   linestyle='-')
    if rs_idx:
        after_rs = max(rs_idx) + 0.5
        if after_rs < n - 0.5:
            ax.axhline(y=after_rs, color='#CCCCCC', linewidth=0.5,
                       linestyle='-')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['label'] for r in ordered])
    ax.set_xlabel('DiD treatment effect (pre\u2192post change difference)')
    region = did.get('country', 'South Asia')
    ax.set_title(f'Difference-in-differences: treatment effects on land cover\nand remote sensing indicators ({region})',
                 fontsize=10, fontweight='bold')

    ax.invert_yaxis()

    # Legend
    legend_handles = [
        plt.scatter([], [], color=_TOL_MUTED['indigo'], marker='D', s=40,
                    edgecolor='#333', label='p < 0.01'),
        plt.scatter([], [], color=_TOL_MUTED['teal'], marker='D', s=40,
                    edgecolor='#333', label='p < 0.05'),
        plt.scatter([], [], color=_TOL_MUTED['sand'], marker='s', s=40,
                    edgecolor='#333', label='p < 0.1'),
        plt.scatter([], [], color=_TOL_MUTED['pale_grey'], marker='o', s=40,
                    edgecolor='#333', label='n.s.'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=8,
              frameon=True, framealpha=0.9)

    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_fig3_forest_plot.png')
    plt.close()
    print("  Saved did_fig3_forest_plot.png")


# ── Figure 4: Parallel Trends Panel ─────────────────────────────────────────

def fig_parallel_trends_panel(df):
    """2x2 panel: parallel trends for built-up, cropland, NTL, SAR VV."""
    apply_style()

    outcomes = [
        ('dw_built_pct', 'Built-up (%)'),
        ('dw_crops_pct', 'Cropland (%)'),
        ('viirs_avg_rad', 'Nighttime light\n(nW/sr/cm\u00b2)'),
        ('sar_vv_db', 'SAR VV backscatter (dB)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.0))
    axes = axes.flatten()

    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for idx, (col, ylabel) in enumerate(outcomes):
        ax = axes[idx]

        for group, color, marker, ls in [
            ('treatment', TREAT_COLOR, 'o', '-'),
            ('control', CONTROL_COLOR, 's', '-'),
        ]:
            sub = df[df['group'] == group]
            means = []
            sems = []
            for tp in TIME_POINTS:
                vals = pd.to_numeric(sub[sub.time_point == tp][col], errors='coerce').dropna()
                means.append(vals.mean())
                sems.append(vals.sem())

            n = sub['site_id'].nunique()
            ax.errorbar(range(4), means, yerr=sems,
                        marker=marker, color=color, linewidth=1.5,
                        markersize=6, capsize=3, capthick=1,
                        label=f'{group.title()} (n={n})', linestyle=ls)

        # Vertical line at construction
        ax.axvline(x=1.5, color='#999999', linewidth=0.6, linestyle=':',
                   alpha=0.7)
        ax.text(1.55, ax.get_ylim()[1], 'construction', fontsize=6,
                color='#999999', va='top', rotation=0)

        ax.set_xticks(range(4))
        ax.set_xticklabels(TP_LABELS, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f'{panel_labels[idx]} {ylabel.split(chr(10))[0]}',
                     fontsize=9, fontweight='bold')

        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('Parallel trends: treatment vs. control sites',
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_fig4_parallel_trends.png')
    plt.close()
    print("  Saved did_fig4_parallel_trends.png")


# ── Figure 5: Change Distributions (Box + Strip) ────────────────────────────

def fig_change_distributions(df):
    """Box + strip plots of pre→post change for key outcomes, by group."""
    apply_style()

    # Compute changes
    pre = df[df.time_point == 'pre_construction'].set_index('site_id')
    post = df[df.time_point == 'post_construction'].set_index('site_id')
    common = pre.index.intersection(post.index)

    outcomes = [
        ('dw_built_pct', 'Built-up\n(\u0394pp)'),
        ('dw_crops_pct', 'Cropland\n(\u0394pp)'),
        ('dw_bare_pct', 'Bare ground\n(\u0394pp)'),
        ('viirs_avg_rad', 'NTL\n(\u0394nW/sr/cm\u00b2)'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 3.0), sharey=False)

    for idx, (col, ylabel) in enumerate(outcomes):
        ax = axes[idx]
        pre_vals = pd.to_numeric(pre.loc[common, col], errors='coerce')
        post_vals = pd.to_numeric(post.loc[common, col], errors='coerce')
        delta = post_vals - pre_vals
        groups = pre.loc[common, 'group']

        treat_delta = delta[groups == 'treatment'].dropna()
        control_delta = delta[groups == 'control'].dropna()

        # Box plots
        bp = ax.boxplot([treat_delta, control_delta],
                        positions=[0, 1], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops={'color': '#333333', 'linewidth': 1.5})

        bp['boxes'][0].set_facecolor(TREAT_COLOR)
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(CONTROL_COLOR)
        bp['boxes'][1].set_alpha(0.6)

        # Strip plots (jittered points)
        np.random.seed(42)
        jitter_t = np.random.normal(0, 0.08, len(treat_delta))
        jitter_c = np.random.normal(0, 0.08, len(control_delta))
        ax.scatter(jitter_t, treat_delta, color=TREAT_COLOR, s=15,
                   alpha=0.7, edgecolor='none', zorder=3)
        ax.scatter(1 + jitter_c, control_delta, color=CONTROL_COLOR, s=15,
                   alpha=0.7, edgecolor='none', zorder=3)

        ax.axhline(y=0, color='#999999', linewidth=0.6, linestyle='--')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Treat', 'Control'], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

        # Annotate means
        for gi, (vals, x) in enumerate([(treat_delta, 0), (control_delta, 1)]):
            m = vals.mean()
            ax.text(x, ax.get_ylim()[1] * 0.95, f'\u03bc={m:.1f}',
                    ha='center', fontsize=7, color='#555555')

    fig.suptitle('Pre \u2192 Post construction change distributions',
                 fontsize=10, fontweight='bold', y=1.03)
    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_fig5_change_distributions.png')
    plt.close()
    print("  Saved did_fig5_change_distributions.png")


# ── Figure 6: Full Temporal Trajectories ─────────────────────────────────────

def fig_temporal_trajectories(df):
    """Line plots of mean LULC classes over 4 time points, faceted by group."""
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.5), sharey=True)

    show_classes = ['dw_built_pct', 'dw_crops_pct', 'dw_trees_pct',
                    'dw_water_pct', 'dw_bare_pct']

    for gi, (group, title) in enumerate([('treatment', 'Treatment sites'),
                                          ('control', 'Control sites')]):
        ax = axes[gi]
        sub = df[df['group'] == group]

        for col in show_classes:
            label, color = DW_COL_MAP[col]
            means = []
            sems = []
            for tp in TIME_POINTS:
                vals = sub[sub.time_point == tp][col].dropna()
                means.append(vals.mean())
                sems.append(vals.sem())

            means = np.array(means)
            sems = np.array(sems)
            x = range(4)

            ax.plot(x, means, color=color, linewidth=1.5, marker='o',
                    markersize=4, label=label)
            ax.fill_between(x, means - sems, means + sems,
                            color=color, alpha=0.15)

        ax.axvline(x=1.5, color='#999999', linewidth=0.6, linestyle=':')
        ax.set_xticks(range(4))
        ax.set_xticklabels(TP_LABELS, fontsize=7)
        ax.set_title(title, fontsize=10, fontweight='bold')
        if gi == 0:
            ax.set_ylabel('Coverage (%)')
        if gi == 1:
            ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Land cover trajectories: treatment vs. control',
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_fig6_trajectories.png')
    plt.close()
    print("  Saved did_fig6_trajectories.png")


# ── Figure 7: Feasibility Score Comparison ───────────────────────────────────

def fig_feasibility(comp):
    """Compare feasibility scores of treatment vs control sites."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))

    control = comp.get('comparison_sites', [])
    treatment = comp.get('treatment_sites', [])

    # Panel A: Feasibility score histograms
    ax = axes[0]
    t_scores = [s['feasibility_score'] for s in treatment if s.get('feasibility_score') is not None]
    c_scores = [s['feasibility_score'] for s in control if s.get('feasibility_score') is not None]

    bins = np.arange(0, 1.15, 0.1)
    # Plot control first (background), then treatment on top (foreground)
    ax.hist(c_scores, bins=bins, alpha=0.6, color=CONTROL_COLOR,
            label=f'Control (n={len(c_scores)})', edgecolor='white')
    ax.hist(t_scores, bins=bins, alpha=0.8, color=TREAT_COLOR,
            label=f'Treatment (n={len(t_scores)})', edgecolor='white')
    ax.set_xlabel('Feasibility score')
    ax.set_ylabel('Count')
    ax.set_title('(a) Site feasibility scores', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)

    # Panel B: GHI comparison
    ax = axes[1]
    t_ghi = [s['ghi_kwh_m2_day'] for s in treatment if s.get('ghi_kwh_m2_day')]
    c_ghi = [s['ghi_kwh_m2_day'] for s in control if s.get('ghi_kwh_m2_day')]

    bp = ax.boxplot([t_ghi, c_ghi], positions=[0, 1], widths=0.5,
                    patch_artist=True, showfliers=True,
                    medianprops={'color': '#333', 'linewidth': 1.5},
                    flierprops={'markersize': 3, 'alpha': 0.5})
    bp['boxes'][0].set_facecolor(TREAT_COLOR)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(CONTROL_COLOR)
    bp['boxes'][1].set_alpha(0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Treatment', 'Control'])
    ax.set_ylabel('GHI (kWh/m\u00b2/day)')
    ax.set_title('(b) Solar irradiance', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, FIG_DIR / 'did_fig7_feasibility.png')
    plt.close()
    print("  Saved did_fig7_feasibility.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create DiD analysis figures")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter to single country (e.g. 'bangladesh')")
    args = parser.parse_args()

    country = args.country
    label = country.title() if country else "South Asia"

    print(f"Loading data ({label})...")
    df, did, db = load_data(country)

    with open(COMPARISON_SITES) as f:
        comp = json.load(f)
    # Filter comparison sites if country specified
    if country:
        comp = {
            'comparison_sites': [s for s in comp.get('comparison_sites', [])
                                 if s['country'].lower() == country.lower()],
            'treatment_sites': [s for s in comp.get('treatment_sites', [])
                                if s['country'].lower() == country.lower()],
        }

    print(f"Panel: {len(df)} rows, {df.site_id.nunique()} sites")
    print(f"Unified DB: {len(db)} entries")
    print(f"DiD results: {len(did['regressions'])} regressions")

    print("\nGenerating figures...")
    fig_integration_summary(db, country)
    fig_lulc_stacked(df)
    fig_did_forest(did)
    fig_parallel_trends_panel(df)
    fig_change_distributions(df)
    fig_temporal_trajectories(df)
    fig_feasibility(comp)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
