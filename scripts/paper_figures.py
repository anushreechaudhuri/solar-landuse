#!/usr/bin/env python3
"""Generate all publication-quality figures for the solar land-use paper.

Outputs PDF figures to paper/figures/ at 300 DPI, A4-width (7.2 in),
Paul Tol Muted palette, no gridlines, no top/right spines.

Usage:
    python scripts/paper_figures.py
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── Import project style module ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from figure_style import (
    apply_style, save_fig, LULC_COLORS, DATASET_COLORS,
    FULL_WIDTH, HALF_WIDTH, DPI, _TOL_MUTED, CLASS_LABELS,
)

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

# ── Outcome styling ──────────────────────────────────────────────────────────
OUTCOME_COLORS = {
    'dw_crops_pct':   _TOL_MUTED['sand'],
    'dw_bare_pct':    _TOL_MUTED['wine'],
    'dw_trees_pct':   _TOL_MUTED['green'],
    'dw_built_pct':   _TOL_MUTED['rose'],
    'dw_water_pct':   _TOL_MUTED['cyan'],
    'ndvi_mean':      _TOL_MUTED['olive'],
    'vlm_solar_panels': '#333333',
    'vlm_crops':      _TOL_MUTED['sand'],
    'vlm_trees':      _TOL_MUTED['green'],
    'vlm_built':      _TOL_MUTED['rose'],
    'vlm_bare':       _TOL_MUTED['wine'],
    'vlm_water':      _TOL_MUTED['cyan'],
    'viirs_avg_rad':  _TOL_MUTED['sand'],
    'pop_mean':       _TOL_MUTED['rose'],
    'bldg_presence':  _TOL_MUTED['wine'],
    'lst_night_c':    _TOL_MUTED['indigo'],
    'evi_mean':       _TOL_MUTED['green'],
    'sar_vh_db':      _TOL_MUTED['teal'],
}

OUTCOME_LABELS = {
    'dw_crops_pct':   'Cropland (%)',
    'dw_bare_pct':    'Bare Ground (%)',
    'dw_trees_pct':   'Tree Cover (%)',
    'dw_built_pct':   'Built-Up (%)',
    'dw_water_pct':   'Water (%)',
    'ndvi_mean':      'NDVI',
    'vlm_solar_panels': 'Solar Panels (%)',
    'vlm_crops':      'VLM Cropland (%)',
    'vlm_trees':      'VLM Trees (%)',
    'vlm_built':      'VLM Built-Up (%)',
    'vlm_bare':       'VLM Bare Ground (%)',
    'vlm_water':      'VLM Water (%)',
    'viirs_avg_rad':  'VIIRS Nightlights (nW)',
    'pop_mean':       'Population Density',
    'bldg_presence':  'Building Presence',
    'lst_night_c':    'Night LST (\u00b0C)',
    'evi_mean':       'EVI',
    'sar_vh_db':      'SAR VH (dB)',
}

COUNTRY_COLORS = {
    'India':       _TOL_MUTED['indigo'],
    'Pakistan':    _TOL_MUTED['teal'],
    'Bangladesh':  _TOL_MUTED['sand'],
    'Sri Lanka':   _TOL_MUTED['rose'],
    'Nepal':       _TOL_MUTED['green'],
    'Bhutan':      _TOL_MUTED['olive'],
}


# ── Data loading ─────────────────────────────────────────────────────────────
def load_panel():
    return pd.read_csv(os.path.join(DATA, 'full_panel.csv'), low_memory=False)


def load_sites():
    with open(os.path.join(DATA, 'unified_solar_db.json')) as f:
        return json.load(f)


def load_event_study(balanced=False):
    fn = 'event_study_balanced.json' if balanced else 'event_study.json'
    with open(os.path.join(DATA, 'event_study_results', fn)) as f:
        return json.load(f)


def load_conflicts():
    with open(os.path.join(DATA, 'lcw_matched_conflicts.json')) as f:
        return json.load(f)


# ── Event study helpers ──────────────────────────────────────────────────────
def extract_coefficients(outcome_data):
    """Extract sorted event-time coefficients from an outcome dict."""
    coefs = outcome_data['coefficients']
    times = sorted([int(k) for k in coefs.keys()])
    result = {
        'time': times,
        'coef': [coefs[str(t)]['coef'] for t in times],
        'ci_low': [coefs[str(t)]['ci_low'] for t in times],
        'ci_high': [coefs[str(t)]['ci_high'] for t in times],
    }
    return result


def plot_event_study_panel(ax, outcome_data, color, label=None, alpha_band=0.2):
    """Plot one event-study coefficient path on an axis."""
    d = extract_coefficients(outcome_data)
    t = np.array(d['time'])
    c = np.array(d['coef'])
    lo = np.array(d['ci_low'])
    hi = np.array(d['ci_high'])

    ax.fill_between(t, lo, hi, alpha=alpha_band, color=color, linewidth=0)
    ax.plot(t, c, 'o-', color=color, markersize=3, linewidth=1.2, label=label)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
    ax.axvline(-0.5, color='grey', linewidth=0.7, linestyle='--')
    ax.set_xlabel('Years Relative to Construction')


def run_event_study(panel, outcome, site_ids=None):
    """Run a within-site event study using two-way demeaning (Mundlak).

    Returns a dict matching the event_study.json structure.
    """
    df = panel.copy()
    if site_ids is not None:
        df = df[df['site_id'].isin(site_ids)]

    df = df.dropna(subset=[outcome, 'event_time', 'site_id', 'year'])
    df = df[(df['event_time'] >= -5) & (df['event_time'] <= 6)]
    df['event_time'] = df['event_time'].astype(int)

    if len(df) < 20:
        return None

    # Two-way demean
    y = df[outcome].values.astype(float)
    site_ids_arr = df['site_id'].values
    years = df['year'].values

    site_map = {s: i for i, s in enumerate(np.unique(site_ids_arr))}
    year_map = {yr: i for i, yr in enumerate(np.unique(years))}
    site_idx = np.array([site_map[s] for s in site_ids_arr])
    year_idx = np.array([year_map[yr] for yr in years])

    # Demean
    site_means = np.zeros(len(site_map))
    np.add.at(site_means, site_idx, y)
    site_counts = np.zeros(len(site_map))
    np.add.at(site_counts, site_idx, 1)
    site_means /= np.maximum(site_counts, 1)

    year_means = np.zeros(len(year_map))
    np.add.at(year_means, year_idx, y)
    year_counts = np.zeros(len(year_map))
    np.add.at(year_counts, year_idx, 1)
    year_means /= np.maximum(year_counts, 1)

    grand_mean = np.mean(y)
    y_dm = y - site_means[site_idx] - year_means[year_idx] + grand_mean

    # Build event-time dummies (reference = -1)
    event_times = sorted(df['event_time'].unique())
    if -1 in event_times:
        event_times = [t for t in event_times if t != -1]

    et_arr = df['event_time'].values
    X = np.zeros((len(y_dm), len(event_times)))
    for j, t in enumerate(event_times):
        X[:, j] = (et_arr == t).astype(float)

    # Demean X
    X_dm = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        sm = np.zeros(len(site_map))
        np.add.at(sm, site_idx, col)
        sm /= np.maximum(site_counts, 1)
        ym = np.zeros(len(year_map))
        np.add.at(ym, year_idx, col)
        ym /= np.maximum(year_counts, 1)
        gm = np.mean(col)
        X_dm[:, j] = col - sm[site_idx] - ym[year_idx] + gm

    # OLS
    try:
        beta = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    resid = y_dm - X_dm @ beta
    n = len(y_dm)
    k = X_dm.shape[1]
    n_sites = len(site_map)

    # Cluster-robust SE (site-level)
    bread = np.linalg.inv(X_dm.T @ X_dm)
    meat = np.zeros((k, k))
    for s_idx_val in range(len(site_map)):
        mask = site_idx == s_idx_val
        if mask.sum() == 0:
            continue
        score = X_dm[mask].T @ resid[mask]
        meat += np.outer(score, score)
    scale = n_sites / (n_sites - 1) * (n - 1) / (n - k)
    vcov = bread @ meat @ bread * scale
    se = np.sqrt(np.diag(vcov))

    # Build results
    from scipy.stats import t as t_dist
    dof = n_sites - 1
    coefficients = {}
    for j, t in enumerate(event_times):
        t_stat = beta[j] / se[j] if se[j] > 0 else 0
        pval = 2 * (1 - t_dist.cdf(abs(t_stat), dof))
        ci_low = beta[j] - 1.96 * se[j]
        ci_high = beta[j] + 1.96 * se[j]
        coefficients[str(t)] = {
            'coef': float(beta[j]),
            'se': float(se[j]),
            'pval': float(pval),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        }
    # Add reference period
    coefficients['-1'] = {
        'coef': 0.0, 'se': 0.0, 'pval': 1.0, 'ci_low': 0.0, 'ci_high': 0.0
    }

    # Pre-trends F-test (joint significance of pre-period coefficients)
    pre_idx = [j for j, t in enumerate(event_times) if t < -1]
    if len(pre_idx) >= 2:
        R = np.zeros((len(pre_idx), k))
        for i, j in enumerate(pre_idx):
            R[i, j] = 1
        Rb = R @ beta
        try:
            f_stat = float(Rb @ np.linalg.inv(R @ vcov @ R.T) @ Rb / len(pre_idx))
            from scipy.stats import f as f_dist
            pre_p = float(1 - f_dist.cdf(f_stat, len(pre_idx), dof))
        except np.linalg.LinAlgError:
            f_stat, pre_p = np.nan, np.nan
    else:
        f_stat, pre_p = np.nan, np.nan

    return {
        'outcome': outcome,
        'coefficients': coefficients,
        'n_obs': int(n),
        'n_sites': int(n_sites),
        'pre_trends_f': float(f_stat),
        'pre_trends_p': float(pre_p),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Study area and sample overview
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_study_area():
    print('  Generating Fig 1: Study area ...')
    sites = load_sites()

    lats, lons, caps, countries = [], [], [], []
    cyears = []
    for s in sites:
        lat = s.get('centroid_lat') or s.get('lat')
        lon = s.get('centroid_lon') or s.get('lon')
        if lat is None or lon is None:
            continue
        lats.append(lat)
        lons.append(lon)
        caps.append(s.get('best_capacity_mw') or 1)
        countries.append(s.get('country', 'Unknown'))
        cy = s.get('best_construction_year')
        if cy and cy > 2000:
            cyears.append(cy)

    df = pd.DataFrame({'lat': lats, 'lon': lons, 'cap': caps, 'country': countries})

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 1.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.3, wspace=0.3)

    # Panel (a): Map
    ax_map = fig.add_subplot(gs[0, :])
    for country, grp in df.groupby('country'):
        color = COUNTRY_COLORS.get(country, '#999999')
        sizes = np.clip(grp['cap'].values * 0.3, 3, 80)
        ax_map.scatter(grp['lon'], grp['lat'], s=sizes, c=color,
                       alpha=0.55, edgecolors='white', linewidths=0.3,
                       label=f"{country} (n={len(grp)})")

    ax_map.set_xlim(60, 98)
    ax_map.set_ylim(5, 38)
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.set_title('(a) Solar installation sites across South Asia')
    ax_map.legend(fontsize=7, loc='upper left', frameon=True, framealpha=0.9)
    ax_map.set_aspect('equal')

    # Draw approximate country outlines as a simple bounding context
    # Just add a light grey box for the region
    ax_map.axhline(y=8, color='#cccccc', linewidth=0.3)

    # Panel (b): Construction year histogram
    ax_yr = fig.add_subplot(gs[1, 0])
    cyears_arr = np.array(cyears)
    bins = np.arange(cyears_arr.min() - 0.5, cyears_arr.max() + 1.5, 1)
    ax_yr.hist(cyears_arr, bins=bins, color=_TOL_MUTED['indigo'], edgecolor='white',
               linewidth=0.5)
    ax_yr.set_xlabel('Construction Year')
    ax_yr.set_ylabel('Number of Sites')
    ax_yr.set_title('(b) Construction year distribution')

    # Panel (c): Capacity histogram (log scale)
    ax_cap = fig.add_subplot(gs[1, 1])
    caps_arr = np.array([c for c in caps if c > 0])
    bins_cap = np.logspace(np.log10(max(caps_arr.min(), 0.1)),
                           np.log10(caps_arr.max()), 30)
    ax_cap.hist(caps_arr, bins=bins_cap, color=_TOL_MUTED['teal'], edgecolor='white',
                linewidth=0.5)
    ax_cap.set_xscale('log')
    ax_cap.set_xlabel('Capacity (MW)')
    ax_cap.set_ylabel('Number of Sites')
    ax_cap.set_title('(c) Capacity distribution')

    save_fig(fig, os.path.join(OUT, 'fig1_study_area.pdf'))
    plt.close(fig)
    print('    -> fig1_study_area.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: DW event study
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_event_study_dw():
    print('  Generating Fig 2: DW event study ...')
    es = load_event_study()

    outcomes = ['dw_crops_pct', 'dw_bare_pct', 'dw_trees_pct',
                'dw_built_pct', 'ndvi_mean', 'dw_water_pct']

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6))
    axes = axes.flatten()

    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        od = es['primary_outcomes'].get(outcome)
        if od is None:
            ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        color = OUTCOME_COLORS.get(outcome, _TOL_MUTED['indigo'])
        plot_event_study_panel(ax, od, color)

        pre_p = od.get('pre_trends_p', np.nan)
        n_sites = od.get('n_sites', '?')
        title = OUTCOME_LABELS.get(outcome, outcome)
        ax.set_title(title)
        if not np.isnan(pre_p):
            ax.annotate(f'Pre-trends p = {pre_p:.3f}',
                        xy=(0.02, 0.97), xycoords='axes fraction',
                        fontsize=6.5, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='none', alpha=0.8))
        ax.annotate(f'n = {n_sites:,} sites',
                    xy=(0.98, 0.97), xycoords='axes fraction',
                    fontsize=6.5, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='none', alpha=0.8))
        if i >= 3:
            ax.set_xlabel('Years Relative to Construction')
        else:
            ax.set_xlabel('')
        ax.set_ylabel('Coefficient (pp)' if 'pct' in outcome else 'Coefficient')

    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, 'fig2_event_study_dw.pdf'))
    plt.close(fig)
    print('    -> fig2_event_study_dw.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: VLM solar detection
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_vlm_solar():
    print('  Generating Fig 3: VLM solar detection ...')
    es = load_event_study()
    panel = load_panel()

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.3))

    # Panel (a): VLM solar event study coefficients
    ax = axes[0]
    vlm_solar = es.get('vlm_outcomes', {}).get('vlm_solar_panels')
    if vlm_solar:
        plot_event_study_panel(ax, vlm_solar, '#333333')
        pre_p = vlm_solar.get('pre_trends_p', np.nan)
        if not np.isnan(pre_p):
            ax.annotate(f'Pre-trends p = {pre_p:.3f}',
                        xy=(0.02, 0.97), xycoords='axes fraction',
                        fontsize=6.5, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='none', alpha=0.8))
    ax.set_title('(a) Event study: Solar panels (%)')
    ax.set_ylabel('Coefficient (pp)')

    # Panel (b): Raw mean solar % by event time
    ax = axes[1]
    df = panel.dropna(subset=['vlm_solar_panels', 'event_time']).copy()
    df = df[(df['event_time'] >= -7) & (df['event_time'] <= 8)]
    means = df.groupby('event_time')['vlm_solar_panels'].mean()
    ax.plot(means.index, means.values, 'o-', color='#333333', markersize=3,
            linewidth=1.2)
    ax.axvline(-0.5, color='grey', linewidth=0.7, linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_xlabel('Years Relative to Construction')
    ax.set_ylabel('Mean Solar Panels (%)')
    ax.set_title('(b) Raw solar % by event time')

    # Panel (c): Detection rates by capacity tier
    ax = axes[2]
    df = panel.dropna(subset=['vlm_solar_panels', 'event_time', 'capacity_mw']).copy()
    bins = [0, 10, 50, 200, np.inf]
    labels = ['<10 MW', '10-50 MW', '50-200 MW', '>200 MW']
    df['tier'] = pd.cut(df['capacity_mw'], bins=bins, labels=labels)
    df_post = df[df['event_time'] >= 0]
    df_pre = df[df['event_time'] < 0]

    tp_rates, fp_rates = [], []
    for tier in labels:
        post_tier = df_post[df_post['tier'] == tier]
        pre_tier = df_pre[df_pre['tier'] == tier]
        tp = (post_tier['vlm_solar_panels'] > 5).mean() * 100 if len(post_tier) > 0 else 0
        fp = (pre_tier['vlm_solar_panels'] > 5).mean() * 100 if len(pre_tier) > 0 else 0
        tp_rates.append(tp)
        fp_rates.append(fp)

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, tp_rates, w, label='TP (post)', color=_TOL_MUTED['teal'])
    ax.bar(x + w/2, fp_rates, w, label='FP (pre)', color=_TOL_MUTED['rose'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=15)
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('(c) Detection by capacity tier')
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, 'fig3_vlm_solar.pdf'))
    plt.close(fig)
    print('    -> fig3_vlm_solar.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: VLM vs DW cross-validation
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_vlm_dw_comparison():
    print('  Generating Fig 4: VLM vs DW comparison ...')
    panel = load_panel()

    pairs = [
        ('vlm_crops', 'dw_crops_pct', 'Cropland'),
        ('vlm_trees', 'dw_trees_pct', 'Trees'),
        ('vlm_built', 'dw_built_pct', 'Built-Up'),
        ('vlm_bare', 'dw_bare_pct', 'Bare Ground'),
        ('vlm_water', 'dw_water_pct', 'Water'),
        ('vlm_grass', 'dw_grass_pct', 'Grassland'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6))
    axes = axes.flatten()

    for i, (vlm_col, dw_col, title) in enumerate(pairs):
        ax = axes[i]
        # Check columns exist
        if vlm_col not in panel.columns:
            ax.set_title(title)
            ax.text(0.5, 0.5, f'{vlm_col} missing', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            continue
        if dw_col not in panel.columns:
            # Try alternate name
            alt = dw_col.replace('_pct', '')
            if alt in panel.columns:
                dw_col = alt
            else:
                ax.set_title(title)
                ax.text(0.5, 0.5, f'{dw_col} missing', ha='center', va='center',
                        transform=ax.transAxes, fontsize=7)
                continue

        df = panel[[vlm_col, dw_col]].dropna()
        if len(df) > 3000:
            df = df.sample(3000, random_state=42)

        ax.scatter(df[dw_col], df[vlm_col], s=2, alpha=0.15,
                   color=_TOL_MUTED['indigo'], rasterized=True)

        # 1:1 line
        lim = max(df[dw_col].max(), df[vlm_col].max(), 1)
        ax.plot([0, lim], [0, lim], '--', color='grey', linewidth=0.8)

        # Pearson r
        r, p = stats.pearsonr(df[dw_col], df[vlm_col])
        ax.annotate(f'r = {r:.2f}', xy=(0.05, 0.92), xycoords='axes fraction',
                    fontsize=7, bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                          ec='none', alpha=0.8))
        ax.set_xlabel(f'DW {title} (%)')
        ax.set_ylabel(f'VLM {title} (%)')
        ax.set_title(title)

    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, 'fig4_vlm_dw_comparison.pdf'))
    plt.close(fig)
    print('    -> fig4_vlm_dw_comparison.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Conflict heterogeneity
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_conflict():
    print('  Generating Fig 5: Conflict heterogeneity ...')
    conflicts = load_conflicts()
    panel = load_panel()

    conflict_site_ids = set()
    for c in conflicts:
        sid = c.get('matched_site_id')
        if sid:
            conflict_site_ids.add(sid)

    all_site_ids = set(panel['site_id'].unique())
    non_conflict_ids = all_site_ids - conflict_site_ids
    # Only keep sites that have construction_year (event_time will be non-null)
    panel_valid = panel.dropna(subset=['event_time'])

    outcomes = ['dw_crops_pct', 'dw_bare_pct', 'dw_trees_pct', 'vlm_solar_panels']
    labels = ['Cropland (%)', 'Bare Ground (%)', 'Tree Cover (%)', 'Solar Panels (%)']

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6))
    axes = axes.flatten()

    for i, (outcome, label) in enumerate(zip(outcomes, labels)):
        ax = axes[i]

        # Run event studies for each group
        es_conflict = run_event_study(panel_valid, outcome,
                                       site_ids=conflict_site_ids)
        es_non = run_event_study(panel_valid, outcome,
                                  site_ids=non_conflict_ids)

        if es_conflict is not None:
            plot_event_study_panel(ax, es_conflict, _TOL_MUTED['rose'],
                                  label=f'Conflict (n={es_conflict["n_sites"]})',
                                  alpha_band=0.12)
        if es_non is not None:
            plot_event_study_panel(ax, es_non, _TOL_MUTED['indigo'],
                                  label=f'Non-conflict (n={es_non["n_sites"]})',
                                  alpha_band=0.12)

        ax.set_title(label)
        ax.set_ylabel('Coefficient (pp)')
        ax.legend(fontsize=6.5, loc='best')

    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, 'fig5_conflict.pdf'))
    plt.close(fig)
    print('    -> fig5_conflict.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: EO event study
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_eo_event_study():
    print('  Generating Fig 6: EO event study ...')
    panel = load_panel()
    panel_valid = panel.dropna(subset=['event_time'])

    outcomes = ['viirs_avg_rad', 'pop_mean', 'bldg_presence',
                'lst_night_c', 'evi_mean', 'sar_vh_db']

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, FULL_WIDTH * 0.6))
    axes = axes.flatten()

    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        if outcome not in panel.columns:
            ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
            ax.text(0.5, 0.5, 'Column missing', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            continue

        es = run_event_study(panel_valid, outcome)
        if es is None:
            ax.set_title(OUTCOME_LABELS.get(outcome, outcome))
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            continue

        color = OUTCOME_COLORS.get(outcome, _TOL_MUTED['indigo'])
        plot_event_study_panel(ax, es, color)

        pre_p = es.get('pre_trends_p', np.nan)
        title = OUTCOME_LABELS.get(outcome, outcome)
        ax.set_title(title)
        if not np.isnan(pre_p):
            ax.annotate(f'Pre-trends p = {pre_p:.3f}',
                        xy=(0.02, 0.97), xycoords='axes fraction',
                        fontsize=6.5, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='none', alpha=0.8))
        ax.annotate(f'n = {es["n_sites"]:,} sites',
                    xy=(0.98, 0.97), xycoords='axes fraction',
                    fontsize=6.5, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='none', alpha=0.8))
        ax.set_ylabel('Coefficient')
        if i >= 3:
            ax.set_xlabel('Years Relative to Construction')
        else:
            ax.set_xlabel('')

    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, 'fig6_eo_event_study.pdf'))
    plt.close(fig)
    print('    -> fig6_eo_event_study.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    apply_style()
    print('Generating paper figures ...')
    print(f'Output directory: {OUT}')

    fig1_study_area()
    fig2_event_study_dw()
    fig3_vlm_solar()
    fig4_vlm_dw_comparison()
    fig5_conflict()
    fig6_eo_event_study()

    print('\nAll figures generated successfully.')


if __name__ == '__main__':
    main()
