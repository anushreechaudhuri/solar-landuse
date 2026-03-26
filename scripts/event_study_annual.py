"""
Within-site event study using annual DW panel (2016-2025).

Each site is its own control (before vs. after construction).
No control group needed — avoids parallel trends assumption.

Specification:
    Y_it = α_i + γ_t + Σ_{k≠-1} β_k · 1(event_time = k) + ε_it

    - α_i: site fixed effects (absorb location, climate, baseline LULC)
    - γ_t: year fixed effects (absorb DW calibration changes, climate trends)
    - β_k: event-time coefficients relative to k=-1 (year before construction)
    - SEs clustered at site level

Usage:
    python scripts/event_study_annual.py
    python scripts/event_study_annual.py --balanced  # restrict to sites with full [-3,+3]
    python scripts/event_study_annual.py --vlm       # include VLM outcomes
    python scripts/event_study_annual.py --conflict-split  # heterogeneity by conflict status
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from figure_style import apply_style, save_fig, LULC_COLORS

# ── Configuration ─────────────────────────────────────────────────────────────

PRIMARY_OUTCOMES = [
    ("dw_crops_pct", "Cropland (%)"),
    ("dw_bare_pct", "Bare ground (%)"),
    ("dw_trees_pct", "Tree cover (%)"),
    ("dw_built_pct", "Built-up (%)"),
    ("dw_water_pct", "Water (%)"),
    ("ndvi_mean", "NDVI"),
]

SECONDARY_OUTCOMES = [
    ("dw_grass_pct", "Grassland (%)"),
    ("dw_shrub_and_scrub_pct", "Shrub/scrub (%)"),
    ("dw_flooded_vegetation_pct", "Flooded veg (%)"),
    ("dw_snow_and_ice_pct", "Snow/ice (%)"),
]

VLM_OUTCOMES = [
    ("vlm_solar_panels", "VLM Solar panels (%)"),
    ("vlm_crops", "VLM Cropland (%)"),
    ("vlm_trees", "VLM Trees (%)"),
    ("vlm_built", "VLM Built-up (%)"),
    ("vlm_bare", "VLM Bare ground (%)"),
    ("vlm_water", "VLM Water (%)"),
]

EVENT_WINDOW = (-5, 6)  # bin endpoints beyond this
REF_PERIOD = -1


# ── Data preparation ──────────────────────────────────────────────────────────

def load_and_prepare(balanced_window=None, min_construction_year=2017,
                     panel_path=None):
    """Load annual panel, merge construction years, compute event time."""
    # Prefer full panel if available
    if panel_path is None:
        if os.path.exists("data/full_panel.csv"):
            panel_path = "data/full_panel.csv"
        elif os.path.exists("data/annual_panel_full.csv"):
            panel_path = "data/annual_panel_full.csv"
        else:
            panel_path = "data/annual_panel.csv"
    print(f"  Using panel: {panel_path}")
    df = pd.read_csv(panel_path)

    # Merge construction_year from unified_solar_db where missing
    with open("data/unified_solar_db.json") as f:
        udb = json.load(f)
    udb_cy = {}
    for e in udb:
        sid = e.get("site_id")
        cy = e.get("best_construction_year") or e.get("construction_year")
        if sid and cy:
            udb_cy[sid] = int(cy)

    df["construction_year"] = df.apply(
        lambda r: r["construction_year"] if pd.notna(r["construction_year"])
        else udb_cy.get(r["site_id"]), axis=1
    )

    # Drop sites without construction year or built before panel starts
    df = df[df["construction_year"].notna()].copy()
    df["construction_year"] = df["construction_year"].astype(int)
    df = df[df["construction_year"] >= min_construction_year]

    # Compute event time
    df["event_time"] = df["year"] - df["construction_year"]

    # Bin event time at endpoints
    lo, hi = EVENT_WINDOW
    df["event_time_binned"] = df["event_time"].clip(lo, hi).astype(int)

    # Balanced sample: only sites with full coverage in requested window
    if balanced_window:
        blo, bhi = balanced_window
        required_ets = set(range(blo, bhi + 1))
        site_ets = df.groupby("site_id")["event_time_binned"].apply(set)
        balanced_sites = site_ets[site_ets.apply(lambda s: required_ets.issubset(s))].index
        df = df[df["site_id"].isin(balanced_sites)]
        # Also restrict event times to the balanced window
        df = df[(df["event_time_binned"] >= blo) & (df["event_time_binned"] <= bhi)]

    print(f"Panel: {len(df)} obs, {df['site_id'].nunique()} sites, "
          f"years {df['year'].min()}-{df['year'].max()}")
    print(f"Event time range: [{df['event_time_binned'].min()}, {df['event_time_binned'].max()}]")

    return df


def merge_vlm_data(df):
    """Merge VLM annual panel data into the main DataFrame."""
    # Check if VLM columns already present (from full_panel.csv)
    if "vlm_solar_panels" in df.columns:
        n_vlm = df["vlm_solar_panels"].notna().sum()
        print(f"  VLM data already in panel: {n_vlm}/{len(df)} obs have VLM classifications")
        return df

    vlm_path = "data/vlm_annual_panel.csv"
    if not os.path.exists(vlm_path):
        print(f"  VLM panel not found at {vlm_path} — run analyze_vlm_results.py first")
        return df

    vlm = pd.read_csv(vlm_path)
    # Rename VLM columns to match our naming convention
    rename_map = {
        "solar_panels": "vlm_solar_panels",
        "crops": "vlm_crops",
        "trees": "vlm_trees",
        "built": "vlm_built",
        "bare": "vlm_bare",
        "water": "vlm_water",
        "grass": "vlm_grass",
        "shrub_and_scrub": "vlm_shrub_and_scrub",
        "flooded_vegetation": "vlm_flooded_vegetation",
        "snow_and_ice": "vlm_snow_and_ice",
    }
    vlm = vlm.rename(columns=rename_map)
    vlm_cols = ["site_id", "year"] + [v for v in rename_map.values() if v in vlm.columns]
    vlm = vlm[vlm_cols]

    df = df.merge(vlm, on=["site_id", "year"], how="left")
    n_vlm = df["vlm_solar_panels"].notna().sum()
    print(f"  Merged VLM data: {n_vlm}/{len(df)} obs have VLM classifications")
    return df


def load_conflict_sites():
    """Load set of site_ids with documented land conflicts."""
    conflict_path = "data/lcw_matched_conflicts.json"
    if not os.path.exists(conflict_path):
        print(f"  Conflict data not found at {conflict_path} — run match_lcw_conflicts.py first")
        return set()

    with open(conflict_path) as f:
        conflicts = json.load(f)

    conflict_ids = set()
    for c in conflicts:
        sid = c.get("matched_site_id")
        if sid and c.get("evidence_of_controversy", True):
            conflict_ids.add(sid)

    print(f"  Loaded {len(conflict_ids)} site_ids with documented conflicts")
    return conflict_ids


# ── Estimation ────────────────────────────────────────────────────────────────

def run_event_study(df, outcome_var, event_window=EVENT_WINDOW, ref_period=REF_PERIOD):
    """
    Two-way FE event study via manual demeaning.

    Equivalent to PanelOLS with entity_effects=True, time_effects=True.
    """
    lo, hi = event_window
    data = df[["site_id", "year", "event_time_binned", outcome_var]].dropna().copy()
    data = data[(data["event_time_binned"] >= lo) & (data["event_time_binned"] <= hi)]

    if len(data) < 100:
        return None

    # Create event-time dummies (excluding reference period)
    event_times = sorted([k for k in range(lo, hi + 1) if k != ref_period])
    for k in event_times:
        data[f"D_{k}"] = (data["event_time_binned"] == k).astype(float)
    dummy_cols = [f"D_{k}" for k in event_times]

    all_cols = [outcome_var] + dummy_cols

    # Two-way demeaning: x_tilde = x - x_bar_i - x_bar_t + x_bar
    site_means = data.groupby("site_id")[all_cols].transform("mean")
    year_means = data.groupby("year")[all_cols].transform("mean")
    grand_mean = data[all_cols].mean()

    demeaned = pd.DataFrame(index=data.index)
    for col in all_cols:
        demeaned[col] = data[col] - site_means[col] - year_means[col] + grand_mean[col]

    y = demeaned[outcome_var]
    X = demeaned[dummy_cols]

    # OLS with site-clustered SEs
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": data["site_id"]})

    # Extract coefficients
    coefficients = {}
    for i, k in enumerate(event_times):
        coefficients[k] = {
            "coef": float(model.params.iloc[i]),
            "se": float(model.bse.iloc[i]),
            "pval": float(model.pvalues.iloc[i]),
            "ci_low": float(model.conf_int().iloc[i, 0]),
            "ci_high": float(model.conf_int().iloc[i, 1]),
        }
    # Reference period
    coefficients[ref_period] = {
        "coef": 0.0, "se": 0.0, "pval": 1.0, "ci_low": 0.0, "ci_high": 0.0
    }

    # Pre-trends F-test: joint test that all pre-treatment coefficients = 0
    pre_indices = [i for i, k in enumerate(event_times) if k < ref_period]
    if len(pre_indices) >= 2:
        r_matrix = np.zeros((len(pre_indices), len(event_times)))
        for j, idx in enumerate(pre_indices):
            r_matrix[j, idx] = 1
        f_test = model.f_test(r_matrix)
        fval = f_test.fvalue
        pre_trends_f = float(fval[0][0]) if hasattr(fval, '__getitem__') and not isinstance(fval, float) else float(fval)
        pval = f_test.pvalue
        pre_trends_p = float(pval) if isinstance(pval, float) else float(pval[0]) if hasattr(pval, '__getitem__') else float(pval)
    else:
        pre_trends_f = None
        pre_trends_p = None

    # Obs per event time
    et_counts = data["event_time_binned"].value_counts().sort_index().to_dict()

    return {
        "outcome": outcome_var,
        "coefficients": coefficients,
        "n_obs": len(data),
        "n_sites": data["site_id"].nunique(),
        "r_squared": float(model.rsquared),
        "pre_trends_f": pre_trends_f,
        "pre_trends_p": pre_trends_p,
        "obs_per_event_time": {int(k): int(v) for k, v in et_counts.items()},
    }


# ── Visualization ─────────────────────────────────────────────────────────────

OUTCOME_COLORS = {
    "dw_crops_pct": "#CC6677",
    "dw_bare_pct": "#DDCC77",
    "dw_trees_pct": "#117733",
    "dw_built_pct": "#882255",
    "dw_water_pct": "#332288",
    "ndvi_mean": "#44AA99",
}


def plot_event_study(results, title_suffix="", output_path=None):
    """Plot 2x3 panel of event study coefficients."""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), sharex=True)
    axes = axes.flatten()

    for idx, (res, (var, label)) in enumerate(zip(results, PRIMARY_OUTCOMES)):
        ax = axes[idx]
        if res is None:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(label, fontsize=9)
            continue

        coefs = res["coefficients"]
        event_times = sorted(coefs.keys())
        betas = [coefs[k]["coef"] for k in event_times]
        ci_low = [coefs[k]["ci_low"] for k in event_times]
        ci_high = [coefs[k]["ci_high"] for k in event_times]

        color = OUTCOME_COLORS.get(var, "#333333")

        # CI band
        ax.fill_between(event_times, ci_low, ci_high, alpha=0.2, color=color)
        # Point estimates
        ax.plot(event_times, betas, "o-", color=color, markersize=4, linewidth=1.5)
        # Zero line
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
        # Treatment onset
        ax.axvline(-0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_title(label, fontsize=9)
        if idx >= 3:
            ax.set_xlabel("Years from construction", fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("Coefficient (pp)" if var != "ndvi_mean" else "Coefficient",
                          fontsize=8)

        # Pre-trends annotation
        if res["pre_trends_p"] is not None:
            p = res["pre_trends_p"]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(0.02, 0.98, f"Pre-trends F: p={p:.3f}{stars}",
                    transform=ax.transAxes, fontsize=6, va="top", ha="left",
                    color="red" if p < 0.05 else "green")

    fig.suptitle(f"Within-Site Event Study: Land Cover Change Around Solar Construction{title_suffix}",
                 fontsize=10, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        save_fig(fig, output_path)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_vlm_event_study(results, output_path=None):
    """Plot VLM event study: solar detection + VLM LULC outcomes."""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), sharex=True)
    axes = axes.flatten()

    vlm_colors = {
        "vlm_solar_panels": "#AA3377",  # DATASET_COLORS['vlm']
        "vlm_crops": "#DDCC77",
        "vlm_trees": "#117733",
        "vlm_built": "#CC6677",
        "vlm_bare": "#882255",
        "vlm_water": "#88CCEE",
    }

    for idx, (res, (var, label)) in enumerate(zip(results, VLM_OUTCOMES)):
        ax = axes[idx]
        if res is None:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(label, fontsize=9)
            continue

        coefs = res["coefficients"]
        event_times = sorted(coefs.keys())
        betas = [coefs[k]["coef"] for k in event_times]
        ci_low = [coefs[k]["ci_low"] for k in event_times]
        ci_high = [coefs[k]["ci_high"] for k in event_times]

        color = vlm_colors.get(var, "#333333")
        ax.fill_between(event_times, ci_low, ci_high, alpha=0.2, color=color)
        ax.plot(event_times, betas, "o-", color=color, markersize=4, linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
        ax.axvline(-0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(label, fontsize=9)
        if idx >= 3:
            ax.set_xlabel("Years from construction", fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel("Coefficient (pp)", fontsize=8)

    fig.suptitle("VLM Event Study: Land Cover Change Around Solar Construction",
                 fontsize=10, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        save_fig(fig, output_path)
        print(f"Saved: {output_path}")
    plt.close()


def plot_conflict_comparison(conflict_results, no_conflict_results, outcomes, output_path=None):
    """Plot side-by-side event studies: conflict vs non-conflict sites."""
    apply_style()
    n_outcomes = len(outcomes)
    fig, axes = plt.subplots(n_outcomes, 2, figsize=(10, 2.5 * n_outcomes), sharex=True)
    if n_outcomes == 1:
        axes = axes.reshape(1, 2)

    for idx, ((var, label), res_c, res_nc) in enumerate(
        zip(outcomes, conflict_results, no_conflict_results)
    ):
        for col, (res, group_label) in enumerate([
            (res_c, "Conflict sites"),
            (res_nc, "Non-conflict sites"),
        ]):
            ax = axes[idx, col]
            if res is None:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                        ha="center", va="center")
                ax.set_title(f"{label} — {group_label}", fontsize=8)
                continue

            coefs = res["coefficients"]
            event_times = sorted(coefs.keys())
            betas = [coefs[k]["coef"] for k in event_times]
            ci_low = [coefs[k]["ci_low"] for k in event_times]
            ci_high = [coefs[k]["ci_high"] for k in event_times]

            color = "#CC6677" if col == 0 else "#4477AA"
            ax.fill_between(event_times, ci_low, ci_high, alpha=0.2, color=color)
            ax.plot(event_times, betas, "o-", color=color, markersize=3, linewidth=1.2)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(-0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

            n_label = f"(n={res['n_sites']})" if res else ""
            ax.set_title(f"{label} — {group_label} {n_label}", fontsize=8)
            if idx == n_outcomes - 1:
                ax.set_xlabel("Years from construction", fontsize=7)
            if col == 0:
                ax.set_ylabel("Coefficient (pp)", fontsize=7)

    fig.suptitle("Heterogeneous Event Study: Conflict vs Non-conflict Sites",
                 fontsize=10, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output_path:
        save_fig(fig, output_path)
        print(f"Saved: {output_path}")
    plt.close()


def plot_pre_post_summary(results, output_path=None):
    """Bar chart of average pre vs post coefficients."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4))

    labels = []
    pre_means = []
    post_means = []
    for res, (var, label) in zip(results, PRIMARY_OUTCOMES):
        if res is None:
            continue
        coefs = res["coefficients"]
        pre_vals = [coefs[k]["coef"] for k in coefs if k < REF_PERIOD]
        post_vals = [coefs[k]["coef"] for k in coefs if k > 0]
        labels.append(label)
        pre_means.append(np.mean(pre_vals) if pre_vals else 0)
        post_means.append(np.mean(post_vals) if post_vals else 0)

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, pre_means, width, label="Pre-construction (avg)", alpha=0.7)
    ax.bar(x + width / 2, post_means, width, label="Post-construction (avg)", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Average coefficient")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.set_title("Average Pre vs Post Treatment Effects", fontsize=10, fontweight="bold")
    plt.tight_layout()

    if output_path:
        save_fig(fig, output_path)
        print(f"Saved: {output_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def print_summary_table(results, all_outcomes, label=""):
    """Print formatted summary table of event study results."""
    print("\n" + "=" * 80)
    print(f"EVENT STUDY RESULTS {label}")
    print("=" * 80)
    print(f"{'Outcome':<25} {'N obs':>8} {'N sites':>8} {'Pre-trends p':>13} {'Post avg':>10} {'Max |β|':>10}")
    print("-" * 80)

    for res, (var, olabel) in zip(results, all_outcomes):
        if res is None:
            continue
        coefs = res["coefficients"]
        post_vals = [coefs[k]["coef"] for k in coefs if k > 0]
        all_betas = [abs(coefs[k]["coef"]) for k in coefs if k != REF_PERIOD]
        pre_p = res["pre_trends_p"]
        pre_str = f"{pre_p:.3f}" if pre_p is not None else "N/A"
        print(f"{olabel:<25} {res['n_obs']:>8} {res['n_sites']:>8} {pre_str:>13} "
              f"{np.mean(post_vals):>+10.3f} {max(all_betas) if all_betas else 0:>10.3f}")


def main():
    parser = argparse.ArgumentParser(description="Within-site event study")
    parser.add_argument("--balanced", action="store_true",
                        help="Restrict to sites with full [-3,+3] coverage")
    parser.add_argument("--min-year", type=int, default=2017,
                        help="Minimum construction year to include")
    parser.add_argument("--vlm", action="store_true",
                        help="Include VLM-derived LULC outcomes")
    parser.add_argument("--conflict-split", action="store_true",
                        help="Run heterogeneous analysis: conflict vs non-conflict sites")
    args = parser.parse_args()

    os.makedirs("data/event_study_results", exist_ok=True)
    os.makedirs("docs/figures/event_study", exist_ok=True)

    balanced_window = (-3, 3) if args.balanced else None
    suffix = "_balanced" if args.balanced else ""
    title_suffix = "\n(Balanced sample, k in [-3, +3])" if args.balanced else ""

    # Load data
    print("Loading and preparing data...")
    df = load_and_prepare(balanced_window=balanced_window,
                          min_construction_year=args.min_year)

    # Merge VLM data if requested
    if args.vlm:
        print("\nMerging VLM data...")
        df = merge_vlm_data(df)

    # ── Standard event studies ───────────────────────────────────────────────
    print("\nRunning event studies (DW outcomes)...")
    results = []
    all_outcomes = PRIMARY_OUTCOMES + SECONDARY_OUTCOMES
    for var, label in all_outcomes:
        print(f"  {label}...", end=" ", flush=True)
        res = run_event_study(df, var)
        if res:
            print(f"n={res['n_obs']}, R²={res['r_squared']:.4f}, "
                  f"pre-trends p={res['pre_trends_p']:.3f}" if res["pre_trends_p"] else "")
        else:
            print("SKIPPED (insufficient data)")
        results.append(res)

    primary_results = results[:len(PRIMARY_OUTCOMES)]
    secondary_results = results[len(PRIMARY_OUTCOMES):]

    # ── VLM event studies ────────────────────────────────────────────────────
    vlm_results = []
    if args.vlm:
        print("\nRunning event studies (VLM outcomes)...")
        for var, label in VLM_OUTCOMES:
            print(f"  {label}...", end=" ", flush=True)
            if var in df.columns:
                res = run_event_study(df, var)
                if res:
                    print(f"n={res['n_obs']}, R²={res['r_squared']:.4f}")
                else:
                    print("SKIPPED (insufficient data)")
            else:
                res = None
                print("SKIPPED (column not found)")
            vlm_results.append(res)

    # ── Save results JSON ────────────────────────────────────────────────────
    output = {
        "specification": {
            "model": "Y_it = alpha_i + gamma_t + sum_k beta_k * D_it^k + eps_it",
            "fixed_effects": ["site (alpha_i)", "year (gamma_t)"],
            "clustering": "site-level",
            "reference_period": REF_PERIOD,
            "event_window": list(EVENT_WINDOW),
            "balanced": args.balanced,
            "balanced_window": list(balanced_window) if balanced_window else None,
            "min_construction_year": args.min_year,
        },
        "sample": {
            "n_sites": df["site_id"].nunique(),
            "n_obs": len(df),
            "construction_year_range": [int(df["construction_year"].min()),
                                        int(df["construction_year"].max())],
            "countries": df["country"].value_counts().to_dict(),
        },
        "primary_outcomes": {res["outcome"]: res for res in primary_results if res},
        "secondary_outcomes": {res["outcome"]: res for res in secondary_results if res},
    }
    if vlm_results:
        output["vlm_outcomes"] = {res["outcome"]: res for res in vlm_results if res}

    json_path = f"data/event_study_results/event_study{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results: {json_path}")

    # Print summary tables
    print_summary_table(results, all_outcomes,
                        f"{'(BALANCED)' if args.balanced else '(FULL SAMPLE)'}")
    if vlm_results:
        print_summary_table(vlm_results, VLM_OUTCOMES, "(VLM OUTCOMES)")

    # Generate figures
    print("\nGenerating figures...")
    plot_event_study(primary_results, title_suffix=title_suffix,
                     output_path=f"docs/figures/event_study/event_study_primary{suffix}.png")
    plot_pre_post_summary(primary_results,
                          output_path=f"docs/figures/event_study/pre_post_summary{suffix}.png")
    if vlm_results:
        plot_vlm_event_study(vlm_results,
                             output_path=f"docs/figures/event_study/event_study_vlm{suffix}.png")

    # ── Conflict heterogeneity ───────────────────────────────────────────────
    if args.conflict_split:
        print("\n" + "=" * 80)
        print("CONFLICT HETEROGENEITY ANALYSIS")
        print("=" * 80)

        conflict_ids = load_conflict_sites()
        if not conflict_ids:
            print("  No conflict data available — skipping heterogeneity analysis")
        else:
            df["has_conflict"] = df["site_id"].isin(conflict_ids)
            df_conflict = df[df["has_conflict"]].copy()
            df_no_conflict = df[~df["has_conflict"]].copy()

            print(f"  Conflict sites: {df_conflict['site_id'].nunique()} "
                  f"({len(df_conflict)} obs)")
            print(f"  Non-conflict sites: {df_no_conflict['site_id'].nunique()} "
                  f"({len(df_no_conflict)} obs)")

            # Run for key outcomes
            conflict_outcomes = PRIMARY_OUTCOMES[:4]  # crops, bare, trees, built
            conflict_results_c = []
            conflict_results_nc = []

            for var, label in conflict_outcomes:
                print(f"\n  {label}:")
                res_c = run_event_study(df_conflict, var)
                res_nc = run_event_study(df_no_conflict, var)
                if res_c:
                    print(f"    Conflict: n={res_c['n_obs']}, sites={res_c['n_sites']}")
                else:
                    print("    Conflict: insufficient data")
                if res_nc:
                    print(f"    Non-conflict: n={res_nc['n_obs']}, sites={res_nc['n_sites']}")
                else:
                    print("    Non-conflict: insufficient data")
                conflict_results_c.append(res_c)
                conflict_results_nc.append(res_nc)

            # Save conflict results
            conflict_output = {
                "conflict_sites": list(conflict_ids),
                "n_conflict": df_conflict["site_id"].nunique(),
                "n_no_conflict": df_no_conflict["site_id"].nunique(),
                "conflict_results": {
                    var: res for (var, _), res in zip(conflict_outcomes, conflict_results_c) if res
                },
                "no_conflict_results": {
                    var: res for (var, _), res in zip(conflict_outcomes, conflict_results_nc) if res
                },
            }
            conflict_json = f"data/event_study_results/conflict_heterogeneity{suffix}.json"
            with open(conflict_json, "w") as f:
                json.dump(conflict_output, f, indent=2)
            print(f"\nSaved: {conflict_json}")

            # Plot
            plot_conflict_comparison(
                conflict_results_c, conflict_results_nc, conflict_outcomes,
                output_path=f"docs/figures/event_study/conflict_comparison{suffix}.png"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
