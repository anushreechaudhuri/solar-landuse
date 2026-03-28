"""
Comprehensive paper analyses for solar land-use project.

Runs all event studies and summary statistics needed for the paper:
1. EO Event Study (VIIRS, population, buildings, LST, EVI, SAR)
2. Conflict Heterogeneity (conflict vs non-conflict sites)
3. Urbanization/Agricultural Preservation Hypothesis (population terciles)
4. Capacity Stratification (<10, 10-50, 50-200, >200 MW)
5. Summary Statistics Table

All results cached as JSON/CSV in data/event_study_results/ and data/.

Usage:
    python scripts/paper_analyses.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from figure_style import apply_style, save_fig

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

EVENT_WINDOW = (-5, 6)
REF_PERIOD = -1
MIN_CONSTRUCTION_YEAR = 2017

# Outcome definitions
EO_OUTCOMES = [
    ("viirs_avg_rad", "VIIRS Nighttime Lights"),
    ("pop_mean", "Population Density"),
    ("bldg_presence", "Building Presence"),
    ("bldg_frac_count", "Building Density"),
    ("lst_day_c", "LST Day (C)"),
    ("lst_night_c", "LST Night (C)"),
    ("evi_mean", "EVI"),
    ("sar_vv_db", "SAR VV (dB)"),
]

DW_OUTCOMES = [
    ("dw_crops_pct", "Cropland (%)"),
    ("dw_bare_pct", "Bare ground (%)"),
    ("dw_trees_pct", "Tree cover (%)"),
    ("dw_built_pct", "Built-up (%)"),
    ("dw_water_pct", "Water (%)"),
    ("ndvi_mean", "NDVI"),
]

VLM_OUTCOMES = [
    ("vlm_solar_panels", "VLM Solar panels (%)"),
    ("vlm_crops", "VLM Cropland (%)"),
    ("vlm_trees", "VLM Trees (%)"),
    ("vlm_built", "VLM Built-up (%)"),
    ("vlm_bare", "VLM Bare ground (%)"),
    ("vlm_water", "VLM Water (%)"),
]

CONFLICT_OUTCOMES = [
    ("dw_crops_pct", "Cropland (%)"),
    ("dw_bare_pct", "Bare ground (%)"),
    ("dw_trees_pct", "Tree cover (%)"),
    ("dw_built_pct", "Built-up (%)"),
    ("vlm_solar_panels", "VLM Solar (%)"),
    ("ndvi_mean", "NDVI"),
]

URBANIZATION_OUTCOMES = [
    ("dw_crops_pct", "Cropland (%)"),
    ("dw_built_pct", "Built-up (%)"),
    ("dw_trees_pct", "Tree cover (%)"),
    ("ndvi_mean", "NDVI"),
    ("viirs_avg_rad", "VIIRS Nighttime Lights"),
]

CAPACITY_OUTCOMES = [
    ("dw_crops_pct", "Cropland (%)"),
    ("dw_bare_pct", "Bare ground (%)"),
    ("dw_trees_pct", "Tree cover (%)"),
    ("dw_built_pct", "Built-up (%)"),
    ("ndvi_mean", "NDVI"),
    ("vlm_solar_panels", "VLM Solar (%)"),
    ("viirs_avg_rad", "VIIRS Nighttime Lights"),
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_panel():
    """Load full panel and prepare event time."""
    panel_path = "data/full_panel.csv"
    print(f"Loading panel: {panel_path}")
    df = pd.read_csv(panel_path)

    # Merge construction_year from unified_solar_db where missing
    with open("data/unified_solar_db.json") as f:
        udb = json.load(f)
    udb_cy = {}
    udb_cap = {}
    for e in udb:
        sid = e.get("site_id")
        cy = e.get("best_construction_year") or e.get("construction_year")
        cap = e.get("best_capacity_mw")
        if sid and cy:
            udb_cy[sid] = int(cy)
        if sid and cap:
            udb_cap[sid] = float(cap)

    df["construction_year"] = df.apply(
        lambda r: r["construction_year"] if pd.notna(r.get("construction_year"))
        else udb_cy.get(r["site_id"]), axis=1
    )

    # Add capacity from unified_solar_db
    if "capacity_mw" not in df.columns or df["capacity_mw"].isna().all():
        df["capacity_mw"] = df["site_id"].map(udb_cap)
    else:
        df["capacity_mw"] = df["capacity_mw"].fillna(df["site_id"].map(udb_cap))

    # Filter
    df = df[df["construction_year"].notna()].copy()
    df["construction_year"] = df["construction_year"].astype(int)
    df = df[df["construction_year"] >= MIN_CONSTRUCTION_YEAR]

    # Event time
    df["event_time"] = df["year"] - df["construction_year"]
    lo, hi = EVENT_WINDOW
    df["event_time_binned"] = df["event_time"].clip(lo, hi).astype(int)

    print(f"  Panel: {len(df)} obs, {df['site_id'].nunique()} sites, "
          f"years {df['year'].min()}-{df['year'].max()}")
    return df, udb


def load_conflict_sites():
    """Load conflict site IDs where evidence_of_controversy is True."""
    path = "data/lcw_matched_conflicts.json"
    with open(path) as f:
        conflicts = json.load(f)
    ids = set()
    for c in conflicts:
        sid = c.get("matched_site_id")
        if sid and c.get("evidence_of_controversy", False):
            ids.add(sid)
    print(f"  Conflict sites with evidence_of_controversy: {len(ids)}")
    return ids


# ── Event study estimation ───────────────────────────────────────────────────

def run_event_study(df, outcome_var, event_window=EVENT_WINDOW, ref_period=REF_PERIOD):
    """Two-way FE event study via manual demeaning, site-clustered SEs."""
    lo, hi = event_window
    data = df[["site_id", "year", "event_time_binned", outcome_var]].dropna().copy()
    data = data[(data["event_time_binned"] >= lo) & (data["event_time_binned"] <= hi)]

    if len(data) < 100 or data["site_id"].nunique() < 5:
        return None

    event_times = sorted([k for k in range(lo, hi + 1) if k != ref_period])
    for k in event_times:
        data[f"D_{k}"] = (data["event_time_binned"] == k).astype(float)
    dummy_cols = [f"D_{k}" for k in event_times]

    all_cols = [outcome_var] + dummy_cols

    # Two-way demeaning
    site_means = data.groupby("site_id")[all_cols].transform("mean")
    year_means = data.groupby("year")[all_cols].transform("mean")
    grand_mean = data[all_cols].mean()

    demeaned = pd.DataFrame(index=data.index)
    for col in all_cols:
        demeaned[col] = data[col] - site_means[col] - year_means[col] + grand_mean[col]

    y = demeaned[outcome_var]
    X = demeaned[dummy_cols]

    try:
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": data["site_id"]})
    except Exception as e:
        print(f"    OLS failed for {outcome_var}: {e}")
        return None

    coefficients = {}
    for i, k in enumerate(event_times):
        coefficients[k] = {
            "coef": float(model.params.iloc[i]),
            "se": float(model.bse.iloc[i]),
            "pval": float(model.pvalues.iloc[i]),
            "ci_low": float(model.conf_int().iloc[i, 0]),
            "ci_high": float(model.conf_int().iloc[i, 1]),
        }
    coefficients[ref_period] = {
        "coef": 0.0, "se": 0.0, "pval": 1.0, "ci_low": 0.0, "ci_high": 0.0
    }

    # Pre-trends F-test
    pre_indices = [i for i, k in enumerate(event_times) if k < ref_period]
    pre_trends_f = None
    pre_trends_p = None
    if len(pre_indices) >= 2:
        r_matrix = np.zeros((len(pre_indices), len(event_times)))
        for j, idx in enumerate(pre_indices):
            r_matrix[j, idx] = 1
        try:
            f_test = model.f_test(r_matrix)
            fval = f_test.fvalue
            pre_trends_f = float(fval[0][0]) if hasattr(fval, '__getitem__') and not isinstance(fval, float) else float(fval)
            pval = f_test.pvalue
            pre_trends_p = float(pval) if isinstance(pval, float) else float(pval[0]) if hasattr(pval, '__getitem__') else float(pval)
        except Exception:
            pass

    et_counts = data["event_time_binned"].value_counts().sort_index().to_dict()

    # Compute post-treatment average
    post_coefs = [coefficients[k]["coef"] for k in coefficients if k > 0]
    post_avg = float(np.mean(post_coefs)) if post_coefs else 0.0

    return {
        "outcome": outcome_var,
        "coefficients": {int(k): v for k, v in coefficients.items()},
        "n_obs": int(len(data)),
        "n_sites": int(data["site_id"].nunique()),
        "r_squared": float(model.rsquared),
        "pre_trends_f": pre_trends_f,
        "pre_trends_p": pre_trends_p,
        "post_avg": post_avg,
        "obs_per_event_time": {int(k): int(v) for k, v in et_counts.items()},
    }


def run_batch(df, outcomes, label=""):
    """Run event studies for a list of outcomes, print summary."""
    results = {}
    for var, olabel in outcomes:
        if var not in df.columns:
            print(f"    {olabel}: SKIPPED (column not in panel)")
            continue
        n_valid = df[var].notna().sum()
        if n_valid < 100:
            print(f"    {olabel}: SKIPPED (only {n_valid} non-null obs)")
            continue
        res = run_event_study(df, var)
        if res:
            results[var] = res
            stars = ""
            if res["pre_trends_p"] is not None:
                p = res["pre_trends_p"]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            pre_str = f"p={res['pre_trends_p']:.3f}{stars}" if res["pre_trends_p"] is not None else "N/A"
            print(f"    {olabel}: n={res['n_obs']}, sites={res['n_sites']}, "
                  f"R2={res['r_squared']:.4f}, pre-trends {pre_str}, post_avg={res['post_avg']:+.4f}")
        else:
            print(f"    {olabel}: FAILED (insufficient data)")
    return results


def save_json(data, path):
    """Save dict as formatted JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ── Print helpers ────────────────────────────────────────────────────────────

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_event_study_table(results):
    """Print a compact summary table of event study results."""
    print(f"  {'Outcome':<25} {'N obs':>7} {'Sites':>6} {'Pre-trends p':>13} "
          f"{'Post avg':>10} {'R2':>7}")
    print("  " + "-" * 72)
    for var, res in results.items():
        pre_p = res.get("pre_trends_p")
        pre_str = f"{pre_p:.3f}" if pre_p is not None else "N/A"
        print(f"  {var:<25} {res['n_obs']:>7} {res['n_sites']:>6} {pre_str:>13} "
              f"{res['post_avg']:>+10.4f} {res['r_squared']:>7.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: EO Event Study
# ══════════════════════════════════════════════════════════════════════════════

def analysis_eo_event_study(df):
    print_header("ANALYSIS 1: EO EVENT STUDY")
    print("  Running event studies for EO variables...")
    results = run_batch(df, EO_OUTCOMES, label="EO")

    output = {
        "specification": {
            "model": "Y_it = alpha_i + gamma_t + sum_k beta_k * D_it^k + eps_it",
            "fixed_effects": ["site", "year"],
            "clustering": "site-level",
            "reference_period": REF_PERIOD,
            "event_window": list(EVENT_WINDOW),
        },
        "n_sites": int(df["site_id"].nunique()),
        "n_obs": int(len(df)),
        "results": results,
    }

    path = "data/event_study_results/event_study_eo.json"
    save_json(output, path)

    print("\n  SUMMARY:")
    print_event_study_table(results)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Conflict Heterogeneity
# ══════════════════════════════════════════════════════════════════════════════

def analysis_conflict_heterogeneity(df):
    print_header("ANALYSIS 2: CONFLICT HETEROGENEITY")

    conflict_ids = load_conflict_sites()
    df["has_conflict"] = df["site_id"].isin(conflict_ids)

    df_c = df[df["has_conflict"]].copy()
    df_nc = df[~df["has_conflict"]].copy()

    print(f"  Conflict sites: {df_c['site_id'].nunique()} ({len(df_c)} obs)")
    print(f"  Non-conflict sites: {df_nc['site_id'].nunique()} ({len(df_nc)} obs)")

    print("\n  --- Conflict sites ---")
    results_c = run_batch(df_c, CONFLICT_OUTCOMES, label="conflict")

    print("\n  --- Non-conflict sites ---")
    results_nc = run_batch(df_nc, CONFLICT_OUTCOMES, label="non-conflict")

    output = {
        "n_conflict_sites": int(df_c["site_id"].nunique()),
        "n_non_conflict_sites": int(df_nc["site_id"].nunique()),
        "conflict_site_ids": sorted(list(conflict_ids)),
        "conflict_results": results_c,
        "non_conflict_results": results_nc,
    }

    path = "data/event_study_results/conflict_heterogeneity.json"
    save_json(output, path)

    # Compare post-treatment averages
    print("\n  COMPARISON (post-treatment average coefficients):")
    print(f"  {'Outcome':<25} {'Conflict':>12} {'Non-conflict':>14} {'Difference':>12}")
    print("  " + "-" * 65)
    for var, label in CONFLICT_OUTCOMES:
        c_avg = results_c.get(var, {}).get("post_avg", float("nan"))
        nc_avg = results_nc.get(var, {}).get("post_avg", float("nan"))
        diff = c_avg - nc_avg if not (np.isnan(c_avg) or np.isnan(nc_avg)) else float("nan")
        print(f"  {label:<25} {c_avg:>+12.4f} {nc_avg:>+14.4f} {diff:>+12.4f}")

    return results_c, results_nc


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Urbanization / Agricultural Preservation Hypothesis
# ══════════════════════════════════════════════════════════════════════════════

def analysis_urbanization(df):
    print_header("ANALYSIS 3: URBANIZATION / AGRICULTURAL PRESERVATION")

    # Compute baseline (pre-construction) population density per site
    pre = df[df["event_time"] < 0].copy()
    baseline_pop = pre.groupby("site_id")["pop_mean"].mean().dropna()

    if len(baseline_pop) < 30:
        print("  Insufficient baseline population data. Skipping.")
        return None

    # Terciles
    tercile_labels = ["low_pop", "mid_pop", "high_pop"]
    tercile_bounds = baseline_pop.quantile([0.0, 1/3, 2/3, 1.0]).values
    baseline_pop_tercile = pd.cut(
        baseline_pop, bins=tercile_bounds, labels=tercile_labels, include_lowest=True
    )

    # Map to dataframe
    df["pop_tercile"] = df["site_id"].map(baseline_pop_tercile)

    print(f"  Baseline population tercile cutoffs: "
          f"{tercile_bounds[1]:.1f}, {tercile_bounds[2]:.1f}")
    for t in tercile_labels:
        sub = df[df["pop_tercile"] == t]
        print(f"    {t}: {sub['site_id'].nunique()} sites, {len(sub)} obs")

    all_results = {}
    for tercile in tercile_labels:
        print(f"\n  --- {tercile.upper()} ---")
        df_t = df[df["pop_tercile"] == tercile].copy()
        results = run_batch(df_t, URBANIZATION_OUTCOMES, label=tercile)
        all_results[tercile] = results

    # Also compute pre-construction baseline means for context
    baseline_stats = {}
    for tercile in tercile_labels:
        t_sites = baseline_pop_tercile[baseline_pop_tercile == tercile].index
        t_pre = pre[pre["site_id"].isin(t_sites)]
        baseline_stats[tercile] = {
            "n_sites": int(len(t_sites)),
            "pop_mean_avg": float(t_pre["pop_mean"].mean()) if "pop_mean" in t_pre else None,
            "dw_crops_pct_avg": float(t_pre["dw_crops_pct"].mean()) if "dw_crops_pct" in t_pre else None,
            "dw_built_pct_avg": float(t_pre["dw_built_pct"].mean()) if "dw_built_pct" in t_pre else None,
            "viirs_avg_avg": float(t_pre["viirs_avg_rad"].mean()) if "viirs_avg_rad" in t_pre else None,
        }

    output = {
        "tercile_cutoffs": {
            "low_mid": float(tercile_bounds[1]),
            "mid_high": float(tercile_bounds[2]),
        },
        "baseline_stats": baseline_stats,
        "tercile_results": all_results,
    }

    path = "data/event_study_results/urbanization_stratification.json"
    save_json(output, path)

    # Summary comparison
    print("\n  CROPLAND POST-TREATMENT AVERAGES BY POPULATION TERCILE:")
    print(f"  {'Tercile':<12} {'Baseline crop%':>15} {'Post avg coef':>15} {'Baseline pop':>15}")
    print("  " + "-" * 60)
    for t in tercile_labels:
        crop_post = all_results[t].get("dw_crops_pct", {}).get("post_avg", float("nan"))
        bs = baseline_stats[t]
        print(f"  {t:<12} {bs['dw_crops_pct_avg']:>15.2f} {crop_post:>+15.4f} {bs['pop_mean_avg']:>15.1f}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Capacity Stratification
# ══════════════════════════════════════════════════════════════════════════════

def analysis_capacity_stratification(df):
    print_header("ANALYSIS 4: CAPACITY STRATIFICATION")

    # Capacity tiers
    tiers = {
        "<10 MW": (0, 10),
        "10-50 MW": (10, 50),
        "50-200 MW": (50, 200),
        ">200 MW": (200, float("inf")),
    }

    # Get per-site capacity
    site_cap = df.groupby("site_id")["capacity_mw"].first()
    n_with_cap = site_cap.notna().sum()
    print(f"  Sites with capacity data: {n_with_cap}/{len(site_cap)}")

    if n_with_cap < 30:
        print("  Insufficient capacity data. Skipping.")
        return None

    all_results = {}
    tier_stats = {}

    for tier_name, (lo, hi) in tiers.items():
        tier_sites = site_cap[(site_cap >= lo) & (site_cap < hi)].index
        df_t = df[df["site_id"].isin(tier_sites)].copy()
        n_sites = df_t["site_id"].nunique()
        tier_stats[tier_name] = {"n_sites": n_sites, "n_obs": len(df_t)}
        print(f"\n  --- {tier_name} ({n_sites} sites, {len(df_t)} obs) ---")

        if n_sites < 5:
            print(f"    SKIPPED (too few sites)")
            all_results[tier_name] = {}
            continue

        results = run_batch(df_t, CAPACITY_OUTCOMES, label=tier_name)
        all_results[tier_name] = results

    output = {
        "tiers": {k: {"range_mw": list(v), "n_sites": tier_stats[k]["n_sites"],
                       "n_obs": tier_stats[k]["n_obs"]}
                  for k, v in tiers.items()},
        "tier_results": all_results,
    }

    path = "data/event_study_results/capacity_stratification.json"
    save_json(output, path)

    # Summary
    print("\n  POST-TREATMENT AVERAGES BY CAPACITY TIER:")
    print(f"  {'Tier':<12} {'Sites':>6} {'Crop post':>11} {'Tree post':>11} {'Built post':>11} {'NDVI post':>11}")
    print("  " + "-" * 65)
    for tier_name in tiers:
        r = all_results.get(tier_name, {})
        ns = tier_stats[tier_name]["n_sites"]
        crop = r.get("dw_crops_pct", {}).get("post_avg", float("nan"))
        tree = r.get("dw_trees_pct", {}).get("post_avg", float("nan"))
        built = r.get("dw_built_pct", {}).get("post_avg", float("nan"))
        ndvi = r.get("ndvi_mean", {}).get("post_avg", float("nan"))
        print(f"  {tier_name:<12} {ns:>6} {crop:>+11.4f} {tree:>+11.4f} "
              f"{built:>+11.4f} {ndvi:>+11.4f}")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════

def analysis_summary_stats(df, udb):
    print_header("ANALYSIS 5: SUMMARY STATISTICS")

    # Site-level info from unified DB
    udb_df = pd.DataFrame(udb)
    sites_in_panel = set(df["site_id"].unique())

    # Filter to sites in panel
    udb_panel = udb_df[udb_df["site_id"].isin(sites_in_panel)].copy()

    n_total = len(udb_panel)
    print(f"  Total sites in panel: {n_total}")

    # Country distribution
    country_dist = df.groupby("site_id")["country"].first().value_counts().to_dict()
    print(f"  Countries: {country_dist}")

    # Capacity distribution
    cap_col = "best_capacity_mw"
    if cap_col in udb_panel.columns:
        caps = udb_panel[cap_col].dropna()
        cap_stats = {
            "n_with_capacity": int(caps.notna().sum()) if hasattr(caps, 'notna') else int(len(caps)),
            "mean": float(caps.mean()),
            "median": float(caps.median()),
            "min": float(caps.min()),
            "max": float(caps.max()),
            "std": float(caps.std()),
            "q25": float(caps.quantile(0.25)),
            "q75": float(caps.quantile(0.75)),
            "tier_counts": {
                "<10 MW": int((caps < 10).sum()),
                "10-50 MW": int(((caps >= 10) & (caps < 50)).sum()),
                "50-200 MW": int(((caps >= 50) & (caps < 200)).sum()),
                ">200 MW": int((caps >= 200).sum()),
            },
        }
    else:
        cap_stats = {"error": "no capacity column found"}

    # Construction year distribution
    cy_col = "best_construction_year"
    if cy_col in udb_panel.columns:
        cys = udb_panel[cy_col].dropna().astype(int)
        cy_dist = cys.value_counts().sort_index().to_dict()
        cy_dist = {int(k): int(v) for k, v in cy_dist.items()}
    else:
        cy_dist = {}

    # Pre-construction LULC means (DW)
    pre = df[df["event_time"] < 0]
    dw_vars = ["dw_crops_pct", "dw_bare_pct", "dw_trees_pct", "dw_built_pct",
               "dw_water_pct", "ndvi_mean"]
    pre_lulc = {}
    for var in dw_vars:
        if var in pre.columns:
            vals = pre[var].dropna()
            pre_lulc[var] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(vals.median()),
                "n": int(len(vals)),
            }

    # Pre-construction EO means
    eo_vars = ["viirs_avg_rad", "pop_mean", "bldg_presence", "bldg_frac_count",
               "lst_day_c", "lst_night_c", "evi_mean", "sar_vv_db"]
    pre_eo = {}
    for var in eo_vars:
        if var in pre.columns:
            vals = pre[var].dropna()
            if len(vals) > 0:
                pre_eo[var] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "median": float(vals.median()),
                    "n": int(len(vals)),
                }

    # VLM coverage
    vlm_vars = ["vlm_solar_panels", "vlm_crops"]
    vlm_coverage = {}
    for var in vlm_vars:
        if var in df.columns:
            vlm_coverage[var] = {
                "n_nonmissing": int(df[var].notna().sum()),
                "n_total": int(len(df)),
                "coverage_pct": float(df[var].notna().mean() * 100),
            }

    output = {
        "n_sites": n_total,
        "n_obs": int(len(df)),
        "panel_years": [int(df["year"].min()), int(df["year"].max())],
        "country_distribution": {str(k): int(v) for k, v in country_dist.items()},
        "capacity_stats": cap_stats,
        "construction_year_distribution": cy_dist,
        "pre_construction_lulc_means": pre_lulc,
        "pre_construction_eo_means": pre_eo,
        "vlm_coverage": vlm_coverage,
    }

    path = "data/paper_summary_stats.json"
    save_json(output, path)

    # Print nicely
    print(f"\n  Panel: {n_total} sites, {len(df)} obs, {df['year'].min()}-{df['year'].max()}")
    print(f"\n  Countries:")
    for c, n in sorted(country_dist.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n}")

    if isinstance(cap_stats, dict) and "mean" in cap_stats:
        print(f"\n  Capacity: mean={cap_stats['mean']:.1f} MW, "
              f"median={cap_stats['median']:.1f} MW, range=[{cap_stats['min']:.1f}, {cap_stats['max']:.1f}]")
        print(f"  Capacity tiers: {cap_stats['tier_counts']}")

    print(f"\n  Construction years: {cy_dist}")

    print(f"\n  Pre-construction LULC means:")
    for var, stats in pre_lulc.items():
        print(f"    {var}: {stats['mean']:.2f} +/- {stats['std']:.2f} (n={stats['n']})")

    print(f"\n  Pre-construction EO means:")
    for var, stats in pre_eo.items():
        print(f"    {var}: {stats['mean']:.3f} +/- {stats['std']:.3f} (n={stats['n']})")

    return output


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs("data/event_study_results", exist_ok=True)

    # Load data
    df, udb = load_panel()

    # 1. EO Event Study
    eo_results = analysis_eo_event_study(df)

    # 2. Conflict Heterogeneity
    conflict_c, conflict_nc = analysis_conflict_heterogeneity(df)

    # 3. Urbanization / Agricultural Preservation
    urban_results = analysis_urbanization(df)

    # 4. Capacity Stratification
    cap_results = analysis_capacity_stratification(df)

    # 5. Summary Statistics
    summary = analysis_summary_stats(df, udb)

    # ── Final summary ────────────────────────────────────────────────────────
    print_header("ALL ANALYSES COMPLETE")
    print("  Cached results:")
    print("    data/event_study_results/event_study_eo.json")
    print("    data/event_study_results/conflict_heterogeneity.json")
    print("    data/event_study_results/urbanization_stratification.json")
    print("    data/event_study_results/capacity_stratification.json")
    print("    data/paper_summary_stats.json")


if __name__ == "__main__":
    main()
