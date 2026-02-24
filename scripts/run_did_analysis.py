"""
Difference-in-Differences analysis for solar farm land-use impacts.

Compares treatment sites (operational solar farms, high/very_high confidence)
against control sites (proposed/cancelled GEM projects that were never built)
using multi-temporal panel data.

Specifications:
  1. Baseline WLS: delta_Y ~ treatment + GHI + capacity + baseline_Y, weighted
  2. Country FE: adds C(country) dummies to pooled regression
  3. Heterogeneity: stratified by capacity, baseline LULC, construction year
  4. Propensity score matching: logistic on observables, NN matching, re-run DiD

Usage:
    python scripts/run_did_analysis.py
    python scripts/run_did_analysis.py --country bangladesh
    python scripts/run_did_analysis.py --psm     # With propensity score matching
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import apply_style, save_fig, FULL_WIDTH, HALF_WIDTH, DPI

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PANEL_CSV = DATA_DIR / "temporal_panel.csv"
UNIFIED_DB = DATA_DIR / "unified_solar_db.json"
OUTPUT_DIR = DATA_DIR / "did_results"

# Confidence weights
CONFIDENCE_WEIGHTS = {
    "very_high": 1.0,
    "high": 0.8,
    "proposed": 0.6,  # Control sites get decent weight
}

# DW class columns
DW_CLASSES = ["water", "trees", "grass", "flooded_vegetation",
              "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]


def load_panel(country_filter=None):
    """Load temporal panel CSV into DataFrame."""
    df = pd.read_csv(PANEL_CSV)

    if country_filter:
        df = df[df["country"].str.lower() == country_filter.lower()].copy()

    print(f"Loaded panel: {len(df)} rows, {df['site_id'].nunique()} sites")
    print(f"  Groups: {df['group'].value_counts().to_dict()}")
    print(f"  Time points: {df['time_point'].value_counts().to_dict()}")

    return df


def compute_change_metrics(df):
    """Compute pre→post change metrics for each site.

    Returns a DataFrame with one row per site, containing:
    - delta_* columns for each outcome variable
    - treatment indicator
    - covariates (GHI, capacity, baseline composition)
    """
    # Pivot: for each site, get pre and post values
    pre = df[df["time_point"] == "pre_construction"].copy()
    post = df[df["time_point"] == "post_construction"].copy()
    baseline = df[df["time_point"] == "baseline"].copy()

    # Merge pre and post on site_id
    pre = pre.set_index("site_id")
    post = post.set_index("site_id")

    # Only keep sites that have both pre and post
    common_sites = pre.index.intersection(post.index)
    print(f"\nSites with both pre and post data: {len(common_sites)}")

    if len(common_sites) == 0:
        print("No sites with both pre and post data!")
        return None

    # Compute deltas
    outcome_cols = [f"dw_{cn}_pct" for cn in DW_CLASSES]
    outcome_cols += ["viirs_avg_rad", "sar_vv_db", "sar_vh_db"]
    # Socioeconomic proxy datasets
    outcome_cols += ["ndvi_mean", "evi_mean", "lst_day_c", "lst_night_c",
                     "pop_sum", "pop_mean", "bldg_presence", "bldg_height_m",
                     "bldg_frac_count"]

    changes = pd.DataFrame(index=common_sites)

    # Site metadata (from pre row)
    for col in ["country", "group", "confidence", "capacity_mw",
                "construction_year", "ghi_kwh_m2_day", "project_name",
                "lat", "lon"]:
        changes[col] = pre.loc[common_sites, col]

    # Treatment indicator
    changes["treatment"] = (changes["group"] == "treatment").astype(int)

    # Confidence weight
    changes["weight"] = changes["confidence"].map(CONFIDENCE_WEIGHTS).fillna(0.3)

    # Change metrics
    for col in outcome_cols:
        if col in pre.columns and col in post.columns:
            pre_vals = pd.to_numeric(pre.loc[common_sites, col], errors="coerce")
            post_vals = pd.to_numeric(post.loc[common_sites, col], errors="coerce")
            changes[f"delta_{col}"] = post_vals - pre_vals
            changes[f"pre_{col}"] = pre_vals
            changes[f"post_{col}"] = post_vals

    # Baseline covariates
    if not baseline.empty:
        baseline = baseline.set_index("site_id")
        baseline_sites = baseline.index.intersection(common_sites)
        for col in outcome_cols:
            if col in baseline.columns:
                changes.loc[baseline_sites, f"baseline_{col}"] = pd.to_numeric(
                    baseline.loc[baseline_sites, col], errors="coerce")

    changes = changes.reset_index()
    return changes


def run_did_regression(changes, outcome_var, label=None):
    """Run a single DiD regression for one outcome variable.

    Model: delta_Y ~ treatment + covariates, weighted by confidence.
    Since we're using pre→post changes, the DiD estimator is just the
    coefficient on 'treatment' (difference in changes between groups).
    """
    dep_var = f"delta_{outcome_var}"
    if dep_var not in changes.columns:
        return None

    data = changes.dropna(subset=[dep_var, "treatment", "weight"])
    if len(data) < 10:
        return None

    # Build formula with available covariates
    covariates = []
    if "ghi_kwh_m2_day" in data.columns and data["ghi_kwh_m2_day"].notna().sum() > 5:
        covariates.append("ghi_kwh_m2_day")
    if "capacity_mw" in data.columns and data["capacity_mw"].notna().sum() > 5:
        covariates.append("capacity_mw")

    # Baseline value of same variable as covariate (controls for level)
    baseline_col = f"baseline_{outcome_var}"
    if baseline_col in data.columns and data[baseline_col].notna().sum() > 5:
        covariates.append(baseline_col)

    cov_str = " + ".join(covariates) if covariates else ""
    formula = f"Q('{dep_var}') ~ treatment"
    if cov_str:
        formula += f" + {cov_str}"

    try:
        model = smf.wls(formula, data=data, weights=data["weight"])
        result = model.fit()

        # Extract key stats
        treat_coef = result.params.get("treatment", np.nan)
        treat_se = result.bse.get("treatment", np.nan)
        treat_pval = result.pvalues.get("treatment", np.nan)

        return {
            "outcome": outcome_var,
            "label": label or outcome_var,
            "n_obs": int(result.nobs),
            "n_treatment": int(data["treatment"].sum()),
            "n_control": int((1 - data["treatment"]).sum()),
            "treatment_coef": float(treat_coef),
            "treatment_se": float(treat_se),
            "treatment_pval": float(treat_pval),
            "treatment_ci_low": float(treat_coef - 1.96 * treat_se),
            "treatment_ci_high": float(treat_coef + 1.96 * treat_se),
            "r_squared": float(result.rsquared),
            "mean_delta_treatment": float(data.loc[data["treatment"]==1, dep_var].mean()),
            "mean_delta_control": float(data.loc[data["treatment"]==0, dep_var].mean()),
            "formula": formula,
            "summary": result.summary().as_text(),
        }
    except Exception as e:
        print(f"  Regression failed for {outcome_var}: {e}")
        return None


def run_did_regression_fe(changes, outcome_var, label=None):
    """Run DiD regression with country fixed effects.

    Model: delta_Y ~ treatment + C(country) + covariates, weighted.
    Only meaningful for pooled (multi-country) analysis.
    """
    dep_var = f"delta_{outcome_var}"
    if dep_var not in changes.columns:
        return None

    data = changes.dropna(subset=[dep_var, "treatment", "weight", "country"])
    if len(data) < 10 or data["country"].nunique() < 2:
        return None

    covariates = ["C(country)"]
    if "ghi_kwh_m2_day" in data.columns and data["ghi_kwh_m2_day"].notna().sum() > 5:
        covariates.append("ghi_kwh_m2_day")
    if "capacity_mw" in data.columns and data["capacity_mw"].notna().sum() > 5:
        covariates.append("capacity_mw")
    baseline_col = f"baseline_{outcome_var}"
    if baseline_col in data.columns and data[baseline_col].notna().sum() > 5:
        covariates.append(baseline_col)

    cov_str = " + ".join(covariates)
    formula = f"Q('{dep_var}') ~ treatment + {cov_str}"

    try:
        model = smf.wls(formula, data=data, weights=data["weight"])
        result = model.fit()

        treat_coef = result.params.get("treatment", np.nan)
        treat_se = result.bse.get("treatment", np.nan)
        treat_pval = result.pvalues.get("treatment", np.nan)

        return {
            "outcome": outcome_var,
            "label": label or outcome_var,
            "n_obs": int(result.nobs),
            "n_treatment": int(data["treatment"].sum()),
            "n_control": int((1 - data["treatment"]).sum()),
            "treatment_coef": float(treat_coef),
            "treatment_se": float(treat_se),
            "treatment_pval": float(treat_pval),
            "treatment_ci_low": float(treat_coef - 1.96 * treat_se),
            "treatment_ci_high": float(treat_coef + 1.96 * treat_se),
            "r_squared": float(result.rsquared),
            "mean_delta_treatment": float(data.loc[data["treatment"]==1, dep_var].mean()),
            "mean_delta_control": float(data.loc[data["treatment"]==0, dep_var].mean()),
            "formula": formula,
            "n_countries": int(data["country"].nunique()),
            "summary": result.summary().as_text(),
        }
    except Exception as e:
        print(f"  FE regression failed for {outcome_var}: {e}")
        return None


def run_heterogeneity_analysis(changes, outcome_var, label=None):
    """Test treatment effect heterogeneity by capacity, baseline LULC, and year.

    Returns dict with stratified coefficients for each dimension.
    """
    dep_var = f"delta_{outcome_var}"
    if dep_var not in changes.columns:
        return None

    data = changes.dropna(subset=[dep_var, "treatment", "weight"]).copy()
    if len(data) < 50:
        return None

    results = {"outcome": outcome_var, "label": label or outcome_var}

    # ── By capacity quartile ──
    cap_data = data.dropna(subset=["capacity_mw"])
    if len(cap_data) > 50 and cap_data["capacity_mw"].nunique() > 3:
        cap_data = cap_data.copy()
        cap_data["cap_q"] = pd.qcut(cap_data["capacity_mw"], 3,
                                     labels=["small", "medium", "large"],
                                     duplicates="drop")
        cap_results = {}
        for q in cap_data["cap_q"].dropna().unique():
            sub = cap_data[cap_data["cap_q"] == q]
            if sub["treatment"].sum() > 5 and (1 - sub["treatment"]).sum() > 3:
                try:
                    formula = f"Q('{dep_var}') ~ treatment"
                    model = smf.wls(formula, data=sub, weights=sub["weight"])
                    res = model.fit()
                    cap_results[str(q)] = {
                        "coef": float(res.params.get("treatment", np.nan)),
                        "pval": float(res.pvalues.get("treatment", np.nan)),
                        "n": int(res.nobs),
                    }
                except Exception:
                    pass
        results["by_capacity"] = cap_results

    # ── By baseline dominant LULC ──
    lulc_cols = [f"baseline_dw_{cn}_pct" for cn in DW_CLASSES
                 if f"baseline_dw_{cn}_pct" in data.columns]
    if lulc_cols:
        lulc_data = data.dropna(subset=lulc_cols).copy()
        if len(lulc_data) > 50:
            lulc_data["dominant_lulc"] = lulc_data[lulc_cols].idxmax(axis=1).str.replace(
                "baseline_dw_", "").str.replace("_pct", "")
            lulc_results = {}
            for lulc in lulc_data["dominant_lulc"].value_counts().head(4).index:
                sub = lulc_data[lulc_data["dominant_lulc"] == lulc]
                if sub["treatment"].sum() > 5 and (1 - sub["treatment"]).sum() > 3:
                    try:
                        formula = f"Q('{dep_var}') ~ treatment"
                        model = smf.wls(formula, data=sub, weights=sub["weight"])
                        res = model.fit()
                        lulc_results[lulc] = {
                            "coef": float(res.params.get("treatment", np.nan)),
                            "pval": float(res.pvalues.get("treatment", np.nan)),
                            "n": int(res.nobs),
                        }
                    except Exception:
                        pass
            results["by_baseline_lulc"] = lulc_results

    # ── By construction year cohort ──
    yr_data = data.dropna(subset=["construction_year"]).copy()
    if len(yr_data) > 50:
        yr_data["year_cohort"] = pd.cut(
            yr_data["construction_year"],
            bins=[2014, 2018, 2021, 2026],
            labels=["early (2015-18)", "mid (2019-21)", "late (2022-25)"],
        )
        yr_results = {}
        for cohort in yr_data["year_cohort"].dropna().unique():
            sub = yr_data[yr_data["year_cohort"] == cohort]
            if sub["treatment"].sum() > 5 and (1 - sub["treatment"]).sum() > 3:
                try:
                    formula = f"Q('{dep_var}') ~ treatment"
                    model = smf.wls(formula, data=sub, weights=sub["weight"])
                    res = model.fit()
                    yr_results[str(cohort)] = {
                        "coef": float(res.params.get("treatment", np.nan)),
                        "pval": float(res.pvalues.get("treatment", np.nan)),
                        "n": int(res.nobs),
                    }
                except Exception:
                    pass
        results["by_construction_year"] = yr_results

    # ── Treatment × covariate interaction ──
    # Test if treatment effect varies with GHI
    if "ghi_kwh_m2_day" in data.columns and data["ghi_kwh_m2_day"].notna().sum() > 50:
        try:
            formula = f"Q('{dep_var}') ~ treatment * ghi_kwh_m2_day"
            model = smf.wls(formula, data=data.dropna(subset=["ghi_kwh_m2_day"]),
                            weights=data.dropna(subset=["ghi_kwh_m2_day"])["weight"])
            res = model.fit()
            interaction_key = "treatment:ghi_kwh_m2_day"
            results["ghi_interaction"] = {
                "coef": float(res.params.get(interaction_key, np.nan)),
                "pval": float(res.pvalues.get(interaction_key, np.nan)),
            }
        except Exception:
            pass

    return results


def run_psm_analysis(changes, outcomes_list):
    """Propensity score matching: match treatment/control on observables.

    1. Estimate P(treatment=1 | X) via logistic regression
    2. Nearest-neighbor match (1:1) within caliper
    3. Re-run DiD on matched sample

    Returns (matched_changes, psm_results, diagnostics).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Covariates for propensity score
    psm_covars = []
    for col in ["ghi_kwh_m2_day", "baseline_dw_crops_pct", "baseline_dw_trees_pct",
                "baseline_dw_built_pct", "baseline_dw_bare_pct",
                "baseline_dw_water_pct", "baseline_viirs_avg_rad"]:
        if col in changes.columns and changes[col].notna().sum() > 50:
            psm_covars.append(col)

    if len(psm_covars) < 3:
        print("  Insufficient covariates for PSM")
        return None, [], {}

    data = changes.dropna(subset=psm_covars + ["treatment"]).copy()
    print(f"\n  PSM: {len(data)} sites with complete covariates")
    print(f"  Covariates: {psm_covars}")

    X = data[psm_covars].values
    y = data["treatment"].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic regression
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_scaled, y)
    data["pscore"] = lr.predict_proba(X_scaled)[:, 1]

    # Check overlap
    treat_ps = data.loc[data["treatment"] == 1, "pscore"]
    ctrl_ps = data.loc[data["treatment"] == 0, "pscore"]
    print(f"  Propensity scores — Treatment: {treat_ps.mean():.3f} "
          f"[{treat_ps.min():.3f}, {treat_ps.max():.3f}]")
    print(f"  Propensity scores — Control: {ctrl_ps.mean():.3f} "
          f"[{ctrl_ps.min():.3f}, {ctrl_ps.max():.3f}]")

    # Nearest-neighbor matching (1:1 without replacement)
    caliper = 0.2 * data["pscore"].std()
    ctrl_indices = data.index[data["treatment"] == 0].tolist()
    treat_indices = data.index[data["treatment"] == 1].tolist()

    matched_pairs = []
    used_ctrl = set()

    for t_idx in treat_indices:
        t_ps = data.loc[t_idx, "pscore"]
        best_dist = float("inf")
        best_ctrl = None

        for c_idx in ctrl_indices:
            if c_idx in used_ctrl:
                continue
            dist = abs(t_ps - data.loc[c_idx, "pscore"])
            if dist < best_dist and dist <= caliper:
                best_dist = dist
                best_ctrl = c_idx

        if best_ctrl is not None:
            matched_pairs.append((t_idx, best_ctrl))
            used_ctrl.add(best_ctrl)

    print(f"  Matched pairs: {len(matched_pairs)} "
          f"(of {len(treat_indices)} treatment, {len(ctrl_indices)} control)")

    if len(matched_pairs) < 20:
        print("  Too few matched pairs for PSM analysis")
        return None, [], {}

    # Build matched dataset
    matched_idx = [t for t, c in matched_pairs] + [c for t, c in matched_pairs]
    matched = data.loc[matched_idx].copy()

    # Run regressions on matched sample
    psm_results = []
    for var, label in outcomes_list:
        dep_var = f"delta_{var}"
        if dep_var not in matched.columns:
            continue
        sub = matched.dropna(subset=[dep_var])
        if len(sub) < 20:
            continue
        try:
            formula = f"Q('{dep_var}') ~ treatment"
            model = smf.ols(formula, data=sub)
            res = model.fit()
            treat_coef = res.params.get("treatment", np.nan)
            treat_se = res.bse.get("treatment", np.nan)
            treat_pval = res.pvalues.get("treatment", np.nan)
            psm_results.append({
                "outcome": var,
                "label": label,
                "treatment_coef": float(treat_coef),
                "treatment_se": float(treat_se),
                "treatment_pval": float(treat_pval),
                "treatment_ci_low": float(treat_coef - 1.96 * treat_se),
                "treatment_ci_high": float(treat_coef + 1.96 * treat_se),
                "n_obs": int(res.nobs),
                "r_squared": float(res.rsquared),
            })
        except Exception:
            pass

    diagnostics = {
        "n_treatment": len(treat_indices),
        "n_control": len(ctrl_indices),
        "n_matched_pairs": len(matched_pairs),
        "caliper": float(caliper),
        "covariates": psm_covars,
        "pscore_overlap": {
            "treatment_mean": float(treat_ps.mean()),
            "control_mean": float(ctrl_ps.mean()),
        },
    }

    # Balance check: standardized mean differences before/after
    balance = {}
    for col in psm_covars:
        # Before matching
        t_mean_pre = data.loc[data["treatment"]==1, col].mean()
        c_mean_pre = data.loc[data["treatment"]==0, col].mean()
        pooled_sd = data[col].std()
        smd_pre = (t_mean_pre - c_mean_pre) / pooled_sd if pooled_sd > 0 else 0

        # After matching
        t_mean_post = matched.loc[matched["treatment"]==1, col].mean()
        c_mean_post = matched.loc[matched["treatment"]==0, col].mean()
        smd_post = (t_mean_post - c_mean_post) / pooled_sd if pooled_sd > 0 else 0

        balance[col] = {
            "smd_before": float(smd_pre),
            "smd_after": float(smd_post),
        }
    diagnostics["balance"] = balance

    return matched, psm_results, diagnostics


def run_all_regressions(changes):
    """Run DiD regressions for all outcome variables."""
    outcomes = [
        # LULC
        ("dw_built_pct", "Built-up (%)"),
        ("dw_crops_pct", "Cropland (%)"),
        ("dw_trees_pct", "Trees (%)"),
        ("dw_bare_pct", "Bare ground (%)"),
        ("dw_water_pct", "Water (%)"),
        ("dw_grass_pct", "Grassland (%)"),
        # Remote sensing
        ("viirs_avg_rad", "Nighttime light (nW/sr/cm\u00b2)"),
        ("sar_vv_db", "SAR VV backscatter (dB)"),
        ("sar_vh_db", "SAR VH backscatter (dB)"),
        # Vegetation
        ("ndvi_mean", "NDVI"),
        ("evi_mean", "EVI"),
        # Temperature
        ("lst_day_c", "Daytime LST (\u00b0C)"),
        ("lst_night_c", "Nighttime LST (\u00b0C)"),
        # Population
        ("pop_sum", "Population (sum, 1km)"),
        ("pop_mean", "Population density"),
        # Buildings
        ("bldg_presence", "Building presence"),
        ("bldg_height_m", "Building height (m)"),
        ("bldg_frac_count", "Building count"),
    ]

    results = []
    for var, label in outcomes:
        print(f"\n  Regression: {label}...")
        res = run_did_regression(changes, var, label)
        if res:
            sig = "***" if res["treatment_pval"] < 0.01 else "**" if res["treatment_pval"] < 0.05 else "*" if res["treatment_pval"] < 0.1 else ""
            print(f"    coef={res['treatment_coef']:.3f} (SE={res['treatment_se']:.3f}), "
                  f"p={res['treatment_pval']:.4f}{sig}, R²={res['r_squared']:.3f}")
            print(f"    Mean Δ: treatment={res['mean_delta_treatment']:.2f}, "
                  f"control={res['mean_delta_control']:.2f}")
            results.append(res)
        else:
            print(f"    Skipped (insufficient data)")

    return results


# ── Visualization ────────────────────────────────────────────────────────────

def plot_coefficient_chart(results, country_label=""):
    """Forest plot of DiD treatment coefficients."""
    apply_style()

    # Filter to results with valid coefficients
    valid = [r for r in results if not np.isnan(r["treatment_coef"])]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, max(3, len(valid) * 0.5)))

    labels = [r["label"] for r in valid]
    coefs = [r["treatment_coef"] for r in valid]
    ci_low = [r["treatment_ci_low"] for r in valid]
    ci_high = [r["treatment_ci_high"] for r in valid]
    pvals = [r["treatment_pval"] for r in valid]

    y_pos = range(len(valid))

    # Color by significance
    colors = []
    for p in pvals:
        if p < 0.01:
            colors.append("#332288")  # Strong significance
        elif p < 0.05:
            colors.append("#44AA99")
        elif p < 0.1:
            colors.append("#DDCC77")
        else:
            colors.append("#DDDDDD")

    ax.barh(y_pos, coefs, color=colors, edgecolor="none", height=0.6, alpha=0.8)

    # Error bars
    for i, (c, lo, hi) in enumerate(zip(coefs, ci_low, ci_high)):
        ax.plot([lo, hi], [i, i], color="black", linewidth=1)

    # Zero line
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("DiD Treatment Effect (pre→post change difference)")

    title = "Difference-in-Differences: Solar Farm Impact"
    if country_label:
        title += f" ({country_label})"
    ax.set_title(title)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#332288", label="p < 0.01"),
        Patch(facecolor="#44AA99", label="p < 0.05"),
        Patch(facecolor="#DDCC77", label="p < 0.1"),
        Patch(facecolor="#DDDDDD", label="not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    save_fig(fig, FIG_DIR / "did_coefficient_chart.png")
    plt.close()


def plot_parallel_trends(changes, outcome_var="dw_built_pct", label="Built-up (%)",
                         country_label=""):
    """Plot pre/post means for treatment vs control (parallel trends check)."""
    apply_style()

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 3.5))

    for group, color, marker in [("treatment", "#CC6677", "o"),
                                 ("control", "#88CCEE", "s")]:
        data = changes[changes["group"] == group]
        if data.empty:
            continue

        pre_col = f"pre_{outcome_var}"
        post_col = f"post_{outcome_var}"

        if pre_col not in data.columns or post_col not in data.columns:
            continue

        pre_mean = data[pre_col].mean()
        post_mean = data[post_col].mean()
        pre_se = data[pre_col].sem()
        post_se = data[post_col].sem()

        ax.errorbar([0, 1], [pre_mean, post_mean],
                    yerr=[pre_se, post_se],
                    marker=marker, label=f"{group} (n={len(data)})",
                    color=color, linewidth=2, markersize=8, capsize=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-construction", "Post-construction"])
    ax.set_ylabel(label)

    title = f"Parallel Trends: {label}"
    if country_label:
        title += f" ({country_label})"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, FIG_DIR / f"did_parallel_trends_{outcome_var}.png")
    plt.close()


def create_regression_table(results):
    """Create a formatted regression results table."""
    rows = []
    for r in results:
        sig = ""
        if r["treatment_pval"] < 0.01:
            sig = "***"
        elif r["treatment_pval"] < 0.05:
            sig = "**"
        elif r["treatment_pval"] < 0.1:
            sig = "*"

        rows.append({
            "Outcome": r["label"],
            "DiD Coef": f"{r['treatment_coef']:.3f}{sig}",
            "SE": f"({r['treatment_se']:.3f})",
            "95% CI": f"[{r['treatment_ci_low']:.3f}, {r['treatment_ci_high']:.3f}]",
            "p-value": f"{r['treatment_pval']:.4f}",
            "R²": f"{r['r_squared']:.3f}",
            "N": r["n_obs"],
            "Treat": r["n_treatment"],
            "Control": r["n_control"],
            "Mean Δ Treat": f"{r['mean_delta_treatment']:.2f}",
            "Mean Δ Control": f"{r['mean_delta_control']:.2f}",
        })

    table_df = pd.DataFrame(rows)
    return table_df


# ── Main ─────────────────────────────────────────────────────────────────────

OUTCOMES_LIST = [
    ("dw_built_pct", "Built-up (%)"),
    ("dw_crops_pct", "Cropland (%)"),
    ("dw_trees_pct", "Trees (%)"),
    ("dw_bare_pct", "Bare ground (%)"),
    ("dw_water_pct", "Water (%)"),
    ("dw_grass_pct", "Grassland (%)"),
    ("viirs_avg_rad", "Nighttime light (nW/sr/cm\u00b2)"),
    ("sar_vv_db", "SAR VV backscatter (dB)"),
    ("sar_vh_db", "SAR VH backscatter (dB)"),
    ("ndvi_mean", "NDVI"),
    ("evi_mean", "EVI"),
    ("lst_day_c", "Daytime LST (\u00b0C)"),
    ("lst_night_c", "Nighttime LST (\u00b0C)"),
    ("pop_sum", "Population (sum, 1km)"),
    ("pop_mean", "Population density"),
    ("bldg_presence", "Building presence"),
    ("bldg_height_m", "Building height (m)"),
    ("bldg_frac_count", "Building count"),
]


def run_analysis(country_filter=None, run_psm=False):
    print("=" * 70)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 70)

    # Load data
    df = load_panel(country_filter)
    country_label = country_filter.title() if country_filter else "South Asia"

    # Country-specific output directory
    if country_filter:
        out_dir = OUTPUT_DIR / country_filter.lower()
    else:
        out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute change metrics
    print("\nComputing change metrics...")
    changes = compute_change_metrics(df)
    if changes is None:
        return

    print(f"\nChange dataset: {len(changes)} sites")
    print(f"  Treatment: {changes['treatment'].sum()}")
    print(f"  Control: {(1 - changes['treatment']).sum()}")

    # ── 1. Baseline regressions ──
    print("\n" + "=" * 70)
    print("1. BASELINE DiD REGRESSIONS (WLS)")
    print("=" * 70)
    results = run_all_regressions(changes)

    if not results:
        print("\nNo valid regression results!")
        return

    table = create_regression_table(results)
    print(f"\n{table.to_string(index=False)}")
    table.to_csv(out_dir / "did_regression_table.csv", index=False)

    results_json = {
        "country": country_label,
        "n_sites": len(changes),
        "n_treatment": int(changes["treatment"].sum()),
        "n_control": int((1 - changes["treatment"]).sum()),
        "regressions": [{k: v for k, v in r.items() if k != "summary"}
                        for r in results],
    }

    # Save full regression summaries
    with open(out_dir / "did_full_summaries.txt", "w") as f:
        for r in results:
            f.write(f"\n{'='*70}\n")
            f.write(f"Outcome: {r['label']}\n")
            f.write(f"{'='*70}\n\n")
            f.write(r["summary"])
            f.write("\n\n")

    # ── 2. Country fixed effects (pooled only) ──
    if not country_filter and changes["country"].nunique() > 1:
        print("\n" + "=" * 70)
        print("2. COUNTRY FIXED EFFECTS (WLS)")
        print("=" * 70)
        fe_results = []
        for var, label in OUTCOMES_LIST:
            res = run_did_regression_fe(changes, var, label)
            if res:
                sig = "***" if res["treatment_pval"] < 0.01 else \
                      "**" if res["treatment_pval"] < 0.05 else \
                      "*" if res["treatment_pval"] < 0.1 else ""
                print(f"  {label}: coef={res['treatment_coef']:.3f}{sig} "
                      f"(p={res['treatment_pval']:.4f}), "
                      f"R²={res['r_squared']:.3f}")
                fe_results.append(res)

        results_json["fe_regressions"] = [
            {k: v for k, v in r.items() if k != "summary"} for r in fe_results
        ]

        # Compare baseline vs FE
        print("\n  Baseline vs Country FE comparison:")
        print(f"  {'Outcome':<30} {'Baseline':>10} {'FE':>10} {'Δ coef':>10}")
        for base_r in results:
            fe_match = [r for r in fe_results if r["outcome"] == base_r["outcome"]]
            if fe_match:
                fe_r = fe_match[0]
                delta = fe_r["treatment_coef"] - base_r["treatment_coef"]
                print(f"  {base_r['label']:<30} "
                      f"{base_r['treatment_coef']:>+10.3f} "
                      f"{fe_r['treatment_coef']:>+10.3f} "
                      f"{delta:>+10.3f}")

        # Save FE summaries
        with open(out_dir / "did_fe_summaries.txt", "w") as f:
            for r in fe_results:
                f.write(f"\n{'='*70}\n")
                f.write(f"Outcome: {r['label']} (Country FE)\n")
                f.write(f"{'='*70}\n\n")
                f.write(r["summary"])
                f.write("\n\n")

    # ── 3. Heterogeneity analysis ──
    print("\n" + "=" * 70)
    print("3. HETEROGENEITY ANALYSIS")
    print("=" * 70)
    # Run on key outcomes only (to keep output manageable)
    key_outcomes_het = [
        ("dw_trees_pct", "Trees (%)"),
        ("dw_bare_pct", "Bare ground (%)"),
        ("dw_built_pct", "Built-up (%)"),
        ("viirs_avg_rad", "NTL"),
        ("sar_vh_db", "SAR VH (dB)"),
        ("ndvi_mean", "NDVI"),
        ("lst_night_c", "Night LST (\u00b0C)"),
    ]
    het_results = {}
    for var, label in key_outcomes_het:
        het = run_heterogeneity_analysis(changes, var, label)
        if het:
            het_results[var] = het
            print(f"\n  {label}:")
            if "by_capacity" in het and het["by_capacity"]:
                print(f"    By capacity: ", end="")
                for k, v in het["by_capacity"].items():
                    sig = "*" if v["pval"] < 0.05 else ""
                    print(f"{k}={v['coef']:+.2f}{sig} (n={v['n']}), ", end="")
                print()
            if "by_baseline_lulc" in het and het["by_baseline_lulc"]:
                print(f"    By baseline LULC: ", end="")
                for k, v in het["by_baseline_lulc"].items():
                    sig = "*" if v["pval"] < 0.05 else ""
                    print(f"{k}={v['coef']:+.2f}{sig}, ", end="")
                print()
            if "by_construction_year" in het and het["by_construction_year"]:
                print(f"    By construction year: ", end="")
                for k, v in het["by_construction_year"].items():
                    sig = "*" if v["pval"] < 0.05 else ""
                    print(f"{k}={v['coef']:+.2f}{sig}, ", end="")
                print()
            if "ghi_interaction" in het:
                gi = het["ghi_interaction"]
                sig = "*" if gi["pval"] < 0.05 else ""
                print(f"    GHI interaction: {gi['coef']:+.3f}{sig} "
                      f"(p={gi['pval']:.3f})")

    results_json["heterogeneity"] = het_results

    # ── 4. Propensity score matching ──
    if run_psm:
        print("\n" + "=" * 70)
        print("4. PROPENSITY SCORE MATCHING")
        print("=" * 70)
        matched, psm_results, psm_diag = run_psm_analysis(
            changes, OUTCOMES_LIST)

        if psm_results:
            print(f"\n  PSM DiD results ({psm_diag['n_matched_pairs']} pairs):")
            print(f"  {'Outcome':<30} {'PSM coef':>10} {'p-value':>10} "
                  f"{'Baseline':>10}")
            for pr in psm_results:
                base_match = [r for r in results
                              if r["outcome"] == pr["outcome"]]
                base_coef = base_match[0]["treatment_coef"] if base_match else np.nan
                sig = "*" if pr["treatment_pval"] < 0.05 else ""
                print(f"  {pr['label']:<30} "
                      f"{pr['treatment_coef']:>+10.3f}{sig} "
                      f"{pr['treatment_pval']:>10.4f} "
                      f"{base_coef:>+10.3f}")

            # Balance diagnostics
            if "balance" in psm_diag:
                print(f"\n  Covariate balance (standardized mean differences):")
                print(f"  {'Covariate':<35} {'Before':>8} {'After':>8}")
                for col, b in psm_diag["balance"].items():
                    flag = " !" if abs(b["smd_after"]) > 0.1 else ""
                    print(f"  {col:<35} {b['smd_before']:>+8.3f} "
                          f"{b['smd_after']:>+8.3f}{flag}")

            results_json["psm"] = {
                "diagnostics": psm_diag,
                "results": psm_results,
            }

    # ── Save all results ──
    with open(out_dir / "did_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    # Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_coefficient_chart(results, country_label)

    key_outcomes = [
        ("dw_built_pct", "Built-up (%)"),
        ("dw_crops_pct", "Cropland (%)"),
        ("viirs_avg_rad", "Nighttime light"),
    ]
    for var, label in key_outcomes:
        plot_parallel_trends(changes, var, label, country_label)

    print(f"\nResults saved to {out_dir}/")
    print(f"Figures saved to {FIG_DIR}/")

    # Key findings summary
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    sig_results = [r for r in results if r["treatment_pval"] < 0.05]
    if sig_results:
        print(f"\nStatistically significant effects (p < 0.05):")
        for r in sig_results:
            direction = "increase" if r["treatment_coef"] > 0 else "decrease"
            print(f"  {r['label']}: {direction} of {abs(r['treatment_coef']):.2f} "
                  f"(p={r['treatment_pval']:.4f})")
    else:
        print("\nNo statistically significant treatment effects found (p < 0.05).")
        marginal = [r for r in results if r["treatment_pval"] < 0.1]
        if marginal:
            print(f"Marginally significant (p < 0.1):")
            for r in marginal:
                print(f"  {r['label']}: coef={r['treatment_coef']:.3f} "
                      f"(p={r['treatment_pval']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Run DiD analysis on solar farm temporal panel data")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter to single country (e.g. 'bangladesh')")
    parser.add_argument("--psm", action="store_true",
                        help="Run propensity score matching analysis")
    args = parser.parse_args()

    run_analysis(country_filter=args.country, run_psm=args.psm)


if __name__ == "__main__":
    main()
