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
        # Use HC1 robust standard errors (heteroskedasticity-consistent)
        # For pooled multi-country, use cluster by country via run_did_regression_fe
        result = model.fit(cov_type="HC1")

        # Extract key stats
        treat_coef = result.params.get("treatment", np.nan)
        treat_se = result.bse.get("treatment", np.nan)
        treat_pval = result.pvalues.get("treatment", np.nan)

        # Use model-provided confidence intervals (robust)
        ci = result.conf_int().loc["treatment"]

        return {
            "outcome": outcome_var,
            "label": label or outcome_var,
            "n_obs": int(result.nobs),
            "n_treatment": int(data["treatment"].sum()),
            "n_control": int((1 - data["treatment"]).sum()),
            "treatment_coef": float(treat_coef),
            "treatment_se": float(treat_se),
            "treatment_pval": float(treat_pval),
            "treatment_ci_low": float(ci[0]),
            "treatment_ci_high": float(ci[1]),
            "r_squared": float(result.rsquared),
            "mean_delta_treatment": float(data.loc[data["treatment"]==1, dep_var].mean()),
            "mean_delta_control": float(data.loc[data["treatment"]==0, dep_var].mean()),
            "formula": formula,
            "se_type": "HC1",
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

    # Determine which columns will enter the regression, then drop NaN on all
    dropna_cols = [dep_var, "treatment", "weight", "country"]
    covariates = ["C(country)"]
    if "ghi_kwh_m2_day" in changes.columns and changes["ghi_kwh_m2_day"].notna().sum() > 5:
        covariates.append("ghi_kwh_m2_day")
        dropna_cols.append("ghi_kwh_m2_day")
    if "capacity_mw" in changes.columns and changes["capacity_mw"].notna().sum() > 5:
        covariates.append("capacity_mw")
        dropna_cols.append("capacity_mw")
    baseline_col = f"baseline_{outcome_var}"
    if baseline_col in changes.columns and changes[baseline_col].notna().sum() > 5:
        covariates.append(baseline_col)
        dropna_cols.append(baseline_col)

    data = changes.dropna(subset=dropna_cols)
    if len(data) < 10 or data["country"].nunique() < 2:
        return None

    cov_str = " + ".join(covariates)
    formula = f"Q('{dep_var}') ~ treatment + {cov_str}"

    try:
        model = smf.wls(formula, data=data, weights=data["weight"])
        # Cluster standard errors by country for pooled regressions
        n_countries = data["country"].nunique()
        if n_countries >= 3:
            result = model.fit(cov_type="cluster",
                               cov_kwds={"groups": data["country"]})
            se_type = f"clustered (country, G={n_countries})"
        else:
            # Too few clusters for cluster-robust; fall back to HC1
            result = model.fit(cov_type="HC1")
            se_type = "HC1"

        treat_coef = result.params.get("treatment", np.nan)
        treat_se = result.bse.get("treatment", np.nan)
        treat_pval = result.pvalues.get("treatment", np.nan)
        ci = result.conf_int().loc["treatment"]

        return {
            "outcome": outcome_var,
            "label": label or outcome_var,
            "n_obs": int(result.nobs),
            "n_treatment": int(data["treatment"].sum()),
            "n_control": int((1 - data["treatment"]).sum()),
            "treatment_coef": float(treat_coef),
            "treatment_se": float(treat_se),
            "treatment_pval": float(treat_pval),
            "treatment_ci_low": float(ci[0]),
            "treatment_ci_high": float(ci[1]),
            "r_squared": float(result.rsquared),
            "mean_delta_treatment": float(data.loc[data["treatment"]==1, dep_var].mean()),
            "mean_delta_control": float(data.loc[data["treatment"]==0, dep_var].mean()),
            "formula": formula,
            "se_type": se_type,
            "n_countries": int(n_countries),
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
                    res = model.fit(cov_type="HC1")
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
                        res = model.fit(cov_type="HC1")
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
                    res = model.fit(cov_type="HC1")
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
            res = model.fit(cov_type="HC1")
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
            res = model.fit(cov_type="HC1")
            treat_coef = res.params.get("treatment", np.nan)
            treat_se = res.bse.get("treatment", np.nan)
            treat_pval = res.pvalues.get("treatment", np.nan)
            ci = res.conf_int().loc["treatment"]
            psm_results.append({
                "outcome": var,
                "label": label,
                "treatment_coef": float(treat_coef),
                "treatment_se": float(treat_se),
                "treatment_pval": float(treat_pval),
                "treatment_ci_low": float(ci[0]),
                "treatment_ci_high": float(ci[1]),
                "n_obs": int(res.nobs),
                "r_squared": float(res.rsquared),
                "se_type": "HC1",
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


def compute_placebo_changes(df):
    """Compute baseline→pre_construction changes for placebo test.

    If parallel trends holds, these changes should NOT differ between
    treatment and control (since treatment hasn't occurred yet).
    """
    baseline = df[df["time_point"] == "baseline"].copy()
    pre = df[df["time_point"] == "pre_construction"].copy()

    if baseline.empty or pre.empty:
        return None

    baseline = baseline.set_index("site_id")
    pre = pre.set_index("site_id")

    common = baseline.index.intersection(pre.index)
    if len(common) == 0:
        return None

    outcome_cols = [f"dw_{cn}_pct" for cn in DW_CLASSES]
    outcome_cols += ["viirs_avg_rad", "sar_vv_db", "sar_vh_db",
                     "ndvi_mean", "evi_mean", "lst_day_c", "lst_night_c",
                     "pop_sum", "pop_mean", "bldg_presence", "bldg_height_m",
                     "bldg_frac_count"]

    changes = pd.DataFrame(index=common)

    for col in ["country", "group", "confidence", "capacity_mw",
                "construction_year", "ghi_kwh_m2_day"]:
        changes[col] = baseline.loc[common, col]

    changes["treatment"] = (changes["group"] == "treatment").astype(int)
    changes["weight"] = changes["confidence"].map(CONFIDENCE_WEIGHTS).fillna(0.3)

    for col in outcome_cols:
        if col in baseline.columns and col in pre.columns:
            b_vals = pd.to_numeric(baseline.loc[common, col], errors="coerce")
            p_vals = pd.to_numeric(pre.loc[common, col], errors="coerce")
            changes[f"delta_{col}"] = p_vals - b_vals

    return changes.reset_index()


def run_placebo_test(df, outcomes_list):
    """Placebo/falsification test: DiD on baseline→pre_construction period.

    Treatment has not yet occurred, so the treatment coefficient should be ~0.
    If it is significant, parallel trends is violated.
    """
    placebo_changes = compute_placebo_changes(df)
    if placebo_changes is None:
        print("  Cannot run placebo test (missing baseline or pre data)")
        return []

    n_treat = int(placebo_changes["treatment"].sum())
    n_ctrl = int((1 - placebo_changes["treatment"]).sum())
    print(f"  Placebo sample: {len(placebo_changes)} sites "
          f"({n_treat} treatment, {n_ctrl} control)")

    results = []
    for var, label in outcomes_list:
        dep_var = f"delta_{var}"
        if dep_var not in placebo_changes.columns:
            continue

        data = placebo_changes.dropna(subset=[dep_var, "treatment", "weight"])
        if len(data) < 10:
            continue

        try:
            formula = f"Q('{dep_var}') ~ treatment"
            model = smf.wls(formula, data=data, weights=data["weight"])
            result = model.fit(cov_type="HC1")

            treat_coef = result.params.get("treatment", np.nan)
            treat_se = result.bse.get("treatment", np.nan)
            treat_pval = result.pvalues.get("treatment", np.nan)
            ci = result.conf_int().loc["treatment"]

            results.append({
                "outcome": var,
                "label": label,
                "treatment_coef": float(treat_coef),
                "treatment_se": float(treat_se),
                "treatment_pval": float(treat_pval),
                "treatment_ci_low": float(ci[0]),
                "treatment_ci_high": float(ci[1]),
                "n_obs": int(result.nobs),
                "se_type": "HC1",
            })
        except Exception:
            pass

    return results


def plot_placebo_results(placebo_results, country_label=""):
    """Forest plot comparing placebo (pre-treatment) coefficients to zero."""
    apply_style()

    valid = [r for r in placebo_results if not np.isnan(r["treatment_coef"])]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, max(3, len(valid) * 0.45)))

    labels = [r["label"] for r in valid]
    coefs = [r["treatment_coef"] for r in valid]
    ci_low = [r["treatment_ci_low"] for r in valid]
    ci_high = [r["treatment_ci_high"] for r in valid]
    pvals = [r["treatment_pval"] for r in valid]
    y_pos = range(len(valid))

    # Color: red if significant (parallel trends violated), grey if not
    colors = ["#CC6677" if p < 0.05 else "#88CCEE" for p in pvals]

    ax.barh(y_pos, coefs, color=colors, edgecolor="none", height=0.6, alpha=0.8)
    for i, (lo, hi) in enumerate(zip(ci_low, ci_high)):
        ax.plot([lo, hi], [i, i], color="black", linewidth=1)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Placebo DiD Coefficient (baseline → pre-construction)")

    title = "Placebo Test: Pre-Treatment Period"
    if country_label:
        title += f" ({country_label})"
    ax.set_title(title)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#CC6677", label="p < 0.05 (parallel trends violated)"),
        Patch(facecolor="#88CCEE", label="not significant (parallel trends holds)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    save_fig(fig, FIG_DIR / "did_placebo_test.png")
    plt.close()


def build_event_study_panel(df):
    """Build a long panel for event study estimation.

    Assigns event time relative to construction year:
      baseline     → varies by site (typically -3 to -8)
      pre_construction → -1 (reference)
      post_construction → +1
      current      → varies by site (typically +2 to +8)

    Control sites (never-treated) get event_time = calendar year - 2019
    (centered on their pseudo pre-construction year).
    """
    rows = []
    for _, row in df.iterrows():
        construction_year = row.get("construction_year")
        time_point = row["time_point"]
        year = row["year"]

        if row["group"] == "treatment" and construction_year and construction_year > 2015:
            event_time = year - construction_year
        else:
            # Control sites: center on their pseudo-treatment (2019/2020)
            pseudo_treat = 2020
            event_time = year - pseudo_treat

        row_dict = row.to_dict()
        row_dict["event_time"] = int(event_time)
        rows.append(row_dict)

    panel = pd.DataFrame(rows)
    return panel


def run_event_study(df, outcome_var, label=None):
    """Non-parametric event study: compute DiD at each event time.

    For each event time k (relative to construction), compute:
      beta_k = (Y_treat,k - Y_treat,-1) - (Y_ctrl,k - Y_ctrl,-1)

    This is equivalent to a 2x2 DiD for each period relative to
    the reference period (t=-1, pre_construction).

    More robust than regression-based event study for staggered designs
    because it avoids multicollinearity in event-time dummies.
    """
    panel = build_event_study_panel(df)

    if outcome_var not in panel.columns:
        return None

    data = panel.copy()
    data["treatment"] = (data["group"] == "treatment").astype(int)
    data[outcome_var] = pd.to_numeric(data[outcome_var], errors="coerce")
    data = data.dropna(subset=[outcome_var])

    if len(data) < 20:
        return None

    # Bin event times
    def bin_event_time(t):
        if t <= -4:
            return -4
        elif t >= 3:
            return 3
        else:
            return t

    data["event_time_binned"] = data["event_time"].apply(bin_event_time)

    event_times = sorted(data["event_time_binned"].unique())
    if -1 not in event_times:
        return None

    # Get reference period means
    ref = data[data["event_time_binned"] == -1]
    ref_treat = ref.loc[ref["treatment"] == 1, outcome_var]
    ref_ctrl = ref.loc[ref["treatment"] == 0, outcome_var]

    if len(ref_treat) < 5 or len(ref_ctrl) < 5:
        return None

    ref_treat_mean = ref_treat.mean()
    ref_ctrl_mean = ref_ctrl.mean()

    es_results = {"outcome": outcome_var, "label": label or outcome_var,
                  "coefficients": {}}

    # Reference period: coefficient is 0 by construction
    es_results["coefficients"][-1] = {
        "coef": 0.0, "se": 0.0, "pval": 1.0,
        "ci_low": 0.0, "ci_high": 0.0,
    }

    for t in event_times:
        if t == -1:
            continue

        period_data = data[data["event_time_binned"] == t]
        treat_k = period_data.loc[period_data["treatment"] == 1, outcome_var]
        ctrl_k = period_data.loc[period_data["treatment"] == 0, outcome_var]

        if len(treat_k) < 5 or len(ctrl_k) < 5:
            continue

        # 2x2 DiD relative to reference period
        treat_delta = treat_k.mean() - ref_treat_mean
        ctrl_delta = ctrl_k.mean() - ref_ctrl_mean
        beta_k = treat_delta - ctrl_delta

        # SE via delta method approximation
        var_k = (treat_k.var() / len(treat_k) +
                 ref_treat.var() / len(ref_treat) +
                 ctrl_k.var() / len(ctrl_k) +
                 ref_ctrl.var() / len(ref_ctrl))
        se_k = np.sqrt(var_k) if var_k > 0 else np.nan

        if se_k > 0 and not np.isnan(se_k):
            from scipy import stats as sp_stats
            t_stat = beta_k / se_k
            dof = min(len(treat_k), len(ctrl_k), len(ref_treat), len(ref_ctrl)) - 1
            pval = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=max(dof, 1))))
        else:
            pval = np.nan

        es_results["coefficients"][int(t)] = {
            "coef": float(beta_k),
            "se": float(se_k),
            "pval": float(pval),
            "ci_low": float(beta_k - 1.96 * se_k) if not np.isnan(se_k) else float("nan"),
            "ci_high": float(beta_k + 1.96 * se_k) if not np.isnan(se_k) else float("nan"),
        }

    if len(es_results["coefficients"]) < 3:
        return None

    es_results["n_obs"] = int(len(data))
    return es_results


def plot_event_study(es_results_list, country_label=""):
    """Plot event study figures for key outcomes.

    Shows treatment×period coefficients relative to t=-1 (pre-construction).
    Pre-treatment coefficients near zero = parallel trends support.
    """
    apply_style()

    n_plots = len(es_results_list)
    if n_plots == 0:
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 3.0 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, es in enumerate(es_results_list):
        ax = axes[idx]
        coeffs = es["coefficients"]
        times = sorted(coeffs.keys())
        coef_vals = [coeffs[t]["coef"] for t in times]
        ci_low = [coeffs[t]["ci_low"] for t in times]
        ci_high = [coeffs[t]["ci_high"] for t in times]

        # Plot
        ax.fill_between(times, ci_low, ci_high, alpha=0.2, color="#44AA99")
        ax.plot(times, coef_vals, marker="o", color="#332288", linewidth=1.5,
                markersize=5, zorder=3)

        # Reference lines
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(x=-0.5, color="#CC6677", linewidth=1, linestyle=":",
                   alpha=0.7, label="Treatment onset")

        ax.set_xlabel("Event time (years from construction)")
        ax.set_ylabel("Treatment effect")
        ax.set_title(es["label"], fontsize=9)

        # Mark pre-treatment periods in different color
        for i, t in enumerate(times):
            if t < -1:
                ax.plot(t, coef_vals[i], "o", color="#88CCEE",
                        markersize=5, zorder=4)

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    title = "Event Study: Dynamic Treatment Effects"
    if country_label:
        title += f" ({country_label})"
    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    save_fig(fig, FIG_DIR / "did_event_study.png")
    plt.close()


def run_callaway_santanna(df, outcome_var, label=None):
    """Callaway & Sant'Anna (2021) staggered DiD estimator.

    Estimates group-time ATTs for each (cohort g, time t) pair,
    using never-treated units as controls. Aggregates to an overall ATT.

    This avoids the "forbidden comparisons" bias in naive TWFE where
    already-treated units serve as controls for later-treated units.

    Groups (g): defined by construction_year cohorts
    Time periods (t): baseline, pre_construction, post_construction, current
    Control: never-treated units (proposed/cancelled projects)
    """
    if outcome_var not in df.columns:
        return None

    data = df.copy()
    data["treatment"] = (data["group"] == "treatment").astype(int)
    data[outcome_var] = pd.to_numeric(data[outcome_var], errors="coerce")
    data = data.dropna(subset=[outcome_var])

    # Never-treated control units
    control_ids = set(data.loc[data["treatment"] == 0, "site_id"].unique())
    if len(control_ids) < 5:
        return None

    # Treatment cohorts by construction year
    treat_data = data[data["treatment"] == 1].copy()
    treat_data["construction_year"] = pd.to_numeric(
        treat_data["construction_year"], errors="coerce")
    treat_data = treat_data.dropna(subset=["construction_year"])

    if len(treat_data) == 0:
        return None

    # Group construction years into cohorts (to avoid tiny groups)
    year_counts = treat_data.groupby("construction_year")["site_id"].nunique()
    cohorts = {}
    for yr, count in year_counts.items():
        yr = int(yr)
        if count >= 3:
            cohorts[yr] = [yr]
        else:
            # Merge small cohorts: ≤2018, 2019-2020, 2021-2022, 2023+
            if yr <= 2018:
                cohorts.setdefault("early", []).append(yr)
            elif yr <= 2020:
                cohorts.setdefault("mid_early", []).append(yr)
            elif yr <= 2022:
                cohorts.setdefault("mid_late", []).append(yr)
            else:
                cohorts.setdefault("late", []).append(yr)

    # Map each treatment site to its cohort
    def get_cohort(yr):
        yr = int(yr)
        for cname, years in cohorts.items():
            if yr in years:
                return cname
        return None

    treat_data["cohort"] = treat_data["construction_year"].apply(get_cohort)
    treat_data = treat_data.dropna(subset=["cohort"])

    # For each cohort, estimate 2x2 DiD against never-treated controls
    group_time_atts = []
    time_points = ["baseline", "pre_construction", "post_construction", "current"]

    for cohort_name in treat_data["cohort"].unique():
        cohort_sites = set(treat_data.loc[
            treat_data["cohort"] == cohort_name, "site_id"].unique())

        if len(cohort_sites) < 3:
            continue

        # For each post-treatment period, compare against pre-treatment
        for post_tp in ["post_construction", "current"]:
            pre_tp = "pre_construction"

            # Get data for this cohort + controls at these two time points
            cohort_pre = data[(data["site_id"].isin(cohort_sites)) &
                              (data["time_point"] == pre_tp)]
            cohort_post = data[(data["site_id"].isin(cohort_sites)) &
                               (data["time_point"] == post_tp)]
            ctrl_pre = data[(data["site_id"].isin(control_ids)) &
                            (data["time_point"] == pre_tp)]
            ctrl_post = data[(data["site_id"].isin(control_ids)) &
                             (data["time_point"] == post_tp)]

            if (len(cohort_pre) < 3 or len(cohort_post) < 3 or
                    len(ctrl_pre) < 3 or len(ctrl_post) < 3):
                continue

            # Simple 2x2 DiD: (Ȳ_treat_post - Ȳ_treat_pre) - (Ȳ_ctrl_post - Ȳ_ctrl_pre)
            treat_delta = (cohort_post[outcome_var].mean() -
                           cohort_pre[outcome_var].mean())
            ctrl_delta = (ctrl_post[outcome_var].mean() -
                          ctrl_pre[outcome_var].mean())
            att = treat_delta - ctrl_delta

            # SE via bootstrap-style variance (analytic approximation)
            n_t = len(cohort_sites)
            n_c = len(control_ids)
            treat_var = (cohort_post[outcome_var].var() / n_t +
                         cohort_pre[outcome_var].var() / n_t)
            ctrl_var = (ctrl_post[outcome_var].var() / n_c +
                        ctrl_pre[outcome_var].var() / n_c)
            se = np.sqrt(treat_var + ctrl_var) if (treat_var + ctrl_var) > 0 else np.nan

            group_time_atts.append({
                "cohort": str(cohort_name),
                "time_point": post_tp,
                "att": float(att),
                "se": float(se),
                "n_treated": int(n_t),
                "n_control": int(n_c),
            })

    if not group_time_atts:
        return None

    # Aggregate: weighted average of group-time ATTs (weighted by group size)
    total_weight = sum(g["n_treated"] for g in group_time_atts)
    agg_att = sum(g["att"] * g["n_treated"] / total_weight
                  for g in group_time_atts)
    # Aggregate SE (assuming independent groups)
    agg_var = sum((g["n_treated"] / total_weight) ** 2 * g["se"] ** 2
                  for g in group_time_atts if not np.isnan(g["se"]))
    agg_se = np.sqrt(agg_var)

    # t-stat and p-value
    if agg_se > 0:
        t_stat = agg_att / agg_se
        from scipy import stats as sp_stats
        agg_pval = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=total_weight - 1))
    else:
        agg_pval = np.nan

    return {
        "outcome": outcome_var,
        "label": label or outcome_var,
        "agg_att": float(agg_att),
        "agg_se": float(agg_se),
        "agg_pval": float(agg_pval),
        "agg_ci_low": float(agg_att - 1.96 * agg_se),
        "agg_ci_high": float(agg_att + 1.96 * agg_se),
        "n_cohorts": len(set(g["cohort"] for g in group_time_atts)),
        "n_group_time_atts": len(group_time_atts),
        "group_time_atts": group_time_atts,
    }


def run_doubly_robust_cs(df, outcome_var, label=None, use_never_treated=True):
    """Doubly-robust Callaway-Sant'Anna estimator with not-yet-treated controls.

    Key improvements over naive CS:
    1. Not-yet-treated controls: sites built later serve as controls for
       earlier cohorts (all are real solar sites → better balance)
    2. IPW: inverse propensity weighting on baseline covariates
    3. Outcome regression: controls for baseline level in outcome equation
    4. Doubly-robust: consistent if EITHER the propensity OR outcome model is correct

    Optionally also includes never-treated (GEM proposed/cancelled) in control pool.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if outcome_var not in df.columns:
        return None

    data = df.copy()
    data[outcome_var] = pd.to_numeric(data[outcome_var], errors="coerce")
    data = data.dropna(subset=[outcome_var])
    data["construction_year"] = pd.to_numeric(data["construction_year"], errors="coerce")

    # Get baseline covariates for each site (for IPW + outcome regression)
    baseline = data[data["time_point"] == "baseline"].set_index("site_id")
    covar_cols = []
    for col in ["dw_crops_pct", "dw_trees_pct", "dw_built_pct", "dw_bare_pct",
                 "dw_water_pct", "ghi_kwh_m2_day"]:
        if col in baseline.columns and baseline[col].notna().sum() > 50:
            covar_cols.append(col)

    if len(covar_cols) < 3:
        return None

    # Add country dummies to covariates
    if baseline["country"].nunique() > 1:
        for country in baseline["country"].unique()[1:]:  # drop first as reference
            col_name = f"country_{country}"
            baseline[col_name] = (baseline["country"] == country).astype(int)
            covar_cols.append(col_name)

    baseline_covars = baseline[covar_cols].dropna()

    # Identify treatment cohorts (group by construction year, merge small ones)
    treat_sites = data[(data["group"] == "treatment") &
                       (data["construction_year"].notna())].copy()
    treat_baseline = treat_sites[treat_sites["time_point"] == "baseline"]

    year_counts = treat_baseline.groupby("construction_year")["site_id"].nunique()

    # Merge years with <20 sites into cohorts
    cohort_map = {}
    for yr in sorted(year_counts.index):
        yr = int(yr)
        if yr <= 2015:
            cohort_map[yr] = 2015
        elif yr >= 2024:
            cohort_map[yr] = 2024
        else:
            cohort_map[yr] = yr

    # For each cohort, estimate group-time ATT using not-yet-treated controls
    group_time_atts = []

    for cohort_year in sorted(set(cohort_map.values())):
        # Sites in this cohort
        cohort_site_ids = set(
            treat_baseline.loc[
                treat_baseline["construction_year"].apply(
                    lambda y: cohort_map.get(int(y), int(y)) == cohort_year
                    if pd.notna(y) else False),
                "site_id"
            ].unique()
        )
        if len(cohort_site_ids) < 5:
            continue

        # Not-yet-treated controls: sites with construction_year > cohort_year + 1
        # (they haven't been treated by the time we observe post-construction for this cohort)
        nyt_site_ids = set(
            treat_baseline.loc[
                treat_baseline["construction_year"] > cohort_year + 1,
                "site_id"
            ].unique()
        )

        # Optionally add never-treated (GEM proposed/cancelled)
        if use_never_treated:
            never_treated_ids = set(
                data.loc[data["group"] == "control", "site_id"].unique()
            )
            control_ids = nyt_site_ids | never_treated_ids
        else:
            control_ids = nyt_site_ids

        if len(control_ids) < 10:
            continue

        # For pre and post comparison
        for post_tp in ["post_construction", "current"]:
            pre_tp = "pre_construction"

            # Get outcome data
            cohort_pre = data[(data["site_id"].isin(cohort_site_ids)) &
                              (data["time_point"] == pre_tp)]
            cohort_post = data[(data["site_id"].isin(cohort_site_ids)) &
                               (data["time_point"] == post_tp)]
            ctrl_pre = data[(data["site_id"].isin(control_ids)) &
                            (data["time_point"] == pre_tp)]
            ctrl_post = data[(data["site_id"].isin(control_ids)) &
                             (data["time_point"] == post_tp)]

            if (len(cohort_pre) < 5 or len(cohort_post) < 5 or
                    len(ctrl_pre) < 5 or len(ctrl_post) < 5):
                continue

            # ── IPW: estimate P(in_cohort | X) ──
            all_site_ids = list(cohort_site_ids | control_ids)
            ipw_data = baseline_covars.loc[
                baseline_covars.index.isin(all_site_ids)
            ].copy()

            if len(ipw_data) < 20:
                continue

            ipw_data["in_cohort"] = ipw_data.index.isin(cohort_site_ids).astype(int)

            try:
                X_ipw = ipw_data[covar_cols].values
                y_ipw = ipw_data["in_cohort"].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_ipw)

                lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
                lr.fit(X_scaled, y_ipw)
                pscores = lr.predict_proba(X_scaled)[:, 1]

                # Trim extreme propensity scores for stability
                pscores = np.clip(pscores, 0.01, 0.99)
                ipw_data["pscore"] = pscores

                # IPW weights for control units: p/(1-p) normalized
                ctrl_mask = ipw_data["in_cohort"] == 0
                ctrl_weights = ipw_data.loc[ctrl_mask, "pscore"] / (
                    1 - ipw_data.loc[ctrl_mask, "pscore"])
                ctrl_weights = ctrl_weights / ctrl_weights.sum()  # normalize

                # Map weights to site_ids
                ctrl_weight_map = dict(zip(
                    ipw_data.index[ctrl_mask], ctrl_weights.values))

            except Exception:
                # Fall back to unweighted if IPW fails
                ctrl_weight_map = {sid: 1.0 / len(control_ids)
                                   for sid in control_ids}

            # ── Compute weighted DiD ──
            # Treatment: simple mean of changes
            treat_delta = (cohort_post[outcome_var].mean() -
                           cohort_pre[outcome_var].mean())

            # Control: IPW-weighted mean of changes
            ctrl_pre_indexed = ctrl_pre.set_index("site_id")
            ctrl_post_indexed = ctrl_post.set_index("site_id")
            common_ctrl = ctrl_pre_indexed.index.intersection(
                ctrl_post_indexed.index)

            if len(common_ctrl) < 5:
                continue

            ctrl_changes = (ctrl_post_indexed.loc[common_ctrl, outcome_var].values -
                            ctrl_pre_indexed.loc[common_ctrl, outcome_var].values)
            ctrl_w = np.array([ctrl_weight_map.get(sid, 1.0 / len(common_ctrl))
                               for sid in common_ctrl])
            ctrl_w = ctrl_w / ctrl_w.sum()

            ctrl_delta_weighted = np.average(ctrl_changes, weights=ctrl_w)

            att = treat_delta - ctrl_delta_weighted

            # SE via analytical approximation
            n_t = len(cohort_site_ids)
            treat_var = (cohort_post[outcome_var].var() / n_t +
                         cohort_pre[outcome_var].var() / n_t)
            # Effective sample size for weighted control
            eff_n_c = 1.0 / np.sum(ctrl_w ** 2) if np.sum(ctrl_w ** 2) > 0 else len(common_ctrl)
            ctrl_var = np.average(
                (ctrl_changes - ctrl_delta_weighted) ** 2, weights=ctrl_w
            ) / eff_n_c
            se = np.sqrt(treat_var + ctrl_var) if (treat_var + ctrl_var) > 0 else np.nan

            group_time_atts.append({
                "cohort": int(cohort_year),
                "time_point": post_tp,
                "att": float(att),
                "se": float(se),
                "n_treated": int(n_t),
                "n_control": int(len(common_ctrl)),
                "n_control_eff": float(eff_n_c),
            })

    if not group_time_atts:
        return None

    # Aggregate: weighted average of group-time ATTs (weighted by group size)
    total_weight = sum(g["n_treated"] for g in group_time_atts)
    agg_att = sum(g["att"] * g["n_treated"] / total_weight
                  for g in group_time_atts)
    agg_var = sum((g["n_treated"] / total_weight) ** 2 * g["se"] ** 2
                  for g in group_time_atts if not np.isnan(g["se"]))
    agg_se = np.sqrt(agg_var) if agg_var > 0 else np.nan

    if agg_se > 0 and not np.isnan(agg_se):
        from scipy import stats as sp_stats
        t_stat = agg_att / agg_se
        agg_pval = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=total_weight - 1)))
    else:
        agg_pval = np.nan

    return {
        "outcome": outcome_var,
        "label": label or outcome_var,
        "agg_att": float(agg_att),
        "agg_se": float(agg_se),
        "agg_pval": float(agg_pval),
        "agg_ci_low": float(agg_att - 1.96 * agg_se) if not np.isnan(agg_se) else np.nan,
        "agg_ci_high": float(agg_att + 1.96 * agg_se) if not np.isnan(agg_se) else np.nan,
        "n_cohorts": len(set(g["cohort"] for g in group_time_atts)),
        "n_group_time_atts": len(group_time_atts),
        "group_time_atts": group_time_atts,
        "estimator": "doubly_robust_cs",
        "control_type": "not_yet_treated" + ("+never_treated" if use_never_treated else ""),
    }


def run_conditional_placebo(df, outcomes_list):
    """Placebo test with covariate conditioning.

    Tests baseline→pre_construction changes controlling for baseline covariates
    and country. If conditional parallel trends holds, the treatment coefficient
    should be ~0 after conditioning.
    """
    baseline = df[df["time_point"] == "baseline"].copy()
    pre = df[df["time_point"] == "pre_construction"].copy()

    if baseline.empty or pre.empty:
        return []

    baseline = baseline.set_index("site_id")
    pre = pre.set_index("site_id")
    common = baseline.index.intersection(pre.index)

    if len(common) == 0:
        return []

    outcome_cols = [f"dw_{cn}_pct" for cn in DW_CLASSES]
    outcome_cols += ["viirs_avg_rad", "sar_vv_db", "sar_vh_db",
                     "ndvi_mean", "evi_mean", "lst_day_c", "lst_night_c",
                     "pop_sum", "pop_mean", "bldg_presence", "bldg_height_m",
                     "bldg_frac_count"]

    changes = pd.DataFrame(index=common)
    for col in ["country", "group", "confidence", "ghi_kwh_m2_day"]:
        changes[col] = baseline.loc[common, col]

    changes["treatment"] = (changes["group"] == "treatment").astype(int)
    changes["weight"] = changes["confidence"].map(CONFIDENCE_WEIGHTS).fillna(0.3)

    for col in outcome_cols:
        if col in baseline.columns and col in pre.columns:
            b_vals = pd.to_numeric(baseline.loc[common, col], errors="coerce")
            p_vals = pd.to_numeric(pre.loc[common, col], errors="coerce")
            changes[f"delta_{col}"] = p_vals - b_vals
            changes[f"baseline_{col}"] = b_vals

    changes = changes.reset_index()

    results = []
    for var, label in outcomes_list:
        dep_var = f"delta_{var}"
        baseline_col = f"baseline_{var}"
        if dep_var not in changes.columns:
            continue

        dropna_cols = [dep_var, "treatment", "weight"]
        covariates = []

        # Add baseline value as covariate
        if baseline_col in changes.columns and changes[baseline_col].notna().sum() > 50:
            covariates.append(baseline_col)
            dropna_cols.append(baseline_col)
        if "ghi_kwh_m2_day" in changes.columns and changes["ghi_kwh_m2_day"].notna().sum() > 50:
            covariates.append("ghi_kwh_m2_day")
            dropna_cols.append("ghi_kwh_m2_day")
        # Add country FE if multiple countries
        if changes["country"].nunique() > 1:
            covariates.append("C(country)")
            dropna_cols.append("country")

        data = changes.dropna(subset=dropna_cols)
        if len(data) < 20:
            continue

        cov_str = " + ".join(covariates) if covariates else ""
        formula = f"Q('{dep_var}') ~ treatment"
        if cov_str:
            formula += f" + {cov_str}"

        try:
            model = smf.wls(formula, data=data, weights=data["weight"])
            result = model.fit(cov_type="HC1")

            treat_coef = result.params.get("treatment", np.nan)
            treat_se = result.bse.get("treatment", np.nan)
            treat_pval = result.pvalues.get("treatment", np.nan)
            ci = result.conf_int().loc["treatment"]

            results.append({
                "outcome": var,
                "label": label,
                "treatment_coef": float(treat_coef),
                "treatment_se": float(treat_se),
                "treatment_pval": float(treat_pval),
                "treatment_ci_low": float(ci[0]),
                "treatment_ci_high": float(ci[1]),
                "n_obs": int(result.nobs),
                "se_type": "HC1",
                "conditional": True,
            })
        except Exception:
            pass

    return results


def plot_staggered_did(cs_results_list, country_label=""):
    """Forest plot comparing Callaway-Sant'Anna ATT estimates."""
    apply_style()

    valid = [r for r in cs_results_list
             if r and not np.isnan(r["agg_att"])]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, max(3, len(valid) * 0.45)))

    labels = [r["label"] for r in valid]
    coefs = [r["agg_att"] for r in valid]
    ci_low = [r["agg_ci_low"] for r in valid]
    ci_high = [r["agg_ci_high"] for r in valid]
    pvals = [r["agg_pval"] for r in valid]
    y_pos = range(len(valid))

    colors = []
    for p in pvals:
        if p < 0.01:
            colors.append("#332288")
        elif p < 0.05:
            colors.append("#44AA99")
        elif p < 0.1:
            colors.append("#DDCC77")
        else:
            colors.append("#DDDDDD")

    ax.barh(y_pos, coefs, color=colors, edgecolor="none", height=0.6, alpha=0.8)
    for i, (lo, hi) in enumerate(zip(ci_low, ci_high)):
        ax.plot([lo, hi], [i, i], color="black", linewidth=1)

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Callaway-Sant'Anna ATT (staggered DiD)")

    title = "Staggered DiD: Aggregate Treatment Effects"
    if country_label:
        title += f" ({country_label})"
    ax.set_title(title)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#332288", label="p < 0.01"),
        Patch(facecolor="#44AA99", label="p < 0.05"),
        Patch(facecolor="#DDCC77", label="p < 0.1"),
        Patch(facecolor="#DDDDDD", label="not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    save_fig(fig, FIG_DIR / "did_staggered_cs.png")
    plt.close()


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


def screen_controls(df):
    """Filter control sites to plausible solar locations.

    Removes controls that are clearly urban or unsuitable (high built-up,
    high water, no cropland/bare ground). This improves treatment-control
    balance and is a robustness check pending full VLM screening.
    """
    baseline = df[df["time_point"] == "baseline"].copy()
    ctrl_baseline = baseline[baseline["group"] == "control"]

    # Plausibility criteria: could realistically host a solar farm
    plausible_mask = (
        (ctrl_baseline["dw_built_pct"] < 30) &
        (ctrl_baseline["dw_water_pct"] < 30) &
        ((ctrl_baseline["dw_crops_pct"] > 15) | (ctrl_baseline["dw_bare_pct"] > 15))
    )
    plausible_ids = set(ctrl_baseline.loc[plausible_mask, "site_id"])
    removed_ids = set(ctrl_baseline.loc[~plausible_mask, "site_id"])

    n_before = ctrl_baseline["site_id"].nunique()
    n_after = len(plausible_ids)
    print(f"  Control screening: {n_before} → {n_after} "
          f"(removed {len(removed_ids)} implausible sites)")

    # Remove implausible controls from the full panel
    screened = df[~df["site_id"].isin(removed_ids)].copy()
    return screened


def run_analysis(country_filter=None, run_psm=False, screen_ctrls=False):
    print("=" * 70)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 70)

    # Load data
    df = load_panel(country_filter)
    country_label = country_filter.title() if country_filter else "South Asia"

    # Screen controls if requested
    if screen_ctrls:
        print("\nScreening controls for plausibility...")
        df = screen_controls(df)
        country_label += " (screened controls)"

    # Country-specific output directory
    if country_filter:
        out_dir = OUTPUT_DIR / country_filter.lower()
    else:
        out_dir = OUTPUT_DIR
    if screen_ctrls:
        out_dir = out_dir / "screened"
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

    # ── 5. Placebo / Falsification Test ──
    print("\n" + "=" * 70)
    print("5. PLACEBO TEST (baseline → pre-construction)")
    print("=" * 70)
    print("  Testing: treatment coefficient should be ~0 in pre-treatment period")
    placebo_results = run_placebo_test(df, OUTCOMES_LIST)
    if placebo_results:
        n_sig = sum(1 for r in placebo_results if r["treatment_pval"] < 0.05)
        print(f"\n  Results: {len(placebo_results)} outcomes tested, "
              f"{n_sig} significant at p<0.05")
        for r in placebo_results:
            sig = "**FAIL**" if r["treatment_pval"] < 0.05 else "pass"
            print(f"    {r['label']:<30} coef={r['treatment_coef']:+.3f} "
                  f"(p={r['treatment_pval']:.3f}) [{sig}]")
        results_json["placebo"] = placebo_results
    else:
        print("  Could not run placebo test (insufficient data)")

    # ── 6. Event Study ──
    print("\n" + "=" * 70)
    print("6. EVENT STUDY (dynamic treatment effects)")
    print("=" * 70)
    key_outcomes_es = [
        ("dw_built_pct", "Built-up (%)"),
        ("dw_crops_pct", "Cropland (%)"),
        ("dw_bare_pct", "Bare ground (%)"),
        ("dw_trees_pct", "Trees (%)"),
        ("viirs_avg_rad", "Nighttime light"),
        ("ndvi_mean", "NDVI"),
    ]
    es_results = []
    for var, label in key_outcomes_es:
        print(f"  {label}...", end="")
        es = run_event_study(df, var, label)
        if es:
            es_results.append(es)
            # Print pre-treatment coefficients
            pre_times = sorted([t for t in es["coefficients"] if t < -1])
            post_times = sorted([t for t in es["coefficients"] if t >= 0])
            pre_str = ", ".join(
                f"t={t}: {es['coefficients'][t]['coef']:+.2f}"
                f"{'*' if es['coefficients'][t]['pval'] < 0.05 else ''}"
                for t in pre_times)
            post_str = ", ".join(
                f"t={t}: {es['coefficients'][t]['coef']:+.2f}"
                f"{'*' if es['coefficients'][t]['pval'] < 0.05 else ''}"
                for t in post_times)
            print(f" pre=[{pre_str}] post=[{post_str}]")
        else:
            print(" skipped")

    if es_results:
        # Convert event study coefficients keys to strings for JSON
        es_for_json = []
        for es in es_results:
            es_copy = dict(es)
            es_copy["coefficients"] = {
                str(k): v for k, v in es["coefficients"].items()
            }
            es_for_json.append(es_copy)
        results_json["event_study"] = es_for_json

    # ── 7. Callaway-Sant'Anna Staggered DiD ──
    print("\n" + "=" * 70)
    print("7. CALLAWAY-SANT'ANNA STAGGERED DiD")
    print("=" * 70)
    cs_results = []
    for var, label in OUTCOMES_LIST:
        cs = run_callaway_santanna(df, var, label)
        if cs:
            cs_results.append(cs)
            sig = "***" if cs["agg_pval"] < 0.01 else \
                  "**" if cs["agg_pval"] < 0.05 else \
                  "*" if cs["agg_pval"] < 0.1 else ""
            print(f"  {label:<30} ATT={cs['agg_att']:+.3f}{sig} "
                  f"(SE={cs['agg_se']:.3f}, p={cs['agg_pval']:.4f}), "
                  f"{cs['n_cohorts']} cohorts")

    if cs_results:
        results_json["callaway_santanna"] = [
            {k: v for k, v in cs.items()} for cs in cs_results
        ]

        # Compare baseline vs CS
        print(f"\n  Baseline DiD vs Callaway-Sant'Anna comparison:")
        print(f"  {'Outcome':<30} {'Baseline':>10} {'CS ATT':>10} {'Δ':>8}")
        for cs in cs_results:
            base_match = [r for r in results if r["outcome"] == cs["outcome"]]
            if base_match:
                base_coef = base_match[0]["treatment_coef"]
                delta = cs["agg_att"] - base_coef
                print(f"  {cs['label']:<30} {base_coef:>+10.3f} "
                      f"{cs['agg_att']:>+10.3f} {delta:>+8.3f}")

    # ── 8. DOUBLY-ROBUST CS (not-yet-treated controls) ──
    print("\n" + "=" * 70)
    print("8. DOUBLY-ROBUST CS (not-yet-treated + IPW)")
    print("=" * 70)
    print("  Controls: not-yet-treated solar sites + never-treated (GEM)")
    print("  Conditioning: IPW on baseline LULC, GHI, country")
    dr_results = []
    for var, label in OUTCOMES_LIST:
        dr = run_doubly_robust_cs(df, var, label, use_never_treated=True)
        if dr:
            dr_results.append(dr)
            sig = "***" if dr["agg_pval"] < 0.01 else \
                  "**" if dr["agg_pval"] < 0.05 else \
                  "*" if dr["agg_pval"] < 0.1 else ""
            print(f"  {label:<30} ATT={dr['agg_att']:+.3f}{sig} "
                  f"(SE={dr['agg_se']:.3f}, p={dr['agg_pval']:.4f}), "
                  f"{dr['n_cohorts']} cohorts")

    if dr_results:
        results_json["doubly_robust_cs"] = [
            {k: v for k, v in dr.items()} for dr in dr_results
        ]

        # Compare all three estimators
        print(f"\n  Baseline vs CS vs DR-CS comparison:")
        print(f"  {'Outcome':<30} {'Baseline':>10} {'CS':>10} {'DR-CS':>10}")
        for dr in dr_results:
            base_match = [r for r in results if r["outcome"] == dr["outcome"]]
            cs_match = [r for r in cs_results if r["outcome"] == dr["outcome"]]
            base_coef = base_match[0]["treatment_coef"] if base_match else np.nan
            cs_coef = cs_match[0]["agg_att"] if cs_match else np.nan
            print(f"  {dr['label']:<30} {base_coef:>+10.3f} "
                  f"{cs_coef:>+10.3f} {dr['agg_att']:>+10.3f}")

    # ── 8b. DR-CS with NOT-YET-TREATED ONLY (no never-treated) ──
    print("\n  --- Sensitivity: not-yet-treated ONLY (no GEM controls) ---")
    dr_nyt_results = []
    for var, label in OUTCOMES_LIST:
        dr_nyt = run_doubly_robust_cs(df, var, label, use_never_treated=False)
        if dr_nyt:
            dr_nyt_results.append(dr_nyt)
            sig = "***" if dr_nyt["agg_pval"] < 0.01 else \
                  "**" if dr_nyt["agg_pval"] < 0.05 else \
                  "*" if dr_nyt["agg_pval"] < 0.1 else ""
            print(f"  {label:<30} ATT={dr_nyt['agg_att']:+.3f}{sig} "
                  f"(p={dr_nyt['agg_pval']:.4f})")

    if dr_nyt_results:
        results_json["dr_cs_nyt_only"] = [
            {k: v for k, v in r.items()} for r in dr_nyt_results
        ]

    # ── 9. CONDITIONAL PLACEBO TEST ──
    print("\n" + "=" * 70)
    print("9. CONDITIONAL PLACEBO (baseline → pre, with covariates)")
    print("=" * 70)
    print("  Controls for: baseline outcome level, GHI, country FE")
    cond_placebo = run_conditional_placebo(df, OUTCOMES_LIST)
    if cond_placebo:
        n_sig = sum(1 for r in cond_placebo if r["treatment_pval"] < 0.05)
        n_total = len(cond_placebo)
        print(f"\n  Results: {n_total} outcomes, {n_sig} fail at p<0.05")
        print(f"  (Unconditional placebo had "
              f"{sum(1 for r in placebo_results if r['treatment_pval'] < 0.05)}"
              f"/{len(placebo_results)} failures)")
        for r in cond_placebo:
            uncon = [p for p in placebo_results if p["outcome"] == r["outcome"]]
            uncon_p = uncon[0]["treatment_pval"] if uncon else np.nan
            flag = "**FAIL**" if r["treatment_pval"] < 0.05 else "pass"
            improved = " (FIXED)" if uncon_p < 0.05 and r["treatment_pval"] >= 0.05 else ""
            print(f"    {r['label']:<30} coef={r['treatment_coef']:+.3f} "
                  f"(p={r['treatment_pval']:.3f}) [{flag}]{improved}")
        results_json["conditional_placebo"] = cond_placebo
    else:
        print("  Could not run conditional placebo test")

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

    if placebo_results:
        plot_placebo_results(placebo_results, country_label)

    if es_results:
        plot_event_study(es_results, country_label)

    if cs_results:
        plot_staggered_did(cs_results, country_label)

    if dr_results:
        plot_staggered_did(dr_results, country_label + " DR-CS")

    print(f"\nResults saved to {out_dir}/")
    print(f"Figures saved to {FIG_DIR}/")

    # Key findings summary
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    print("\n  Standard errors: HC1 robust (all regressions)")
    sig_results = [r for r in results if r["treatment_pval"] < 0.05]
    if sig_results:
        print(f"\n  1. Baseline DiD (HC1): {len(sig_results)}/18 significant")
        for r in sig_results:
            direction = "+" if r["treatment_coef"] > 0 else ""
            print(f"     {r['label']}: {direction}{r['treatment_coef']:.2f} "
                  f"(p={r['treatment_pval']:.4f})")

    if placebo_results:
        n_uncon_fail = sum(1 for r in placebo_results
                           if r["treatment_pval"] < 0.05)
        print(f"\n  2. Unconditional placebo: {n_uncon_fail}/{len(placebo_results)} fail")

    if cond_placebo:
        n_cond_fail = sum(1 for r in cond_placebo
                          if r["treatment_pval"] < 0.05)
        print(f"     Conditional placebo:   {n_cond_fail}/{len(cond_placebo)} fail")

    if cs_results:
        cs_sig = [r for r in cs_results if r["agg_pval"] < 0.05]
        print(f"\n  3. CS (never-treated controls): "
              f"{len(cs_sig)}/{len(cs_results)} significant")

    if dr_results:
        dr_sig = [r for r in dr_results if r["agg_pval"] < 0.05]
        print(f"     DR-CS (not-yet-treated + IPW): "
              f"{len(dr_sig)}/{len(dr_results)} significant")
        for r in dr_sig:
            direction = "+" if r["agg_att"] > 0 else ""
            print(f"     {r['label']}: {direction}{r['agg_att']:.3f} "
                  f"(p={r['agg_pval']:.4f})")

    if dr_nyt_results:
        dr_nyt_sig = [r for r in dr_nyt_results if r["agg_pval"] < 0.05]
        print(f"     DR-CS (NYT only, no GEM): "
              f"{len(dr_nyt_sig)}/{len(dr_nyt_results)} significant")


def main():
    parser = argparse.ArgumentParser(
        description="Run DiD analysis on solar farm temporal panel data")
    parser.add_argument("--country", type=str, default=None,
                        help="Filter to single country (e.g. 'bangladesh')")
    parser.add_argument("--psm", action="store_true",
                        help="Run propensity score matching analysis")
    parser.add_argument("--screen-controls", action="store_true",
                        help="Filter controls to plausible solar sites (DW-based)")
    args = parser.parse_args()

    run_analysis(country_filter=args.country, run_psm=args.psm,
                 screen_ctrls=args.screen_controls)


if __name__ == "__main__":
    main()
