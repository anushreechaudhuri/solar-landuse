"""
Analyze full-dataset VLM classification results from Gemini 2.5 Flash.

Loads per-image JSON results, merges with site metadata and DW panel,
produces solar detection statistics, cross-validation with DW, and
publication-ready figures.

Usage:
    python scripts/analyze_vlm_results.py
    python scripts/analyze_vlm_results.py --skip-figures
    python scripts/analyze_vlm_results.py --vlm-dir data/vlm_results --output-dir docs/figures/vlm

Download results from Modal first:
    modal volume get solar-landuse-data vlm_results/ data/vlm_results/
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from figure_style import apply_style, save_fig, DATASET_COLORS, LULC_COLORS, FULL_WIDTH, HALF_WIDTH

# ── Constants ────────────────────────────────────────────────────────────────

LULC_FIELDS = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice", "solar_panels",
]

DETECTION_THRESHOLDS = [1, 2, 5, 10, 20]

CAPACITY_TIERS = [
    ("<10 MW", 0, 10),
    ("10-50 MW", 10, 50),
    ("50-200 MW", 50, 200),
    (">200 MW", 200, float("inf")),
]

# Map VLM field names to DW column names for cross-validation
VLM_TO_DW = {
    "crops": "dw_crops_pct",
    "trees": "dw_trees_pct",
    "built": "dw_built_pct",
    "water": "dw_water_pct",
    "bare": "dw_bare_pct",
    "grass": "dw_grass_pct",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_vlm_results(vlm_dir):
    """Load all VLM JSON result files, filtering out errors.

    Returns:
        df: DataFrame with site_id, year, LULC percentages, solar_visible, description
        n_total: total files found
        n_errors: files with errors
        error_patterns: dict of error type -> count
    """
    files = sorted(glob.glob(os.path.join(vlm_dir, "*.json")))
    if not files:
        return None, 0, 0, {}

    records = []
    n_errors = 0
    error_patterns = {}

    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            n_errors += 1
            err_key = type(e).__name__
            error_patterns[err_key] = error_patterns.get(err_key, 0) + 1
            continue

        if "error" in data:
            n_errors += 1
            err_msg = str(data["error"])
            # Truncate long error messages for grouping
            err_key = err_msg[:80] if len(err_msg) > 80 else err_msg
            error_patterns[err_key] = error_patterns.get(err_key, 0) + 1
            continue

        record = {
            "site_id": data.get("site_id", ""),
            "year": int(data.get("year", 0)),
            "model": data.get("model", ""),
            "solar_visible": bool(data.get("solar_visible", False)),
            "description": data.get("description", ""),
        }
        for field in LULC_FIELDS:
            val = data.get(field, 0)
            try:
                record[field] = float(val)
            except (ValueError, TypeError):
                record[field] = 0.0

        records.append(record)

    if not records:
        return None, len(files), n_errors, error_patterns

    df = pd.DataFrame(records)
    return df, len(files), n_errors, error_patterns


def load_site_metadata(db_path="data/unified_solar_db.json"):
    """Load site metadata from unified_solar_db.json."""
    with open(db_path) as f:
        entries = json.load(f)

    records = []
    for e in entries:
        records.append({
            "site_id": e.get("site_id", ""),
            "country": e.get("country", ""),
            "capacity_mw": e.get("best_capacity_mw"),
            "construction_year": e.get("best_construction_year"),
            "treatment_group": e.get("treatment_group", ""),
            "lat": e.get("centroid_lat"),
            "lon": e.get("centroid_lon"),
        })
    return pd.DataFrame(records)


def load_dw_panel(panel_path="data/annual_panel.csv"):
    """Load DW annual panel data."""
    return pd.read_csv(panel_path)


# ── Analysis functions ───────────────────────────────────────────────────────

def compute_detection_rates(df, thresholds=DETECTION_THRESHOLDS):
    """Compute TP and FP rates at various solar_panels thresholds.

    TP = % of post-construction images with solar_panels > threshold
    FP = % of pre-construction images with solar_panels > threshold

    Requires event_time column (year - construction_year).
    """
    # Post-construction: event_time >= 0
    post = df[df["event_time"] >= 0]
    # Pre-construction: event_time < 0
    pre = df[df["event_time"] < 0]

    results = []
    for thresh in thresholds:
        tp_rate = (post["solar_panels"] > thresh).mean() * 100 if len(post) > 0 else 0
        fp_rate = (pre["solar_panels"] > thresh).mean() * 100 if len(pre) > 0 else 0
        results.append({
            "threshold_pct": thresh,
            "tp_rate": tp_rate,
            "fp_rate": fp_rate,
            "n_post": len(post),
            "n_pre": len(pre),
        })
    return pd.DataFrame(results)


def compute_detection_by_capacity(df, threshold=5):
    """Compute detection rates by capacity tier."""
    results = []
    for tier_name, lo, hi in CAPACITY_TIERS:
        mask = (df["capacity_mw"] >= lo) & (df["capacity_mw"] < hi)
        tier = df[mask]
        if len(tier) == 0:
            results.append({
                "tier": tier_name,
                "n_sites": 0, "n_images": 0,
                "pre_detection": 0, "post_detection": 0,
            })
            continue

        pre = tier[tier["event_time"] < 0]
        post = tier[tier["event_time"] >= 0]
        pre_det = (pre["solar_panels"] > threshold).mean() * 100 if len(pre) > 0 else 0
        post_det = (post["solar_panels"] > threshold).mean() * 100 if len(post) > 0 else 0

        results.append({
            "tier": tier_name,
            "n_sites": tier["site_id"].nunique(),
            "n_images": len(tier),
            "pre_detection": pre_det,
            "post_detection": post_det,
        })
    return pd.DataFrame(results)


def compute_solar_by_event_time(df):
    """Mean solar_panels % by event_time, with 95% CI."""
    grouped = df.groupby("event_time")["solar_panels"]
    stats = grouped.agg(["mean", "std", "count"]).reset_index()
    stats.columns = ["event_time", "mean", "std", "count"]
    # 95% CI = mean +/- 1.96 * std / sqrt(n)
    stats["ci"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
    stats["ci_low"] = stats["mean"] - stats["ci"]
    stats["ci_high"] = stats["mean"] + stats["ci"]
    return stats


def cross_validate_with_dw(vlm_df, dw_df):
    """Merge VLM and DW panels, compute R^2 for matching classes."""
    # Standardize DW columns
    merged = vlm_df.merge(
        dw_df[["site_id", "year"] + list(VLM_TO_DW.values())],
        on=["site_id", "year"],
        how="inner",
    )

    correlations = {}
    for vlm_col, dw_col in VLM_TO_DW.items():
        if vlm_col not in merged.columns or dw_col not in merged.columns:
            continue
        valid = merged[[vlm_col, dw_col]].dropna()
        if len(valid) < 10:
            continue
        ss_res = ((valid[vlm_col] - valid[dw_col]) ** 2).sum()
        ss_tot = ((valid[dw_col] - valid[dw_col].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Also compute Pearson correlation
        corr = valid[vlm_col].corr(valid[dw_col])

        correlations[vlm_col] = {
            "r2": r2,
            "pearson_r": corr,
            "n": len(valid),
            "vlm_mean": valid[vlm_col].mean(),
            "dw_mean": valid[dw_col].mean(),
        }

    return merged, correlations


# ── Figure functions ─────────────────────────────────────────────────────────

def plot_solar_by_event_time(stats, output_path):
    """Line plot of mean solar_panels % by event_time with 95% CI."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4))

    color = DATASET_COLORS["vlm"]

    ax.fill_between(
        stats["event_time"], stats["ci_low"], stats["ci_high"],
        alpha=0.2, color=color,
    )
    ax.plot(
        stats["event_time"], stats["mean"],
        "o-", color=color, markersize=4, linewidth=1.5,
        label="Mean solar_panels %",
    )

    ax.axvline(-0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Construction year")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")

    ax.set_xlabel("Years from construction")
    ax.set_ylabel("Solar panels (%)")
    ax.set_title("VLM Solar Detection by Event Time", fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_detection_rates(det_df, output_path):
    """Bar chart of TP vs FP rates at different thresholds."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4))

    x = np.arange(len(det_df))
    width = 0.35

    tp_color = DATASET_COLORS["vlm"]
    fp_color = "#DDDDDD"

    bars_tp = ax.bar(
        x - width / 2, det_df["tp_rate"], width,
        label="True positive (post-construction)", color=tp_color, alpha=0.85,
    )
    bars_fp = ax.bar(
        x + width / 2, det_df["fp_rate"], width,
        label="False positive (pre-construction)", color=fp_color,
        edgecolor="#888888", linewidth=0.8,
    )

    # Add value labels on bars
    for bar in bars_tp:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=7)
    for bar in bars_fp:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f">{t}%" for t in det_df["threshold_pct"]])
    ax.set_xlabel("Solar panel detection threshold")
    ax.set_ylabel("Detection rate (%)")
    ax.set_title("Solar Detection: True Positive vs False Positive Rates", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(det_df["tp_rate"].max(), 100) * 1.15)

    plt.tight_layout()
    save_fig(fig, output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_vlm_dw_cropland(merged, output_path):
    """Scatter plot of VLM crops vs DW crops, colored by event_time."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, FULL_WIDTH * 0.8))

    valid = merged[["crops", "dw_crops_pct", "event_time"]].dropna()
    if len(valid) == 0:
        plt.close()
        print(f"  Skipped {output_path}: no matching data")
        return

    # Subsample if too many points for readability
    if len(valid) > 5000:
        valid = valid.sample(5000, random_state=42)

    norm = Normalize(vmin=valid["event_time"].min(), vmax=valid["event_time"].max())
    cmap = plt.cm.RdYlBu_r

    sc = ax.scatter(
        valid["dw_crops_pct"], valid["crops"],
        c=valid["event_time"], cmap=cmap, norm=norm,
        s=6, alpha=0.4, edgecolors="none",
    )

    # 1:1 reference line
    lims = [0, max(valid["dw_crops_pct"].max(), valid["crops"].max()) * 1.05]
    ax.plot(lims, lims, "--", color="gray", linewidth=0.8, alpha=0.7, label="1:1 line")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Event time (years from construction)", fontsize=8)

    # Compute R^2 for annotation
    ss_res = ((valid["crops"] - valid["dw_crops_pct"]) ** 2).sum()
    ss_tot = ((valid["dw_crops_pct"] - valid["dw_crops_pct"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = valid["crops"].corr(valid["dw_crops_pct"])
    ax.text(0.05, 0.95, f"r = {corr:.3f}\nR$^2$ = {r2:.3f}\nn = {len(valid):,}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("DW cropland (%)")
    ax.set_ylabel("VLM cropland (%)")
    ax.set_title("VLM vs Dynamic World: Cropland Estimates", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    save_fig(fig, output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_capacity_tier_detection(tier_df, output_path):
    """Bar chart of solar detection by capacity tier, pre vs post."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4))

    x = np.arange(len(tier_df))
    width = 0.35

    post_color = DATASET_COLORS["vlm"]
    pre_color = "#DDDDDD"

    bars_post = ax.bar(
        x - width / 2, tier_df["post_detection"], width,
        label="Post-construction", color=post_color, alpha=0.85,
    )
    bars_pre = ax.bar(
        x + width / 2, tier_df["pre_detection"], width,
        label="Pre-construction", color=pre_color,
        edgecolor="#888888", linewidth=0.8,
    )

    # Value labels
    for bar in bars_post:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=7)
    for bar in bars_pre:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    # Annotate site counts
    for i, row in tier_df.iterrows():
        ax.text(i, -3, f"n={row['n_sites']}", ha="center", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(tier_df["tier"])
    ax.set_xlabel("Solar farm capacity")
    ax.set_ylabel("Detection rate (%) at >5% threshold")
    ax.set_title("Solar Detection by Capacity Tier", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=-6, top=max(tier_df["post_detection"].max(), 10) * 1.15)

    plt.tight_layout()
    save_fig(fig, output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# ── Summary printing ─────────────────────────────────────────────────────────

def print_summary(vlm_df, n_total, n_errors, error_patterns, det_df,
                  tier_df, et_stats, correlations):
    """Print comprehensive statistics for RESULTS.md documentation."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("VLM FULL-DATASET ANALYSIS SUMMARY")
    print(sep)

    # 1. Coverage
    print("\n--- Data Coverage ---")
    n_success = len(vlm_df) if vlm_df is not None else 0
    print(f"  Total files:     {n_total:,}")
    print(f"  Successful:      {n_success:,} ({n_success / n_total * 100:.1f}%)" if n_total > 0 else "")
    print(f"  Errors:          {n_errors:,} ({n_errors / n_total * 100:.1f}%)" if n_total > 0 else "")
    if vlm_df is not None:
        print(f"  Unique sites:    {vlm_df['site_id'].nunique():,}")
        print(f"  Year range:      {vlm_df['year'].min()}-{vlm_df['year'].max()}")
        print(f"  Images/site:     {vlm_df.groupby('site_id').size().mean():.1f} avg")

    if error_patterns:
        print("\n--- Error Patterns ---")
        for err, count in sorted(error_patterns.items(), key=lambda x: -x[1])[:10]:
            print(f"  [{count:>5}] {err}")

    if vlm_df is None:
        return

    # 2. LULC distribution
    print("\n--- LULC Distribution (all images, mean %) ---")
    for field in LULC_FIELDS:
        mean_val = vlm_df[field].mean()
        std_val = vlm_df[field].std()
        print(f"  {field:<20s}  {mean_val:6.2f} +/- {std_val:5.2f}")

    # 3. Solar detection
    print("\n--- Solar Detection Rates ---")
    has_event_time = "event_time" in vlm_df.columns and vlm_df["event_time"].notna().any()
    if has_event_time:
        post = vlm_df[vlm_df["event_time"] >= 0]
        pre = vlm_df[vlm_df["event_time"] < 0]
        print(f"  Post-construction images: {len(post):,}")
        print(f"  Pre-construction images:  {len(pre):,}")
        print(f"  solar_visible=True (post): {post['solar_visible'].sum():,} "
              f"({post['solar_visible'].mean() * 100:.1f}%)")
        print(f"  solar_visible=True (pre):  {pre['solar_visible'].sum():,} "
              f"({pre['solar_visible'].mean() * 100:.1f}%)")

    if det_df is not None and len(det_df) > 0:
        print(f"\n  {'Threshold':>10}  {'TP rate':>10}  {'FP rate':>10}  {'TP-FP':>10}")
        print(f"  {'-' * 44}")
        for _, row in det_df.iterrows():
            diff = row["tp_rate"] - row["fp_rate"]
            print(f"  >{row['threshold_pct']:>7.0f}%  {row['tp_rate']:>9.1f}%  "
                  f"{row['fp_rate']:>9.1f}%  {diff:>+9.1f}pp")

    # 4. Capacity tiers
    if tier_df is not None and len(tier_df) > 0:
        print("\n--- Detection by Capacity Tier (>5% threshold) ---")
        print(f"  {'Tier':<12}  {'Sites':>6}  {'Post det':>10}  {'Pre det':>10}")
        print(f"  {'-' * 42}")
        for _, row in tier_df.iterrows():
            print(f"  {row['tier']:<12}  {row['n_sites']:>6}  "
                  f"{row['post_detection']:>9.1f}%  {row['pre_detection']:>9.1f}%")

    # 5. Event-time step change
    if et_stats is not None and len(et_stats) > 0:
        print("\n--- Solar % by Event Time (key periods) ---")
        for _, row in et_stats.iterrows():
            k = int(row["event_time"])
            if -3 <= k <= 5:
                marker = " <-- construction" if k == 0 else ""
                print(f"  k={k:>+3d}:  {row['mean']:5.2f}% +/- {row['ci']:.2f}  "
                      f"(n={int(row['count']):,}){marker}")

    # 6. Cross-validation with DW
    if correlations:
        print("\n--- Cross-Validation with Dynamic World ---")
        print(f"  {'Class':<20}  {'Pearson r':>10}  {'R^2':>8}  {'VLM mean':>10}  {'DW mean':>10}  {'n':>8}")
        print(f"  {'-' * 70}")
        for cls, stats in sorted(correlations.items()):
            print(f"  {cls:<20}  {stats['pearson_r']:>10.3f}  {stats['r2']:>8.3f}  "
                  f"{stats['vlm_mean']:>9.1f}%  {stats['dw_mean']:>9.1f}%  {stats['n']:>8,}")

    print(f"\n{sep}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze full-dataset VLM classification results",
    )
    parser.add_argument(
        "--vlm-dir", default="data/vlm_results",
        help="Directory containing VLM result JSON files (default: data/vlm_results)",
    )
    parser.add_argument(
        "--output-dir", default="docs/figures/vlm",
        help="Directory for output figures (default: docs/figures/vlm)",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation, only compute statistics",
    )
    args = parser.parse_args()

    # ── 1. Load VLM results ──────────────────────────────────────────────
    print("Loading VLM results...")
    if not os.path.isdir(args.vlm_dir):
        print(f"\nERROR: VLM results directory not found: {args.vlm_dir}")
        print("Download from Modal first:")
        print("  modal volume get solar-landuse-data vlm_results/ data/vlm_results/")
        sys.exit(1)

    vlm_df, n_total, n_errors, error_patterns = load_vlm_results(args.vlm_dir)

    if vlm_df is None or len(vlm_df) == 0:
        print(f"\nERROR: No valid VLM results found in {args.vlm_dir}")
        print(f"  Total files scanned: {n_total}")
        print(f"  Errors: {n_errors}")
        if error_patterns:
            print("  Error patterns:")
            for err, cnt in sorted(error_patterns.items(), key=lambda x: -x[1])[:5]:
                print(f"    [{cnt}] {err}")
        print("\nDownload from Modal:")
        print("  modal volume get solar-landuse-data vlm_results/ data/vlm_results/")
        sys.exit(1)

    print(f"  Loaded {len(vlm_df):,} successful results from {n_total:,} files "
          f"({n_errors:,} errors)")

    # ── 2. Merge with site metadata ──────────────────────────────────────
    print("\nMerging with site metadata...")
    meta_df = load_site_metadata()
    vlm_df = vlm_df.merge(
        meta_df[["site_id", "country", "capacity_mw", "construction_year",
                 "treatment_group", "lat", "lon"]],
        on="site_id",
        how="left",
    )

    # Compute event_time for sites with known construction year
    has_cy = vlm_df["construction_year"].notna()
    vlm_df.loc[has_cy, "event_time"] = (
        vlm_df.loc[has_cy, "year"] - vlm_df.loc[has_cy, "construction_year"]
    ).astype(int)

    n_with_cy = vlm_df[has_cy]["site_id"].nunique()
    n_total_sites = vlm_df["site_id"].nunique()
    print(f"  Sites with construction year: {n_with_cy:,}/{n_total_sites:,}")

    # Assign capacity tier
    def assign_tier(mw):
        if pd.isna(mw):
            return "Unknown"
        for name, lo, hi in CAPACITY_TIERS:
            if lo <= mw < hi:
                return name
        return "Unknown"

    vlm_df["capacity_tier"] = vlm_df["capacity_mw"].apply(assign_tier)

    # Save panel CSV
    panel_path = "data/vlm_annual_panel.csv"
    vlm_df.to_csv(panel_path, index=False)
    print(f"  Saved: {panel_path} ({len(vlm_df):,} rows)")

    # ── 3. Solar detection analysis ──────────────────────────────────────
    print("\nComputing solar detection rates...")
    # Filter to sites with known construction year for TP/FP analysis
    det_subset = vlm_df[vlm_df["event_time"].notna()].copy()

    det_df = None
    tier_df = None
    et_stats = None

    if len(det_subset) > 0:
        det_df = compute_detection_rates(det_subset)
        tier_df = compute_detection_by_capacity(det_subset, threshold=5)
        et_stats = compute_solar_by_event_time(det_subset)
        # Restrict event-time stats to a reasonable window
        et_stats = et_stats[(et_stats["event_time"] >= -7) & (et_stats["event_time"] <= 8)]
        print(f"  Detection rates computed on {len(det_subset):,} images "
              f"({det_subset['site_id'].nunique():,} sites)")
    else:
        print("  WARNING: No images with known construction year for detection analysis")

    # ── 4. Cross-validation with DW ──────────────────────────────────────
    print("\nCross-validating with Dynamic World...")
    merged = None
    correlations = {}
    dw_path = "data/annual_panel.csv"
    if os.path.exists(dw_path):
        dw_df = load_dw_panel(dw_path)
        merged, correlations = cross_validate_with_dw(vlm_df, dw_df)
        print(f"  Matched {len(merged):,} VLM-DW pairs")
        if correlations:
            for cls, stats in sorted(correlations.items()):
                print(f"    {cls}: r={stats['pearson_r']:.3f}, R^2={stats['r2']:.3f}")
    else:
        print(f"  WARNING: DW panel not found at {dw_path}, skipping cross-validation")

    # ── 5. Figures ───────────────────────────────────────────────────────
    if not args.skip_figures:
        print("\nGenerating figures...")
        os.makedirs(args.output_dir, exist_ok=True)

        if et_stats is not None and len(et_stats) > 0:
            plot_solar_by_event_time(
                et_stats,
                os.path.join(args.output_dir, "vlm_solar_by_event_time.png"),
            )

        if det_df is not None and len(det_df) > 0:
            plot_detection_rates(
                det_df,
                os.path.join(args.output_dir, "vlm_solar_detection_rates.png"),
            )

        if merged is not None and len(merged) > 0:
            plot_vlm_dw_cropland(
                merged,
                os.path.join(args.output_dir, "vlm_dw_cropland_comparison.png"),
            )

        if tier_df is not None and len(tier_df) > 0:
            plot_capacity_tier_detection(
                tier_df,
                os.path.join(args.output_dir, "vlm_solar_by_capacity.png"),
            )
    else:
        print("\nSkipping figures (--skip-figures)")

    # ── 6. Summary ───────────────────────────────────────────────────────
    print_summary(
        vlm_df, n_total, n_errors, error_patterns,
        det_df, tier_df, et_stats, correlations,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
