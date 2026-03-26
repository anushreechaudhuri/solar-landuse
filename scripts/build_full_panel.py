"""
Build unified analysis panel from Modal pipeline outputs.

Merges DW annual panel, EO annual panel, and VLM results into a single
analysis-ready dataset. Caches all intermediate outputs so partial runs
are never lost.

Usage:
    python scripts/build_full_panel.py
    python scripts/build_full_panel.py --vlm-dir data/vlm_results_full/vlm_results
    python scripts/build_full_panel.py --skip-vlm  # just merge DW + EO
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

VLM_LULC_FIELDS = [
    "water", "trees", "grass", "flooded_vegetation", "crops",
    "shrub_and_scrub", "built", "bare", "snow_and_ice", "solar_panels",
]


# ── VLM parsing ──────────────────────────────────────────────────────────────

def parse_vlm_results(vlm_dir, cache_path="data/vlm_annual_panel.csv"):
    """Parse all VLM JSON files into a DataFrame. Caches to CSV.

    Returns (df, stats_dict) where stats_dict has success/error counts.
    """
    files = sorted(glob.glob(os.path.join(vlm_dir, "*.json")))
    if not files:
        print(f"  No VLM files found in {vlm_dir}")
        return None, {"total": 0, "success": 0, "errors": 0}

    print(f"  Found {len(files):,} VLM JSON files")

    records = []
    n_errors = 0
    error_types = {}

    for i, fpath in enumerate(files):
        if i > 0 and i % 5000 == 0:
            print(f"    ... parsed {i:,}/{len(files):,} ({len(records):,} valid)")

        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            n_errors += 1
            continue

        if "error" in data:
            n_errors += 1
            err_key = str(data["error"])[:40]
            error_types[err_key] = error_types.get(err_key, 0) + 1
            continue

        record = {
            "site_id": data.get("site_id", ""),
            "year": int(data.get("year", 0)),
            "solar_visible": bool(data.get("solar_visible", False)),
        }
        for field in VLM_LULC_FIELDS:
            try:
                record[field] = float(data.get(field, 0))
            except (ValueError, TypeError):
                record[field] = 0.0
        records.append(record)

    if not records:
        return None, {"total": len(files), "success": 0, "errors": n_errors,
                       "error_types": error_types}

    df = pd.DataFrame(records)

    # Cache
    df.to_csv(cache_path, index=False)
    print(f"  Cached VLM panel: {cache_path} ({len(df):,} rows)")

    stats = {
        "total": len(files),
        "success": len(records),
        "errors": n_errors,
        "error_types": error_types,
        "n_sites": df["site_id"].nunique(),
        "year_range": (int(df["year"].min()), int(df["year"].max())),
    }
    return df, stats


# ── Panel merging ────────────────────────────────────────────────────────────

def build_panel(dw_path, eo_path, vlm_df=None, db_path="data/unified_solar_db.json"):
    """Merge DW, EO, and VLM panels into a unified DataFrame."""
    print("\nLoading DW panel...")
    dw = pd.read_csv(dw_path)
    print(f"  DW panel: {len(dw):,} rows, {dw['site_id'].nunique():,} sites")

    print("Loading EO panel...")
    eo = pd.read_csv(eo_path)
    print(f"  EO panel: {len(eo):,} rows, {eo['site_id'].nunique():,} sites")

    # Merge DW + EO
    panel = dw.merge(eo, on=["site_id", "year"], how="outer", suffixes=("", "_eo"))
    print(f"  DW+EO merged: {len(panel):,} rows")

    # Merge VLM if available
    if vlm_df is not None and len(vlm_df) > 0:
        # Prefix VLM columns to avoid collisions
        vlm_cols = {f: f"vlm_{f}" for f in VLM_LULC_FIELDS}
        vlm_cols["solar_visible"] = "vlm_solar_visible"
        vlm_renamed = vlm_df.rename(columns=vlm_cols)
        panel = panel.merge(
            vlm_renamed, on=["site_id", "year"], how="left"
        )
        n_vlm = panel["vlm_solar_panels"].notna().sum()
        print(f"  VLM merged: {n_vlm:,}/{len(panel):,} rows have VLM data")

    # Merge construction year from unified_solar_db
    print("Loading site metadata...")
    with open(db_path) as f:
        udb = json.load(f)

    meta = {}
    for e in udb:
        sid = e.get("site_id")
        if not sid:
            continue
        meta[sid] = {
            "best_construction_year": e.get("best_construction_year"),
            "best_capacity_mw": e.get("best_capacity_mw"),
            "country": e.get("country"),
            "centroid_lat": e.get("centroid_lat"),
            "centroid_lon": e.get("centroid_lon"),
            "treatment_group": e.get("treatment_group"),
        }

    meta_df = pd.DataFrame.from_dict(meta, orient="index")
    meta_df.index.name = "site_id"
    meta_df = meta_df.reset_index()

    # Fill missing construction_year from metadata
    panel = panel.merge(
        meta_df[["site_id", "best_construction_year", "best_capacity_mw",
                 "treatment_group", "centroid_lat", "centroid_lon"]],
        on="site_id", how="left"
    )

    # Use best_construction_year where construction_year is missing
    if "construction_year" in panel.columns:
        panel["construction_year"] = panel["construction_year"].fillna(
            panel["best_construction_year"]
        )
    else:
        panel["construction_year"] = panel["best_construction_year"]

    if "capacity_mw" not in panel.columns:
        panel["capacity_mw"] = panel["best_capacity_mw"]
    else:
        panel["capacity_mw"] = panel["capacity_mw"].fillna(panel["best_capacity_mw"])

    # Compute event_time
    has_cy = panel["construction_year"].notna()
    panel.loc[has_cy, "construction_year"] = panel.loc[has_cy, "construction_year"].astype(int)
    panel["event_time"] = np.where(
        has_cy,
        panel["year"] - panel["construction_year"],
        np.nan
    )

    # Drop helper columns
    panel = panel.drop(columns=["best_construction_year", "best_capacity_mw"], errors="ignore")

    # Summary
    n_sites = panel["site_id"].nunique()
    n_with_cy = panel[has_cy]["site_id"].nunique()
    print(f"\nFinal panel: {len(panel):,} rows, {n_sites:,} sites")
    print(f"  Sites with construction year: {n_with_cy:,}")
    if "country" in panel.columns:
        countries = panel.drop_duplicates("site_id")["country"].value_counts()
        for c, n in countries.head(5).items():
            print(f"  {c}: {n:,} sites")

    return panel


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build unified analysis panel")
    parser.add_argument("--dw-path", default="data/annual_panel_full.csv")
    parser.add_argument("--eo-path", default="data/eo_annual_panel_full.csv")
    parser.add_argument("--vlm-dir", default="data/vlm_results_full/vlm_results")
    parser.add_argument("--vlm-cache", default="data/vlm_annual_panel.csv")
    parser.add_argument("--output", default="data/full_panel.csv")
    parser.add_argument("--skip-vlm", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    # ── Parse VLM ────────────────────────────────────────────────────────
    vlm_df = None
    vlm_stats = None

    if not args.skip_vlm:
        # Try cache first
        if os.path.exists(args.vlm_cache):
            print(f"Loading cached VLM panel from {args.vlm_cache}...")
            vlm_df = pd.read_csv(args.vlm_cache)
            print(f"  Loaded {len(vlm_df):,} rows, {vlm_df['site_id'].nunique():,} sites")

            # Check if VLM dir has more files than cache
            if os.path.isdir(args.vlm_dir):
                n_files = len(glob.glob(os.path.join(args.vlm_dir, "*.json")))
                if n_files > len(vlm_df) * 1.1:  # >10% more files
                    print(f"  VLM dir has {n_files:,} files vs {len(vlm_df):,} cached — re-parsing")
                    vlm_df, vlm_stats = parse_vlm_results(args.vlm_dir, args.vlm_cache)
        elif os.path.isdir(args.vlm_dir):
            print(f"Parsing VLM results from {args.vlm_dir}...")
            vlm_df, vlm_stats = parse_vlm_results(args.vlm_dir, args.vlm_cache)
        else:
            print(f"WARNING: No VLM data at {args.vlm_dir} or {args.vlm_cache}")

    # ── Build merged panel ───────────────────────────────────────────────
    panel = build_panel(args.dw_path, args.eo_path, vlm_df)

    # ── Save ─────────────────────────────────────────────────────────────
    panel.to_csv(args.output, index=False)
    elapsed = time.time() - t0
    print(f"\nSaved: {args.output} ({len(panel):,} rows, {elapsed:.1f}s)")

    # ── Quick VLM stats ──────────────────────────────────────────────────
    if vlm_stats:
        print(f"\nVLM parsing stats:")
        print(f"  Total files: {vlm_stats['total']:,}")
        print(f"  Successful: {vlm_stats['success']:,} "
              f"({vlm_stats['success']/vlm_stats['total']*100:.1f}%)")
        print(f"  Errors: {vlm_stats['errors']:,}")
        if vlm_stats.get("error_types"):
            print("  Top error types:")
            for e, c in sorted(vlm_stats["error_types"].items(), key=lambda x: -x[1])[:3]:
                print(f"    [{c:>5}] {e}")

    # Save stats JSON for documentation
    stats_path = "data/panel_build_stats.json"
    stats = {
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(panel),
        "n_sites": int(panel["site_id"].nunique()),
        "n_years": int(panel["year"].nunique()),
        "year_range": [int(panel["year"].min()), int(panel["year"].max())],
        "has_vlm": vlm_df is not None,
        "n_vlm_valid": int(len(vlm_df)) if vlm_df is not None else 0,
        "vlm_stats": vlm_stats if vlm_stats else None,
        "columns": list(panel.columns),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved build stats: {stats_path}")


if __name__ == "__main__":
    main()
