#!/usr/bin/env python3
"""
Trajectory Growth and Velocity Analysis

Computes horizontal and vertical growth metrics plus velocity evolution for storm trajectories.
Metrics include absolute and percentage changes, with per-timestamp velocity tracking.
"""

import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm as cmc


RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
PATHWAY_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/df_pathways_merged_no_dominance.csv")
MIDPOINT_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/crop_midpoints_from_nc.csv")
OUT_DIR = os.path.join(OUTPUT_PATH, "growth_velocity_analysis")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
RATE_PLOTS_DIR = os.path.join(PLOTS_DIR, "rate_of_change")
VELOCITY_DIST_DIR = os.path.join(PLOTS_DIR, "velocity_distributions")
TEMPORAL_DIR = os.path.join(PLOTS_DIR, "temporal_evolution")

DOMAIN_EXTENT = (5.0, 16.0, 42.0, 51.5)  # lon_min, lon_max, lat_min, lat_max
CROP_HALF_SIDE_DEG = 2.0
EARTH_RADIUS_M = 6371000.0  # meters

# plotting font sizes
LABEL_FS = 14
TICK_FS = 12
TITLE_FS = 15
LEGEND_FS = 12


def parse_crop_metadata(crop_name: str) -> pd.Series:
    """Parse storm ID and datetime from crop filename."""
    basename = os.path.basename(str(crop_name))
    pattern = (
        r"^storm(?P<storm_id>\d+)_"
        r"(?P<datetime>\d{4}-\d{2}-\d{2}T\d{2}-\d{2})_"
    )
    match = re.match(pattern, basename)
    if match is None:
        return pd.Series({"parsed_storm_id": pd.NA, "parsed_datetime": pd.NaT})

    dt = pd.to_datetime(
        match.group("datetime"),
        format="%Y-%m-%dT%H-%M",
        errors="coerce",
    )
    return pd.Series({
        "parsed_storm_id": int(match.group("storm_id")),
        "parsed_datetime": dt,
    })


def add_centers_from_midpoint_csv(df: pd.DataFrame, midpoint_csv: str) -> pd.DataFrame:
    """Add center_lat and center_lon from midpoint CSV."""
    if not os.path.exists(midpoint_csv):
        return df

    midpoint_df = pd.read_csv(midpoint_csv, low_memory=False)
    if "crop" not in midpoint_df.columns:
        return df

    midpoint_df = midpoint_df.copy()
    midpoint_df["crop_key"] = midpoint_df["crop"].astype(str).map(os.path.basename)
    midpoint_df = midpoint_df.drop_duplicates(subset=["crop_key"], keep="last")

    out = df.copy()
    if "crop" in out.columns:
        out["crop_key"] = out["crop"].astype(str).map(os.path.basename)
    else:
        return out

    keep_cols = [c for c in ["crop_key", "center_lat", "center_lon"] if c in midpoint_df.columns]
    if len(keep_cols) < 3:
        return out

    out = out.merge(midpoint_df[keep_cols], how="left", on="crop_key")
    return out


def load_pathway_df(pathway_csv: str, midpoint_csv: str) -> pd.DataFrame:
    """Load and preprocess pathway data."""
    df = pd.read_csv(pathway_csv, low_memory=False)

    if "crop" in df.columns:
        parsed = df["crop"].apply(parse_crop_metadata)
        df = pd.concat([df, parsed], axis=1)
    else:
        df["parsed_storm_id"] = pd.NA
        df["parsed_datetime"] = pd.NaT

    df = df.dropna(subset=["parsed_storm_id", "parsed_datetime"])
    df["parsed_storm_id"] = df["parsed_storm_id"].astype(int)

    df = add_centers_from_midpoint_csv(df, midpoint_csv)
    df = df.dropna(subset=["center_lat", "center_lon"])

    df["dt"] = pd.to_datetime(df["parsed_datetime"], errors="coerce")
    df["trajectory_id"] = pd.to_numeric(df["parsed_storm_id"], errors="coerce").astype("Int64")
    df["center_lat"] = pd.to_numeric(df["center_lat"], errors="coerce")
    df["center_lon"] = pd.to_numeric(df["center_lon"], errors="coerce")

    for metric_col in ["cma", "cth50", "cth_very_high", "cot_thick"]:
        if metric_col in df.columns:
            df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    if "pathway" not in df.columns:
        df["pathway"] = "unknown"

    if "t_aligned" not in df.columns:
        df["t_aligned"] = 0
    else:
        df["t_aligned"] = pd.to_numeric(df["t_aligned"], errors="coerce")

    df = df.dropna(subset=["trajectory_id", "dt", "center_lat", "center_lon"])
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df = df.sort_values(["trajectory_id", "dt"]).reset_index(drop=True)

    return df


def style_two_axis_plot(ax: plt.Axes) -> None:
    """Apply cleaner plot style."""
    ax.grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", labelsize=11)


def compute_rate_of_change_per_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute horizontal and vertical rate of change (absolute and percentage) per trajectory.
    
    Metrics:
    - cma_rate_abs, cma_rate_pct: Area cloud growth
    - cth_very_high_rate_abs, cth_very_high_rate_pct: Core height growth
    - cot_thick_rate_abs, cot_thick_rate_pct: Optical thickness growth
    - cth50_rate_abs, cth50_rate_pct: Vertical cloud top height growth
    """
    records = []
    metric_specs = {
        "cma": "cma",
        "cth_very_high": "cth_very_high",
        "cot_thick": "cot_thick",
        # optional: include cot50 if present in the dataframe
        "cot50": "cot50",
        "cth50": "cth50",
    }

    for traj_id, g in df.groupby("trajectory_id"):
        g = g.sort_values("dt").copy()
        pathway = str(g["pathway"].iloc[0])
        n_timestamps = len(g)

        record = {
            "trajectory_id": int(traj_id),
            "pathway": pathway,
            "n_timestamps": int(n_timestamps),
        }

        # Compute rate of change for each metric
        for metric_name, col_name in metric_specs.items():
            if col_name not in g.columns:
                record[f"{metric_name}_rate_abs"] = np.nan
                record[f"{metric_name}_rate_pct"] = np.nan
                continue

            series = pd.to_numeric(g[col_name], errors="coerce")
            if series.dropna().empty:
                record[f"{metric_name}_rate_abs"] = np.nan
                record[f"{metric_name}_rate_pct"] = np.nan
                continue

            initial = series.iloc[0]
            final = series.iloc[-1]
            abs_change = final - initial

            # absolute change
            record[f"{metric_name}_rate_abs"] = float(abs_change) if np.isfinite(abs_change) else np.nan

            # percentage change (existing behavior)
            if np.isfinite(initial) and initial != 0 and np.isfinite(final):
                pct_change = (final - initial) / initial * 100.0
                record[f"{metric_name}_rate_pct"] = float(pct_change)
                # fractional change (unitless, e.g., fraction of initial)
                record[f"{metric_name}_rate_frac"] = float((final - initial) / initial)
            else:
                record[f"{metric_name}_rate_pct"] = np.nan
                record[f"{metric_name}_rate_frac"] = np.nan

        records.append(record)

    return pd.DataFrame(records)


def print_pathway_coverage_summary(df: pd.DataFrame) -> None:
    """Print total trajectories per pathway and valid counts for each rate metric."""
    rate_cols = [
        "cma_rate_abs",
        "cma_rate_pct",
        "cth_very_high_rate_abs",
        "cth_very_high_rate_pct",
        "cot_thick_rate_abs",
        "cot_thick_rate_pct",
        "cth50_rate_abs",
        "cth50_rate_pct",
    ]

    print("\n=== PATHWAY COVERAGE SUMMARY ===")
    pathway_counts = df.groupby("pathway")["trajectory_id"].nunique().sort_index()
    print("Total trajectories per pathway:")
    for pathway, count in pathway_counts.items():
        print(f"  {pathway}: {count}")

    print("\nValid trajectories per metric and pathway:")
    for metric_col in rate_cols:
        if metric_col not in df.columns:
            continue
        print(f"  {metric_col}:")
        counts = df.groupby("pathway")[metric_col].apply(lambda s: s.notna().sum()).sort_index()
        for pathway, count in counts.items():
            print(f"    {pathway}: {count}")


def compute_velocity_per_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute velocity (rate of change between consecutive timestamps) per trajectory.
    
    Velocity computed as: (value_t - value_t-1) / (time_t - time_t-1)
    - cma_velocity: change in cloud area per hour
    - cth_very_high_velocity: change in core height per hour
    - cot_thick_velocity: change in optical thickness per hour
    - cth50_velocity_ms: change in cth50 in m/s
    """
    records = []
    metric_specs = {
        "cma": "cma",
        "cth_very_high": "cth_very_high",
        "cot_thick": "cot_thick",
        "cot50": "cot50",
        "cth50": "cth50",
    }

    for traj_id, g in df.groupby("trajectory_id"):
        g = g.sort_values("dt").copy()
        pathway = str(g["pathway"].iloc[0])

        # Iterate through consecutive timestamps
        for i in range(1, len(g)):
            prev_row = g.iloc[i - 1]
            curr_row = g.iloc[i]

            dt_seconds = (curr_row["dt"] - prev_row["dt"]).total_seconds()
            if dt_seconds <= 0:
                continue

            dt_hours = dt_seconds / 3600.0
            t_aligned = pd.to_numeric(curr_row["t_aligned"], errors="coerce")

            record = {
                "trajectory_id": int(traj_id),
                "pathway": pathway,
                "t_aligned": float(t_aligned) if np.isfinite(t_aligned) else np.nan,
                "timestamp_index": int(i),
                "dt_hours": float(dt_hours),
            }

            # Compute velocity for each metric
            for metric_name, col_name in metric_specs.items():
                if col_name not in g.columns:
                    record[f"{metric_name}_velocity"] = np.nan
                    continue

                prev_val = pd.to_numeric(prev_row[col_name], errors="coerce")
                curr_val = pd.to_numeric(curr_row[col_name], errors="coerce")

                if not (np.isfinite(prev_val) and np.isfinite(curr_val)):
                    record[f"{metric_name}_velocity"] = np.nan
                    continue

                delta = curr_val - prev_val
                velocity = delta / dt_hours
                record[f"{metric_name}_velocity"] = float(velocity)

                # fractional velocity per hour (relative to previous value)
                if prev_val != 0 and np.isfinite(prev_val):
                    record[f"{metric_name}_velocity_frac_per_hour"] = float((delta / prev_val) / dt_hours)
                else:
                    record[f"{metric_name}_velocity_frac_per_hour"] = np.nan

            # Special case: cth50_velocity in m/s
            if "cth50" in g.columns:
                prev_cth = pd.to_numeric(g.iloc[i - 1]["cth50"], errors="coerce")
                curr_cth = pd.to_numeric(g.iloc[i]["cth50"], errors="coerce")

                if np.isfinite(prev_cth) and np.isfinite(curr_cth):
                    cth_diff_m = (curr_cth - prev_cth)  # cth50 is already in meters
                    velocity_ms = cth_diff_m / dt_seconds
                    record["cth50_velocity_ms"] = float(velocity_ms)
                    # also provide cth50 velocity in meters/hour for plotting convenience
                    record["cth50_velocity_mph"] = float(cth_diff_m / dt_hours)
                else:
                    record["cth50_velocity_ms"] = np.nan
                    record["cth50_velocity_mph"] = np.nan
            else:
                record["cth50_velocity_ms"] = np.nan

            records.append(record)

    return pd.DataFrame(records)


def print_rate_statistics(df: pd.DataFrame):
    """Print rate of change statistics overall and per pathway."""
    rate_metrics = {
        "cma_rate_abs": "CMA rate (absolute)",
        "cma_rate_pct": "CMA rate (%)",
        "cth_very_high_rate_abs": "CTH Very High rate (absolute)",
        "cth_very_high_rate_pct": "CTH Very High rate (%)",
        "cot_thick_rate_abs": "COT Thick rate (absolute)",
        "cot_thick_rate_pct": "COT Thick rate (%)",
        "cth50_rate_abs": "CTH50 rate (absolute, meters)",
        "cth50_rate_pct": "CTH50 rate (%)",
    }

    print("\n=== RATE OF CHANGE STATISTICS ===")
    for metric_col, label in rate_metrics.items():
        values = df[metric_col].dropna()
        print(f"\n{label}:")
        if len(values) == 0:
            print("  No valid values")
            continue
        print(
            f"  Overall: N={len(values)}, mean={values.mean():.4f}, std={values.std():.4f}, "
            f"min={values.min():.4f}, max={values.max():.4f}"
        )
        print("  Per pathway:")
        for pathway in sorted(df["pathway"].astype(str).unique()):
            subset = df.loc[df["pathway"].astype(str) == pathway, metric_col].dropna()
            if len(subset) == 0:
                print(f"    {pathway}: N=0")
                continue
            print(
                f"    {pathway}: N={len(subset)}, mean={subset.mean():.4f}, std={subset.std():.4f}, "
                f"min={subset.min():.4f}, max={subset.max():.4f}"
            )


def print_velocity_statistics(df: pd.DataFrame):
    """Print velocity statistics overall and per pathway."""
    velocity_metrics = {
        "cma_velocity": "CMA velocity (per hour)",
        "cth_very_high_velocity": "CTH Very High velocity (per hour)",
        "cot_thick_velocity": "COT Thick velocity (per hour)",
        "cth50_velocity_ms": "CTH50 velocity (m/s)",
    }

    print("\n=== VELOCITY STATISTICS ===")
    for metric_col, label in velocity_metrics.items():
        values = df[metric_col].dropna()
        print(f"\n{label}:")
        if len(values) == 0:
            print("  No valid values")
            continue
        print(
            f"  Overall: N={len(values)}, mean={values.mean():.4f}, std={values.std():.4f}, "
            f"min={values.min():.4f}, max={values.max():.4f}"
        )
        print("  Per pathway:")
        for pathway in sorted(df["pathway"].astype(str).unique()):
            subset = df.loc[df["pathway"].astype(str) == pathway, metric_col].dropna()
            if len(subset) == 0:
                print(f"    {pathway}: N=0")
                continue
            print(
                f"    {pathway}: N={len(subset)}, mean={subset.mean():.4f}, std={subset.std():.4f}, "
                f"min={subset.min():.4f}, max={subset.max():.4f}"
            )


def plot_rate_distribution(df: pd.DataFrame, metric: str, title: str, unit: str, out_dir: str):
    """Plot overall and per-pathway rate of change distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vals = df[metric].dropna().values

    # Overall histogram
    axes[0].hist(vals, bins=30, color="#377eb8", edgecolor="black", alpha=0.8)
    axes[0].set_title(f"{title} - Overall", fontsize=TITLE_FS, fontweight="bold")
    axes[0].set_xlabel(unit, fontsize=LABEL_FS)
    axes[0].set_ylabel("Count", fontsize=LABEL_FS)
    axes[0].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(axis="both", which="major", labelsize=TICK_FS)
    axes[0].text(
        0.98, 0.97,
        f"N={len(vals)}\nMean={vals.mean():.4f}\nStd={vals.std():.4f}",
        transform=axes[0].transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    # Per-pathway boxplot
    pathways = sorted(df["pathway"].astype(str).unique())
    box_data = [df.loc[df["pathway"].astype(str) == p, metric].dropna().values for p in pathways]
    axes[1].boxplot(box_data, labels=pathways, showfliers=False)
    axes[1].set_title(f"{title} - Per Pathway", fontsize=TITLE_FS, fontweight="bold")
    axes[1].set_ylabel(unit, fontsize=LABEL_FS)
    axes[1].set_xlabel("Pathway", fontsize=LABEL_FS)
    axes[1].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].tick_params(axis="both", which="major", labelsize=TICK_FS)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    outfile = os.path.join(out_dir, f"{metric}_distribution.png")
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


def plot_velocity_distribution(df: pd.DataFrame, metric: str, title: str, unit: str, out_dir: str):
    """Plot overall and per-pathway velocity distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vals = df[metric].dropna().values

    # Overall histogram
    axes[0].hist(vals, bins=30, color="#e41a1c", edgecolor="black", alpha=0.8)
    axes[0].set_title(f"{title} - Overall", fontsize=TITLE_FS, fontweight="bold")
    axes[0].set_xlabel(unit, fontsize=LABEL_FS)
    axes[0].set_ylabel("Count", fontsize=LABEL_FS)
    axes[0].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(axis="both", which="major", labelsize=TICK_FS)
    axes[0].text(
        0.98, 0.97,
        f"N={len(vals)}\nMean={vals.mean():.4f}\nStd={vals.std():.4f}",
        transform=axes[0].transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    # Per-pathway boxplot
    pathways = sorted(df["pathway"].astype(str).unique())
    box_data = [df.loc[df["pathway"].astype(str) == p, metric].dropna().values for p in pathways]
    axes[1].boxplot(box_data, labels=pathways, showfliers=False)
    axes[1].set_title(f"{title} - Per Pathway", fontsize=TITLE_FS, fontweight="bold")
    axes[1].set_ylabel(unit, fontsize=LABEL_FS)
    axes[1].set_xlabel("Pathway", fontsize=LABEL_FS)
    axes[1].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].tick_params(axis="both", which="major", labelsize=TICK_FS)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    outfile = os.path.join(out_dir, f"{metric}_distribution.png")
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


def plot_velocity_temporal_evolution(df: pd.DataFrame, metric: str, title: str, unit: str, out_dir: str):
    """Plot velocity vs t_aligned overall and per pathway."""
    # Overall density (hexbin) plot
    fig, ax = plt.subplots(figsize=(12, 6))
    vals = df[metric].dropna()
    t_aligned = df.loc[vals.index, "t_aligned"].values

    hb = ax.hexbin(t_aligned, vals.values, gridsize=80, cmap="viridis", mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("counts", fontsize=LABEL_FS)

    ax.set_xlabel("t_aligned", fontsize=LABEL_FS)
    ax.set_ylabel(unit, fontsize=LABEL_FS)
    ax.set_title(f"{title} - Temporal Density (Overall)", fontsize=TITLE_FS, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    style_two_axis_plot(ax)

    plt.tight_layout()
    outfile = os.path.join(out_dir, f"{metric}_temporal_overall_density.png")
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")

    # Per-pathway - create separate figures, with thin lines per trajectory
    plot_per_pathway_trajectory_lines(df, metric, title, unit, out_dir)


def plot_per_pathway_trajectory_lines(df: pd.DataFrame, metric: str, title: str, unit: str, out_dir: str):
    """Create one plot per pathway; each trajectory is a thin line of metric vs t_aligned."""
    pathways = sorted(df["pathway"].astype(str).unique())
    for pathway in pathways:
        mask_path = df["pathway"].astype(str) == pathway
        subset = df.loc[mask_path & df[metric].notna(), :].copy()
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # background: thin, very transparent trajectory lines
        for traj_id, g in subset.groupby("trajectory_id"):
            g = g.sort_values("t_aligned")
            ax.plot(g["t_aligned"].values, g[metric].values, color="#444444", linewidth=0.6, alpha=0.08)

        # compute median and 25/75 percentiles per t_aligned
        t_vals = np.sort(subset["t_aligned"].dropna().unique())
        medians = []
        p25 = []
        p75 = []
        t_plot = []
        for t in t_vals:
            vals = subset.loc[subset["t_aligned"] == t, metric].dropna().values
            if len(vals) == 0:
                continue
            t_plot.append(t)
            medians.append(np.nanmedian(vals))
            p25.append(np.nanpercentile(vals, 25))
            p75.append(np.nanpercentile(vals, 75))

        if len(t_plot) > 0:
            t_plot = np.array(t_plot)
            medians = np.array(medians)
            p25 = np.array(p25)
            p75 = np.array(p75)

            # median bold line
            ax.plot(t_plot, medians, color="#d73027", linewidth=2.4, alpha=0.95, label="median")
            # IQR filled
            ax.fill_between(t_plot, p25, p75, color="#d73027", alpha=0.25, label="25-75%")

        ax.set_xlabel("t_aligned", fontsize=LABEL_FS)
        ax.set_ylabel(unit, fontsize=LABEL_FS)
        ax.set_title(f"{title} - {pathway}", fontsize=TITLE_FS, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
        style_two_axis_plot(ax)
        ax.legend(fontsize=LEGEND_FS)

        # Add stats box for this pathway
        vals_all = subset[metric].dropna()
        n_vals = len(vals_all)
        n_traj = subset["trajectory_id"].nunique()
        if n_vals > 0:
            mean_v = vals_all.mean()
            std_v = vals_all.std()
            min_v = vals_all.min()
            max_v = vals_all.max()
            stats_txt = (
                f"N={n_vals}\nTraj={n_traj}\nmean={mean_v:.4f}\nstd={std_v:.4f}\nmin={min_v:.4f}\nmax={max_v:.4f}"
            )
            ax.text(
                0.98, 0.98, stats_txt,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=TICK_FS, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
            )

        plt.tight_layout()
        outfile = os.path.join(out_dir, f"{metric}_temporal_{pathway.replace(' ', '_')}.png")
        plt.savefig(outfile, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outfile}")


def main():
    for dir_path in [OUT_DIR, PLOTS_DIR, RATE_PLOTS_DIR, VELOCITY_DIST_DIR, TEMPORAL_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Loading pathways: {PATHWAY_CSV}")
    df = load_pathway_df(PATHWAY_CSV, MIDPOINT_CSV)
    print(f"Rows after preprocessing: {len(df)}")
    print(f"Trajectories: {df['trajectory_id'].nunique()}")

    print("\nComputing rate of change per trajectory...")
    df_rate = compute_rate_of_change_per_trajectory(df)
    print(f"Rate results: {len(df_rate)} trajectories")

    rate_csv = os.path.join(OUT_DIR, "trajectory_rate_of_change.csv")
    df_rate.to_csv(rate_csv, index=False)
    print(f"Saved: {rate_csv}")

    print_pathway_coverage_summary(df_rate)
    print_rate_statistics(df_rate)

    print("\nComputing velocity per timestamp...")
    df_velocity = compute_velocity_per_timestamp(df)
    print(f"Velocity results: {len(df_velocity)} timestamp transitions")

    velocity_csv = os.path.join(OUT_DIR, "trajectory_velocity_per_timestamp.csv")
    df_velocity.to_csv(velocity_csv, index=False)
    print(f"Saved: {velocity_csv}")

    print_velocity_statistics(df_velocity)

    print(f"\nCreating rate of change plots in: {RATE_PLOTS_DIR}")
    plot_rate_distribution(df_rate, "cma_rate_abs", "CMA Rate (Absolute)", "CMA", RATE_PLOTS_DIR)
    plot_rate_distribution(df_rate, "cma_rate_pct", "CMA Rate (Percent)", "%", RATE_PLOTS_DIR)
    # fractional change (unitless fraction of initial)
    if "cma_rate_frac" in df_rate.columns:
        plot_rate_distribution(df_rate, "cma_rate_frac", "CMA Rate (Fraction)", "fraction", RATE_PLOTS_DIR)
    plot_rate_distribution(df_rate, "cth_very_high_rate_abs", "CTH Very High Rate (Absolute)", "CTH VH", RATE_PLOTS_DIR)
    plot_rate_distribution(df_rate, "cth_very_high_rate_pct", "CTH Very High Rate (Percent)", "%", RATE_PLOTS_DIR)
    if "cth_very_high_rate_frac" in df_rate.columns:
        plot_rate_distribution(df_rate, "cth_very_high_rate_frac", "CTH Very High Rate (Fraction)", "fraction", RATE_PLOTS_DIR)
    plot_rate_distribution(df_rate, "cot_thick_rate_abs", "COT Thick Rate (Absolute)", "COT Thick", RATE_PLOTS_DIR)
    plot_rate_distribution(df_rate, "cot_thick_rate_pct", "COT Thick Rate (Percent)", "%", RATE_PLOTS_DIR)
    if "cot_thick_rate_frac" in df_rate.columns:
        plot_rate_distribution(df_rate, "cot_thick_rate_frac", "COT Thick Rate (Fraction)", "fraction", RATE_PLOTS_DIR)
    if "cot50_rate_frac" in df_rate.columns:
        plot_rate_distribution(df_rate, "cot50_rate_frac", "COT50 Rate (Fraction)", "fraction", RATE_PLOTS_DIR)
    plot_rate_distribution(df_rate, "cth50_rate_abs", "CTH50 Rate (Absolute)", "meters", RATE_PLOTS_DIR)
    plot_rate_distribution(df_rate, "cth50_rate_pct", "CTH50 Rate (Percent)", "%", RATE_PLOTS_DIR)
    if "cth50_rate_frac" in df_rate.columns:
        plot_rate_distribution(df_rate, "cth50_rate_frac", "CTH50 Rate (Fraction)", "fraction", RATE_PLOTS_DIR)

    print(f"\nCreating velocity distributions in: {VELOCITY_DIST_DIR}")
    plot_velocity_distribution(df_velocity, "cma_velocity", "CMA Velocity", "per hour", VELOCITY_DIST_DIR)
    plot_velocity_distribution(df_velocity, "cth_very_high_velocity", "CTH Very High Velocity", "per hour", VELOCITY_DIST_DIR)
    plot_velocity_distribution(df_velocity, "cot_thick_velocity", "COT Thick Velocity", "per hour", VELOCITY_DIST_DIR)
    plot_velocity_distribution(df_velocity, "cth50_velocity_ms", "CTH50 Velocity", "m/s", VELOCITY_DIST_DIR)
    # fractional velocities per hour (unit: fraction/hour)
    if "cma_velocity_frac_per_hour" in df_velocity.columns:
        plot_velocity_distribution(df_velocity, "cma_velocity_frac_per_hour", "CMA Velocity (Fraction/hour)", "fraction/hour", VELOCITY_DIST_DIR)
    if "cth_very_high_velocity_frac_per_hour" in df_velocity.columns:
        plot_velocity_distribution(df_velocity, "cth_very_high_velocity_frac_per_hour", "CTH Very High Velocity (Fraction/hour)", "fraction/hour", VELOCITY_DIST_DIR)
    if "cot_thick_velocity_frac_per_hour" in df_velocity.columns:
        plot_velocity_distribution(df_velocity, "cot_thick_velocity_frac_per_hour", "COT Thick Velocity (Fraction/hour)", "fraction/hour", VELOCITY_DIST_DIR)
    if "cot50_velocity_frac_per_hour" in df_velocity.columns:
        plot_velocity_distribution(df_velocity, "cot50_velocity_frac_per_hour", "COT50 Velocity (Fraction/hour)", "fraction/hour", VELOCITY_DIST_DIR)

    print(f"\nCreating temporal evolution plots in: {TEMPORAL_DIR}")
    # absolute velocities
    plot_velocity_temporal_evolution(df_velocity, "cma_velocity", "CMA Velocity", "per hour", TEMPORAL_DIR)
    plot_velocity_temporal_evolution(df_velocity, "cth_very_high_velocity", "CTH Very High Velocity", "per hour", TEMPORAL_DIR)
    plot_velocity_temporal_evolution(df_velocity, "cot_thick_velocity", "COT Thick Velocity", "per hour", TEMPORAL_DIR)
    plot_velocity_temporal_evolution(df_velocity, "cth50_velocity_ms", "CTH50 Velocity", "m/s", TEMPORAL_DIR)
    # fractional velocities per hour
    if "cma_velocity_frac_per_hour" in df_velocity.columns:
        plot_velocity_temporal_evolution(df_velocity, "cma_velocity_frac_per_hour", "CMA Velocity (Fraction/hour)", "fraction/hour", TEMPORAL_DIR)
    if "cth_very_high_velocity_frac_per_hour" in df_velocity.columns:
        plot_velocity_temporal_evolution(df_velocity, "cth_very_high_velocity_frac_per_hour", "CTH Very High Velocity (Fraction/hour)", "fraction/hour", TEMPORAL_DIR)
    if "cot_thick_velocity_frac_per_hour" in df_velocity.columns:
        plot_velocity_temporal_evolution(df_velocity, "cot_thick_velocity_frac_per_hour", "COT Thick Velocity (Fraction/hour)", "fraction/hour", TEMPORAL_DIR)
    if "cot50_velocity_frac_per_hour" in df_velocity.columns:
        plot_velocity_temporal_evolution(df_velocity, "cot50_velocity_frac_per_hour", "COT50 Velocity (Fraction/hour)", "fraction/hour", TEMPORAL_DIR)

    print(f"\nAll analysis complete!")
    print(f"Results saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
