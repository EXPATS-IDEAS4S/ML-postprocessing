import os
import re
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm as cmc


RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
PATHWAY_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/df_pathways_merged_no_dominance.csv")
MIDPOINT_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/crop_midpoints_from_nc.csv")
OUT_DIR = os.path.join(OUTPUT_PATH, "trajectory_duration_distance")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
DOMAIN_EXTENT = (5.0, 16.0, 42.0, 51.5)  # lon_min, lon_max, lat_min, lat_max
CROP_HALF_SIDE_DEG = 2.0

# Cloud property files
TEST_PERCENTILES_CSV = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj/merged_crops_stats_cot_percentiles.csv"
TEST_FRACTIONS_CSV = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj/merged_crops_stats_cot_cth_fractions.csv"
TRAIN_CVC_CSV = f"/data1/fig/{RUN_NAME}/epoch_800/all/merged_crops_stats_all_cvc.csv"
TRAIN_FRACTIONS_CSV = f"/data1/fig/{RUN_NAME}/epoch_800/all/merged_crops_stats_all_cvc_cot_cth_fractions.csv"


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

    for metric_col in ["precipitation99", "max_precip_intensity", "max_hail_intensity"]:
        if metric_col in df.columns:
            df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    if "pathway" not in df.columns:
        df["pathway"] = "unknown"

    df = df.dropna(subset=["trajectory_id", "dt", "center_lat", "center_lon"])
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df = df.sort_values(["trajectory_id", "dt"]).reset_index(drop=True)

    return df


def haversine_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in degrees between two points using Haversine formula."""
    R = 6371.0  # Earth radius in km
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    distance_km = R * c
    
    # Convert km to degrees (approximate: 1 degree ≈ 111 km at equator)
    distance_degrees = distance_km / 111.0
    return distance_degrees


def style_two_axis_plot(ax: plt.Axes) -> None:
    """Apply a cleaner two-axis style: keep only left/bottom spines and add a light grid."""
    ax.grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", labelsize=11)


def normalize_pathway_label(value: str) -> str:
    """Normalize pathway labels so '1-2' and '1 -> 2' can be matched consistently."""
    return str(value).replace(" ", "").replace("->", "-")


def get_precip_color_values(df: pd.DataFrame) -> pd.Series:
    """Return the preferred precipitation metric for marker coloring."""
    if "traj_max_precipitation99" in df.columns:
        return pd.to_numeric(df["traj_max_precipitation99"], errors="coerce")
    if "precipitation99" in df.columns:
        return pd.to_numeric(df["precipitation99"], errors="coerce")
    if "max_precip_intensity" in df.columns:
        return pd.to_numeric(df["max_precip_intensity"], errors="coerce")
    if "n_precip" in df.columns:
        return pd.to_numeric(df["n_precip"], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def get_hail_size_values(df: pd.DataFrame) -> pd.Series:
    """Return hail metric used to size markers."""
    if "traj_max_hail_intensity" in df.columns:
        return pd.to_numeric(df["traj_max_hail_intensity"], errors="coerce")
    if "max_hail_intensity" in df.columns:
        return pd.to_numeric(df["max_hail_intensity"], errors="coerce")
    if "mean_hail_intensity" in df.columns:
        return pd.to_numeric(df["mean_hail_intensity"], errors="coerce")
    if "n_hail" in df.columns:
        return pd.to_numeric(df["n_hail"], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def hail_size_for_plot(hail_values: pd.Series) -> np.ndarray:
    """Map hail values to discrete plotting sizes using [0-5), [5-10), and [10, +inf)."""
    hail = pd.to_numeric(hail_values, errors="coerce").to_numpy(dtype=float)
    sizes = np.full(hail.shape, 30.0)
    sizes[(hail >= 0) & (hail < 5)] = 30.0
    sizes[(hail >= 5) & (hail < 10)] = 90.0
    sizes[hail >= 10] = 170.0
    sizes[~np.isfinite(hail)] = 30.0
    return sizes


def count_boundary_near_crops(df: pd.DataFrame) -> pd.DataFrame:
    """Count crops per pathway whose midpoint is within the padding distance of the domain boundary."""
    lon_min, lon_max, lat_min, lat_max = DOMAIN_EXTENT
    boundary_mask = (
        (df["center_lon"] <= lon_min + CROP_HALF_SIDE_DEG)
        | (df["center_lon"] >= lon_max - CROP_HALF_SIDE_DEG)
        | (df["center_lat"] <= lat_min + CROP_HALF_SIDE_DEG)
        | (df["center_lat"] >= lat_max - CROP_HALF_SIDE_DEG)
    )

    summary = (
        df.assign(boundary_crop=boundary_mask)
        .groupby("pathway", dropna=False)["boundary_crop"]
        .agg(total_crops="count", boundary_crops="sum")
        .reset_index()
    )
    summary["boundary_fraction"] = summary["boundary_crops"] / summary["total_crops"]
    return summary


def load_cloud_properties_test() -> dict:
    """Load cloud properties from test files: cma, cth50, cot50, cth_very_high, cot_thick."""
    test_props = {
        "cma": [],
        "cth50": [],
        "cot50": [],
        "cth_very_high": [],
        "cot_thick": []
    }
    
    # Load cma and raw percentiles (cth50, cot50) from percentiles file
    if os.path.exists(TEST_PERCENTILES_CSV):
        df_pct = pd.read_csv(TEST_PERCENTILES_CSV, low_memory=False)
        
        # Extract cma (from "None" column where var == "cma")
        df_cma = df_pct[df_pct["var"] == "cma"].copy()
        if not df_cma.empty:
            test_props["cma"] = pd.to_numeric(df_cma["None"], errors="coerce").dropna().values
        
        # Extract cth50 (from "50" column where var == "cth")
        df_cth = df_pct[df_pct["var"] == "cth"].copy()
        if not df_cth.empty:
            test_props["cth50"] = pd.to_numeric(df_cth["50"], errors="coerce").dropna().values
        
        # Extract cot50 (from "50" column where var == "cot_percentiles")
        df_cot = df_pct[df_pct["var"] == "cot_percentiles"].copy()
        if not df_cot.empty:
            test_props["cot50"] = pd.to_numeric(df_cot["50"], errors="coerce").dropna().values
    
    # Load fractions (cth_very_high, cot_thick) from fractions file
    if os.path.exists(TEST_FRACTIONS_CSV):
        df_frac = pd.read_csv(TEST_FRACTIONS_CSV, low_memory=False)
        
        # Extract cth_very_high
        df_cth_vh = df_frac[df_frac["var"] == "cth_very_high"].copy()
        if not df_cth_vh.empty:
            test_props["cth_very_high"] = pd.to_numeric(df_cth_vh["None"], errors="coerce").dropna().values
        
        # Extract cot_thick
        df_cot_th = df_frac[df_frac["var"] == "cot_thick"].copy()
        if not df_cot_th.empty:
            test_props["cot_thick"] = pd.to_numeric(df_cot_th["None"], errors="coerce").dropna().values
    
    return test_props


def load_cloud_properties_train() -> dict:
    """Load cloud properties from training files: cma, cth50, cot50, cth_very_high, cot_thick."""
    train_props = {
        "cma": [],
        "cth50": [],
        "cot50": [],
        "cth_very_high": [],
        "cot_thick": []
    }
    
    # Load cma and raw percentiles (cth50, cot50) from CVC file
    if os.path.exists(TRAIN_CVC_CSV):
        df_cvc = pd.read_csv(TRAIN_CVC_CSV, low_memory=False)
        
        # Extract cma (from "None" column where var == "cma")
        df_cma = df_cvc[df_cvc["var"] == "cma"].copy()
        if not df_cma.empty:
            train_props["cma"] = pd.to_numeric(df_cma["None"], errors="coerce").dropna().values
        
        # Extract cth50 (from "50" column where var == "cth")
        df_cth = df_cvc[df_cvc["var"] == "cth"].copy()
        if not df_cth.empty:
            train_props["cth50"] = pd.to_numeric(df_cth["50"], errors="coerce").dropna().values
        
        # Extract cot50 (from "50" column where var == "cot")
        df_cot = df_cvc[df_cvc["var"] == "cot"].copy()
        if not df_cot.empty:
            train_props["cot50"] = pd.to_numeric(df_cot["50"], errors="coerce").dropna().values
    
    # Load fractions (cth_very_high, cot_thick) from fractions file
    if os.path.exists(TRAIN_FRACTIONS_CSV):
        df_frac = pd.read_csv(TRAIN_FRACTIONS_CSV, low_memory=False)
        
        # Extract cth_very_high
        df_cth_vh = df_frac[df_frac["var"] == "cth_very_high"].copy()
        if not df_cth_vh.empty:
            train_props["cth_very_high"] = pd.to_numeric(df_cth_vh["None"], errors="coerce").dropna().values
        
        # Extract cot_thick
        df_cot_th = df_frac[df_frac["var"] == "cot_thick"].copy()
        if not df_cot_th.empty:
            train_props["cot_thick"] = pd.to_numeric(df_cot_th["None"], errors="coerce").dropna().values
    
    return train_props


def print_cloud_property_comparison(test_props: dict, train_props: dict):
    """Print a comparison table of cloud properties between test and train sets."""
    print("\n=== CLOUD PROPERTY STATISTICS (TEST vs TRAIN) ===\n")
    
    properties = ["cma", "cth_very_high", "cot_thick", "cot50", "cth50"]
    
    # Print header
    print(f"{'Property':<20} {'Set':<10} {'N':<8} {'Mean':<12} {'Std':<12}")
    print("-" * 62)
    
    for prop in properties:
        test_vals = test_props.get(prop, np.array([]))
        train_vals = train_props.get(prop, np.array([]))
        
        if len(test_vals) > 0:
            test_mean = np.mean(test_vals)
            test_std = np.std(test_vals)
            print(f"{prop:<20} {'TEST':<10} {len(test_vals):<8} {test_mean:<12.4f} {test_std:<12.4f}")
        else:
            print(f"{prop:<20} {'TEST':<10} {'N/A':<8} {'N/A':<12} {'N/A':<12}")
        
        if len(train_vals) > 0:
            train_mean = np.mean(train_vals)
            train_std = np.std(train_vals)
            print(f"{'':<20} {'TRAIN':<10} {len(train_vals):<8} {train_mean:<12.4f} {train_std:<12.4f}")
        else:
            print(f"{'':<20} {'TRAIN':<10} {'N/A':<8} {'N/A':<12} {'N/A':<12}")
        
        print()



def compute_duration_distance_per_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trajectory duration (hours) and distance (degrees) for each trajectory."""
    records = []

    for traj_id, g in df.groupby("trajectory_id"):
        g = g.sort_values("dt").copy()
        
        # Duration in hours
        start_time = g["dt"].iloc[0]
        end_time = g["dt"].iloc[-1]
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        
        # Distance from first to last crop (in degrees)
        lat_start = g["center_lat"].iloc[0]
        lon_start = g["center_lon"].iloc[0]
        lat_end = g["center_lat"].iloc[-1]
        lon_end = g["center_lon"].iloc[-1]
        distance_degrees = haversine_degrees(lat_start, lon_start, lat_end, lon_end)
        
        # Get pathway
        pathway = str(g["pathway"].iloc[0])
        n_timestamps = len(g)

        precip_max = np.nan
        if "precipitation99" in g.columns:
            precip_max = pd.to_numeric(g["precipitation99"], errors="coerce").max()

        hail_max = np.nan
        if "max_hail_intensity" in g.columns:
            hail_max = pd.to_numeric(g["max_hail_intensity"], errors="coerce").max()
        
        records.append({
            "trajectory_id": int(traj_id),
            "pathway": pathway,
            "duration_hours": float(duration_hours),
            "distance_degrees": float(distance_degrees),
            "n_timestamps": int(n_timestamps),
            "start_time": start_time,
            "end_time": end_time,
            "traj_max_precipitation99": float(precip_max) if pd.notna(precip_max) else np.nan,
            "traj_max_hail_intensity": float(hail_max) if pd.notna(hail_max) else np.nan,
        })

    return pd.DataFrame(records)


def compute_growth_per_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first-to-last growth for selected cloud properties on each trajectory."""
    growth_specs = {
        "area_growth": "cma",
        # vertical handled separately to produce both abs and percent
        "core_growth_cth10p": "cth_very_high",
        "core_growth_cot30p": "cot_thick",
    }

    records = []

    for traj_id, g in df.groupby("trajectory_id"):
        g = g.sort_values("dt").copy()
        record = {
            "trajectory_id": int(traj_id),
            "pathway": str(g["pathway"].iloc[0]),
        }

        # handle vertical growth separately (both absolute and percent)
        if "cth50" in g.columns:
            s_cth = pd.to_numeric(g["cth50"], errors="coerce")
            if not s_cth.dropna().empty:
                initial_cth = s_cth.iloc[0]
                final_cth = s_cth.iloc[-1]
                # absolute difference
                if np.isfinite(initial_cth) and np.isfinite(final_cth):
                    record["vertical_growth_abs"] = float(final_cth - initial_cth)
                else:
                    record["vertical_growth_abs"] = np.nan

                # percent change
                if np.isfinite(initial_cth) and initial_cth != 0 and np.isfinite(final_cth):
                    record["vertical_growth"] = float((final_cth - initial_cth) / initial_cth * 100.0)
                else:
                    record["vertical_growth"] = np.nan
            else:
                record["vertical_growth_abs"] = np.nan
                record["vertical_growth"] = np.nan
        else:
            record["vertical_growth_abs"] = np.nan
            record["vertical_growth"] = np.nan

        for growth_name, source_col in growth_specs.items():
            # compute percent change relative to initial value: (final - initial) / initial * 100
            if source_col not in g.columns:
                record[growth_name] = np.nan
                continue

            series = pd.to_numeric(g[source_col], errors="coerce")
            if series.dropna().empty:
                record[growth_name] = np.nan
                continue

            initial = series.iloc[0]
            final = series.iloc[-1]
            if not np.isfinite(initial) or initial == 0:
                record[growth_name] = np.nan
            else:
                record[growth_name] = float((final - initial) / initial * 100.0)

        records.append(record)

    return pd.DataFrame(records)


def plot_growth_distribution(df: pd.DataFrame, metric: str, title: str, out_dir: str, filename_prefix: str, unit_label: str):
    """Plot overall and per-pathway growth distributions for one metric."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    values = df[metric].dropna().values
    axes[0].hist(values, bins=30, color="#4daf4a", edgecolor="black", alpha=0.8)
    axes[0].set_title(f"{title} (all pathways)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(unit_label)
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["left"].set_linewidth(1.5)
    axes[0].spines["bottom"].set_linewidth(1.5)
    axes[0].tick_params(axis="both", which="major", labelsize=10)
    if len(values) > 0:
        axes[0].text(
            0.98,
            0.97,
            f"N={len(values)}\nMean={values.mean():.4f}\nStd={values.std():.4f}",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    pathways = sorted(df["pathway"].astype(str).unique())
    box_data = [df.loc[df["pathway"].astype(str) == p, metric].dropna().values for p in pathways]
    axes[1].boxplot(box_data, labels=pathways, showfliers=False)
    axes[1].set_title(f"{title} per pathway", fontsize=12, fontweight="bold")
    axes[1].set_ylabel(unit_label)
    axes[1].set_xlabel("Pathway")
    axes[1].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["left"].set_linewidth(1.5)
    axes[1].spines["bottom"].set_linewidth(1.5)
    axes[1].tick_params(axis="both", which="major", labelsize=10)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    outfile = os.path.join(out_dir, f"{filename_prefix}.png")
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


def print_growth_statistics(df: pd.DataFrame):
    """Print growth statistics overall and per pathway."""
    growth_labels = {
        "area_growth": "Area growth % (cma final - initial) / initial *100",
        "vertical_growth": "Vertical growth % (cth50 final - initial) / initial *100",
        "vertical_growth_abs": "Vertical growth ABS (cth50 final - initial)",
        "core_growth_cth10p": "Core growth CTH10+ % (cth_very_high final - initial) / initial *100",
        "core_growth_cot30p": "Core growth COT30+ % (cot_thick final - initial) / initial *100",
    }

    print("\n=== GROWTH STATISTICS ===")
    for metric, label in growth_labels.items():
        values = df[metric].dropna()
        print(f"\n{label}:")
        if len(values) == 0:
            print("  No valid values")
            continue
        print(f"  Overall: N={len(values)}, mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
        print("  Per pathway:")
        for pathway in sorted(df["pathway"].astype(str).unique()):
            subset = df.loc[df["pathway"].astype(str) == pathway, metric].dropna()
            if len(subset) == 0:
                print(f"    {pathway}: N=0")
                continue
            print(
                f"    {pathway}: N={len(subset)}, mean={subset.mean():.4f}, std={subset.std():.4f}, "
                f"min={subset.min():.4f}, max={subset.max():.4f}"
            )


def compute_vertical_velocity_per_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vertical velocity (cth50 change in m/s) for each crop in each trajectory."""
    records = []

    for traj_id, g in df.groupby("trajectory_id"):
        g = g.sort_values("dt").copy()
        pathway = str(g["pathway"].iloc[0])

        # Check if we have required columns
        if "cth50" not in g.columns or "t_aligned" not in g.columns or "dt" not in g.columns:
            continue

        cth_vals = pd.to_numeric(g["cth50"], errors="coerce")
        t_aligned_vals = pd.to_numeric(g["t_aligned"], errors="coerce")
        datetimes = g["dt"]

        if cth_vals.dropna().empty or t_aligned_vals.dropna().empty:
            continue

        # Compute velocities between consecutive timestamps
        for i in range(len(g) - 1):
            cth_current = cth_vals.iloc[i]
            cth_next = cth_vals.iloc[i + 1]
            t_aligned = t_aligned_vals.iloc[i]
            dt_current = datetimes.iloc[i]
            dt_next = datetimes.iloc[i + 1]

            # Skip if either value is missing
            if not (np.isfinite(cth_current) and np.isfinite(cth_next)):
                continue

            # Compute time difference in seconds
            time_diff_seconds = (dt_next - dt_current).total_seconds()
            if time_diff_seconds <= 0:
                continue

            # Velocity in meters per second (cth50 is in meters)
            velocity_ms = (cth_next - cth_current) / time_diff_seconds

            records.append({
                "trajectory_id": int(traj_id),
                "pathway": pathway,
                "t_aligned": float(t_aligned),
                "vertical_velocity_ms": float(velocity_ms),
            })

    return pd.DataFrame(records)


def plot_vertical_velocity_scatter(df: pd.DataFrame, out_dir: str):
    """Plot vertical velocity vs t_aligned: one overall plot and one per pathway."""
    if df.empty:
        print("No velocity data available for plotting")
        return
    
    pathways = sorted(df["pathway"].astype(str).unique())
    
    # Overall scatter plot with all pathways
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(pathways)))
    
    for pathway, color in zip(pathways, colors):
        mask = df["pathway"].astype(str) == pathway
        subset = df[mask]
        ax.scatter(
            subset["t_aligned"],
            subset["vertical_velocity_ms"],
            label=pathway,
            alpha=0.6,
            s=30,
            color=color
        )
    
    ax.set_xlabel("Time within trajectory (aligned, hours)", fontsize=12)
    ax.set_ylabel("Vertical velocity (m/s)", fontsize=12)
    ax.set_title("Vertical Velocity vs Time (All Pathways)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=1)
    
    plt.tight_layout()
    outfile = os.path.join(out_dir, "vertical_velocity_all_pathways.png")
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")
    
    # Per-pathway scatter plots
    for pathway, color in zip(pathways, colors):
        mask = df["pathway"].astype(str) == pathway
        subset = df[mask]
        
        if len(subset) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            subset["t_aligned"],
            subset["vertical_velocity_ms"],
            alpha=0.7,
            s=40,
            color=color
        )
        
        ax.set_xlabel("Time within trajectory (aligned, hours)", fontsize=12)
        ax.set_ylabel("Vertical velocity (m/s)", fontsize=12)
        ax.set_title(f"Vertical Velocity vs Time - Pathway {pathway}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        
        # Add statistics box
        vel_data = subset["vertical_velocity_ms"].dropna()
        stats_text = f"N={len(vel_data)}\nMean={vel_data.mean():.4f} m/s\nStd={vel_data.std():.4f} m/s\nMin={vel_data.min():.4f} m/s\nMax={vel_data.max():.4f} m/s"
        ax.text(
            0.98, 0.97,
            stats_text,
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            fontsize=10
        )
        
        plt.tight_layout()
        pathway_safe = str(pathway).replace(" ", "_").replace("->", "to").replace("/", "_")
        outfile = os.path.join(out_dir, f"vertical_velocity_pathway_{pathway_safe}.png")
        plt.savefig(outfile, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outfile}")


def plot_vertical_velocity_hexbin(df: pd.DataFrame, out_dir: str):
    """Plot vertical velocity as hexbin (2D density) overall and top pathways."""
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall hexbin
    hb = axes[0].hexbin(df["t_aligned"], df["vertical_velocity_ms"], gridsize=25, cmap="YlOrRd", mincnt=1)
    axes[0].axhline(0, color="blue", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[0].set_title("Vertical Velocity Density (all pathways)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Trajectory Time (t_aligned)")
    axes[0].set_ylabel("Vertical Velocity (m/s)")
    cb = plt.colorbar(hb, ax=axes[0])
    cb.set_label("Count", fontsize=10)

    # Per-pathway: select top 3 pathways by count
    pathway_counts = df["pathway"].value_counts()
    top_pathways = pathway_counts.head(3).index.tolist()

    for pathway in top_pathways:
        mask = df["pathway"].astype(str) == pathway
        subset = df[mask]
        if len(subset) < 5:
            continue
        axes[1].hexbin(subset["t_aligned"], subset["vertical_velocity_ms"], gridsize=20, cmap="Blues", mincnt=1, alpha=0.7)

    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[1].set_title(f"Vertical Velocity Density (top 3 pathways: {', '.join(str(p) for p in top_pathways[:3])})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Trajectory Time (t_aligned)")
    axes[1].set_ylabel("Vertical Velocity (m/s)")

    plt.tight_layout()
    outfile = os.path.join(out_dir, "vertical_velocity_hexbin_density.png")
    plt.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")


def plot_duration_distribution(df: pd.DataFrame, out_dir: str):
    """Plot trajectory duration distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    vals = df["duration_hours"].dropna().values
    axes[0].hist(vals, bins=30, color="#377eb8", edgecolor="black", alpha=0.8)
    axes[0].set_title("Duration distribution (all pathways)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Duration (hours)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["left"].set_linewidth(1.5)
    axes[0].spines["bottom"].set_linewidth(1.5)
    axes[0].tick_params(axis="both", which="major", labelsize=10)
    axes[0].text(0.98, 0.97, f"N={len(vals)}\nMean={vals.mean():.1f}h\nStd={vals.std():.1f}h", 
                 transform=axes[0].transAxes, ha="right", va="top", 
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # Per pathway boxplot
    pathways = sorted(df["pathway"].astype(str).unique())
    box_data = [df.loc[df["pathway"].astype(str) == p, "duration_hours"].dropna().values for p in pathways]
    axes[1].boxplot(box_data, labels=pathways, showfliers=False)
    axes[1].set_title("Duration distribution per pathway", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Duration (hours)")
    axes[1].set_xlabel("Pathway")
    axes[1].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["left"].set_linewidth(1.5)
    axes[1].spines["bottom"].set_linewidth(1.5)
    axes[1].tick_params(axis="both", which="major", labelsize=10)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_duration_distributions.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, 'trajectory_duration_distributions.png')}")


def plot_distance_distribution(df: pd.DataFrame, out_dir: str):
    """Plot trajectory distance distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    vals = df["distance_degrees"].dropna().values
    axes[0].hist(vals, bins=30, color="#e41a1c", edgecolor="black", alpha=0.8)
    axes[0].set_title("Distance distribution (all pathways)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Distance (degrees)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["left"].set_linewidth(1.5)
    axes[0].spines["bottom"].set_linewidth(1.5)
    axes[0].tick_params(axis="both", which="major", labelsize=10)
    axes[0].text(0.98, 0.97, f"N={len(vals)}\nMean={vals.mean():.2f}°\nStd={vals.std():.2f}°", 
                 transform=axes[0].transAxes, ha="right", va="top", 
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # Per pathway boxplot
    pathways = sorted(df["pathway"].astype(str).unique())
    box_data = [df.loc[df["pathway"].astype(str) == p, "distance_degrees"].dropna().values for p in pathways]
    axes[1].boxplot(box_data, labels=pathways, showfliers=False)
    axes[1].set_title("Distance distribution per pathway", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Distance (degrees)")
    axes[1].set_xlabel("Pathway")
    axes[1].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["left"].set_linewidth(1.5)
    axes[1].spines["bottom"].set_linewidth(1.5)
    axes[1].tick_params(axis="both", which="major", labelsize=10)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_distance_distributions.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, 'trajectory_distance_distributions.png')}")


def plot_duration_distance_scatter(df: pd.DataFrame, out_dir: str):
    """Plot scatter plot of duration vs distance per pathway."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    pathways = sorted(df["pathway"].astype(str).unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(pathways)))
    
    for pathway, color in zip(pathways, colors):
        mask = df["pathway"].astype(str) == pathway
        subset = df[mask]
        ax.scatter(subset["duration_hours"], subset["distance_degrees"], 
                  label=pathway, alpha=0.6, s=50, color=color)
    
    ax.set_xlabel("Duration (hours)", fontsize=11)
    ax.set_ylabel("Distance (degrees)", fontsize=11)
    ax.set_title("Trajectory Duration vs Distance per Pathway", fontsize=12, fontweight="bold")
    style_two_axis_plot(ax)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_duration_vs_distance_scatter.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, 'trajectory_duration_vs_distance_scatter.png')}")


def plot_pathway_pair_scatter(df: pd.DataFrame, out_dir: str):
    """Plot duration vs distance for pathway 2 and pathway 1-2 in a two-panel figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    panel_specs = [
        ("2", "c) DC", axes[0]),
        ("1-2", "d) EC-> DC", axes[1]),
    ]

    pathway_norm = df["pathway"].astype(str).map(normalize_pathway_label)
    precip_values = get_precip_color_values(df)
    hail_values = get_hail_size_values(df)
    precip_array = precip_values.to_numpy(dtype=float)
    finite_precip = np.isfinite(precip_array)
    shared_vmin = 10.0
    shared_vmax = 50.0
    scatter_mappable = None

    for pathway, title, ax in panel_specs:
        subset = df.loc[pathway_norm == normalize_pathway_label(pathway)].copy()
        subset_precip = precip_values.loc[subset.index]
        subset_hail = hail_values.loc[subset.index]
        subset_sizes = hail_size_for_plot(subset_hail)
        subset_precip_array = subset_precip.to_numpy(dtype=float)
        valid_precip_mask = np.isfinite(subset_precip_array)
        missing_precip_mask = ~valid_precip_mask

        if subset.empty:
            ax.text(0.5, 0.5, f"No data for pathway {pathway}", ha="center", va="center")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Duration (hours)", fontsize=12)
            ax.set_ylabel("Distance (degrees)", fontsize=12)
            style_two_axis_plot(ax)
            continue

        if "traj_max_hail_intensity" in subset.columns:
            subset_hail = pd.to_numeric(subset["traj_max_hail_intensity"], errors="coerce")
        else:
            subset_hail = pd.Series(np.nan, index=subset.index)
        subset_sizes = hail_size_for_plot(subset_hail)

        if valid_precip_mask.any():
            scatter_mappable = ax.scatter(
                subset.loc[valid_precip_mask, "duration_hours"],
                subset.loc[valid_precip_mask, "distance_degrees"],
                s=subset_sizes[valid_precip_mask],
                alpha=0.75,
                c=subset_precip_array[valid_precip_mask],
                cmap=cmc.hawaii,
                vmin=shared_vmin,
                vmax=shared_vmax,
                edgecolors="none",
            )
        # Do not plot trajectories without precipitation99 values in the scatter panels.
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Duration (hours)", fontsize=13)
        ax.set_ylabel("Distance (degrees)", fontsize=13)
        style_two_axis_plot(ax)

        ax.tick_params(axis="both", which="major", labelsize=13)

    if scatter_mappable is not None:
        cbar = fig.colorbar(scatter_mappable, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("Max Rain Rate (mm/h)", fontsize=12)
        cbar.set_ticks([10, 20, 30, 40, 50])
        cbar.ax.tick_params(labelsize=11)

    size_legend_values = [
        (30.0, "[0-5) cm"),
        (90.0, "[5-10) cm"),
        (170.0, "[10-) cm")
    ]
    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=np.sqrt(size) / 1.4,
            markerfacecolor="0.3",
            markeredgecolor="0.3",
            label=label,
        )
        for size, label in size_legend_values
    ]
    axes[1].legend(
        handles=legend_handles,
        title="Hail max diameter",
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        edgecolor="0.8",
        fontsize=11,
        title_fontsize=12,
    )

    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.12, top=0.9, wspace=0.22)
    filename = "trajectory_duration_distance_pathway_2_and_1_2_scatter.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, filename)}")


def plot_per_pathway_distributions(df: pd.DataFrame, out_dir: str):
    """Create individual distribution plots for each pathway."""
    pathways = sorted(df["pathway"].astype(str).unique())
    
    for pathway in pathways:
        mask = df["pathway"].astype(str) == pathway
        subset = df[mask]
        
        if len(subset) == 0:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Duration
        vals_dur = subset["duration_hours"].dropna().values
        axes[0].hist(vals_dur, bins=min(20, max(5, len(vals_dur)//2)), 
                    color="#377eb8", edgecolor="black", alpha=0.8)
        axes[0].set_title(f"{pathway} - Duration (N={len(vals_dur)})", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Duration (hours)")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["left"].set_linewidth(1.5)
        axes[0].spines["bottom"].set_linewidth(1.5)
        axes[0].tick_params(axis="both", which="major", labelsize=10)
        if len(vals_dur) > 0:
            axes[0].text(0.98, 0.97, f"Mean={vals_dur.mean():.1f}h\nStd={vals_dur.std():.1f}h\nMin={vals_dur.min():.1f}h\nMax={vals_dur.max():.1f}h", 
                        transform=axes[0].transAxes, ha="right", va="top", 
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=9)
        
        # Distance
        vals_dist = subset["distance_degrees"].dropna().values
        axes[1].hist(vals_dist, bins=min(20, max(5, len(vals_dist)//2)), 
                    color="#e41a1c", edgecolor="black", alpha=0.8)
        axes[1].set_title(f"{pathway} - Distance (N={len(vals_dist)})", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Distance (degrees)")
        axes[1].set_ylabel("Count")
        axes[1].grid(True, alpha=0.3, color="grey", linestyle="-", linewidth=0.5)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["left"].set_linewidth(1.5)
        axes[1].spines["bottom"].set_linewidth(1.5)
        axes[1].tick_params(axis="both", which="major", labelsize=10)
        if len(vals_dist) > 0:
            axes[1].text(0.98, 0.97, f"Mean={vals_dist.mean():.2f}°\nStd={vals_dist.std():.2f}°\nMin={vals_dist.min():.2f}°\nMax={vals_dist.max():.2f}°", 
                        transform=axes[1].transAxes, ha="right", va="top", 
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=9)
        
        plt.tight_layout()
        filename = f"pathway_{pathway.replace('/', '_').replace(' ', '_')}_distributions.png"
        plt.savefig(os.path.join(out_dir, filename), dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {os.path.join(out_dir, filename)}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Loading pathways: {PATHWAY_CSV}")
    df = load_pathway_df(PATHWAY_CSV, MIDPOINT_CSV)
    print(f"Rows after preprocessing: {len(df)}")
    print(f"Trajectories: {df['trajectory_id'].nunique()}")

    print("Computing duration and distance per trajectory...")
    df_results = compute_duration_distance_per_trajectory(df)
    print(f"Results: {len(df_results)} trajectories")

    # Save results
    results_csv = os.path.join(OUT_DIR, "trajectory_duration_distance.csv")
    df_results.to_csv(results_csv, index=False)
    print(f"\nSaved results: {results_csv}")

    # Summary statistics
    print("\n=== OVERALL STATISTICS ===")
    print(f"\nDuration (hours):")
    print(f"  Mean: {df_results['duration_hours'].mean():.2f}h")
    print(f"  Median: {df_results['duration_hours'].median():.2f}h")
    print(f"  Std: {df_results['duration_hours'].std():.2f}h")
    print(f"  Min: {df_results['duration_hours'].min():.2f}h")
    print(f"  Max: {df_results['duration_hours'].max():.2f}h")

    print(f"\nDistance (degrees):")
    print(f"  Mean: {df_results['distance_degrees'].mean():.3f}°")
    print(f"  Median: {df_results['distance_degrees'].median():.3f}°")
    print(f"  Std: {df_results['distance_degrees'].std():.3f}°")
    print(f"  Min: {df_results['distance_degrees'].min():.3f}°")
    print(f"  Max: {df_results['distance_degrees'].max():.3f}°")

    print("\n=== PER PATHWAY STATISTICS ===")
    for pathway in sorted(df_results["pathway"].unique()):
        mask = df_results["pathway"] == pathway
        subset = df_results[mask]
        print(f"\n{pathway} (N={len(subset)}):")
        print(f"  Duration: {subset['duration_hours'].mean():.2f}±{subset['duration_hours'].std():.2f}h")
        print(f"  Travel distance: {subset['distance_degrees'].mean():.3f}±{subset['distance_degrees'].std():.3f}°")

    print("\n=== BOUNDARY-NEAR CROPS (padding effect, 2° from domain edge) ===")
    boundary_summary = count_boundary_near_crops(df)
    for _, row in boundary_summary.sort_values("pathway").iterrows():
        print(
            f"{row['pathway']}: {int(row['boundary_crops'])}/{int(row['total_crops'])} crops "
            f"near boundaries ({row['boundary_fraction']:.1%})"
        )

    print("\nComputing growth metrics per trajectory...")
    df_growth = compute_growth_per_trajectory(df)
    print(f"Growth results: {len(df_growth)} trajectories")

    growth_csv = os.path.join(OUT_DIR, "trajectory_growth.csv")
    df_growth.to_csv(growth_csv, index=False)
    print(f"Saved growth results: {growth_csv}")

    print_growth_statistics(df_growth)

    print("\nComputing vertical velocity per trajectory...")
    df_velocity = compute_vertical_velocity_per_trajectory(df)
    print(f"Velocity results: {len(df_velocity)} crop transitions")

    velocity_csv = os.path.join(OUT_DIR, "trajectory_vertical_velocity.csv")
    df_velocity.to_csv(velocity_csv, index=False)
    print(f"Saved velocity results: {velocity_csv}")

    # Load and print cloud property statistics
    print("\nLoading cloud property data...")
    test_props = load_cloud_properties_test()
    train_props = load_cloud_properties_train()
    print_cloud_property_comparison(test_props, train_props)

    # Create plots
    print(f"\nCreating plots in: {PLOTS_DIR}")
    plot_duration_distribution(df_results, PLOTS_DIR)
    plot_distance_distribution(df_results, PLOTS_DIR)
    plot_duration_distance_scatter(df_results, PLOTS_DIR)
    plot_pathway_pair_scatter(df_results, PLOTS_DIR)
    plot_per_pathway_distributions(df_results, PLOTS_DIR)

    plot_growth_distribution(
        df_growth,
        metric="area_growth",
        title="Area Growth (percent)",
        out_dir=PLOTS_DIR,
        filename_prefix="trajectory_area_growth_distributions",
        unit_label="%",
    )
    # vertical: absolute and percent
    plot_growth_distribution(
        df_growth,
        metric="vertical_growth_abs",
        title="Vertical Growth (absolute cth50 difference)",
        out_dir=PLOTS_DIR,
        filename_prefix="trajectory_vertical_growth_abs_distributions",
        unit_label="cth50 (abs)",
    )
    plot_growth_distribution(
        df_growth,
        metric="vertical_growth",
        title="Vertical Growth (percent)",
        out_dir=PLOTS_DIR,
        filename_prefix="trajectory_vertical_growth_distributions",
        unit_label="%",
    )
    plot_growth_distribution(
        df_growth,
        metric="core_growth_cth10p",
        title="Core Growth CTH10+ (percent)",
        out_dir=PLOTS_DIR,
        filename_prefix="trajectory_core_growth_cth10p_distributions",
        unit_label="%",
    )
    plot_growth_distribution(
        df_growth,
        metric="core_growth_cot30p",
        title="Core Growth COT30+ (percent)",
        out_dir=PLOTS_DIR,
        filename_prefix="trajectory_core_growth_cot30p_distributions",
        unit_label="%",
    )

    print(f"\nCreating velocity plots...")
    if not df_velocity.empty:
        plot_vertical_velocity_scatter(df_velocity, PLOTS_DIR)
        plot_vertical_velocity_hexbin(df_velocity, PLOTS_DIR)
    else:
        print("No velocity data to plot.")

    print(f"\nAll plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
