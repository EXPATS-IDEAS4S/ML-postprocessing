import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
PATHWAY_CSV = os.path.join(BASE_DIR, "pathway_analysis/df_pathways_merged_no_dominance.csv")
OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"
OUT_DIR = os.path.join(BASE_DIR, "pathway_orography_analysis")

CLASS_ID_TO_SHORT = {1: "EC", 2: "DC", 4: "OA"}
CLASS_COLORS = {1: "#fdae61", 2: "#d73027", 4: "#8b46a1"}
TARGET_CLASSES = [1, 2, 4]
TOP_N_EXTREME = 50


def sample_dem_at_points(df, orog_ds, lat_col="lat", lon_col="lon", dem_var="DEM"):
    """Sample DEM at point locations using nearest-neighbor lookup."""
    sampled = orog_ds[dem_var].sel(
        lat=xr.DataArray(df[lat_col].values, dims="points"),
        lon=xr.DataArray(df[lon_col].values, dims="points"),
        method="nearest",
    )
    return sampled.values


def load_and_prepare(pathway_csv, orog_path):
    df = pd.read_csv(pathway_csv, low_memory=False)

    required = ["label", "lat", "lon", "pathway", "storm_id", "datetime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in pathway CSV: {missing}")

    df = df[df["label"].isin(TARGET_CLASSES)].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["t_aligned"] = pd.to_numeric(df.get("t_aligned", np.nan), errors="coerce")
    df["t_bin"] = pd.to_numeric(df.get("t_bin", np.nan), errors="coerce")
    df = df.dropna(subset=["datetime", "lat", "lon"])

    ds_orog = xr.open_dataset(orog_path)
    if "DEM" not in ds_orog.data_vars:
        raise ValueError(f"DEM variable not found in {orog_path}. Variables: {list(ds_orog.data_vars)}")
    if "lat" not in ds_orog.coords or "lon" not in ds_orog.coords:
        raise ValueError(f"lat/lon coords not found in {orog_path}. Coords: {list(ds_orog.coords)}")

    df["dem"] = sample_dem_at_points(df, ds_orog)
    df["class_short"] = df["label"].map(CLASS_ID_TO_SHORT)

    return df


def select_extreme_subset(df, metric_col, n_top=50, fallback_col=None):
    """Select top-N rows by a metric column (events-based selection)."""
    if metric_col not in df.columns:
        if fallback_col is None or fallback_col not in df.columns:
            raise ValueError(
                f"Metric column '{metric_col}' not found and fallback '{fallback_col}' unavailable."
            )
        metric_col = fallback_col

    out = df.dropna(subset=[metric_col]).copy()
    if out.empty:
        return out
    return out.nlargest(n_top, metric_col)


def plot_dem_distribution(df, out_file):
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    bins = np.linspace(float(df["dem"].min()), float(df["dem"].max()), 60)
    for class_id in TARGET_CLASSES:
        sub = df[df["label"] == class_id]
        if sub.empty:
            continue
        ax.hist(
            sub["dem"].values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=CLASS_COLORS[class_id],
            label=CLASS_ID_TO_SHORT[class_id],
        )

    ax.set_xlabel("DEM (m)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Class", fontsize=10)
    ax.set_title("DEM Distribution by Convective Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_quantile_series(df_sub, time_col, value_col):
    q = (
        df_sub.groupby(time_col)[value_col]
        .quantile([0.25, 0.5, 0.75])
        .unstack(level=1)
        .rename(columns={0.25: "q25", 0.5: "q50", 0.75: "q75"})
        .sort_index()
    )
    return q


def get_time_axis_column(df):
    """Return time axis column with fallback computed from datetime per storm."""
    if "t_aligned" in df.columns and df["t_aligned"].notna().any():
        return "t_aligned"
    if "t_bin" in df.columns and df["t_bin"].notna().any():
        return "t_bin"

    # Fallback: relative time in 15-min units from storm start.
    if "datetime" not in df.columns or "storm_id" not in df.columns:
        raise ValueError("Cannot derive fallback time axis without datetime and storm_id columns.")

    out = df.copy()
    out["time_rel_15min"] = (
        out["datetime"] - out.groupby("storm_id")["datetime"].transform("min")
    ).dt.total_seconds() / 900.0
    return "time_rel_15min", out


def plot_time_aligned_by_pathway(df, out_dir):
    """Create one time-aligned plot per pathway with rows (DEM, lat, lon)."""
    if df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)

    time_info = get_time_axis_column(df)
    if isinstance(time_info, tuple):
        time_col, df = time_info
    else:
        time_col = time_info

    pathways = (
        df.groupby("pathway")["storm_id"].nunique().sort_values(ascending=False).index.tolist()
    )
    if len(pathways) == 0:
        raise ValueError("No pathways available for time-aligned plotting.")

    metrics = [
        ("dem", "DEM (m)"),
        ("lat", "Latitude"),
        ("lon", "Longitude"),
    ]

    for pathway in pathways:
        df_pw = df[df["pathway"] == pathway].copy()
        n_storms = df_pw["storm_id"].nunique()

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)

        for row_idx, (metric, y_label) in enumerate(metrics):
            ax = axes[row_idx]
            sub = df_pw.dropna(subset=[time_col, metric])
            if not sub.empty:
                q = _compute_quantile_series(sub, time_col, metric)
                x = q.index.values
                ax.plot(
                    x,
                    q["q50"].values,
                    color="black",
                    linewidth=2.2,
                )
                ax.fill_between(
                    x,
                    q["q25"].values,
                    q["q75"].values,
                    color="grey",
                    alpha=0.25,
                )

            if row_idx == 0:
                ax.set_title(f"{pathway} (n_storms={n_storms})", fontsize=12)
            ax.set_ylabel(y_label, fontsize=10)
            if row_idx == 2:
                ax.set_xlabel(time_col, fontsize=10)
            ax.grid(alpha=0.25, linestyle="--")
            ax.tick_params(labelsize=9)

        fig.suptitle("Time-Aligned Evolution (all classes combined, median and IQR)", fontsize=13, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.965])

        safe_pathway = str(pathway).replace(" ", "_").replace("/", "-")
        out_file = os.path.join(out_dir, f"time_aligned_{safe_pathway}.png")
        plt.savefig(out_file, dpi=220, bbox_inches="tight")
        plt.close(fig)


def compute_pathway_table(df):
    # Base location stats per pathway
    pathway_loc = (
        df.groupby("pathway")
        .agg(
            n_storms=("storm_id", "nunique"),
            mean_dem=("dem", "mean"),
            std_dem=("dem", "std"),
            mean_lat=("lat", "mean"),
            std_lat=("lat", "std"),
            mean_lon=("lon", "mean"),
            std_lon=("lon", "std"),
        )
        .reset_index()
    )

    # Per-storm trajectory gradients (final - initial)
    grad_records = []
    for (pathway, storm_id), g in df.sort_values("datetime").groupby(["pathway", "storm_id"]):
        g = g.dropna(subset=["lat", "lon", "dem", "datetime"])
        if g.empty:
            continue
        first = g.iloc[0]
        last = g.iloc[-1]
        grad_records.append(
            {
                "pathway": pathway,
                "storm_id": storm_id,
                "delta_lat": float(last["lat"] - first["lat"]),
                "delta_lon": float(last["lon"] - first["lon"]),
                "delta_dem": float(last["dem"] - first["dem"]),
            }
        )

    grad_df = pd.DataFrame(grad_records)
    if grad_df.empty:
        grad_stats = pd.DataFrame(columns=[
            "pathway",
            "mean_delta_lat",
            "std_delta_lat",
            "mean_delta_lon",
            "std_delta_lon",
            "mean_delta_dem",
            "std_delta_dem",
        ])
    else:
        grad_stats = (
            grad_df.groupby("pathway")
            .agg(
                mean_delta_lat=("delta_lat", "mean"),
                std_delta_lat=("delta_lat", "std"),
                mean_delta_lon=("delta_lon", "mean"),
                std_delta_lon=("delta_lon", "std"),
                mean_delta_dem=("delta_dem", "mean"),
                std_delta_dem=("delta_dem", "std"),
            )
            .reset_index()
        )

    out = pathway_loc.merge(grad_stats, on="pathway", how="left")
    out = out.sort_values("n_storms", ascending=False).reset_index(drop=True)

    # Create display-friendly mean +/- std columns.
    def pm(mean_col, std_col):
        return out.apply(
            lambda r: f"{r[mean_col]:.2f} +/- {r[std_col]:.2f}"
            if pd.notna(r[mean_col]) and pd.notna(r[std_col])
            else "NA",
            axis=1,
        )

    display = pd.DataFrame(
        {
            "pathway": out["pathway"],
            "n_storms": out["n_storms"].astype(int),
            "DEM_mean+/-std": pm("mean_dem", "std_dem"),
            "lat_mean+/-std": pm("mean_lat", "std_lat"),
            "lon_mean+/-std": pm("mean_lon", "std_lon"),
            "dDEM_mean+/-std": pm("mean_delta_dem", "std_delta_dem"),
            "dlat_mean+/-std": pm("mean_delta_lat", "std_delta_lat"),
            "dlon_mean+/-std": pm("mean_delta_lon", "std_delta_lon"),
        }
    )

    return out, display, grad_df


def render_table_figure(df_table_display, out_file):
    fig, ax = plt.subplots(figsize=(18, min(0.45 * len(df_table_display) + 2, 20)))
    ax.axis("off")

    table = ax.table(
        cellText=df_table_display.values,
        colLabels=df_table_display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title("Pathway Orography and Trajectory Gradient Summary", fontsize=14, pad=12)

    plt.tight_layout()
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta_boxplots(grad_df, pathway_order, out_file):
    """Quick pathway comparison using boxplots of trajectory deltas."""
    if grad_df.empty:
        return

    grad_df = grad_df.copy()
    grad_df["pathway"] = pd.Categorical(grad_df["pathway"], categories=pathway_order, ordered=True)

    metrics = [
        ("delta_dem", "delta DEM (m)"),
        ("delta_lat", "delta lat"),
        ("delta_lon", "delta lon"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.9 * len(pathway_order)), 10), sharex=True)

    for ax, (metric, ylab) in zip(axes, metrics):
        data = [
            grad_df.loc[grad_df["pathway"] == pw, metric].dropna().values
            for pw in pathway_order
        ]
        ax.boxplot(data, labels=pathway_order, showfliers=False)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.tick_params(axis="y", labelsize=9)

    axes[-1].tick_params(axis="x", labelrotation=35, labelsize=9)
    axes[-1].set_xlabel("Pathway", fontsize=10)
    fig.suptitle("Trajectory Gradient Boxplots by Pathway", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analysis_for_subset(df_subset, subset_name, out_root):
    """Generate all requested outputs for a dataframe subset."""
    subset_dir = os.path.join(out_root, subset_name)
    os.makedirs(subset_dir, exist_ok=True)

    if df_subset.empty:
        print(f"[{subset_name}] Subset is empty. Skipping.")
        return

    out_dem_dist = os.path.join(subset_dir, "dem_distribution_by_class.png")
    plot_dem_distribution(df_subset, out_dem_dist)

    out_time_dir = os.path.join(subset_dir, "time_aligned_pathway_plots")
    plot_time_aligned_by_pathway(df_subset, out_time_dir)

    table_numeric, table_display, grad_df = compute_pathway_table(df_subset)
    out_table_csv = os.path.join(subset_dir, "pathway_dem_gradient_summary_numeric.csv")
    table_numeric.to_csv(out_table_csv, index=False)

    out_table_fmt_csv = os.path.join(subset_dir, "pathway_dem_gradient_summary_mean_pm_std.csv")
    table_display.to_csv(out_table_fmt_csv, index=False)

    out_table_png = os.path.join(subset_dir, "pathway_dem_gradient_summary_table.png")
    render_table_figure(table_display, out_table_png)

    pathway_order = table_numeric["pathway"].tolist()
    out_box = os.path.join(subset_dir, "trajectory_gradient_boxplots_by_pathway.png")
    plot_delta_boxplots(grad_df, pathway_order, out_box)

    print(f"[{subset_name}] Saved outputs:")
    print(f" - {out_dem_dist}")
    print(f" - {out_time_dir}")
    print(f" - {out_table_csv}")
    print(f" - {out_table_fmt_csv}")
    print(f" - {out_table_png}")
    print(f" - {out_box}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pathway dataframe together with DEM/orography."
    )
    parser.add_argument("--pathway-csv", default=PATHWAY_CSV)
    parser.add_argument("--orog-path", default=OROG_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_and_prepare(args.pathway_csv, args.orog_path)
    print(f"Loaded {len(df)} rows after filtering to classes {TARGET_CLASSES}")

    df_top_precip = select_extreme_subset(
        df,
        metric_col="max_precip_intensity",
        fallback_col="precipitation99",
        n_top=TOP_N_EXTREME,
    )
    df_top_hail = select_extreme_subset(
        df,
        metric_col="max_hail_intensity",
        n_top=TOP_N_EXTREME,
    )

    run_analysis_for_subset(df, "all", args.out_dir)
    run_analysis_for_subset(df_top_precip, f"top{TOP_N_EXTREME}_precip", args.out_dir)
    run_analysis_for_subset(df_top_hail, f"top{TOP_N_EXTREME}_hail", args.out_dir)


if __name__ == "__main__":
    main()
