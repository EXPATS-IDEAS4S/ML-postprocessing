import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
PATHWAY_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/df_pathways_merged_no_dominance.csv")
MIDPOINT_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/crop_midpoints_from_nc.csv")
OUT_DIR = os.path.join(OUTPUT_PATH, "pathway_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Compare only these two pathways.
PATHWAY_COMPARE = {
    "1 -> 2": "EC -> DC",
    "2": "DC persistence",
}

COLORS = {
    "1 -> 2": "tab:blue",
    "2": "tab:orange",
}


def parse_crop_metadata(crop_name):
    basename = os.path.basename(str(crop_name))
    pattern = (
        r"^storm(?P<storm_id>\d+)_"
        r"(?P<datetime>\d{4}-\d{2}-\d{2}T\d{2}-\d{2})_"
    )
    match = re.match(pattern, basename)
    if match is None:
        return pd.Series({"parsed_storm_id": pd.NA, "parsed_datetime": pd.NaT})

    dt = pd.to_datetime(match.group("datetime"), format="%Y-%m-%dT%H-%M", errors="coerce")
    return pd.Series({
        "parsed_storm_id": int(match.group("storm_id")),
        "parsed_datetime": dt,
    })


def add_centers_from_midpoint_csv(df, midpoint_csv):
    midpoint_df = pd.read_csv(midpoint_csv)
    midpoint_df["crop_key"] = midpoint_df["crop"].astype(str).map(os.path.basename)
    midpoint_df = midpoint_df.drop_duplicates(subset=["crop_key"], keep="last")

    out = df.copy()
    out["crop_key"] = out["crop"].astype(str).map(os.path.basename)
    return out.merge(
        midpoint_df[["crop_key", "center_lat", "center_lon"]],
        how="left",
        on="crop_key",
    )


def angle_from_east_deg(dlon, dlat):
    # 0 deg = east, 90 = north, 180 = west, 270 = south.
    return (np.degrees(np.arctan2(dlat, dlon)) + 360.0) % 360.0


def build_metrics_for_pathway(df_pw):
    metrics = {
        "utc_hour": [],
        "duration_h": [],
        "distance_deg": [],
        "direction_deg": [],
        "max_hail_cm": [],
        "max_rain_mm_h": [],
    }

    for _, g in df_pw.groupby("parsed_storm_id"):
        g = g.sort_values("parsed_datetime")
        if len(g) < 2:
            continue

        times = pd.to_datetime(g["parsed_datetime"], errors="coerce")
        g = g.loc[~times.isna()].copy()
        times = pd.to_datetime(g["parsed_datetime"], errors="coerce")
        if len(g) < 2:
            continue

        # 1) UTC hour distribution from actual timestamps (all crops).
        hours = times.dt.hour.to_numpy(dtype=float) + times.dt.minute.to_numpy(dtype=float) / 60.0
        metrics["utc_hour"].extend(hours.tolist())

        # 2) Duration in hours per trajectory.
        duration_h = (times.max() - times.min()).total_seconds() / 3600.0
        metrics["duration_h"].append(duration_h)

        # 3) Travel distance (sum of consecutive midpoint displacements) in degrees.
        lons = pd.to_numeric(g["center_lon"], errors="coerce").to_numpy(dtype=float)
        lats = pd.to_numeric(g["center_lat"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(lons) & np.isfinite(lats)
        lons = lons[valid]
        lats = lats[valid]

        if len(lons) >= 2:
            dlon = np.diff(lons)
            dlat = np.diff(lats)
            step_dist = np.sqrt(dlon**2 + dlat**2)
            metrics["distance_deg"].append(float(np.nansum(step_dist)))

            # 4) Direction distribution for consecutive midpoint displacements.
            step_angles = angle_from_east_deg(dlon, dlat)
            metrics["direction_deg"].extend(step_angles.tolist())

        # 5) Max hail diameter (assumed cm from dataset convention).
        if "max_hail_intensity" in g.columns:
            hail_vals = pd.to_numeric(g["max_hail_intensity"], errors="coerce")
            hail_max = hail_vals.max()
            if pd.notna(hail_max):
                metrics["max_hail_cm"].append(float(hail_max))

        # 6) Max rain rate in mm/h.
        rain_max = np.nan
        if "max_precip_intensity" in g.columns:
            rain_vals = pd.to_numeric(g["max_precip_intensity"], errors="coerce")
            rain_max = rain_vals.max()
        if pd.isna(rain_max) and "precipitation99" in g.columns:
            rain_vals = pd.to_numeric(g["precipitation99"], errors="coerce")
            rain_max = rain_vals.max()
        if pd.notna(rain_max):
            metrics["max_rain_mm_h"].append(float(rain_max))

    return metrics


def plot_distributions(metrics_by_pathway, output_file):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    axes = axes.flatten()

    panel_specs = [
        ("utc_hour", "a) UTC time of crops", "UTC hour", np.arange(0, 25, 1.0)),
        ("duration_h", "b) Trajectory duration", "Hours", 30),
        ("distance_deg", "c) Trajectory travel distance", "Distance (deg)", 30),
        ("direction_deg", "d) Direction of displacement", "Angle (deg)", np.arange(0, 361, 15)),
        ("max_hail_cm", "e) Max hail diameter", "cm", 30),
        ("max_rain_mm_h", "f) Max rain rate", "mm/h", 30),
    ]

    for ax, (metric_key, title, xlabel, bins) in zip(axes, panel_specs):
        for pathway_value, label in PATHWAY_COMPARE.items():
            values = np.asarray(metrics_by_pathway[pathway_value][metric_key], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue

            ax.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=2.1,
                color=COLORS[pathway_value],
                label=f"{label} (n={len(values)})",
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)

        # Style: hide top/right spines, thicken left/bottom.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.8)
        ax.spines["bottom"].set_linewidth(1.8)

        if metric_key == "direction_deg":
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])

    axes[-1].legend(frameon=False, fontsize=10, loc="upper right")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.94, wspace=0.22, hspace=0.28)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(PATHWAY_CSV):
        raise FileNotFoundError(f"Missing pathway CSV: {PATHWAY_CSV}")
    if not os.path.exists(MIDPOINT_CSV):
        raise FileNotFoundError(f"Missing midpoint CSV: {MIDPOINT_CSV}")

    df = pd.read_csv(PATHWAY_CSV, low_memory=False)

    parsed = df["crop"].apply(parse_crop_metadata)
    df = pd.concat([df, parsed], axis=1)
    df = df.dropna(subset=["parsed_storm_id", "parsed_datetime"])
    df["parsed_storm_id"] = df["parsed_storm_id"].astype(int)
    df = add_centers_from_midpoint_csv(df, MIDPOINT_CSV)

    metrics_by_pathway = {}
    for pathway_value in PATHWAY_COMPARE.keys():
        df_pw = df[df["pathway"] == pathway_value].copy()
        metrics_by_pathway[pathway_value] = build_metrics_for_pathway(df_pw)

    output_file = os.path.join(OUT_DIR, "pathway_2x3_distributions_ecdc_vs_dc.png")
    plot_distributions(metrics_by_pathway, output_file)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
