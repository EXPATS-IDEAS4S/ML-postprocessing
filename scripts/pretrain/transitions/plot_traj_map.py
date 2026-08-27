import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmcrameri.cm as cmc

# ==================================================
# IMPORT PROJECT UTILITIES
# ==================================================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from transitions.plot_utils import plot_orography_map

# ==================================================
# CONFIGURATION
# ==================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
os.makedirs(OUTPUT_PATH, exist_ok=True)

OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"
PATHWAY_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/df_pathways_merged_no_dominance.csv")
MIDPOINT_CSV = os.path.join(OUTPUT_PATH, "pathway_analysis/crop_midpoints_from_nc.csv")

# open csv pathway analysis
pathway_df = pd.read_csv(PATHWAY_CSV, low_memory=False)

def parse_crop_metadata(crop_name):
    """Extract storm_id and datetime from crop filename."""
    basename = os.path.basename(str(crop_name))
    pattern = (
        r"^storm(?P<storm_id>\d+)_"
        r"(?P<datetime>\d{4}-\d{2}-\d{2}T\d{2}-\d{2})_"
    )
    match = re.match(pattern, basename)
    if match is None:
        return pd.Series({
            "parsed_storm_id": pd.NA,
            "parsed_datetime": pd.NaT,
        })

    dt = pd.to_datetime(match.group("datetime"), format="%Y-%m-%dT%H-%M", errors="coerce")
    return pd.Series({
        "parsed_storm_id": int(match.group("storm_id")),
        "parsed_datetime": dt,
    })


def add_centers_from_midpoint_csv(df, midpoint_csv):
    """Add center_lat/center_lon by merging precomputed midpoint CSV on crop filename."""
    midpoint_df = pd.read_csv(midpoint_csv)
    midpoint_df["crop_key"] = midpoint_df["crop"].astype(str).map(os.path.basename)
    midpoint_df = midpoint_df.drop_duplicates(subset=["crop_key"], keep="last")

    df = df.copy()
    df["crop_key"] = df["crop"].astype(str).map(os.path.basename)
    merged = df.merge(
        midpoint_df[["crop_key", "center_lat", "center_lon", "status"]],
        how="left",
        on="crop_key",
    )
    return merged


parsed_cols = pathway_df["crop"].apply(parse_crop_metadata)
pathway_df = pd.concat([pathway_df, parsed_cols], axis=1)
pathway_df = pathway_df.dropna(subset=["parsed_storm_id", "parsed_datetime"])
    # Debug statement removed
pathway_df["parsed_storm_id"] = pathway_df["parsed_storm_id"].astype(int)

if not os.path.exists(MIDPOINT_CSV):
    raise FileNotFoundError(
        f"Midpoint CSV not found: {MIDPOINT_CSV}. "
        "Run build_crop_midpoint_csv.py first."
    )

pathway_df = add_centers_from_midpoint_csv(pathway_df, midpoint_csv=MIDPOINT_CSV)
    # Debug statement removed
    # Debug statement removed
pathway_df = pathway_df.dropna(subset=["center_lat", "center_lon"])
pathway_df = pathway_df.dropna(subset=["parsed_datetime"])

if "t_bin" not in pathway_df.columns:
    raise KeyError("Column 't_bin' not found in pathway CSV. Alignment requires existing t_bin values.")

pathway_df["t_bin"] = pd.to_numeric(pathway_df["t_bin"], errors="coerce")
pathway_df = pathway_df.dropna(subset=["t_bin"])

print(
    f"Loaded {len(pathway_df)} points with center coords from crop nc across "
    f"{pathway_df['parsed_storm_id'].nunique()} unique trajectories"
)

day_hours = pathway_df["parsed_datetime"].dt.hour
daytime_mask = (day_hours >= 5) & (day_hours < 17)
nighttime_mask = ~daytime_mask
print(
    f"Daytime test samples (05-16 UTC): {int(daytime_mask.sum())}; "
    f"nighttime test samples (17-04 UTC): {int(nighttime_mask.sum())}"
)

MAP_EXTENT = [7, 14, 44, 49.5]  # lon_min, lon_max, lat_min, lat_max
PATHWAY_CLASS_MAP = {1: "EC", 2: "DC", 4: "OA"}
ALIGNED_HOUR_VMIN = 10.0
ALIGNED_HOUR_VMAX = 24.0


def pathway_to_class_label(pathway_value):
    """Convert pathway string like '1->2' into class labels like 'EC->DC'."""
    tokens = [t.strip() for t in str(pathway_value).split("->")]
    class_tokens = []
    for token in tokens:
        try:
            num = int(token)
            class_tokens.append(PATHWAY_CLASS_MAP.get(num, token))
        except ValueError:
            class_tokens.append(token)
    return "->".join(class_tokens)

# Create subfolder for plots
PLOT_SUBFOLDER = os.path.join(OUTPUT_PATH, "trajectory_plots_by_pathway")
os.makedirs(PLOT_SUBFOLDER, exist_ok=True)


def plot_trajectories_on_ax(ax, df):
    """Plot all trajectories contained in df on an existing cartopy axis."""

    draw_base_map(ax)

    cmap = cmc.glasgow_r
    norm = mcolors.Normalize(vmin=0.0, vmax=24.0)

    unique_storm_ids = df["parsed_storm_id"].unique()
    for storm_id in unique_storm_ids:
        storm_data = df[df["parsed_storm_id"] == storm_id].sort_values("parsed_datetime")
        if len(storm_data) <= 1:
            continue

        lons = storm_data["center_lon"].values
        lats = storm_data["center_lat"].values
        times = pd.to_datetime(storm_data["parsed_datetime"], errors="coerce")
        if times.isna().all():
            continue

        hour_values = times.dt.hour.to_numpy() + times.dt.minute.to_numpy() / 60.0

        if len(lons) < 2:
            continue

        for idx in range(len(lons) - 1):
            ax.plot(
                lons[idx:idx + 2],
                lats[idx:idx + 2],
                color=cmap(norm(hour_values[idx])),
                linewidth=1.6,
                alpha=0.98,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )


def draw_base_map(ax):
    """Draw orography and map decorations for trajectory panels."""
    plot_orography_map(
        OROG_PATH,
        ax=ax,
        var_name="DEM",
        extent=MAP_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.6,
    )

    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.set_xticks([7, 9, 11, 13], crs=ccrs.PlateCarree())
    ax.set_yticks([45, 47, 49], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))
    ax.plot(
        [MAP_EXTENT[0], MAP_EXTENT[1]],
        [47.0, 47.0],
        color="red",
        linewidth=1.2,
        linestyle="-",
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=20,
    )
    ax.tick_params(axis="both", labelsize=14)
    gl = ax.gridlines(draw_labels=False, alpha=0.3)


def plot_trajectories_map(df, title, output_file):
    """Plot all trajectories contained in df and save to output_file."""
    fig = plt.figure(figsize=(8.5, 6.2))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    plot_trajectories_on_ax(ax, df)

    plt.title(title, fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def extract_trajectory_series(df):
    """Return a list of per-storm trajectories aligned using existing CSV t_bin."""
    trajectories = []
    for storm_id, storm_data in df.groupby("parsed_storm_id"):
        storm_data = storm_data.sort_values(["t_bin", "parsed_datetime"])
        if len(storm_data) < 2:
            continue

        lons = storm_data["center_lon"].to_numpy(dtype=float)
        lats = storm_data["center_lat"].to_numpy(dtype=float)
        times = pd.to_datetime(storm_data["parsed_datetime"], errors="coerce")
        hour_values = times.dt.hour.to_numpy() + times.dt.minute.to_numpy() / 60.0
        rel_steps = storm_data["t_bin"].to_numpy(dtype=float)

        south_count = int((lats < 47.0).sum())
        north_count = int((lats >= 47.0).sum())
        region = "south" if south_count > north_count else "north"
        lon_steps = np.diff(lons)
        mean_lon_step = float(np.nanmean(lon_steps)) if len(lon_steps) > 0 else 0.0
        direction = "east" if mean_lon_step >= 0 else "west"

        trajectories.append({
            "storm_id": int(storm_id),
            "rel_steps": rel_steps,
            "lon": lons,
            "lat": lats,
            "hour": hour_values,
            "region": region,
            "direction": direction,
        })
    return trajectories


def compute_aligned_stats(trajectories):
    """Compute mean lon-lat-time at each aligned t_bin."""
    if len(trajectories) == 0:
        return None

    rel_list = []
    lon_list = []
    lat_list = []
    hour_list = []
    for tr in trajectories:
        rel_list.extend(tr["rel_steps"].tolist())
        lon_list.extend(tr["lon"].tolist())
        lat_list.extend(tr["lat"].tolist())
        hour_list.extend(tr["hour"].tolist())

    stats_df = pd.DataFrame({
        "rel_step": rel_list,
        "lon": lon_list,
        "lat": lat_list,
        "hour": hour_list,
    })

    grouped = stats_df.groupby("rel_step", as_index=False).agg(
        lon_mean=("lon", "mean"),
        lon_std=("lon", "std"),
        lat_mean=("lat", "mean"),
        lat_std=("lat", "std"),
        hour_mean=("hour", "mean"),
        n=("lon", "size"),
    )
    grouped = grouped.sort_values("rel_step")
    grouped = grouped[grouped["n"] >= 10].copy()
    grouped[["lon_std", "lat_std"]] = grouped[["lon_std", "lat_std"]].fillna(0.0)
    if grouped.empty:
        return None
    return grouped


def plot_aligned_mean_traj_on_map(
    ax,
    stats,
    linewidth=2.8,
    linestyle="-",
    marker_start="^",
    marker_mid="o",
    marker_end="s",
    markersize=34,
    zorder=15,
):
    """Plot aligned-bin mean trajectory on map with time coloring."""
    if stats is None or stats.empty:
        return

    cmap = cmc.glasgow_r
    norm = mcolors.Normalize(vmin=ALIGNED_HOUR_VMIN, vmax=ALIGNED_HOUR_VMAX)

    lon_mean = stats["lon_mean"].to_numpy()
    lat_mean = stats["lat_mean"].to_numpy()
    hour_mean = stats["hour_mean"].to_numpy()

    if len(lon_mean) >= 2:
        for idx in range(len(lon_mean) - 1):
            ax.plot(
                lon_mean[idx:idx + 2],
                lat_mean[idx:idx + 2],
                color=cmap(norm(hour_mean[idx])),
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.98,
                transform=ccrs.PlateCarree(),
                zorder=zorder,
            )

    npts = len(lon_mean)
    if npts > 0:
        idx_start = 0
        idx_mid = npts // 2
        idx_end = npts - 1
        marker_specs = [
            (idx_start, marker_start),
            (idx_mid, marker_mid),
            (idx_end, marker_end),
        ]
        for marker_idx, marker_style in marker_specs:
            ax.scatter(
                [lon_mean[marker_idx]],
                [lat_mean[marker_idx]],
                c=[hour_mean[marker_idx]],
                cmap=cmap,
                norm=norm,
                s=markersize,
                marker=marker_style,
                edgecolors="k",
                linewidths=0.55,
                alpha=0.98,
                transform=ccrs.PlateCarree(),
                zorder=zorder + 0.6,
            )


def add_right_colorbar(fig, axes, label):
    """Add a shared colorbar to the right of a multiplot."""
    sm = plt.cm.ScalarMappable(cmap=cmc.glasgow_r, norm=mcolors.Normalize(vmin=ALIGNED_HOUR_VMIN, vmax=ALIGNED_HOUR_VMAX))
    sm.set_array([])
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.18, 0.015, 0.64])
    fig.colorbar(sm, cax=cax, label=label)


def plot_aligned_mean_multiplot(pathway_specs, output_file):
    """Plot midpoint-aligned mean trajectories on orography maps."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.6),
        sharex=False,
        sharey=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for i, ((pathway_value, panel_title), ax) in enumerate(zip(pathway_specs, axes)):
        pathway_data = pathway_df[pathway_df["pathway"] == pathway_value]
        trajectories = extract_trajectory_series(pathway_data)
        stats = compute_aligned_stats(trajectories)

        draw_base_map(ax)

        if stats is None or stats.empty:
            ax.text(0.5, 0.5, "No valid trajectories", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        plot_aligned_mean_traj_on_map(
            ax,
            stats,
            linewidth=3.4,
            linestyle="-",
            marker_start="^",
            marker_mid="o",
            marker_end="s",
            markersize=32,
        )
        ax.set_title(f"{panel_title} ({len(trajectories)} storms)", fontsize=11, fontweight="bold")

        if i == 1:
            ax.set_yticks([], crs=ccrs.PlateCarree())
            ax.yaxis.set_ticklabels([])

    add_right_colorbar(fig, axes, "Mean UTC hour (aligned bin)")
    fig.suptitle("Midpoint-aligned mean trajectories", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.05, bottom=0.08, top=0.88, wspace=0.10)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_mean_north_south_multiplot(pathway_specs, output_file):
    """Plot midpoint-aligned mean trajectories split by North/South on maps."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.6),
        sharex=False,
        sharey=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for i, ((pathway_value, panel_title), ax) in enumerate(zip(pathway_specs, axes)):
        pathway_data = pathway_df[pathway_df["pathway"] == pathway_value]
        trajectories = extract_trajectory_series(pathway_data)
        north_trajs = [tr for tr in trajectories if tr["region"] == "north"]
        south_trajs = [tr for tr in trajectories if tr["region"] == "south"]

        draw_base_map(ax)

        plotted_any = False
        for region_name, trajs, linestyle, marker, z in [
            ("north", north_trajs, "-", "o", 16),
            ("south", south_trajs, "--", "s", 15),
        ]:
            stats = compute_aligned_stats(trajs)
            if stats is None or stats.empty:
                continue

            plot_aligned_mean_traj_on_map(
                ax,
                stats,
                linewidth=3.4,
                linestyle=linestyle,
                marker_start="^",
                marker_mid="o",
                marker_end="s",
                markersize=32,
                zorder=z,
            )
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No valid trajectories", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.set_title(
            f"{panel_title} (N={len(north_trajs)}, S={len(south_trajs)})",
            fontsize=11,
            fontweight="bold",
        )

        if i == 1:
            ax.set_yticks([], crs=ccrs.PlateCarree())
            ax.yaxis.set_ticklabels([])

    add_right_colorbar(fig, axes, "Mean UTC hour (aligned bin)")

    handles = [
        Line2D([0], [0], color="k", linestyle="-", linewidth=2.2, label="North"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=2.2, label="South"),
    ]
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=9)

    fig.suptitle("Midpoint-aligned mean trajectories split by North/South", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.05, bottom=0.08, top=0.88, wspace=0.10)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_mean_traj_on_map(ax, stats, color, linestyle="-", linewidth=3.0, zorder=15):
    """Plot an aligned mean trajectory with fixed styling and no time colormap."""
    if stats is None or stats.empty:
        return

    lon_mean = stats["lon_mean"].to_numpy()
    lat_mean = stats["lat_mean"].to_numpy()

    if len(lon_mean) >= 2:
        for idx in range(len(lon_mean) - 1):
            ax.plot(
                lon_mean[idx:idx + 2],
                lat_mean[idx:idx + 2],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.98,
                transform=ccrs.PlateCarree(),
                zorder=zorder,
            )


def plot_grouped_mean_plus_std_traj_on_map(ax, stats, color, linestyle="-", linewidth=1.0, zorder=13):
    """Plot thin mean +/- std trajectories in map space."""
    if stats is None or stats.empty:
        return

    lon_mean = stats["lon_mean"].to_numpy()
    lat_mean = stats["lat_mean"].to_numpy()
    lon_std = stats["lon_std"].to_numpy()
    lat_std = stats["lat_std"].to_numpy()
    lon_std = np.nan_to_num(lon_std, nan=0.0)
    lat_std = np.nan_to_num(lat_std, nan=0.0)

    for offset_sign, alpha in [(1.0, 0.45), (-1.0, 0.35)]:
        lon_vals = lon_mean + offset_sign * lon_std
        lat_vals = lat_mean + offset_sign * lat_std
        if len(lon_vals) >= 2:
            for idx in range(len(lon_vals) - 1):
                ax.plot(
                    lon_vals[idx:idx + 2],
                    lat_vals[idx:idx + 2],
                    color=color,
                    linewidth=linewidth,
                    linestyle=":",
                    alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    zorder=zorder,
                )


def plot_grouped_mean_with_markers_on_map(ax, stats, color, linestyle="-", linewidth=3.2, zorder=15):
    """Plot mean trajectory with big start/end markers and small intermediate markers."""
    if stats is None or stats.empty:
        return

    lon_mean = stats["lon_mean"].to_numpy()
    lat_mean = stats["lat_mean"].to_numpy()

    if len(lon_mean) >= 2:
        for idx in range(len(lon_mean) - 1):
            ax.plot(
                lon_mean[idx:idx + 2],
                lat_mean[idx:idx + 2],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.98,
                transform=ccrs.PlateCarree(),
                zorder=zorder,
            )

    npts = len(lon_mean)
    if npts > 0:
        if npts > 2:
            mid_indices = list(range(1, npts - 1))
            ax.scatter(
                lon_mean[mid_indices],
                lat_mean[mid_indices],
                color=color,
                s=10,
                marker="o",
                edgecolors="k",
                linewidths=0.25,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=zorder + 0.35,
            )

        start_end_specs = [(0, "^"), (npts - 1, "s")]
        for idx, marker_style in start_end_specs:
            ax.scatter(
                [lon_mean[idx]],
                [lat_mean[idx]],
                color=color,
                s=26,
                marker=marker_style,
                edgecolors="k",
                linewidths=0.45,
                alpha=0.98,
                transform=ccrs.PlateCarree(),
                zorder=zorder + 0.5,
            )


def plot_aligned_direction_north_south_multiplot(pathway_specs, output_file):
    """Plot midpoint-aligned mean trajectories split by North/South and East/West."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.5),
        sharex=False,
        sharey=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    direction_styles = {
        "east": {"color": "tab:blue"},
        "west": {"color": "tab:orange"},
    }
    region_linestyles = {
        "north": "-",
        "south": "--",
    }

    for i, ((pathway_value, panel_title), ax) in enumerate(zip(pathway_specs, axes)):
        pathway_data = pathway_df[pathway_df["pathway"] == pathway_value]
        trajectories = extract_trajectory_series(pathway_data)

        north_trajs = [tr for tr in trajectories if tr["region"] == "north"]
        south_trajs = [tr for tr in trajectories if tr["region"] == "south"]
        east_trajs = [tr for tr in trajectories if tr["direction"] == "east"]
        west_trajs = [tr for tr in trajectories if tr["direction"] == "west"]
        north_east_trajs = [tr for tr in north_trajs if tr["direction"] == "east"]
        north_west_trajs = [tr for tr in north_trajs if tr["direction"] == "west"]
        south_east_trajs = [tr for tr in south_trajs if tr["direction"] == "east"]
        south_west_trajs = [tr for tr in south_trajs if tr["direction"] == "west"]

        draw_base_map(ax)

        plotted_any = False
        for region_name, region_trajs in [("north", north_trajs), ("south", south_trajs)]:
            for direction_name in ["east", "west"]:
                split_trajs = [tr for tr in region_trajs if tr["direction"] == direction_name]
                stats = compute_aligned_stats(split_trajs)
                if stats is None or stats.empty:
                    continue

                plot_grouped_mean_with_markers_on_map(
                    ax,
                    stats,
                    color=direction_styles[direction_name]["color"],
                    linestyle=region_linestyles[region_name],
                    linewidth=3.0,
                    zorder=15 if region_name == "north" else 14,
                )
                plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No valid trajectories", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.set_title(
            (
                f"{panel_title} (NE={len(north_east_trajs)}, NW={len(north_west_trajs)}, "
                f"SE={len(south_east_trajs)}, SW={len(south_west_trajs)})"
            ),
            fontsize=11,
            fontweight="bold",
        )

        if i == 1:
            ax.set_yticks([], crs=ccrs.PlateCarree())
            ax.yaxis.set_ticklabels([])

    direction_handles = [
        Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=2.2, label="North, eastward"),
        Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=2.2, label="South, eastward"),
        Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=2.2, label="North, westward"),
        Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=2.2, label="South, westward"),
    ]
    marker_handles = [
        Line2D([0], [0], color="k", marker="^", linestyle="None", markersize=6, label="Start"),
        Line2D([0], [0], color="k", marker="s", linestyle="None", markersize=6, label="End"),
    ]
    marker_legend = axes[0].legend(handles=marker_handles, loc="lower right", frameon=True, fontsize=8)
    marker_legend.get_frame().set_edgecolor("black")
    marker_legend.get_frame().set_linewidth(0.8)
    axes[0].add_artist(marker_legend)
    direction_legend = axes[0].legend(handles=direction_handles, loc="lower left", frameon=True, fontsize=8, title="Direction / region")
    direction_legend.get_frame().set_edgecolor("black")
    direction_legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.88, wspace=0.10)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ==================================================
# PLOT COMBINED TRAJECTORY MAP
# ==================================================
print("Plotting combined trajectory map...")

# Create a dedicated two-panel plot for EC->DC (left) and DC (right).
combined_pathways = [
    ("1 -> 2", "a) EC -> DC"),
    ("2", "b) DC"),
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.4, 4.8),
    subplot_kw={"projection": ccrs.PlateCarree()},
)
fig.subplots_adjust(wspace=0.10)

for i, (ax, (pathway_value, panel_title)) in enumerate(zip(axes, combined_pathways)):
    pathway_data = pathway_df[pathway_df["pathway"] == pathway_value]
    print(f"Plotting pathway {pathway_value} with {len(pathway_data)} points...")

    n_storms_in_pathway = pathway_data["parsed_storm_id"].nunique()
    plot_trajectories_on_ax(ax, pathway_data)

    # remove latitude ticks on right panel
    if i == 1:
        ax.set_yticks([], crs=ccrs.PlateCarree())
        ax.yaxis.set_ticklabels([])

    ax.set_title(f"{panel_title} ({n_storms_in_pathway} storms)", fontsize=13, pad=10, fontweight="bold")

combined_output_file = os.path.join(PLOT_SUBFOLDER, "pathways_1-2_and_2_multiplot.png")
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.90, wspace=0.10)
sm = plt.cm.ScalarMappable(cmap=cmc.glasgow_r, norm=mcolors.Normalize(vmin=0.0, vmax=24.0))
sm.set_array([])
fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="UTC hour")
plt.savefig(combined_output_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saving combined multiplot to {combined_output_file}...")

# ==================================================
# PLOT MIDPOINT-ALIGNED AVERAGE TRAJECTORIES
# ==================================================
aligned_output_file = os.path.join(PLOT_SUBFOLDER, "pathways_aligned_median_iqr_multiplot.png")
print(f"Saving midpoint-aligned mean trajectory multiplot to {aligned_output_file}...")
plot_aligned_mean_multiplot(combined_pathways, aligned_output_file)

aligned_ns_output_file = os.path.join(PLOT_SUBFOLDER, "pathways_aligned_median_iqr_north_south_multiplot.png")
print(f"Saving North/South midpoint-aligned mean trajectory multiplot to {aligned_ns_output_file}...")
plot_aligned_mean_north_south_multiplot(combined_pathways, aligned_ns_output_file)

aligned_dir_output_file = os.path.join(PLOT_SUBFOLDER, "pathways_aligned_mean_north_south_east_west_multiplot.png")
print(f"Saving North/South/East-West midpoint-aligned mean trajectory multiplot to {aligned_dir_output_file}...")
plot_aligned_direction_north_south_multiplot(combined_pathways, aligned_dir_output_file)

# ==================================================
# PLOT TRAJECTORIES BY PATHWAY
# ==================================================
# print("Plotting trajectories by pathway...")
#
# all_storms = pathway_df["parsed_storm_id"].nunique()
# all_output_file = os.path.join(PLOT_SUBFOLDER, "all_trajectories.png")
# print(f"Saving all trajectories ({all_storms} storms) to {all_output_file}...")
# plot_trajectories_map(
#     pathway_df,
#     title=f"All Storm Trajectories ({all_storms} storms)",
#     output_file=all_output_file,
# )
#
# unique_pathways = pathway_df["pathway"].unique()
# print(f"Found {len(unique_pathways)} unique pathways")
#
# for pathway_id in unique_pathways:
#     pathway_data = pathway_df[pathway_df["pathway"] == pathway_id]
#     n_storms_in_pathway = pathway_data["parsed_storm_id"].nunique()
#     pathway_label = pathway_to_class_label(pathway_id)
#
#     output_file = os.path.join(PLOT_SUBFOLDER, f"pathway_{pathway_id}_trajectories.png")
#     print(f"Saving pathway {pathway_id} ({n_storms_in_pathway} storms) to {output_file}...")
#     plot_trajectories_map(
#         pathway_data,
#         title=f"Pathway {pathway_label} - Storm Trajectories ({n_storms_in_pathway} storms)",
#         output_file=output_file,
#     )

print(f"Done! All plots saved to {PLOT_SUBFOLDER}")
