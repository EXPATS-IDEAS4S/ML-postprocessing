import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    # Debug statement removed


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

    plot_orography_map(
        OROG_PATH,
        ax=ax,
        var_name="DEM",
        extent=MAP_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.6,
    )

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

    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.set_xticks([7, 9, 11, 13], crs=ccrs.PlateCarree())
    ax.set_yticks([45, 47, 49], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))
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

# ==================================================
# PLOT COMBINED TRAJECTORY MAP
# ==================================================
print("Plotting combined trajectory map...")

# Create a dedicated two-panel plot for pathway 2 and pathway 1->2.
combined_pathways = [
    ("2", "a) DC"),
    ("1 -> 2", "b) EC -> DC"),
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.2, 4.8),
    subplot_kw={"projection": ccrs.PlateCarree()},
)
#increase space between subplots
fig.subplots_adjust(wspace=0.12)

for i, (ax, (pathway_value, panel_title)) in enumerate(zip(axes, combined_pathways)):
    pathway_data = pathway_df[pathway_df["pathway"] == pathway_value]
    print(f"Plotting pathway {pathway_value} with {len(pathway_data)} points...")

    n_storms_in_pathway = pathway_data["parsed_storm_id"].nunique()
    plot_trajectories_on_ax(ax, pathway_data)
    # remove latitude ticks on the second (right-hand) panel
    if i == 1:
        ax.set_yticks([], crs=ccrs.PlateCarree())
        ax.yaxis.set_ticklabels([])
    ax.set_title(f"{panel_title} ({n_storms_in_pathway} storms)", fontsize=13, pad=10, fontweight="bold")

combined_output_file = os.path.join(PLOT_SUBFOLDER, "pathways_2_and_1-2_multiplot.png")
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.90, wspace=0.10)
sm = plt.cm.ScalarMappable(cmap=cmc.glasgow_r, norm=mcolors.Normalize(vmin=0.0, vmax=24.0))
sm.set_array([])
fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="UTC hour")
plt.savefig(combined_output_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saving combined multiplot to {combined_output_file}...")

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
