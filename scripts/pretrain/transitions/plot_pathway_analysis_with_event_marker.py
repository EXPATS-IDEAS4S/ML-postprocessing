import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import cmcrameri.cm as cmc
from scipy.stats import gaussian_kde
from PIL import Image
from datetime import timedelta
from datetime import datetime
import glob
import ast
import xarray as xr


mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["text.usetex"] = False

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO

RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUT_DIR = f"{BASE_DIR}/pathway_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
SELECTED_CLASSES = [1,2,4]
# Use simple class labels without dominance split
CLASSES_LABELS = [str(c) for c in SELECTED_CLASSES]  # ['1', '2', '4']

path = f"{OUT_DIR}/df_pathways_merged_no_dominance.csv"
df = pd.read_csv(path, low_memory=False)
print(df)#.columns.tolist())
print(df.columns.tolist())

#print the storm with the max hail and precip
top_hail_row = df.loc[df["max_hail_intensity"].idxmax()]
top_precip_row = df.loc[df["precipitation99"].idxmax()]
print("Top hail pathway:")
print(top_hail_row[["pathway_id", "pathway_prob", "pathway", "max_hail_intensity", 'datetime', 'lat', 'lon']])
print("Top precip pathway:")
print(top_precip_row[["pathway_id", "pathway_prob", "pathway", "precipitation99", 'datetime', 'lat', 'lon']])



#open train and centroids tsne
csv_tnse_file = "tsne_all_vectors_with_centroids.csv"
df_tsne = pd.read_csv(os.path.join(BASE_DIR, csv_tnse_file))
df_tsne_centroids = df_tsne[df_tsne["vector_type"] == "CENTROID"]
df_tsne_train = df_tsne[df_tsne["vector_type"] == "TRAIN"]


cloud_items_ordered = sorted(
    CLOUD_CLASS_INFO.items(),
    key=lambda x: x[1]["order"]
)


labels_ordered = [lbl for lbl, _ in cloud_items_ordered]
short_labels = [info["short"] for _, info in cloud_items_ordered]
colors_ordered = [info["color"] for _, info in cloud_items_ordered]
print(colors_ordered)


selected_labels_ordered = [lbl for lbl in labels_ordered if lbl in SELECTED_CLASSES]
selected_short_labels = [short_labels[i] for i, lbl in enumerate(labels_ordered) if lbl in SELECTED_CLASSES]
selected_colors_ordered = [colors_ordered[i] for i, lbl in enumerate(labels_ordered) if lbl in SELECTED_CLASSES]
print(f"Selected labels: {selected_labels_ordered}")
print(f"Selected short labels: {selected_short_labels}")
print(f"Selected colors: {selected_colors_ordered}")



#add color column to df_tsne_train based on label
#remove invalid labels -100
df_tsne_train = df_tsne_train[df_tsne_train["label"] != -100]
print(df_tsne_train.columns.tolist())

colors = {lbl: info["color"] for lbl, info in CLOUD_CLASS_INFO.items()
}
df_tsne_train["color"] = df_tsne_train["label"].map(colors)
#print(df_tsne_train["color"])

def stat_by_time(df, value_col, stat="mean"):
    if stat == "mean":
        return (
            df.groupby("t_bin")[value_col]
            .mean()
            .reset_index()
        )
    elif stat == "sum":
        return (
            df.groupby("t_bin")[value_col]
            .sum()
            .reset_index()
        )
    elif stat == "max":
        return (
            df.groupby("t_bin")[value_col]
            .max()
            .reset_index()
        )

def normalize_coords(coords):
    if coords is None or (isinstance(coords, float) and np.isnan(coords)):
        return None
    if isinstance(coords, str):
        try:
            return ast.literal_eval(coords)
        except (ValueError, SyntaxError):
            return None
    return coords

def load_nc_as_image(nc_path):
    try:
        ds = xr.open_dataset(nc_path)
        var_name = "IR_108" if "IR_108" in ds.data_vars else list(ds.data_vars)[0]
        data = ds[var_name].values
        if data.ndim == 3:
            data = data[0]
        lat_vals = ds["lat"].values if "lat" in ds.coords else None
        lon_vals = ds["lon"].values if "lon" in ds.coords else None
        ds.close()
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_max > data_min:
            data_norm = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            data_norm = np.zeros_like(data, dtype=np.uint8)
        return data_norm, lat_vals, lon_vals
    except Exception:
        return None, None, None

def latlon_to_pixel(lat, lon, lat_vals, lon_vals):
    if lat_vals is None or lon_vals is None:
        return None
    lat_min, lat_max = np.min(lat_vals), np.max(lat_vals)
    lon_min, lon_max = np.min(lon_vals), np.max(lon_vals)
    if lat < lat_min or lat > lat_max or lon < lon_min or lon > lon_max:
        return None
    y_idx = int(np.argmin(np.abs(lat_vals - lat)))
    x_idx = int(np.argmin(np.abs(lon_vals - lon)))
    return x_idx, y_idx





def compute_class_persistence(df_pw):
    records = []

    for storm_id, g in df_pw.sort_values("datetime").groupby("storm_id"):
        for state, gg in g.groupby("state"):
            dt_series = pd.to_datetime(gg["datetime"], errors="coerce")
            if dt_series.isna().all():
                continue
            duration = (dt_series.max() - dt_series.min()).total_seconds() / 3600
            records.append({
                "storm_id": storm_id,
                "state": state,
                "duration": duration
            })

    return pd.DataFrame(records)


def trajectory_radius_km(g):
    coords = g[["lat", "lon"]].dropna()
    if coords.empty:
        return np.nan

    lat0 = coords["lat"].mean()
    lon0 = coords["lon"].mean()
    lat = np.radians(coords["lat"].values)
    lon = np.radians(coords["lon"].values)
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * np.nanmax(c)



stats_cols = [
    "pathway_id",
    "pathway",
    "storm_id",
    "storm_type",
    "lat",
    "lon",
    "precipitation99",
    "euclid_msg_grid",
    "cot_thick",
    "cth_very_high",
    "n_precip",
    "n_hail",
    "max_precip_intensity",
    "max_hail_intensity",
    "t_bin",
]

df_stats = df[stats_cols].copy()

storm_radius = (
    df_stats.groupby(["pathway_id", "pathway", "storm_id"]).apply(trajectory_radius_km)
    .reset_index(name="radius_km")
)

storm_stats = (
    df_stats.groupby(["pathway_id", "pathway", "storm_id"]).agg(
        storm_type=("storm_type", "first"),
        max_precipitation99=("precipitation99", "max"),
        total_euclid_msg_grid=("euclid_msg_grid", "sum"),
        mean_cot_thick=("cot_thick", "mean"),
        mean_cth_very_high=("cth_very_high", "mean"),
        total_n_precip=("n_precip", "sum"),
        total_n_hail=("n_hail", "sum"),
        max_precip_intensity=("max_precip_intensity", "max"),
        max_hail_intensity=("max_hail_intensity", "max"),
        duration_hours=("t_bin", lambda x: x.max() - x.min()),
    )
    .reset_index()
)

storm_stats = storm_stats.merge(
    storm_radius,
    on=["pathway_id", "pathway", "storm_id"],
    how="left"
)

pathway_stats = (
    storm_stats.groupby(["pathway_id", "pathway"]).agg(
        max_precipitation99=("max_precipitation99", "max"),
        max_total_euclid_msg_grid=("total_euclid_msg_grid", "max"),
        mean_cot_thick=("mean_cot_thick", "mean"),
        mean_cth_very_high=("mean_cth_very_high", "mean"),
        max_n_precip=("total_n_precip", "max"),
        max_n_hail=("total_n_hail", "max"),
        max_max_precip_intensity=("max_precip_intensity", "max"),
        max_max_hail_intensity=("max_hail_intensity", "max"),
        mean_duration_hours=("duration_hours", "mean"),
        mean_radius_km=("radius_km", "mean"),
    )
    .reset_index()
)

#add alsonumber of strorms per pathway
pathway_stats["n_storms"] = (
    storm_stats.groupby(["pathway_id", "pathway"])["storm_id"]
    .nunique()
    .values
)

print(pathway_stats)
pathway_stats.to_csv(
    os.path.join(OUT_DIR, "pathway_stats_summary.csv"),
    index=False
)

# --- Aggregate statistics ---
scatter_stats = (
    storm_stats.groupby(["pathway_id", "pathway"]).agg(
        mean_precipitation99=("max_precipitation99", "mean"),
        mean_hail_intensity=("max_hail_intensity", "mean"),
        max_precipitation99=("max_precipitation99", "max"),
        max_hail_intensity=("max_hail_intensity", "max"),
        mean_duration_hours=("duration_hours", "mean"),
        max_duration_hours=("duration_hours", "max"),
    )
    .reset_index()
)

#add number of storms per pathway to scatter_stats
scatter_stats["n_storms"] = (
    storm_stats.groupby(["pathway_id", "pathway"])["storm_id"]
    .nunique()
    .values
)

scatter_stats["mean_hail_intensity"] = scatter_stats["mean_hail_intensity"].fillna(0)

# --- Identify extreme pathways ---
top_hail_row = scatter_stats.loc[
    scatter_stats["max_hail_intensity"].idxmax()
]

top_precip_row = scatter_stats.loc[
    scatter_stats["max_precipitation99"].idxmax()
]

selected_pathway_ids = list(
    dict.fromkeys(
        [int(top_hail_row["pathway_id"]),
         int(top_precip_row["pathway_id"])]
    )
)
#add event type in selected_pathway_ids to make a dict with ids and event type
selected_pathway_info = {
    int(top_hail_row["pathway_id"]): "HAIL",
    int(top_precip_row["pathway_id"]): "PRECIP"
}
print(f"Selected pathway IDs: {selected_pathway_ids}")
print(f"Selected pathway info: {selected_pathway_info}")

# --- Most common pathways by storm count (for multiplot) ---
most_common = pathway_stats.sort_values("n_storms", ascending=False).head(2)
most_common_pathway_info = {
    int(row["pathway_id"]): f"COMMON_{i+1}"
    for i, row in most_common.reset_index(drop=True).iterrows()
}
print(f"Most common pathway IDs: {list(most_common_pathway_info.keys())}")

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=False)

# Create separate normalizations for mean and max duration
norm_mean = mpl.colors.Normalize(
    vmin=scatter_stats["mean_duration_hours"].min(),
    vmax=scatter_stats["mean_duration_hours"].max(),
)
norm_max = mpl.colors.Normalize(
    vmin=scatter_stats["max_duration_hours"].min(),
    vmax=scatter_stats["max_duration_hours"].max(),
)
cmap = cmc.imola

# List of norms corresponding to each subplot
norms = [norm_mean, norm_max]

for ax, x_col, y_col, duration_col, title, norm in zip(
    axes,
    ["mean_precipitation99", "max_precipitation99"],
    ["mean_hail_intensity", "max_hail_intensity"],
    ['mean_duration_hours', 'max_duration_hours'],
    ["a) Mean Intensities", "b) Max Intensities"],
    norms,
):

    for _, row in scatter_stats.iterrows():
        size = 40 + 400 * row["n_storms"] / scatter_stats["n_storms"].max()

        ax.scatter(
            row[x_col],
            row[y_col],
            s=size,
            c=[cmap(norm(row[duration_col]))],
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
            zorder=3,
        )

        # --- Label only extreme pathways ---
        if int(row["pathway_id"]) in selected_pathway_ids:
            ax.text(
                row[x_col],
                row[y_col],
                str(int(row["pathway_id"])),
                fontsize=9,
                weight="bold",
                ha="left",
                va="bottom",
                c="black",
            )

    ax.set_xlabel("Rain Rate P99 (mm/h)", fontsize=10)
    ax.set_ylabel("Max Hail Size (cm)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

# --- Create separate colorbars for each subplot ---
cbar_mean = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm_mean, cmap=cmap),
    ax=axes[0],
    orientation="vertical",
    fraction=0.03,
    pad=0.04,
)
cbar_mean.set_label("Mean Duration (hours)", fontsize=10)
cbar_mean.ax.tick_params(labelsize=11)

cbar_max = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm_max, cmap=cmap),
    ax=axes[1],
    orientation="vertical",
    fraction=0.03,
    pad=0.04,
)
cbar_max.set_label("Max Duration (hours)", fontsize=10)
cbar_max.ax.tick_params(labelsize=11)

fig.tight_layout()

fig.savefig(
    os.path.join(OUT_DIR, "pathway_scatter_summary_mean_and_max_no_dom.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# --- Print summary ---
print("Top hail pathway:")
print(top_hail_row[["pathway_id", "pathway", "max_hail_intensity"]])

print("Top precip pathway:")
print(top_precip_row[["pathway_id", "pathway", "max_precipitation99"]])

# Define crop directory for use in pathway loop
crop_dir = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/nc/1"

for pathway_id, event_type in most_common_pathway_info.items():
    df_pw = df[df["pathway_id"] == pathway_id].copy()
    #number of unique storms in this pathway
    n_storms = df_pw["storm_id"].nunique()
    #pathway description
    pathway = df_pw["pathway"].iloc[0]
    
    if df_pw.empty:
        continue
    path = df_pw["pathway"].iloc[0]
    print(f"Plotting pathway {pathway_id} ({event_type}) with path {path}")
    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(
        nrows=5,
        ncols=2,
        height_ratios=[1, 0.75, 0.3, 0.28, 0.28],
        hspace=0.45,
        wspace=0.3,
        top=0.94
    )

    ax_feat  = fig.add_subplot(gs[0, 0])
    ax_box   = fig.add_subplot(gs[0, 1])
    ax_cloud = fig.add_subplot(gs[1, :])
    ax_events = fig.add_subplot(gs[2, :])
    ax_crops_precip = fig.add_subplot(gs[3, :])
    ax_crops_hail = fig.add_subplot(gs[4, :])

    # ==========================================================
    # FEATURE SPACE (density contours on training embedding)
    # ==========================================================
    # Plot training embedding as background
    ax_feat.scatter(
        df_tsne_train["tsne_dim_1"],
        df_tsne_train["tsne_dim_2"],
        color=df_tsne_train["color"].values,
        s=2,
        alpha=0.1,
        linewidth=0
    )
    
    # Compute density contours from test (pathway-specific) vectors
    test_vectors = df_pw[["tsne_dim_1", "tsne_dim_2"]].values
    if len(test_vectors) > 3:  # Only if enough points for KDE
        try:
            kde = gaussian_kde(test_vectors.T)
            
            # Create grid for contour plot
            x_min, x_max = df_tsne_train["tsne_dim_1"].min(), df_tsne_train["tsne_dim_1"].max()
            y_min, y_max = df_tsne_train["tsne_dim_2"].min(), df_tsne_train["tsne_dim_2"].max()
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde(positions).reshape(xx.shape)
            
            # Calculate density levels for gradient fill
            density_values = kde(test_vectors.T)
            percentile_50 = np.percentile(density_values, 50)
            
            # Plot filled contour with gradient inside 50th percentile
            contourf = ax_feat.contourf(
                xx,
                yy,
                zz,
                levels=10,
                cmap="Blues",
                alpha=0.6,
                vmin=percentile_50,
                vmax=zz.max()
            )
            # Add 50th percentile boundary line
            ax_feat.contour(
                xx,
                yy,
                zz,
                levels=[percentile_50],
                colors="darkblue",
                linewidths=2.0,
                alpha=0.9
            )
        except:
            pass
    
    # Plot centroids as triangles
    for _, row in df_tsne_centroids.iterrows():
        cls_label = int(row["label"])
        #if cls_label in SELECTED_CLASSES:
        ax_feat.scatter(
            row["tsne_dim_1"],
            row["tsne_dim_2"],
            marker="^",
            s=200,
            color=CLOUD_CLASS_INFO[cls_label]["color"],
            edgecolor="black",
            linewidth=1.5,
            zorder=10
        )

    ax_feat.set_title("a) Feature space density contours", fontsize=13, fontweight="bold")
    ax_feat.set_xticks([])
    ax_feat.set_yticks([])

    # ==========================================================
    # CLASS PERSISTENCE HISTOGRAM
    # ==========================================================
    pers = compute_class_persistence(df_pw)
    pers = pers[pers["state"].isin(SELECTED_CLASSES)]
    
    # Create histogram for each class with overlapping 1-hour bins
    max_duration = pers["duration"].max()
    n_bins = int(np.ceil(max_duration))
    bin_edges = np.arange(0, max_duration + 1, 1)
    bar_width = bin_edges[1] - bin_edges[0]
    
    for i, cls in enumerate(SELECTED_CLASSES):
        class_data = pers.loc[pers["state"] == cls, "duration"]
        if len(class_data) > 0:
            counts, _ = np.histogram(class_data, bins=bin_edges)
            # Overlap bars in each bin
            bar_x = bin_edges[:-1]
            ax_box.bar(
                bar_x,
                counts,
                width=bar_width,
                label=selected_short_labels[i],
                facecolor="none",
                edgecolor=selected_colors_ordered[i],
                linewidth=1.2,
                alpha=0.7
            )
    
    ax_box.set_xlabel("Duration (hours)", fontsize=12)
    ax_box.set_ylabel("Count", fontsize=12)
    ax_box.set_title("b) Class persistence duration", fontsize=13, fontweight="bold")
    ax_box.tick_params(axis="x", labelsize=12)
    ax_box.tick_params(axis="y", labelsize=12)
    ax_box.grid(ls="--", alpha=0.3, axis="y")
    #use x ticks label (0,2,4,6,8,10,12,14,16) and set xlim to 0-10
    ax_box.set_xticks([0,2,4,6,8,10,12,14,16,18])
    #ax_box.set_xlim(0, 10)
    #ax_box.legend(fontsize=10)

    # ==========================================================
    # TIME-ALIGNED PANEL 1: CLOUD PROPERTIES ONLY
    # ==========================================================

    # Highlight dominant transition window for two-class pathways
    transition_window = None
    path_states = []
    for state in pathway.split(" -> "):
        try:
            path_states.append(int(state))
        except:
            continue

    if len(path_states) == 2:
        s_from, s_to = path_states
        transition_times = []
        for _, g in df_pw.sort_values("t_bin").groupby("storm_id"):
            seen_from = False
            for _, row in g.iterrows():
                try:
                    row_state = int(row["state"])
                except:
                    continue
                if row_state == s_from:
                    seen_from = True
                elif seen_from and row_state == s_to:
                    transition_times.append(row["t_bin"])
                    break

        if transition_times:
            t_low, t_high = np.percentile(transition_times, [25, 75])
            t_med = float(np.median(transition_times))
            transition_window = (t_low, t_high, t_med)

    # ----------------------------------------------------------
    # Cloud properties with median + 25th/75th percentiles
    # ----------------------------------------------------------
    # Compute median and percentiles for cloud properties
    def stat_by_time_percentile(df, value_col, percentile):
        return (
            df.groupby("t_bin")[value_col]
            .quantile(percentile)
            .reset_index()
            .rename(columns={value_col: f"{value_col}_p{int(percentile*100)}"})
        )
    
    median_cc = stat_by_time_percentile(df_pw, "cma", 0.5)
    p25_cc = stat_by_time_percentile(df_pw, "cma", 0.25)
    p75_cc = stat_by_time_percentile(df_pw, "cma", 0.75)
    
    median_cth = stat_by_time_percentile(df_pw, "cth_very_high", 0.5)
    p25_cth = stat_by_time_percentile(df_pw, "cth_very_high", 0.25)
    p75_cth = stat_by_time_percentile(df_pw, "cth_very_high", 0.75)
    
    median_cot = stat_by_time_percentile(df_pw, "cot_thick", 0.5)
    p25_cot = stat_by_time_percentile(df_pw, "cot_thick", 0.25)
    p75_cot = stat_by_time_percentile(df_pw, "cot_thick", 0.75)
    
    # Plot median trends
    ax_cloud.plot(median_cc["t_bin"], median_cc["cma_p50"], lw=2, label="CC", color="C0")
    ax_cloud.plot(median_cth["t_bin"], median_cth["cth_very_high_p50"], lw=2, label="CTH very high", color="C1")
    ax_cloud.plot(median_cot["t_bin"], median_cot["cot_thick_p50"], lw=2, label="COT thick", color="C2")
    
    # Fill between 25th and 75th percentiles with low alpha
    ax_cloud.fill_between(
        median_cc["t_bin"],
        p25_cc["cma_p25"],
        p75_cc["cma_p75"],
        color="C0",
        alpha=0.2
    )
    ax_cloud.fill_between(
        median_cth["t_bin"],
        p25_cth["cth_very_high_p25"],
        p75_cth["cth_very_high_p75"],
        color="C1",
        alpha=0.2
    )
    ax_cloud.fill_between(
        median_cot["t_bin"],
        p25_cot["cot_thick_p25"],
        p75_cot["cot_thick_p75"],
        color="C2",
        alpha=0.2
    )
    
    if transition_window is not None:
        t_low, t_high, t_med = transition_window
        ax_cloud.axvspan(t_low, t_high, color="gray", alpha=0.2, zorder=0)
        ax_cloud.axvline(t_med, color="gray", ls="--", lw=1)

    ax_cloud.set_ylabel("Cloud properties", fontsize=12)
    #ax_cloud.set_xlabel("Aligned time (hours)", fontsize=12)
    ax_cloud.set_title("c) Time-aligned cloud properties", fontsize=13, fontweight="bold")
    ax_cloud.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_cloud.grid(ls="--", alpha=0.3)
    ax_cloud.tick_params(axis="both", labelsize=12)
    ax_cloud.legend(fontsize=10, loc="lower right")
    
    # ==========================================================
    # TIME-ALIGNED PANEL 2: EVENTS SPLIT BY TYPE
    # ==========================================================
    
    # Mean rain and hail events separately
    mean_rain = stat_by_time(df_pw, "n_precip", stat="mean") 
    mean_hail = stat_by_time(df_pw, "n_hail", stat="mean")
   
    
    ax_events.plot(
        mean_rain["t_bin"],
        mean_rain["n_precip"],
        color="steelblue",
        lw=2,
        label="Rain events"
    )

    ax_events.plot(
        mean_hail["t_bin"],
        mean_hail["n_hail"],
        color="indianred",
        lw=2,
        label="Hail Events"
    )

    
    if transition_window is not None:
        t_low, t_high, t_med = transition_window
        ax_events.axvspan(t_low, t_high, color="gray", alpha=0.2, zorder=0)
        ax_events.axvline(t_med, color="gray", ls="--", lw=1)

    ax_events.set_ylabel("Event count", fontsize=12)
    ax_events.set_xlabel("Aligned time (hours)", fontsize=12)
    ax_events.set_title("d) Time-aligned event counts by type", fontsize=13, fontweight="bold")
    ax_events.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_events.grid(ls="--", alpha=0.3)
    ax_events.tick_params(axis="both", labelsize=12)
    ax_events.legend(fontsize=10, loc="upper right")

    # ==========================================================
    # CROP IMAGES FOR EXTREME PRECIP AND HAIL EVENTS
    # ==========================================================
    # Find extreme precip and hail within this pathway
    extreme_precip_pw_idx = df_pw["precipitation99"].idxmax()
    extreme_hail_pw_idx = df_pw["max_hail_intensity"].idxmax()
    
    extreme_precip_pw_row = df_pw.loc[extreme_precip_pw_idx]
    extreme_hail_pw_row = df_pw.loc[extreme_hail_pw_idx]
    
    precip_pw_storm_id = int(extreme_precip_pw_row["storm_id"].replace('storm', '')) if 'storm' in str(extreme_precip_pw_row["storm_id"]) else int(extreme_precip_pw_row["storm_id"])
    hail_pw_storm_id = int(extreme_hail_pw_row["storm_id"].replace('storm', '')) if 'storm' in str(extreme_hail_pw_row["storm_id"]) else int(extreme_hail_pw_row["storm_id"])
    
    def plot_crop_row(ax, storm_id, trajectory_df, crop_dir, event_col, ax_events, precip_coords=None, hail_coords=None, n_crops=6, gap_pixels=20):
        """Plot crop images: show evenly spaced frames with class-colored borders and event markers."""
        ax.clear()
        
        # Find nearby images around the target time
        search_pattern = f"{crop_dir}/storm{storm_id}_*_*.nc"
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            ax.text(0.5, 0.5, "No images found", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Find images and their times
        image_times = []
        for fpath in matching_files:
            basename = os.path.basename(fpath)
            parts = basename.split('_')
            if len(parts) >= 2:
                try:
                    dt_str = parts[1]
                    img_dt = None
                    for fmt in ['%Y-%m-%dT%H-%M', '%Y%m%dT%H%M%S']:
                        try:
                            img_dt = datetime.strptime(dt_str.replace('-', ''), fmt)
                            break
                        except:
                            pass
                    if img_dt is None:
                        dt_clean = dt_str.replace('-', '')
                        img_dt = datetime.strptime(dt_clean[:15], '%Y%m%dT%H%M%S')
                    
                    # Find matching t_aligned from trajectory_df by matching datetime
                    img_dt_pd = pd.to_datetime(img_dt)
                    trajectory_datetimes = pd.to_datetime(trajectory_df["datetime"])
                    time_diffs = (trajectory_datetimes - img_dt_pd).abs()
                    matching_idx = time_diffs.idxmin()
                    t_al = float(trajectory_df.loc[matching_idx, "t_aligned"])
                    event_val = trajectory_df.loc[matching_idx, event_col]
                    try:
                        state_val = int(trajectory_df.loc[matching_idx, "state"])
                    except:
                        state_val = None
                    state_color = "black"
                    if state_val in CLOUD_CLASS_INFO:
                        state_color = CLOUD_CLASS_INFO[state_val]["color"]
                    
                    image_times.append((fpath, img_dt, t_al, event_val, state_color))
                except:
                    pass
        
        if not image_times:
            ax.text(0.5, 0.5, "No valid images", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Sort by t_aligned
        image_times.sort(key=lambda x: x[2])
        
        # Select n_crops equally spaced across entire duration
        if len(image_times) <= n_crops:
            selected_images = image_times
        else:
            # Calculate indices for equal spacing
            indices = np.linspace(0, len(image_times) - 1, n_crops, dtype=int)
            selected_images = [image_times[int(i)] for i in indices]
        
        if not selected_images:
            ax.text(0.5, 0.5, "No images selected", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Load images and create composite
        loaded_images = []
        titles = []
        
        colors = []
        t_aligned_vals = []
        lat_grids = []
        lon_grids = []
        for fpath, img_dt, t_al, _event_val, state_color in selected_images:
            try:
                img_array, lat_vals, lon_vals = load_nc_as_image(fpath)
                if img_array is None:
                    continue
                loaded_images.append(img_array)
                lat_grids.append(lat_vals)
                lon_grids.append(lon_vals)
                titles.append(f"{t_al:+.1f}h")
                colors.append(state_color)
                t_aligned_vals.append(t_al)
            except:
                pass
        
        if not loaded_images:
            ax.text(0.5, 0.5, "No images loaded", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # Composite images horizontally with gaps
        h, w = loaded_images[0].shape[:2]
        
        # Create composite with gaps
        gap = gap_pixels
        total_width = w * len(loaded_images) + gap * (len(loaded_images) - 1)
        composite = np.full((h, total_width), 255, dtype=loaded_images[0].dtype)
        
        # Place images with gaps
        current_x = 0
        for idx, img in enumerate(loaded_images):
            composite[:, current_x:current_x + w] = img[:, :w]
            current_x += w + gap
        
        # Display composite
        ax.imshow(composite, cmap="gray", aspect="auto", origin="upper")
        
        # Draw colored frames for each crop (class color) and event markers
        precip_coords = normalize_coords(precip_coords)
        hail_coords = normalize_coords(hail_coords)
        current_x = 0
        for idx, color in enumerate(colors):
            rect = mpatches.Rectangle(
                (current_x, 0),
                w,
                h,
                fill=False,
                linewidth=3,
                edgecolor=color
            )
            ax.add_patch(rect)
            
            lat_vals = lat_grids[idx]
            lon_vals = lon_grids[idx]
            if precip_coords:
                for lat, lon in precip_coords:
                    pos = latlon_to_pixel(lat, lon, lat_vals, lon_vals)
                    if pos is None:
                        continue
                    x_idx, y_idx = pos
                    ax.plot(current_x + x_idx, y_idx, marker="o", color="cyan", markersize=6,
                            markeredgecolor="darkblue", markeredgewidth=1, alpha=0.8)
            if hail_coords:
                for lat, lon in hail_coords:
                    pos = latlon_to_pixel(lat, lon, lat_vals, lon_vals)
                    if pos is None:
                        continue
                    x_idx, y_idx = pos
                    ax.plot(current_x + x_idx, y_idx, marker="s", color="red", markersize=6,
                            markeredgecolor="darkred", markeredgewidth=1, alpha=0.8)
            current_x += w + gap
        
        # Draw dotted lines on the event plot aligned with each crop time
        if ax_events is not None:
            for t_al, color in zip(t_aligned_vals, colors):
                ax_events.axvline(x=t_al, color=color, ls=":", lw=2, alpha=0.9)
        
        # Add vertical lines at gaps for visual separation
        current_x = w
        for idx in range(len(loaded_images) - 1):
            ax.axvline(x=current_x - 0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
            current_x += w + gap
        
        # Set ticks at image centers
        tick_positions = []
        current_x = w / 2
        for idx in range(len(loaded_images)):
            tick_positions.append(current_x)
            current_x += w + gap
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(titles, fontsize=10)
        ax.set_yticks([])
    
    # Display precipitation crops
    precip_coords = extreme_precip_pw_row.get("precip_event_coords", None)
    hail_coords = extreme_hail_pw_row.get("hail_event_coords", None)
    plot_crop_row(ax_crops_precip, precip_pw_storm_id, df_pw, crop_dir, "n_precip", ax_events, precip_coords=precip_coords, hail_coords=None)
    precip_dt = pd.to_datetime(extreme_precip_pw_row["datetime"])
    precip_val = extreme_precip_pw_row["precipitation99"]
    ax_crops_precip.set_title(
        f"{precip_dt.strftime('%Y-%m-%d')} | Max rain rate: {precip_val:.2f}",
        fontsize=11,
        fontweight="bold"
    )
    ax_crops_precip.set_ylabel("Precip", fontsize=11, fontweight="bold")
    ax_crops_precip.yaxis.set_label_position("left")
    
    # Display hail crops
    plot_crop_row(ax_crops_hail, hail_pw_storm_id, df_pw, crop_dir, "n_hail", ax_events, precip_coords=None, hail_coords=hail_coords)
    hail_dt = pd.to_datetime(extreme_hail_pw_row["datetime"])
    hail_val = extreme_hail_pw_row["max_hail_intensity"]
    ax_crops_hail.set_title(
        f"{hail_dt.strftime('%Y-%m-%d')} | Max hail diameter: {hail_val:.2f}",
        fontsize=11,
        fontweight="bold"
    )
    ax_crops_hail.set_ylabel("Hail", fontsize=11, fontweight="bold")
    ax_crops_hail.yaxis.set_label_position("left")

    # ==========================================================
    # TITLE + SAVE
    # ==========================================================
    #use the actual class names in the pathway description instead of the numbers
    pathway_desc = []
    for state in pathway.split(" -> "):
        try:
            state_int = int(state)
            idx = selected_labels_ordered.index(state_int)
            pathway_desc.append(selected_short_labels[idx])
        except:
            pathway_desc.append(state)
    pathway_desc_str = " → ".join(pathway_desc)
    fig.suptitle(
        f"Pathway: {pathway_desc_str}" + f" - n_storms: {n_storms}",
        fontsize=18,
        y=1.00
    )

    fname = f"pathway_{pathway_id}_no_dom.png"
    fig.savefig(
        os.path.join(OUT_DIR, fname),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# ==========================================================
# SEPARATE CROP IMAGE FIGURES FOR EXTREME EVENTS
# ==========================================================
from PIL import Image
import glob
from datetime import datetime, timedelta

def load_image_by_datetime(storm_id, target_dt, crop_dir, tolerance_hours=2):
    """Find and load the closest NC crop to the target datetime."""
    try:
        # Search for images matching this storm
        search_pattern = f"{crop_dir}/storm{storm_id}_*_*.nc"
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            return None
        
        # Find the closest image by datetime in filename
        closest_file = None
        closest_diff = float('inf')
        
        for fpath in matching_files:
            # Extract datetime from filename (format: storm{id}_{datetime}_...)
            basename = os.path.basename(fpath)
            # Find the datetime part (after first underscore, before next underscore or .nc)
            parts = basename.split('_')
            if len(parts) >= 2:
                try:
                    # Try to parse datetime (format: YYYYMMDDTHHMMSS or similar)
                    dt_str = parts[1]  # e.g., "2024-06-18T11-00"
                    # Parse with different formats
                    img_dt = None
                    for fmt in ['%Y-%m-%dT%H-%M', '%Y%m%dT%H%M%S']:
                        try:
                            img_dt = datetime.strptime(dt_str.replace('-', ''), fmt)
                            break
                        except:
                            pass
                    
                    if img_dt is None:
                        # Try alternative parsing
                        dt_clean = dt_str.replace('-', '')
                        img_dt = datetime.strptime(dt_clean[:15], '%Y%m%dT%H%M%S')
                    
                    # Calculate difference in hours
                    diff = abs((img_dt - target_dt).total_seconds() / 3600)
                    if diff < closest_diff and diff <= tolerance_hours:
                        closest_diff = diff
                        closest_file = fpath
                except:
                    pass
        
        if closest_file and closest_diff <= tolerance_hours:
            img_array, _lat_vals, _lon_vals = load_nc_as_image(closest_file)
            return img_array
    except Exception as e:
        pass
    
    return None

# Find the most extreme precip and hail storms across all pathways
extreme_precip_idx = df["precipitation99"].idxmax()
extreme_hail_idx = df["max_hail_intensity"].idxmax()

extreme_precip_row = df.loc[extreme_precip_idx]
extreme_hail_row = df.loc[extreme_hail_idx]

# Extract storm IDs (remove 'storm' prefix if present)
precip_storm_id_str = extreme_precip_row["storm_id"]
hail_storm_id_str = extreme_hail_row["storm_id"]
precip_storm_id = int(precip_storm_id_str.replace('storm', '')) if 'storm' in str(precip_storm_id_str) else int(precip_storm_id_str)
hail_storm_id = int(hail_storm_id_str.replace('storm', '')) if 'storm' in str(hail_storm_id_str) else int(hail_storm_id_str)

def create_crop_image_figure(storm_id, storm_trajectory_df, event_type, crop_dir):
    """Create a figure with all crop images for a specific trajectory."""
    # Get all unique datetimes from this trajectory
    datetimes = pd.to_datetime(storm_trajectory_df["datetime"]).unique()
    datetimes = sorted(datetimes)
    
    # Calculate the mean datetime to use as reference for t_aligned
    mean_dt = pd.to_datetime(storm_trajectory_df["datetime"]).mean()
    
    # Load images for each unique datetime
    images = {}
    for dt in datetimes:
        img = load_image_by_datetime(storm_id, dt, crop_dir, tolerance_hours=0.5)
        if img is not None:
            images[dt] = img
    
    if not images:
        print(f"No images found for {event_type} storm {storm_id}")
        return None
    
    print(f"Found {len(images)} images for {event_type} storm {storm_id}")
    
    # Create grid figure with images
    n_cols = min(len(images), 6)  # Max 6 columns
    n_rows = int(np.ceil(len(images) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*2.5))
    axes = np.atleast_2d(axes).flatten()
    
    for idx, (dt, img) in enumerate(sorted(images.items())):
        ax = axes[idx]
        ax.imshow(img, cmap="gray")
        # Calculate t_aligned (hours from mean datetime)
        t_aligned = (dt - mean_dt).total_seconds() / 3600
        ax.set_title(f"t = {t_aligned:.1f} h", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].set_visible(False)
    
    # fig.suptitle(
    #     f"Extreme {event_type} Event (Storm {storm_id}) - Crop Images",
    #     fontsize=14,
    #     fontweight="bold"
    # )
    fig.tight_layout()
    
    return fig

# Get trajectory data for extreme events
precip_trajectory = df[df["storm_id"] == extreme_precip_row["storm_id"]]
hail_trajectory = df[df["storm_id"] == extreme_hail_row["storm_id"]]

# Create precipitation figure
fig_precip = create_crop_image_figure(precip_storm_id, precip_trajectory, "Precipitation", crop_dir)
if fig_precip is not None:
    fig_precip.savefig(
        os.path.join(OUT_DIR, "extreme_precip_crop_images.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_precip)
    print(f"Saved extreme precipitation crop images for storm {precip_storm_id}")

# Create hail figure
fig_hail = create_crop_image_figure(hail_storm_id, hail_trajectory, "Hail", crop_dir)
if fig_hail is not None:
    fig_hail.savefig(
        os.path.join(OUT_DIR, "extreme_hail_crop_images.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_hail)
    print(f"Saved extreme hail crop images for storm {hail_storm_id}")


# ==========================================================
# TOP 50 PRECIPITATION AND HAIL EVENTS
# ==========================================================

# Get top 50 precipitation events
top_precip_events = storm_stats.nlargest(50, "max_precipitation99")[["pathway", "max_precipitation99"]].reset_index(drop=True)
top_precip_events.index = range(1, len(top_precip_events) + 1)

# Get top 50 hail events
top_hail_events = storm_stats.nlargest(50, "max_hail_intensity")[["pathway", "max_hail_intensity"]].reset_index(drop=True)
top_hail_events.index = range(1, len(top_hail_events) + 1)

# Pathway type lookup
pathway_type_map = (
    df.dropna(subset=["pathway_type"])
    .drop_duplicates(subset=["pathway"])
    .set_index("pathway")["pathway_type"]
)

# Precipitation pathway presence stats
precip_pathway_counts = top_precip_events["pathway"].value_counts()
precip_unique_pathways = precip_pathway_counts.size
precip_type_counts_occ = (
    top_precip_events["pathway"].map(pathway_type_map).fillna("unknown").value_counts()
)
precip_type_counts_unique = (
    precip_pathway_counts.index.to_series().map(pathway_type_map).fillna("unknown").value_counts()
)

# Hail pathway presence stats
hail_pathway_counts = top_hail_events["pathway"].value_counts()
hail_unique_pathways = hail_pathway_counts.size
hail_type_counts_occ = (
    top_hail_events["pathway"].map(pathway_type_map).fillna("unknown").value_counts()
)
hail_type_counts_unique = (
    hail_pathway_counts.index.to_series().map(pathway_type_map).fillna("unknown").value_counts()
)

print("Top 50 precip: unique pathways =", precip_unique_pathways)
print("Top 50 precip pathway ranking (count in top 50):")
print(precip_pathway_counts)
print("Top 50 precip pathway types (occurrences):")
print(precip_type_counts_occ)
print("Top 50 precip pathway types (unique pathways):")
print(precip_type_counts_unique)

print("Top 50 hail: unique pathways =", hail_unique_pathways)
print("Top 50 hail pathway ranking (count in top 50):")
print(hail_pathway_counts)
print("Top 50 hail pathway types (occurrences):")
print(hail_type_counts_occ)
print("Top 50 hail pathway types (unique pathways):")
print(hail_type_counts_unique)

# --- Visualization: Top pathways in Top 50 ---
def pathway_to_short_names(pathway):
    states = pathway.split(" -> ")
    short_names = []
    for state in states:
        try:
            state_int = int(state)
            idx = selected_labels_ordered.index(state_int)
            short_names.append(selected_short_labels[idx])
        except:
            short_names.append(state)
    return " → ".join(short_names)

top_k = 8
precip_top_k = precip_pathway_counts.head(top_k)
hail_top_k = hail_pathway_counts.head(top_k)
# print(f"Top {top_k} pathways in top 50 precipitation events:")
# print(precip_top_k)
# print(f"Top {top_k} pathways in top 50 hail events:")
# print(hail_top_k)


# --- Combined: Top pathways + Pathway types (2x2 compact layout) ---
type_labels = ["persistence", "binary", "trinary"]
precip_type_vals = [precip_type_counts_occ.get(lbl, 0) for lbl in type_labels]
hail_type_vals = [hail_type_counts_occ.get(lbl, 0) for lbl in type_labels]

fig = plt.figure(figsize=(9, 5))
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35, height_ratios=[1.2, 0.6])

# --- Row 1: Top pathways (barh) ---
# Precip barh (top left)
ax00 = fig.add_subplot(gs[0, 0])
precip_top_k_labels = [pathway_to_short_names(p) for p in precip_top_k.index]
ax00.barh(
    list(reversed(precip_top_k_labels)),
    list(reversed(precip_top_k.values)),
    color="steelblue",
    edgecolor="black",
    linewidth=0.6
)
ax00.set_title("a) Rain: Top 8 Pathways", fontsize=10, fontweight="bold")
ax00.set_xlabel("Count in Top 50", fontsize=9)
#make labelticks on x axis integers only
ax00.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax00.set_xlim(0, 21)
ax00.tick_params(axis="both", labelsize=8)

# Hail barh (top right)
ax01 = fig.add_subplot(gs[0, 1])
hail_top_k_labels = [pathway_to_short_names(p) for p in hail_top_k.index]
ax01.barh(
    list(reversed(hail_top_k_labels)),
    list(reversed(hail_top_k.values)),
    color="indianred",
    edgecolor="black",
    linewidth=0.6
)
ax01.set_title("b) Hail: Top 8 Pathways", fontsize=10, fontweight="bold")
ax01.set_xlabel("Count in Top 50", fontsize=9)
ax01.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax01.set_xlim(0, 21)
ax01.tick_params(axis="both", labelsize=8)

# --- Row 2: Pathway types (bar) ---
# Precip types (bottom left)
ax10 = fig.add_subplot(gs[1, 0])
ax10.bar(type_labels, precip_type_vals, color="steelblue", edgecolor="black", linewidth=0.6)
ax10.set_title("c) Rain: Pathway Types", fontsize=10, fontweight="bold")
ax10.set_ylabel("Count in Top 50", fontsize=9)
ax10.tick_params(axis="x", rotation=20, labelsize=8)
ax10.tick_params(axis="y", labelsize=8)

# Hail types (bottom right)
ax11 = fig.add_subplot(gs[1, 1])
ax11.bar(type_labels, hail_type_vals, color="indianred", edgecolor="black", linewidth=0.6)
ax11.set_title("d) Hail: Pathway Types", fontsize=10, fontweight="bold")
ax11.tick_params(axis="x", rotation=20, labelsize=8)
ax11.tick_params(axis="y", labelsize=8)

# Share y-axis for pathway types (bottom row)
ax10.set_ylim(ax11.get_ylim())

fig.savefig(
    os.path.join(OUT_DIR, "top50_pathway_ranking_and_types.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Find the most common pathway in top 20 for each event type
most_common_precip_pathway = top_precip_events["pathway"].value_counts().idxmax()
most_common_hail_pathway = top_hail_events["pathway"].value_counts().idxmax()

# Find the most extreme (max value) pathway for each event type
extreme_precip_pathway = top_precip_events.loc[top_precip_events["max_precipitation99"].idxmax(), "pathway"]
extreme_hail_pathway = top_hail_events.loc[top_hail_events["max_hail_intensity"].idxmax(), "pathway"]

# Create figure with 2 rows
fig, axes = plt.subplots(2, 1, figsize=(14, 6))

# --- Top 20 Precipitation Events ---
ax_precip = axes[0]
# Assign colors based on whether pathway is most common
colors_precip = []
for pathway in top_precip_events["pathway"].values:
    if pathway == most_common_precip_pathway:
        colors_precip.append(selected_colors_ordered[0])  # Color most common pathway
    else:
        colors_precip.append("lightgray")

bars_precip = ax_precip.bar(
    range(len(top_precip_events)),
    top_precip_events["max_precipitation99"].values,
    color=colors_precip,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.8
)

# Make the first bar (most extreme) have bold black borders
bars_precip[0].set_edgecolor("black")
bars_precip[0].set_linewidth(3)

ax_precip.set_ylabel("Max Rain \n Rate (mm/h)", fontsize=13, fontweight="bold")
#ax_precip.set_xlabel("Pathway Rank", fontsize=13, fontweight="bold")
ax_precip.set_title("a) Top 50 Most Intense Precipitation Events", fontsize=14, fontweight="bold")
ax_precip.set_xticks(range(len(top_precip_events)))
# Set x-axis labels to rankings (convert pathway to short class names)
rankings_precip = []
for pathway in top_precip_events["pathway"].values:
    states = pathway.split(" -> ")
    short_names = []
    for state in states:
        # state is like '1', '2', '4'
        try:
            state_int = int(state)
            idx = selected_labels_ordered.index(state_int)
            short_names.append(selected_short_labels[idx])
        except:
            short_names.append(state)
    rankings_precip.append(" → ".join(short_names))
ax_precip.set_xticklabels(rankings_precip, rotation=45, ha="right", fontsize=10)
ax_precip.set_yticks([0, 25, 50])
ax_precip.tick_params(axis="y", labelsize=12)
ax_precip.tick_params(axis="x", labelsize=12)
ax_precip.grid(axis="y", alpha=0.3, ls="--")
ax_precip.set_axisbelow(True)

# --- Top 20 Hail Events ---
ax_hail = axes[1]
# Assign colors based on whether pathway is most common
colors_hail = []
for pathway in top_hail_events["pathway"].values:
    if pathway == most_common_hail_pathway:
        colors_hail.append(selected_colors_ordered[1])  # Color most common pathway
    else:
        colors_hail.append("lightgray")

bars_hail = ax_hail.bar(
    range(len(top_hail_events)),
    top_hail_events["max_hail_intensity"].values,
    color=colors_hail,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.8
)

# Make the first bar (most extreme) have bold black borders
bars_hail[0].set_edgecolor("black")
bars_hail[0].set_linewidth(3)

ax_hail.set_ylabel("Max Hail \n Diameter (cm)", fontsize=13, fontweight="bold")
#ax_hail.set_xlabel("Pathway Rank", fontsize=13, fontweight="bold")
ax_hail.set_title("b) Top 50 Most Intense Hail Events", fontsize=14, fontweight="bold")
ax_hail.set_xticks(range(len(top_hail_events)))
# Set x-axis labels to rankings (convert pathway to short class names)
rankings_hail = []
for pathway in top_hail_events["pathway"].values:
    states = pathway.split(" -> ")
    short_names = []
    for state in states:
        # state is like '1', '2', '4'
        try:
            state_int = int(state)
            idx = selected_labels_ordered.index(state_int)
            short_names.append(selected_short_labels[idx])
        except:
            short_names.append(state)
    rankings_hail.append(" → ".join(short_names))
ax_hail.set_xticklabels(rankings_hail, rotation=45, ha="right", fontsize=10)
ax_hail.set_yticks([0, 10, 20])
ax_hail.tick_params(axis="y", labelsize=12)
ax_hail.tick_params(axis="x", labelsize=12)
ax_hail.grid(axis="y", alpha=0.3, ls="--")
ax_hail.set_axisbelow(True)

fig.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "top_extremes_by_pathway_no_dom.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Top 10 precipitation events:")
print(top_precip_events)
print("\nTop 10 hail events:")
print(top_hail_events)
