import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cmcrameri.cm as cmc


mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["text.usetex"] = False

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO

RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUT_DIR = f"{BASE_DIR}/pathway_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
SELECTED_CLASSES = [1,2,4]
CLASSES_6 = [
    "1_low", "1_high",
    "2_low", "2_high",
    "4_low", "4_high",
]

path = f"{OUT_DIR}/df_pathways_merged.csv"
df = pd.read_csv(path, low_memory=False)
print(df)#.columns.tolist())
print(df.columns.tolist())


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
    elif stat == "std":
        return (
            df.groupby("t_bin")[value_col]
            .std()
            .reset_index()
        )
    else:
        raise ValueError(f"Unsupported stat: {stat}")




def compute_class_persistence(df_pw):
    records = []

    for storm_id, g in df_pw.sort_values("datetime").groupby("storm_id"):
        for state, gg in g.groupby("state"):
            duration = gg["t_bin"].max() - gg["t_bin"].min()
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
    "pathway_prob",
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
    df_stats.groupby(["pathway_id","pathway_prob", "pathway", "storm_id"]).apply(trajectory_radius_km)
    .reset_index(name="radius_km")
)

storm_stats = (
    df_stats.groupby(["pathway_id", "pathway_prob", "pathway", "storm_id"]).agg(
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
    on=["pathway_id", "pathway_prob", "pathway", "storm_id"],
    how="left"
)

pathway_stats = (
    storm_stats.groupby(["pathway_id", "pathway_prob", "pathway"]).agg(
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
    storm_stats.groupby(["pathway_id", "pathway_prob", "pathway"])["storm_id"]
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
    storm_stats.groupby(["pathway_id", "pathway_prob", "pathway"]).agg(
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
    storm_stats.groupby(["pathway_id", "pathway_prob", "pathway"])["storm_id"]
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

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=False)

norm = mpl.colors.Normalize(
    vmin=scatter_stats["mean_duration_hours"].min(),
    vmax=scatter_stats["mean_duration_hours"].max(),
)
cmap = cmc.imola

for ax, x_col, y_col, duration_col, title in zip(
    axes,
    ["mean_precipitation99", "max_precipitation99"],
    ["mean_hail_intensity", "max_hail_intensity"],
    ['mean_duration_hours', 'max_duration_hours'],
    ["a) Mean Intensities", "b) Max Intensities"],

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

# --- Shared colorbar ---
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    orientation="vertical",
    fraction=0.03,
    pad=0.04,
)
cbar.set_label("Storm Duration (hours)", fontsize=10)
cbar.ax.tick_params(labelsize=9)
#move colorbar to the right of the figure (more to the right and center)
cbar.ax.set_position([1.02, 0.15, 0.03, 0.7])

fig.tight_layout()

fig.savefig(
    os.path.join(OUT_DIR, "pathway_scatter_summary_mean_and_max.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# --- Print summary ---
print("Top hail pathway:")
print(top_hail_row[["pathway_id", "pathway_prob", "pathway", "max_hail_intensity"]])

print("Top precip pathway:")
print(top_precip_row[["pathway_id", "pathway_prob", "pathway", "max_precipitation99"]])



for pathway_id, event_type in selected_pathway_info.items():
    df_pw = df[df["pathway_id"] == pathway_id].copy()
    #number of unique storms in this pathway
    n_storms = df_pw["storm_id"].nunique()
    
    if df_pw.empty:
        continue
    path = df_pw["pathway"].iloc[0]
    print(f"Plotting pathway {pathway_id} ({event_type}) with path {path}")
    fig = plt.figure(figsize=(9, 10))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1, 0.75, 0.5],
        hspace=0.35,
        wspace=0.3,
        top=0.94
    )

    ax_feat  = fig.add_subplot(gs[0, 0])
    ax_box   = fig.add_subplot(gs[0, 1])
    ax_cloud = fig.add_subplot(gs[1, :])
    ax_events = fig.add_subplot(gs[2, :])

    # ==========================================================
    # FEATURE SPACE (background + trajectories)
    # ==========================================================
    ax_feat.scatter(
        df_tsne_train["tsne_dim_1"],
        df_tsne_train["tsne_dim_2"],
        color=df_tsne_train["color"].values,
        s=2,
        alpha=0.05,
        linewidth=0
    )

    for storm_id, g in df_pw.groupby("storm_id"):
        ax_feat.plot(
            g["tsne_dim_1"],
            g["tsne_dim_2"],
            color="black",
            alpha=0.25,
            lw=1
        )

    ax_feat.set_title("a) Feature space trajectories", fontsize=13, fontweight="bold")
    ax_feat.set_xticks([])
    ax_feat.set_yticks([])

    # ==========================================================
    # CLASS PERSISTENCE BOXPLOT
    # ==========================================================
    pers = compute_class_persistence(df_pw)
    pers = pers[pers["state"].isin(CLASSES_6)]

    data = [
        pers.loc[pers["state"] == cls, "duration"]
        for cls in CLASSES_6
    ]

    bp = ax_box.boxplot(
        data,
        labels=CLASSES_6,
        showfliers=False,
        patch_artist=True
    )
    for patch, cls in zip(bp["boxes"], CLASSES_6):
        base_label = int(str(cls).split("_")[0])
        patch.set_facecolor(CLOUD_CLASS_INFO[base_label]["color"])
        patch.set_edgecolor("black")
    ax_box.set_ylabel("Persistence duration (hours)", fontsize=12)
    ax_box.set_title("b) Class persistence", fontsize=13, fontweight="bold")
    ax_box.tick_params(axis="x", rotation=45, labelsize=11)
    ax_box.tick_params(axis="y", labelsize=11)
    ax_box.grid(ls="--", alpha=0.3)

    # ==========================================================
    # TIME-ALIGNED PANEL 1: CLOUD PROPERTIES ONLY
    # ==========================================================

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
    
    ax_cloud.set_ylabel("Cloud properties", fontsize=12)
    ax_cloud.set_xlabel("Aligned time (hours)", fontsize=12)
    ax_cloud.set_title("c) Time-aligned cloud properties", fontsize=13, fontweight="bold")
    ax_cloud.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_cloud.grid(ls="--", alpha=0.3)
    ax_cloud.tick_params(axis="both", labelsize=11)
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
        color="C0",
        lw=2,
        label="Rain events"
    )

    ax_events.plot(
        mean_hail["t_bin"],
        mean_hail["n_hail"],
        color="C1",
        lw=2,
        label="Hail Events"
    )

    
    ax_events.set_ylabel("Event count", fontsize=12)
    ax_events.set_xlabel("Aligned time (hours)", fontsize=12)
    ax_events.set_title("d) Time-aligned event counts by type", fontsize=13, fontweight="bold")
    ax_events.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_events.grid(ls="--", alpha=0.3)
    ax_events.tick_params(axis="both", labelsize=11)
    ax_events.legend(fontsize=10, loc="upper right")



    # ==========================================================
    # TITLE + SAVE
    # ==========================================================
    fig.suptitle(
        f"Pathway: " + "".join(path) + f" - n_storms: {n_storms}",
        fontsize=18,
        y=1.01
    )

    fname = f"pathway_{pathway_id}.png"
    fig.savefig(
        os.path.join(OUT_DIR, fname),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# ==========================================================
# TOP 20 PRECIPITATION AND HAIL EVENTS
# ==========================================================

# Get top 20 precipitation events
top_precip_events = storm_stats.nlargest(20, "max_precipitation99")[["pathway", "max_precipitation99"]].reset_index(drop=True)
top_precip_events.index = range(1, len(top_precip_events) + 1)

# Get top 20 hail events
top_hail_events = storm_stats.nlargest(20, "max_hail_intensity")[["pathway", "max_hail_intensity"]].reset_index(drop=True)
top_hail_events.index = range(1, len(top_hail_events) + 1)

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
#ax_precip.set_xlabel("Pathway", fontsize=13, fontweight="bold")
ax_precip.set_title("a) Top 20 Most Intense Precipitation Events", fontsize=14, fontweight="bold")
ax_precip.set_xticks(range(len(top_precip_events)))
ax_precip.set_xticklabels(top_precip_events["pathway"].values, rotation=45, ha="right", fontsize=10)
ax_precip.set_yticks([0, 25, 50])
ax_precip.tick_params(axis="y", labelsize=11)
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
#ax_hail.set_xlabel("Pathway", fontsize=13, fontweight="bold")
ax_hail.set_title("b) Top 20 Most Intense Hail Events", fontsize=14, fontweight="bold")
ax_hail.set_xticks(range(len(top_hail_events)))
ax_hail.set_xticklabels(top_hail_events["pathway"].values, rotation=45, ha="right", fontsize=10)
ax_hail.set_yticks([0, 10, 20])
ax_hail.tick_params(axis="y", labelsize=11)
ax_hail.grid(axis="y", alpha=0.3, ls="--")
ax_hail.set_axisbelow(True)

fig.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "top_10_extremes_by_pathway.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Top 10 precipitation events:")
print(top_precip_events)
print("\nTop 10 hail events:")
print(top_hail_events)
