#!/usr/bin/env python3
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
path_to_dir = f"/data1/fig/{RUN_NAME}/epoch_800/closest/"
merged_path = os.path.join(path_to_dir, "merged_crops_stats_cvc_imergtime_closest_1000.csv")

# Path for hourly occurrence data
all_dir = f"/data1/fig/{RUN_NAME}/epoch_800/all/"
crop_list_file = "crop_list_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv"
merged_all_path = os.path.join(all_dir, "merged_crops_stats_all_cvc_cot_cth_fractions.csv")

output_dir = os.path.join(path_to_dir, "training_set_characterization")
os.makedirs(output_dir, exist_ok=True)

# Use the 100 closest samples to each centroid (per class)
N_CLOSEST = 1000

# Marker styling
SAMPLE_SIZE = 28
SAMPLE_ALPHA = 0.8
CLOSEST_SIZE = 140
CENTROID_SIZE = 180
FONTSIZE = 13
LEGEND_FONTSIZE = 10

# Make utils/plotting visible
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/plotting")
from class_colors import CLOUD_CLASS_INFO  # noqa: E402

# === VARIABLES ===
VARIABLES = {
    "cth": {"label": "CTH Median (km)", "vmin": 3.5, "vmax": 9.5, "logscale": False, "mult": 0.001, "offset": 0, "value_col": "50"},
    "cot": {"label": "COT Median", "vmin": 1, "vmax": 150, "logscale": True, "mult": 1, "offset": 0, "value_col": "50"},
}

VARS = ["cth", "cot"]

# ==========================================================================
# LOAD DATA FOR SCATTER PLOTS (CTH vs COT)
# ==========================================================================
print(f"Loading scatter plot data from: {merged_path}")
df = pd.read_csv(merged_path)
print(f"Loaded: {merged_path} ({df.shape})")

# === EXTRACT VALUES ===
subset = df[df["var"].isin(VARS)].copy()

subset["value"] = pd.NA
for var_name, settings in VARIABLES.items():
    mask = subset["var"] == var_name
    subset.loc[mask, "value"] = subset.loc[mask, settings["value_col"]]
    subset.loc[mask, "value"] = (
        pd.to_numeric(subset.loc[mask, "value"], errors="coerce")
        * settings["mult"]
        + settings["offset"]
    )

# Pivot to one row per crop with cth/cot as columns
wide = (
    subset.pivot_table(
        index=["crop_index", "label", "distance"],
        columns="var",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

# Keep only rows with all variables available
wide = wide.dropna(subset=VARS)

# Pick N closest per class (smaller distance means closer)
wide = (
    wide.sort_values(["label", "distance"], ascending=[True, True])
    .groupby("label", as_index=False)
    .head(N_CLOSEST)
)

# ==========================================================================
# LOAD DATA FOR HOURLY OCCURRENCE
# ==========================================================================
print(f"\nLoading hourly occurrence data from: {os.path.join(all_dir, crop_list_file)}")
df_hourly = pd.read_csv(os.path.join(all_dir, crop_list_file))

# Extract time information
def extract_time_from_path(path):
    """Extract datetime from filename like: 2021-07-18T12:15:00_xx.nc"""
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", os.path.basename(path))
    return pd.to_datetime(match.group(1)) if match else pd.NaT

df_hourly["time"] = df_hourly["path"].apply(extract_time_from_path)
df_hourly = df_hourly.dropna(subset=["time"])
df_hourly["hour"] = df_hourly["time"].dt.hour
df_hourly["date"] = df_hourly["time"].dt.date
df_hourly["crop"] = df_hourly["path"].apply(os.path.basename)

# Build nighttime flag like in plot_sample_counts.py
df_night_src = pd.read_csv(merged_all_path, low_memory=False)
night_var_mask = df_night_src["var"] == "nighttime_scene"

if night_var_mask.any():
    df_night = (
        df_night_src.loc[night_var_mask, ["crop", "None"]]
        .rename(columns={"None": "nighttime"})
        .drop_duplicates(subset=["crop"])
    )
    df_hourly = df_hourly.merge(df_night, on="crop", how="left")
    df_hourly["nighttime"] = df_hourly["nighttime"].fillna("False")
else:
    # Fallback: infer nighttime from hour when nighttime_scene is unavailable
    inferred_night = (df_hourly["hour"] < 6) | (df_hourly["hour"] >= 18)
    df_hourly["nighttime"] = np.where(inferred_night, "True", "False")

# remove invalid labels if present
df_hourly = df_hourly[df_hourly["label"] != -100]

# Count occurrences per hour and label
occ_counts = (
    df_hourly.groupby(["hour", "label"])
       .size()
       .unstack(fill_value=0)
)

# Ensure all 24 hours exist
occ_counts = occ_counts.reindex(np.arange(24), fill_value=0)

# ==========================================================================
# LOAD DATA FOR CMA BOXPLOT
# ==========================================================================
print(f"\nLoading CMA data from: {merged_all_path}")
df_cma = pd.read_csv(merged_all_path)
df_cma = df_cma[df_cma["var"] == "cma"]

cloud_items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
labels_ordered = [lbl for lbl, _ in cloud_items]
short_labels = [info["short"] for _, info in cloud_items]
colors = {lbl: info["color"] for lbl, info in cloud_items}

# ==========================================================================
# CLASS-WISE STATS (MEAN/STD) FOR CC, CTH, COT
# ==========================================================================
cc_stats = (
    df_cma.assign(cc_pct=pd.to_numeric(df_cma["None"], errors="coerce") * 100)
    .groupby("label")["cc_pct"]
    .agg(cc_mean="mean", cc_std="std")
)

cth_cot_stats = (
    wide.groupby("label")[["cth", "cot"]]
    .agg(["mean", "std"])
)
cth_cot_stats.columns = [f"{var}_{stat}" for var, stat in cth_cot_stats.columns]

stats_df = (
    pd.DataFrame(index=labels_ordered)
    .join(cc_stats, how="left")
    .join(cth_cot_stats, how="left")
)
stats_df.index.name = "label"
stats_df = stats_df.reset_index()
stats_df["short"] = stats_df["label"].map(lambda x: CLOUD_CLASS_INFO.get(x, {}).get("short", str(x)))

stats_display = stats_df[[
    "label",
    "short",
    "cc_mean",
    "cc_std",
    "cth_mean",
    "cth_std",
    "cot_mean",
    "cot_std",
]]

print("\nClass-wise mean/std summary (CC %, CTH km, COT):")
print(stats_display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

stats_txt_path = os.path.join(output_dir, "training_set_characterization_class_stats.txt")
with open(stats_txt_path, "w", encoding="utf-8") as f:
    f.write("Class-wise mean/std summary (CC %, CTH km, COT)\n")
    f.write("Computed from: CC=cma over all crops, CTH/COT=selected closest samples\n\n")
    f.write(stats_display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    f.write("\n")

print(f"Saved class-wise stats to: {stats_txt_path}")

# ==========================================================================
# SETUP FIGURE WITH GRIDSPEC (2 rows x 4 cols)
# ==========================================================================
fig = plt.figure(figsize=(16, 7))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.5)

# Row 0
ax_counts = fig.add_subplot(gs[0, 0])      # Sample counts
ax_hourly = fig.add_subplot(gs[0, 1:])     # Hourly occurrence (spans 3 columns)

# Row 1
ax_boxplot = fig.add_subplot(gs[1, 0])     # CMA boxplot
ax_scatter = [fig.add_subplot(gs[1, i]) for i in range(1, 4)]  # 3 scatter plots

# ==========================================================================
# PLOT 1: SAMPLE COUNTS (DAY/NIGHT)
# ==========================================================================
print("\nCreating sample counts plot...")

# Identify daytime / nighttime
df_hourly_counts = df_hourly.copy()
night_flag = df_hourly_counts["nighttime"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
daytime_counts = df_hourly_counts[~night_flag]["label"].value_counts().reindex(labels_ordered).fillna(0)
nighttime_counts = df_hourly_counts[night_flag]["label"].value_counts().reindex(labels_ordered).fillna(0)

x = np.arange(len(labels_ordered))
width = 0.7

# Daytime bars (bottom)
ax_counts.bar(
    x,
    daytime_counts.values * 1e-3,  # Scale to thousands
    width=width,
    color=[colors[lbl] for lbl in labels_ordered],
    label="Daytime"
)

# Nighttime bars (stacked, hatched)
ax_counts.bar(
    x,
    nighttime_counts.values * 1e-3,  # Scale to thousands
    width=width,
    bottom=daytime_counts.values * 1e-3,
    color=[colors[lbl] for lbl in labels_ordered],
    hatch="///",
    edgecolor="black",
    linewidth=0.5,
    label="Nighttime"
)

ax_counts.set_ylabel("# Samples (10$^3$)", fontsize=FONTSIZE)
ax_counts.set_xticks(x)
ax_counts.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=FONTSIZE)
ax_counts.tick_params(axis='y', labelsize=FONTSIZE)
ax_counts.set_yticks([0, 10, 20, 30])
ax_counts.set_title("a) Samples per Class", fontsize=FONTSIZE, fontweight="bold")
ax_counts.tick_params(axis='both', which='major', labelsize=FONTSIZE)
ax_counts.spines['top'].set_visible(False)
ax_counts.spines['right'].set_visible(False)
ax_counts.spines['bottom'].set_linewidth(1.5)
ax_counts.spines['left'].set_linewidth(1.5)
ax_counts.grid(color='grey', alpha=0.5, linestyle='--')
ax_counts.legend(fontsize=LEGEND_FONTSIZE, loc="upper center", framealpha=0.8)

# ==========================================================================
# PLOT 2: HOURLY OCCURRENCE (STACKED BAR)
# ==========================================================================
print("Creating hourly occurrence plot...")

bottom = np.zeros(len(occ_counts.index))

for label, info in cloud_items:
    color = info["color"]
    values = occ_counts[label].values

    ax_hourly.bar(
        occ_counts.index,
        values * 1e-3,
        bottom=bottom,
        color=color,
        edgecolor="black",
        linewidth=0.3,
    )
    bottom += values * 1e-3

ax_hourly.set_title("b) Hourly Occurrence Count", fontsize=FONTSIZE, fontweight="bold")
ax_hourly.set_xlabel("Hour (UTC)", fontsize=FONTSIZE)
ax_hourly.set_ylabel("# Occurrences (10$^3$)", fontsize=FONTSIZE)
ax_hourly.set_xticks(range(0, 24, 1))
ax_hourly.tick_params(axis='both', which='major', labelsize=FONTSIZE)
ax_hourly.spines['top'].set_visible(False)
ax_hourly.spines['right'].set_visible(False)
ax_hourly.spines['bottom'].set_linewidth(1.5)
ax_hourly.spines['left'].set_linewidth(1.5)
ax_hourly.grid(color='grey', alpha=0.5, linestyle='--')

# ==========================================================================
# PLOT 3: CMA BOXPLOT
# ==========================================================================
print("Creating CMA boxplot...")

box_data = []
box_positions = []
box_colors = []

for j, (label, color_info) in enumerate(cloud_items, start=1):
    subset = df_cma[df_cma["label"] == label]
    data = pd.to_numeric(subset["None"], errors="coerce").dropna()
    data = data * 100  # Convert to percentage
    
    box_data.append(data.values)
    box_positions.append(j)
    box_colors.append(color_info["color"])

bp = ax_boxplot.boxplot(
    box_data,
    positions=box_positions,
    widths=0.4,
    patch_artist=True,
    showfliers=False
)

# Apply colors
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp.get(element, []), color='black', linewidth=1.2)

ax_boxplot.set_ylabel("CC (%)", fontsize=FONTSIZE)
ax_boxplot.set_xticks(box_positions)
ax_boxplot.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=FONTSIZE)
ax_boxplot.tick_params(axis='both', which='major', labelsize=FONTSIZE)
ax_boxplot.set_title("c) CC Distribution", fontsize=FONTSIZE, fontweight="bold")
ax_boxplot.spines['top'].set_visible(False)
ax_boxplot.spines['right'].set_visible(False)
ax_boxplot.spines['bottom'].set_linewidth(1.5)
ax_boxplot.spines['left'].set_linewidth(1.5)
ax_boxplot.grid(color='grey', alpha=0.5, linestyle='--')
ax_boxplot.set_ylim(0, 100)

# ==========================================================================
# PLOTS 4-6: SCATTER PLOTS (COT vs CTH)
# ==========================================================================
print("Creating scatter plots...")

# === GROUPS (by CLOUD_CLASS_INFO short names) ===
short_to_label = {info["short"]: lbl for lbl, info in CLOUD_CLASS_INFO.items()}

GROUPS = {
    "d) Low-Level Clouds": ["CS", "SC"],
    "e) Mid-Level Clouds": ["MC1", "MC2"],
    "f) Convective Clouds": ["EC", "DC", "OA"],
}

for ax, (group_name, short_names) in zip(ax_scatter, GROUPS.items()):

    group_labels = [short_to_label[s] for s in short_names if s in short_to_label]
    group_df = wide[wide["label"].isin(group_labels)]

    for lbl in group_labels:
        cls_df = group_df[group_df["label"] == lbl]
        color = CLOUD_CLASS_INFO[lbl]["color"]
        short = CLOUD_CLASS_INFO[lbl]["short"]

        if cls_df.empty:
            continue

        ax.scatter(
            cls_df["cot"],
            cls_df["cth"],
            color=color,
            label=short,
            s=SAMPLE_SIZE,
            alpha=SAMPLE_ALPHA,
            edgecolors="none",
        )

        # Centroid proxy (mean of selected samples)
        centroid_cth = cls_df["cth"].mean()
        centroid_cot = cls_df["cot"].mean()
        ax.scatter(
            centroid_cot,
            centroid_cth,
            color=color,
            marker="X",
            s=CENTROID_SIZE,
            edgecolors="black",
            linewidth=0.8,
            zorder=4,
        )

    ax.set_title(group_name, fontsize=FONTSIZE, fontweight="bold")
    ax.set_xlabel(VARIABLES["cot"]["label"], fontsize=FONTSIZE)
    
    # Set y label only for the first subplot
    #if ax == ax_scatter[0]:
    ax.set_ylabel(VARIABLES["cth"]["label"], fontsize=FONTSIZE)
    
    # Set ticks labels size
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    ax.set_xlim(VARIABLES["cot"]["vmin"], VARIABLES["cot"]["vmax"])
    ax.set_ylim(VARIABLES["cth"]["vmin"], VARIABLES["cth"]["vmax"])

    if VARIABLES["cot"]["logscale"]:
        ax.set_xscale("log")
    
    # Apply visual design improvements: clean spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.grid(color='grey', alpha=0.5, linestyle='--')
    
    # Add frame around legend (and make it smaller)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, loc="upper right", fancybox=False, shadow=False)

# ==========================================================================
# SAVE FIGURE
# ==========================================================================
output_path = os.path.join(output_dir, "training_set_characterization.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n✅ Saved comprehensive training set characterization plot to: {output_path}")

