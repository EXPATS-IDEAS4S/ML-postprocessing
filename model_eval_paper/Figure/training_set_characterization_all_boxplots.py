#!/usr/bin/env python3
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
path_to_dir = f"/data1/fig/{RUN_NAME}/epoch_800/all/"
merged_path = os.path.join(path_to_dir, "merged_crops_stats_all_bt_perc.csv")

# Path for hourly occurrence data
all_dir = f"/data1/fig/{RUN_NAME}/epoch_800/all/"
crop_list_file = "crop_list_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv"
merged_all_path = os.path.join(all_dir, "merged_crops_stats_all_cvc_cot_cth_fractions.csv")

output_dir = os.path.join(path_to_dir, "training_set_characterization")
os.makedirs(output_dir, exist_ok=True)

# Use all available samples per class

# Marker styling
SAMPLE_SIZE = 28
SAMPLE_ALPHA = 0.8
CLOSEST_SIZE = 140
CENTROID_SIZE = 180
FONTSIZE = 14
LEGEND_FONTSIZE = 12

# Make utils/plotting visible
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/plotting")
from class_colors import CLOUD_CLASS_INFO  # noqa: E402

# === VARIABLES ===
VARIABLES = {
    "cth": {"label": "CTH Median (km)", "vmin": 0.5, "vmax": 12.5, "logscale": False, "mult": 0.001, "offset": 0, "value_col": "50"},
    "cot": {"label": "COT Median", "vmin": 1, "vmax": 70, "logscale": True, "mult": 1, "offset": 0, "value_col": "50"},
}

VARS = ["cth", "cot"]


def extract_time_from_path(path):
    """Extract datetime from filename like: 2021-07-18T12:15:00_xx.nc"""
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", os.path.basename(path))
    return pd.to_datetime(match.group(1)) if match else pd.NaT


def style_axes(ax, title):
    ax.set_title(title, fontsize=FONTSIZE, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.grid(color="grey", alpha=0.5, linestyle="--")


def draw_class_boxplot(ax, frame, value_column, title, ylabel, ylim=None, log_scale=False, yticks=None):
    box_data = []
    box_positions = []
    box_colors = []

    for j, (label, color_info) in enumerate(cloud_items, start=1):
        subset = frame[frame["label"] == label]
        data = pd.to_numeric(subset[value_column], errors="coerce").dropna()

        box_data.append(data.values)
        box_positions.append(j)
        box_colors.append(color_info["color"])

    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.4,
        patch_artist=True,
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for element in ["whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp.get(element, []), color="black", linewidth=1.2)

    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_xticks(box_positions)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=FONTSIZE)

    if log_scale:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    style_axes(ax, title)


# ==========================================================================
# LOAD DATA FOR SCATTER PLOTS (CTH vs COT)
# ==========================================================================
print(f"Loading scatter plot data from: {merged_path}")
df = pd.read_csv(merged_path)
print(f"Loaded: {merged_path} ({df.shape})")

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

wide = (
    subset.pivot_table(
        index=["crop_index", "label", "distance"],
        columns="var",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

wide = wide.dropna(subset=VARS)
wide = wide.sort_values(["label", "distance"], ascending=[True, True])

# ==========================================================================
# LOAD DATA FOR HOURLY OCCURRENCE
# ==========================================================================
print(f"\nLoading hourly occurrence data from: {os.path.join(all_dir, crop_list_file)}")
df_hourly = pd.read_csv(os.path.join(all_dir, crop_list_file))

df_hourly["time"] = df_hourly["path"].apply(extract_time_from_path)
df_hourly = df_hourly.dropna(subset=["time"])
df_hourly["hour"] = df_hourly["time"].dt.hour
df_hourly["date"] = df_hourly["time"].dt.date
df_hourly["crop"] = df_hourly["path"].apply(os.path.basename)

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
    inferred_night = (df_hourly["hour"] < 6) | (df_hourly["hour"] >= 18)
    df_hourly["nighttime"] = np.where(inferred_night, "True", "False")

df_hourly = df_hourly[df_hourly["label"] != -100]

occ_counts = (
    df_hourly.groupby(["hour", "label"])
       .size()
       .unstack(fill_value=0)
)

occ_counts = occ_counts.reindex(np.arange(24), fill_value=0)

# ==========================================================================
# LOAD DATA FOR CMA BOXPLOT
# ==========================================================================
print(f"\nLoading CMA data from: {merged_all_path}")
df_cma = pd.read_csv(merged_all_path)
df_cma = df_cma[df_cma["var"] == "cma"].copy()
df_cma["cc_pct"] = pd.to_numeric(df_cma["None"], errors="coerce") * 100

cloud_items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
labels_ordered = [lbl for lbl, _ in cloud_items]
short_labels = [info["short"] for _, info in cloud_items]
colors = {lbl: info["color"] for lbl, info in cloud_items}

cc_stats = (
    df_cma.groupby("label")["cc_pct"]
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

stats_txt_path = os.path.join(output_dir, "training_set_characterization_class_stats_all_boxplots.txt")
with open(stats_txt_path, "w", encoding="utf-8") as f:
    f.write("Class-wise mean/std summary (CC %, CTH km, COT)\n")
    f.write("Computed from: CC=cma over all crops, CTH/COT=all available samples\n\n")
    f.write(stats_display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    f.write("\n")

print(f"Saved class-wise stats to: {stats_txt_path}")

# ==========================================================================
# SETUP FIGURE WITH GRIDSPEC (2 rows x 3 cols)
# ==========================================================================
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

ax_counts = fig.add_subplot(gs[0, 0])
ax_hourly = fig.add_subplot(gs[0, 1:])
ax_boxplot_cc = fig.add_subplot(gs[1, 0])
ax_boxplot_cth = fig.add_subplot(gs[1, 1])
ax_boxplot_cot = fig.add_subplot(gs[1, 2])

print("\nCreating sample counts plot...")

df_hourly_counts = df_hourly.copy()
night_flag = df_hourly_counts["nighttime"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
daytime_counts = df_hourly_counts[~night_flag]["label"].value_counts().reindex(labels_ordered).fillna(0)
nighttime_counts = df_hourly_counts[night_flag]["label"].value_counts().reindex(labels_ordered).fillna(0)

x = np.arange(len(labels_ordered))
width = 0.7

ax_counts.bar(
    x,
    daytime_counts.values * 1e-3,
    width=width,
    color=[colors[lbl] for lbl in labels_ordered],
    label="Daytime",
)

ax_counts.bar(
    x,
    nighttime_counts.values * 1e-3,
    width=width,
    bottom=daytime_counts.values * 1e-3,
    color=[colors[lbl] for lbl in labels_ordered],
    hatch="///",
    edgecolor="black",
    linewidth=0.5,
    label="Nighttime",
)

ax_counts.set_ylabel("# Samples (10$^3$)", fontsize=FONTSIZE)
ax_counts.set_xticks(x)
ax_counts.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=FONTSIZE)
ax_counts.tick_params(axis="y", labelsize=FONTSIZE)
ax_counts.set_yticks([0, 10, 20, 30])
ax_counts.set_title("a) Samples per Class", fontsize=FONTSIZE, fontweight="bold")
ax_counts.tick_params(axis="both", which="major", labelsize=FONTSIZE)
ax_counts.spines["top"].set_visible(False)
ax_counts.spines["right"].set_visible(False)
ax_counts.spines["bottom"].set_linewidth(1.5)
ax_counts.spines["left"].set_linewidth(1.5)
ax_counts.grid(color="grey", alpha=0.5, linestyle="--")
ax_counts.legend(fontsize=LEGEND_FONTSIZE, loc="upper center", framealpha=0.8)

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
ax_hourly.set_xticklabels([f"{i}" for i in range(0, 24, 1)], rotation=45, ha="right", fontsize=FONTSIZE)
ax_hourly.tick_params(axis="both", which="major", labelsize=FONTSIZE)
ax_hourly.spines["top"].set_visible(False)
ax_hourly.spines["right"].set_visible(False)
ax_hourly.spines["bottom"].set_linewidth(1.5)
ax_hourly.spines["left"].set_linewidth(1.5)
ax_hourly.grid(color="grey", alpha=0.5, linestyle="--")

print("Creating CC boxplot...")
draw_class_boxplot(
    ax_boxplot_cc,
    df_cma,
    "cc_pct",
    "c) CC Distribution",
    "CC (%)",
    ylim=(0, 100),
)

print("Creating CTH boxplot...")
draw_class_boxplot(
    ax_boxplot_cth,
    wide,
    "cth",
    "d) CTH Distribution",
    VARIABLES["cth"]["label"],
    ylim=(VARIABLES["cth"]["vmin"], VARIABLES["cth"]["vmax"]),
)

print("Creating COT boxplot...")
draw_class_boxplot(
    ax_boxplot_cot,
    wide,
    "cot",
    "e) COT Distribution",
    VARIABLES["cot"]["label"],
    ylim=(1, 50),
    log_scale=True,
    yticks=[1, 2, 5, 10, 20, 50],
)

output_path = os.path.join(output_dir, "training_set_characterization_all_boxplots.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n✅ Saved comprehensive training set characterization plot to: {output_path}")#!/usr/bin/env python3
