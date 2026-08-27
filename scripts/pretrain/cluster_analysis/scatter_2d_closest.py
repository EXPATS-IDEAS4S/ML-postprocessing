#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc"
path_to_dir = f"/data1/fig/{RUN_NAME}/epoch_800/all/"
merged_path = os.path.join(path_to_dir, "crops_stats_vars-cth-cma-cot-precipitation-euclid_msg_grid_stats-50-99-25-75_frames-1_coords-datetime_dcv2_resnet_k7_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_all_140371.csv")
crop_list_path = os.path.join(path_to_dir, "crop_list_dcv2_resnet_k7_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_all_140371.csv")

output_dir = os.path.join(path_to_dir, "scatter_2d_cth_cot_samples")
os.makedirs(output_dir, exist_ok=True)

# Use the 100 closest samples to each centroid (per class)
N_CLOSEST = 1000

# Marker styling
SAMPLE_SIZE = 28
SAMPLE_ALPHA = 0.5
CLOSEST_SIZE = 140
CENTROID_SIZE = 180

# Make utils/plotting visible
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/utils/plotting")
from class_colors import CLOUD_CLASS_INFO  # noqa: E402

# === VARIABLES ===
VARIABLES = {
    "cth": {"label": "Cloud Top Height (km)", "vmin": 3.5, "vmax": 9.5, "logscale": False, "mult": 0.001, "offset": 0, "value_col": "50"},
    "cot": {"label": "Cloud Optical Thickness", "vmin": 1, "vmax": 150, "logscale": True, "mult": 1, "offset": 0, "value_col": "50"},
    #"cma": {"label": "Cloud Cover", "vmin": 0, "vmax": 1, "logscale": False, "mult": 1, "offset": 0, "value_col": "None"}
}

VARS = ["cth", "cot"]

# === LOAD DATA ===
df = pd.read_csv(merged_path)
crop_list_df = pd.read_csv(crop_list_path)

print(f"Loaded stats: {merged_path} ({df.shape})")
print(f"Loaded crop list: {crop_list_path} ({crop_list_df.shape})")

# Merge label (and distance) from crop list using crop filename extracted from path.
crop_list_df = crop_list_df.copy()
crop_list_df["crop"] = crop_list_df["path"].astype(str).map(os.path.basename)
crop_meta = crop_list_df[["crop", "label", "distance"]].drop_duplicates(subset=["crop"], keep="first")

df = df.merge(crop_meta, on="crop", how="left", suffixes=("", "_from_crop_list"))
missing_label = df["label"].isna().sum()
print(f"Rows with missing merged label: {missing_label}/{len(df)}")

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
        index=["crop", "label", "distance"],
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

# === GROUPS (by CLOUD_CLASS_INFO short names) ===
short_to_label = {info["short"]: lbl for lbl, info in CLOUD_CLASS_INFO.items()}

GROUPS = {
    "b) Low-Level Clouds": ["CS", "SC"],
    "c) Mid-Level Clouds": ["MC1", "MC2"],
    "d) Convective Clouds": ["EC", "DC", "OA"],
}

# === PLOT ===
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)

for ax, (group_name, short_names) in zip(axes, GROUPS.items()):

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

        # # Closest sample (minimum distance)
        # closest_row = cls_df.loc[cls_df["distance"].idxmin()]
        # ax.scatter(
        #     closest_row["cth"],
        #     closest_row["cot"],
        #     color=color,
        #     marker="X",
        #     s=CENTROID_SIZE,
        #     edgecolors="black",
        #     linewidth=0.8,
        #     zorder=4,
        # )

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

    ax.set_title(group_name, fontsize=12, fontweight="bold")
    ax.set_xlabel(VARIABLES["cot"]["label"], fontsize=12)
    #set y title only for the first subplot
    if ax == axes[0]:
        ax.set_ylabel(VARIABLES["cth"]["label"], fontsize=12)
    #set tickslabels size 
    ax.tick_params(axis="both", which="major", labelsize=11)

    ax.set_xlim(VARIABLES["cot"]["vmin"], VARIABLES["cot"]["vmax"])
    ax.set_ylim(VARIABLES["cth"]["vmin"], VARIABLES["cth"]["vmax"])

    if VARIABLES["cot"]["logscale"]:
        ax.set_xscale("log")
    #add frame around legend (and make it smaller)
    ax.legend(frameon=True, fontsize=8, loc="upper right", fancybox=False, shadow=False)

plt.tight_layout()
output_path = os.path.join(output_dir, "scatter_2d_cth_cot_samples_groups.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved 2D scatter plot to: {output_path}")
