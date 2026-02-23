import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/utils/plotting')
from class_colors import CLOUD_CLASS_INFO, COLORS_PER_CLASS

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
path_to_dir = f"/data1/fig/{RUN_NAME}/epoch_800/all/"
merged_path = os.path.join(path_to_dir, "merged_crops_stats_all_bt_perc.csv")

output_dir = os.path.join(path_to_dir, "boxplots_new/")
os.makedirs(output_dir, exist_ok=True)


# Variables to plot
VARIABLES = [
    {"var": "bt", "label": "Brightness Temperature (K)", "percentile": 50, "scale": 1, "log": False, "vmin": 230, "vmax": 300},
    {"var": "bt", "label": "Brightness Temperature (K)", "percentile": 99, "scale": 1, "log": False, "vmin": 200, "vmax": 300},
    {"var": "bt", "label": "Brightness Temperature (K)", "percentile": "IQR", "scale": 1, "log": False, "vmin": 0, "vmax": 60},
]


# -----------------------------
# === Helper functions ===
# -----------------------------

# -----------------------------
# === LOAD DATA ===
# -----------------------------
df = pd.read_csv(merged_path)
print(f"Loaded: {merged_path} ({df.shape})")

labels = sorted(df["label"].unique())

items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
print(items)

n_vars = len(VARIABLES) 
print(f"Creating multi-variable boxplot with {n_vars} variables...")
fig, axes = plt.subplots(n_vars, 1, figsize=(2, 1.5*n_vars), sharex=True)


# ---------- Other rows: Variables ----------
for i, info in enumerate(VARIABLES):
    var = info["var"]
    percentile = info["percentile"]

    print(f"Processing boxplot for variable: {var} – {info}")

    ax = axes[i]

    # select variable
    df_var = df[df['var'] == var]

    box_data = []
    box_positions = []
    box_colors = []

    for j, (label, info_color) in enumerate(items, start=1):
        subset = df_var[df_var["label"] == label].copy()

        if info["percentile"] is not None:
            colname = f"{info['percentile']}"
        else:
            colname = "None"

        print(f"Label {label}: using column '{colname}' for boxplot.")
        
        if colname == 'IQR':
            q75 = pd.to_numeric(subset["75"], errors="coerce").dropna()
            q25 = pd.to_numeric(subset["25"], errors="coerce").dropna()
            data = q75 - q25
        else:
            data = pd.to_numeric(subset[colname], errors="coerce").dropna()

        if data.empty:
            continue

        # apply scale
        data = data * info["scale"]

        box_data.append(data.values)
        box_positions.append(j)
        box_colors.append(info_color["color"])

        bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.6,
        patch_artist=True,   # REQUIRED for coloring
        showfliers=False
        )

        # apply colors
        for patch, (label, info_color) in zip(bp["boxes"], items):
            color = info_color["color"]
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # optional: color medians and whiskers
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.)

    #clean_label = info["label"].replace("\n", "")

    # ---- Axis styling ----
    if i == 1:
        ax.set_ylabel(f"{info['label']}", fontsize=11)
    
    if info["percentile"] == "IQR":
        ax.set_title(f"c) {info['percentile']}", fontsize=11, fontweight="bold")
    elif info["percentile"] == 99:
        ax.set_title(f"b) Extreme", fontsize=11, fontweight="bold")
    else:
        ax.set_title(f"a) Median", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=11)
    #set only integer y ticks
    yticks = np.linspace(info["vmin"], info["vmax"], num=5)
    ax.set_yticks(yticks.astype(int))

    if info["log"]:
        ax.set_yscale("log")

    if "vmin" in info and "vmax" in info:
        ax.set_ylim(info["vmin"], info["vmax"])

# ---- X axis shared ----
#axes[-1].set_xlabel("Cloud ", fontsize=11)
axes[-1].set_xticks(range(1, len(labels) + 1))
#take the class name from CLOUD_CLASS_INFO
class_names = [CLOUD_CLASS_INFO[label]["short"] for label, _ in items]
axes[-1].set_xticklabels(class_names, fontsize=10, rotation=45)#, ha="right")

#increase space between subplots
plt.subplots_adjust(hspace=0.4)


plt.savefig(
    os.path.join(output_dir, f"boxplot_BT_perc.png"),
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)
plt.close()

print("Saved multi-variable boxplot.")

