import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
input_dir = "/data1/fig/dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective/epoch_800/all/"
filename = "crop_list_dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective_all_81270.csv"

output_dir = input_dir + "/diurnal_cycle_plots/"
os.makedirs(output_dir, exist_ok=True)

YLIM = 32  # Y-axis limit

# === CLASS COLORS ===
COLORS_PER_CLASS = {
    '0': 'darkgray',
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green',
    '9': 'goldenrod',
    '10': 'magenta',
    '11': 'dodgerblue',
    '12': 'darkorange',
    '13': 'olive',
    '14': 'crimson',
}

# === LOAD DATA ===
df = pd.read_csv(input_dir + filename)

# === Extract hour from file path ===
def extract_hour_from_path(path):
    """Extract hour (0–23) from filename like 2021-07-18T12:15:00_xx.nc"""
    match = re.search(r"T(\d{2}):(\d{2}):\d{2}", os.path.basename(path))
    return int(match.group(1)) if match else None

df["hour"] = df["path"].apply(extract_hour_from_path)
df = df.dropna(subset=["hour"])
df["hour"] = df["hour"].astype(int)

labels = sorted(df["label"].unique())

# ===========================================================
# === CORRECT NORMALIZATION: % OF EACH LABEL WITHIN EACH HOUR
# ===========================================================

# Count samples per hour per label
counts = df.groupby(["hour", "label"]).size().unstack(fill_value=0)

# Ensure all 24 hours exist
all_hours = np.arange(24)
counts = counts.reindex(all_hours, fill_value=0)

# Normalize row-wise → percentage of each label in each hour
totals_per_hour = counts.sum(axis=1).replace(0, np.nan)
rel_pct = counts.div(totals_per_hour, axis=0) * 100
rel_pct = rel_pct.fillna(0)

# =============================
# === PLOT ALL LABELS TOGETHER
# =============================
plt.figure(figsize=(6, 3))

for label in rel_pct.columns:
    color = COLORS_PER_CLASS.get(str(label), "black")
    plt.plot(rel_pct.index, rel_pct[label], linewidth=2.5, color=color)

plt.title("Diurnal Cycle", fontsize=14, fontweight="bold")
plt.xlabel("Hour (UTC)", fontsize=14)
plt.ylabel("%", fontsize=14)
plt.xticks(range(0, 24, 2), fontsize=14)
plt.ylim(0, YLIM)
plt.grid(alpha=0.3)

output_file = os.path.join(output_dir, "diurnal_cycle_all_label.png")
plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print(f"✅ Saved: {output_file}")

# ======================================
# === PLOT INDIVIDUAL LABEL FIGURES
# ======================================
for label in rel_pct.columns:

    plt.figure(figsize=(6, 3))
    color = COLORS_PER_CLASS.get(str(label), "black")

    plt.plot(rel_pct.index, rel_pct[label], linewidth=2.5, color=color)

    plt.title(f"Diurnal Cycle – Label {label}", fontsize=14, fontweight="bold")
    plt.xlabel("Hour (UTC)", fontsize=14)
    plt.ylabel("%", fontsize=14)
    plt.xticks(range(0, 24, 2), fontsize=14)
    plt.ylim(0, YLIM)
    plt.grid(alpha=0.3)

    output_file = os.path.join(output_dir, f"diurnal_cycle_label_{label}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

    print(f"✅ Saved: {output_file}")
