import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/utils/plotting')
from class_colors import CLOUD_CLASS_INFO, COLORS_PER_CLASS

# =========================
# CONFIG
# =========================
input_dir = "/data1/fig/dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc/epoch_800/all/"
filename = "crop_list_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv"

output_dir = os.path.join(input_dir, "diurnal_cycle_plots_new_colors")
os.makedirs(output_dir, exist_ok=True)

YLIM = 45  # Y-axis limit
FONTSIZE = 16

print(CLOUD_CLASS_INFO)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(os.path.join(input_dir, filename))

# =========================
# EXTRACT TIME INFORMATION
# =========================
def extract_time_from_path(path):
    """
    Extract datetime from filename like:
    2021-07-18T12:15:00_xx.nc
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", os.path.basename(path))
    return pd.to_datetime(match.group(1)) if match else pd.NaT

df["time"] = df["path"].apply(extract_time_from_path)
df = df.dropna(subset=["time"])

df["hour"] = df["time"].dt.hour
df["date"] = df["time"].dt.date

# =========================
# SORT CHRONOLOGICALLY
# =========================
df = df.sort_values(["date", "time"]).reset_index(drop=True)

# =========================
# GROUP CONSECUTIVE LABELS
# =========================
# A new occurrence starts when:
# - label changes
# - OR hour changes
# - OR date changes
# df["label_change"] = (
#     (df["label"] != df["label"].shift()) |
#     (df["hour"] != df["hour"].shift()) |
#     (df["date"] != df["date"].shift())
# )

#df["occurrence_id"] = df["label_change"].cumsum()

# =========================
# BUILD OCCURRENCE TABLE
# =========================
# occ = (
#     df.groupby("occurrence_id")
#       .agg(
#           label=("label", "first"),
#           hour=("hour", "first"),
#       )
#       .reset_index(drop=True)
# )

# =========================
# COUNT OCCURRENCES
# =========================
occ_counts = (
    df.groupby(["hour", "label"])
       .size()
       .unstack(fill_value=0)
)

# Ensure all 24 hours exist
occ_counts = occ_counts.reindex(np.arange(24), fill_value=0)

# =========================
# NORMALIZE BY TOTAL OCCURRENCES PER HOUR
# =========================
# total_occ_per_hour = occ_counts.sum(axis=1).replace(0, np.nan)
# rel_pct = occ_counts.div(total_occ_per_hour, axis=0) * 100
# rel_pct = rel_pct.fillna(0)

# =========================
# PLOT ALL LABELS TOGETHER
# =========================
# plt.figure(figsize=(6, 3))




# for label in occ_counts.columns:
#     color = CLOUD_CLASS_INFO.get(label, "black")["color"]
#     plt.plot(
#         occ_counts.index,
#         occ_counts[label],
#         linewidth=2.5,
#         color=color,
#         label=str(label),
#     )

# plt.title("Diurnal Cycle", fontsize=14, fontweight="bold")
# plt.xlabel("Hour (UTC)", fontsize=14)
# plt.ylabel("label occurrences", fontsize=14)
# plt.xticks(range(0, 24, 2), fontsize=14)
# plt.ylim(0, YLIM)
# plt.grid(alpha=0.3)

# output_file = os.path.join(output_dir, "diurnal_cycle_all_labels_occurrence.png")
# plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
# plt.close()

# print(f"✅ Saved: {output_file}")

# # =========================
# # PLOT INDIVIDUAL LABELS
# # =========================
# for label in occ_counts.columns:
#     plt.figure(figsize=(6, 3))

#     color = CLOUD_CLASS_INFO.get(label, "black")['color']
#     plt.plot(
#         occ_counts.index,
#         occ_counts[label],
#         linewidth=2.5,
#         color=color,
#     )

#     plt.title(f"Diurnal Cycle – Label {label}",
#               fontsize=14, fontweight="bold")
#     plt.xlabel("Hour (UTC)", fontsize=FONTSIZE)
#     plt.ylabel("label occurrences", fontsize=FONTSIZE)
#     plt.xticks(range(0, 24, 2), fontsize=FONTSIZE)
#     plt.ylim(0, YLIM)
#     plt.grid(alpha=0.3)

#     output_file = os.path.join(
#         output_dir,
#         f"diurnal_cycle_label_{label}_occurrence.png"
#     )
#     plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
#     plt.close()

#     print(f"✅ Saved: {output_file}")

# print("🎉 Done — occurrence-based diurnal cycle computed correctly.")


# =========================
# STACKED BAR + PIE SUMMARY
# =========================
fig, (ax_bar, ax_pie) = plt.subplots(
    1, 2,
    figsize=(8, 3),
    gridspec_kw={"width_ratios": [3, 1]}
)

# ---------- STACKED BAR ----------
bottom = np.zeros(len(occ_counts.index))

items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
print(items)

for label, info in items:
    print(f"Plotting label {label} with color {info['color']}")
    print(info)
   
    color = info["color"]
    values = occ_counts[label].values

    ax_bar.bar(
        occ_counts.index,
        values,
        bottom=bottom,
        color=color,
        edgecolor="black",
        linewidth=0.3,
    )
    bottom += values

ax_bar.set_title("Hourly Occurrence Count", fontsize=FONTSIZE, fontweight="bold")
ax_bar.set_xlabel("Hour (UTC)", fontsize=FONTSIZE)
ax_bar.set_ylabel("# Occurrences", fontsize=FONTSIZE)
#rotate x ticks 45 degrees for better visibility
ax_bar.set_xticks(range(0, 24, 1))
ax_bar.tick_params(axis='x', rotation=45, labelsize=10)
ax_bar.tick_params(axis='y', labelsize=12)
ax_bar.grid(axis="y", alpha=0.3)

# ---------- PIE CHART ----------
total_occ = occ_counts.sum(axis=0)
labels = []
sizes = []
colors = []

total = total_occ.sum()

for label, info in items:
    value = total_occ[label]
    if value > 0:
        pct = 100 * value / total
        labels.append(f"Class {label}\n{pct:.1f}%")
        sizes.append(value)
        colors.append(info["color"])

ax_pie.pie(
    sizes,
    labels=labels,
    colors=colors,
    startangle=90,
    radius=0.5,                     # 🔹 smaller pie
    labeldistance=1.2,             # 🔹 keeps labels close
    textprops={"fontsize": 8, "weight": "bold"},
)

ax_pie.set_title(
    "Overall sample\n distribution",
    fontsize=13,
    fontweight="bold",
    y=0.95
)

ax_pie.axis("equal")  # keep it circular


# ---------- SAVE ----------
output_file = os.path.join(
    output_dir,
    "diurnal_cycle_occurrences_stacked_bar_with_pie.png"
)
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print(f"✅ Saved stacked bar + pie plot: {output_file}")

