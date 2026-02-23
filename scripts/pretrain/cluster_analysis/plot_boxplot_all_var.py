import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import sys

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/utils/plotting')
from class_colors import CLOUD_CLASS_INFO, COLORS_PER_CLASS

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
path_to_dir = f"/data1/fig/{RUN_NAME}/epoch_800/all/"
merged_path = os.path.join(path_to_dir, "merged_crops_stats_all_cvc_cot_cth_fractions.csv")


# crop_list_path = os.path.join(path_to_dir, "crop_list_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv")
# lc_path = os.path.join(path_to_dir, "crops_stats_vars-cth-cma-precipitation-euclid_msg_grid_stats-50-99-25-75_frames-1_coords-datetime_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv")
# df_label = pd.read_csv(crop_list_path)
# #extract crop from path
# df_label['crop'] = df_label['path'].apply(lambda x: os.path.basename(x))
# df_label = df_label[['crop', 'label']]
# df_lc = pd.read_csv(lc_path)
# print(df_lc)
# #merge df_lc with df_label on crop column
# df_lc = pd.merge(df_lc, df_label, on='crop', how='inner')

# #check values mean  of var 'euclid_msg_grid' for each label
# for label in df_lc['label'].unique():
#     subset = df_lc[df_lc['label'] == label]
#     mean_lc = subset[subset['var']=='euclid_msg_grid']['None'].median()
#     print(f"Label {label}: Mean Lightning Count = {mean_lc}")


PLOT_TYPE = "boxplot"  # boxplot or violin
MIN_SAMPLES_PER_CLASS = 50

output_dir = os.path.join(path_to_dir, f"{PLOT_TYPE}s/new/")
os.makedirs(output_dir, exist_ok=True)

cloud_items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])

# Variables to plot
VARIABLES = [
    {"var": "cma", "short_name": "d) CC", "label": "Cloud Cover (%)", "percentile": None, "scale": 100, "log": False, "vmin": 0, "vmax": 100},
    #{"var": "cth", "short_name": "e) CTH50", "label": "Cloud Top \n Height (km)", "percentile": 50, "scale": 0.001, "log": False, "vmin": 0, "vmax": 12},
    {"var": "cth_very_high", "short_name": "e) CTH(10+)", "label": "Percentage \n Area (%)", "percentile": None, "scale": 100, "log": False, "vmin": 0, "vmax": 100},
    #{"var": "cth", "short_name": "CTH99", "label": "Cloud Top \n Height (km)", "percentile": 99, "scale": 0.001, "log": False, "vmin": 0, "vmax": 15},
    #{"var": "ccv", "short_name": "CVC", "label": "Convective \n Cover (%)", "percentile": None, "scale": 100, "log": True, "vmin": 1, "vmax": 100},
    #{"var": "cot", "short_name": "COT50", "label": "Cloud Optical \n Thickness", "percentile": 50, "scale": 1, "log": False, "vmin": 0, "vmax": 35},
    #{"var": "cot", "short_name": "COT99", "label": "Cloud Optical \n Thickness", "percentile": 99, "scale": 1, "log": False, "vmin": 0, "vmax": 155},
    #{"var": "cot_thin", "short_name": "COT(0-5)", "label": "Percentage \n Area (%)", "percentile": None, "scale": 100, "log": False, "vmin": 0, "vmax": 100},
    #{"var": "cot_medium", "short_name": "COT(5-30)", "label": "Percentage \n Area (%)", "percentile": None, "scale": 100, "log": False, "vmin": 0, "vmax": 100},
    {"var": "cot_thick", "short_name": "f) COT(30+)", "label": "Percentage \n Area (%)", "percentile": None, "scale": 100, "log": False, "vmin": 0, "vmax": 70},
    #{"var": "precip_area", "short_name": "PA", "label": "Precipitating \n Area (%)", "percentile": None, "scale": 100, "log": True, "vmin": 1, "vmax": 100},
    #{"var": "precipitation", "short_name": "RR99", "label": "Rain Rate \n (mm/h)", "percentile": 99, "scale": 1, "log": True, "vmin": 0.1, "vmax": 40},
    #{"var": "euclid_msg_grid", "short_name": "LC", "label": "Lightning \n Count", "percentile": None, "scale": 1, "log": True, "vmin": 1, "vmax": 200},
]

#define function to print in a text file box values stat (median, q1, q3, whiskers) and number of samples
def save_boxplot_stats(box_data, labels, info, output_dir):
    stats_file = os.path.join(output_dir, f"boxplot_stats_{info['var']}_{info['percentile']}.txt")
    with open(stats_file, "w") as f:
        f.write(f"Boxplot statistics for {info['short_name']} ({info['var']}, percentile={info['percentile']})\n")
        f.write("Label\tQ1\tMedian\tQ3\tWhisker Low\tWhisker High\tN Samples\n")
        for i in range(len(box_data)):
            q1 = np.percentile(box_data[i], 25)
            q3 = np.percentile(box_data[i], 75)
            median = np.median(box_data[i])
            whisker_low = q1 - 1.5 * (q3 - q1)
            whisker_high = q3 + 1.5 * (q3 - q1)
            n_samples = len(box_data[i])
            f.write(f"{labels[i]}\t{q1:.2f}\t{median:.2f}\t{q3:.2f}\t{whisker_low:.2f}\t{whisker_high:.2f}\t{n_samples}\n")


def perform_statistical_tests(box_data, labels, info, output_dir):

    # ---- Global test ----
    H, p_global = kruskal(*box_data)

    print(f"\n{info['short_name']} – Kruskal–Wallis H={H:.2f}, p={p_global:.2e}")

    # ---- Pairwise tests ----
    pairs = []
    pvals = []

    for i in range(len(box_data)):
        for j in range(i + 1, len(box_data)):
            U, p = mannwhitneyu(box_data[i], box_data[j], alternative="two-sided")
            pairs.append((labels[i], labels[j]))
            pvals.append(p)

    # ---- Multiple testing correction ----
    reject, pvals_corr, _, _ = multipletests(pvals, method="holm")

    for (c1, c2), p_corr, r in zip(pairs, pvals_corr, reject):
        if r:
            print(f"  {c1} vs {c2}: p_corr={p_corr:.2e}")

    #save results to a text file
    results_file = os.path.join(output_dir, f"stat_tests_{info['var']}_{info['percentile']}.txt")
    with open(results_file, "w") as f:
        f.write(f"{info['short_name']} – Kruskal–Wallis H={H:.2f}, p={p_global:.2e}\n")
        f.write("Significant pairwise differences after Holm correction:\n")
        for (c1, c2), p_corr, r in zip(pairs, pvals_corr, reject):
            if r:
                f.write(f"  {c1} vs {c2}: p_corr={p_corr:.2e}\n")




# -----------------------------
# === Helper functions ===
# -----------------------------

# -----------------------------
# === LOAD DATA ===
# -----------------------------
df = pd.read_csv(merged_path)
print(df['var'].unique())

#print all unique values of var column
print(df['var'].unique())
#print the values for var == 'cth_very_high'

print(f"Loaded: {merged_path} ({df.shape})")

labels = sorted(df["label"].unique())

# ------------------------------------------------------
# === MULTI-VARIABLE DIURNAL CYCLE (SUBPLOTS) ===
# ------------------------------------------------------

n_vars = len(VARIABLES)
print(f"Creating multi-variable boxplot with {n_vars} variables...")

n_rows = 1
n_cols = 3

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(8, 1.5),
    sharex=True
)

if n_rows == 1:
    axes = np.array([axes])  # make it 2D for consistency

for idx, info in enumerate(VARIABLES):

    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    var = info["var"]
    percentile = info["percentile"]

    print(f"Processing {var} – {info}")

    df_var = df[df["var"] == var]
    df_cma = df[df["var"] == "cma"]


    box_data = []
    box_positions = []
    box_colors = []

    for j, (label, color_info) in enumerate(cloud_items, start=1):

        subset = df_var[df_var["label"] == label]
        cma_subset = df_cma[df_cma["label"] == label]
        print(cma_subset["None"])#.describe())

        if var != "cma":
            # align by crop id (not by index)
            subset = subset.merge(
                cma_subset[["crop", "None"]].rename(columns={"None": "cma_val"}),
                on="crop",
                how="inner"
            )
            subset["cma_val"] = pd.to_numeric(subset["cma_val"], errors="coerce")
            #subset = subset[subset["cma_val"] >= 0.05]

        colname = str(percentile) if percentile is not None else "None"
        data = pd.to_numeric(subset[colname], errors="coerce").dropna()
        #print(data)

        if len(data) <= MIN_SAMPLES_PER_CLASS:
            #make an empty series
            data = pd.Series(dtype=float)

        
        data = data * info["scale"]
        print(f"Label {label}: using column '{colname}' for boxplot with {len(data)} samples.")
        print(data.median(), data.quantile(0.25))

        box_data.append(data.values)
        box_positions.append(j)
        box_colors.append(color_info["color"])

    # Skip plotting if not enough data
    # if len(box_data) < MIN_SAMPLES_PER_CLASS:
    #     #print(f"Not enough data for variable {var}, skipping plot.")
    #     ax.set_axis_off()
    #     continue
    #save_boxplot_stats(box_data, labels, info, output_dir)
    #perform_statistical_tests(box_data, labels, info, output_dir)
    
    if PLOT_TYPE == "boxplot":
        #print(box_data)
        bp = ax.boxplot(
            box_data,
            #showmeans=False,
            #showmedians=True,
            positions=box_positions,
            widths=0.4,
            patch_artist=True,
            showfliers=False
        )
        # #print the boxplot values
        # for i in range(len(box_data)):
        #     q1 = np.percentile(box_data[i], 25)
        #     q3 = np.percentile(box_data[i], 75)
        #     median = np.median(box_data[i])
        #     #extreme values, q1 - 1.5*IQR and q3 + 1.5*IQR
        #     print(f"Box {i+1} (Label {labels[i]}): Q1={q1:.2f}, Median={median:.2f}, Q3={q3:.2f}")
        #     print(f"  Whiskers: {q1 - 1.5 * (q3 - q1):.2f} to {q3 + 1.5 * (q3 - q1):.2f}")
        #     #print the number of samples
        #     print(f"Number of samples: {len(box_data[i])}")
            
    elif PLOT_TYPE == "violin":
        print(box_data)
        #use violin plot instead of boxplot (black lines)
        bp = ax.violinplot(
            box_data,
            positions=box_positions,
            widths=0.6,
            showmeans=True,
            showmedians=True,
            showextrema=False
        )
    else:
        raise ValueError(f"Unknown PLOT_TYPE: {PLOT_TYPE}")
    

    # ---- Apply colors ----
    if PLOT_TYPE == "violin":
        for patch, color in zip(bp["bodies"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        #make median lines black and thicker
        bp["cmedians"].set_color("black")
        bp["cmedians"].set_linewidth(1.5)
       
        
    else:
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)


    

    # ---- Titles from the short names
    ax.set_title(info["short_name"], fontsize=13, fontweight="bold")
 

    # ---- Y label on all left panels ----
    ax.set_ylabel(info["label"], fontsize=12)


    # ---- Y ticks ----
    yticks = np.linspace(info["vmin"], info["vmax"], 5)
    ax.set_yticks(yticks.astype(int))
    ax.tick_params(axis="y", labelsize=12)

    if info["log"]:
        ax.set_yscale("log")
        #set y tickes manually
        #y_tickes = [10**i for i in range(int(np.log10(info["vmin"])), int(np.log10(info["vmax"])) + 1)]
        #ax.set_yticks(y_tickes)

    ax.set_ylim(info["vmin"], info["vmax"])
    ax.grid(axis="y", alpha=0.3)

# ---- X axis: bottom row only ----
for ax in axes[-1, :]:
    #ax.set_xlabel("Class Label", fontsize=12)
    ax.set_xticks(range(1, len(labels) + 1))
    #get label name from items (short)
    class_names = [color_info["short"] for label, color_info in cloud_items]
    ax.set_xticklabels(class_names, fontsize=12, rotation=45)#, ha="right")

# ---- Remove x tick labels elsewhere ----
for ax in axes[:-1, :].flat:
    ax.tick_params(labelbottom=False)

plt.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.08,
    top=0.92,
    wspace=0.6,
    hspace=0.25
)

var_names = "_".join([v["var"] for v in VARIABLES])
outfile = os.path.join(output_dir, f"{PLOT_TYPE}_all_var_{var_names}.png")

plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Saved multi-variable {PLOT_TYPE} → {outfile}")


