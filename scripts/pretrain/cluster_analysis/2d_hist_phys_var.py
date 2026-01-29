#!/usr/bin/env python3
"""
Generate per-label transparent histograms for each variable.
- For 'cth' and 'cot': plot percentiles (25, 50, 75, 99)
- For 'cma', 'precipitation', 'euclid_msg_grid': plot 'None' (categorical)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
RUN_NAME = "dcv2_vit_k10_ir108_100x100_2013-2020_3xrandomcrops_1xtimestamp_cma_nc"
crop_sel = "closest"
epoch = "epoch_800"
n_subsets = 1000 
path_to_dir = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/"
merged_path = os.path.join(path_to_dir, f"merged_crops_stats_alltime_{crop_sel}_{n_subsets}.csv")

# === COLOR MAP ===
COLORS_PER_CLASS = {
    '0': 'darkgray', '1': 'darkslategrey', '2': 'peru', '3': 'orangered',
    '4': 'lightcoral', '5': 'deepskyblue', '6': 'purple', '7': 'lightblue',
    '8': 'green', '9': 'goldenrod', '10': 'magenta', '11': 'dodgerblue',
    '12': 'darkorange', '13': 'olive', '14': 'crimson'
}

# === VARIABLES TO PLOT ===
VARIABLES = {
    "cth": {"label": "Cloud Top Height (km)", "vmin": 7.5, "vmax": 12.3, "logscale": False, "offset": 0, "mult": 0.001},
    #"cth": {"label": "Cloud Top Height (km)", "vmin": 3, "vmax": 10, "logscale": False, "offset": 0, "mult": 0.001},
    "cot": {"label": "Cloud Optical Thickness", "vmin": 1, "vmax": 150, "logscale": True, "offset": 0, "mult": 1},
    "cma": {"label": "Cloud Cover (%)", "vmin": 0, "vmax": 100, "logscale": False, "offset": 0, "mult": 100},
    "ccv": {"label": "Convective Cloud Cover (%)", "vmin": 0.02, "vmax": 70, "logscale": True, "offset": 0, "mult": 1},
    #"precipitation": {"label": "Total Precipitation (mm)", "vmin": 5, "vmax": 5000, "logscale": True, "offset": 0, "mult": 0.5},
    "precipitation": {"label": "Rain Rate (mm/h)", "vmin": 0.5, "vmax": 12, "logscale": True, "offset": 0, "mult": 1},
    "euclid_msg_grid": {"label": "Total Lightning Count", "vmin": 0.1, "vmax": 160, "logscale": True, "offset": 0, "mult": 1},
}

PERCENTILE_VARS = ["cth", "cot", "precipitation"]
CATEGORICAL_VARS = ["cma", "euclid_msg_grid", "ccv"]
PERCENTILE_COLS =  ["99"] #["25", "50", "75", "99"]
CATEGORICAL_COL = "None"



#check if merged_path exists
if not os.path.exists(merged_path):
    #open crop list
    crop_list_path = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/crop_list_{RUN_NAME}_{crop_sel}_{n_subsets}.csv"
    df_crops = pd.read_csv(crop_list_path)
    print(df_crops.head())
    #extrect crop from path
    df_crops['crop'] = df_crops['path'].apply(lambda x: os.path.basename(x))
    #open stats file
    stats_path = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/crops_stats_vars-cth-cma-cot-precipitation-lightning_stats-50-99-25-75_frames-1_coords-datetime_{RUN_NAME}_{crop_sel}_{n_subsets}CA.csv"
    df_stats = pd.read_csv(stats_path)
    print(df_stats.head())
    #merge dataframes on crop
    df = pd.merge(df_crops, df_stats, on='crop', how='inner')
    #save merged dataframe
    print(f"Merged dataframe shape: {df.shape}")
    df.to_csv(merged_path, index=False)
    print(f"Saved merged dataframe to: {merged_path}")
    exit()
    
# === LOAD DATA ===
df = pd.read_csv(merged_path)
print(f"✅ Loaded dataframe: {merged_path} ({df.shape})")



# === OUTPUT DIR ===
plots_dir = os.path.join(path_to_dir, f"per_label_distributions_{n_subsets}_subsets")
os.makedirs(plots_dir, exist_ok=True)

# === MAIN LOOP ===
labels = sorted(df["label"].unique())
print(f"Found {len(labels)} labels: {labels}")

for label in labels:
    label_str = str(int(label))
    color = COLORS_PER_CLASS.get(label_str, "gray")
    label_dir = os.path.join(plots_dir, f"label_{label_str}")
    os.makedirs(label_dir, exist_ok=True)

    df_label = df[df["label"] == label]

    for var, var_label in VARIABLES.items():
        df_var = df_label[df_label["var"] == var]
        if df_var.empty:
            print(f"⚠️ No data for var='{var}' and label={label_str}")
            continue

        # Handle percentiles or categorical column
        if var in ["cth", "cot", "precipitation"]:
            for pctl in PERCENTILE_COLS:
                if pctl not in df_var.columns:
                    continue
                data = pd.to_numeric(df_var[pctl], errors="coerce").dropna()
                if data.empty:
                    continue

                plt.figure(figsize=(6, 4))
                sns.histplot(data, bins=30, kde=True, color=color, alpha=0.7)
                plt.title(f"Label {label_str} – {var_label} (p{pctl})", fontsize=14, fontweight="bold")
                plt.xlabel(var_label, fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.gcf().patch.set_alpha(0.0)
                plt.savefig(
                    os.path.join(label_dir, f"{var}_p{pctl}_label_{label_str}.png"),
                    dpi=200, bbox_inches="tight", transparent=True
                )
                plt.close()

        else:
            if CATEGORICAL_COL not in df_var.columns:
                continue
            data = pd.to_numeric(df_var[CATEGORICAL_COL], errors="coerce").dropna()
            if data.empty:
                continue

            plt.figure(figsize=(6, 4))
            sns.histplot(data, bins=30, kde=True, color=color, alpha=0.7)
            plt.title(f"Label {label_str} – {var_label}", fontsize=14, fontweight="bold")
            plt.xlabel(var_label, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.gcf().patch.set_alpha(0.0)
            plt.savefig(
                os.path.join(label_dir, f"{var}_label_{label_str}.png"),
                dpi=200, bbox_inches="tight", transparent=True
            )
            plt.close()

    print(f"💾 Saved histograms for label {label_str}")

print(f"✅ All per-label histograms saved in: {plots_dir}")
