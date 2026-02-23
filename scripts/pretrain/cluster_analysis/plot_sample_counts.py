"""
Compute fractional area of Cloud Optical Thickness (COT) categories for each crop
and add them to the existing CSV file.

Also, generate a stacked bar plot showing the composition of COT categories per cloud class.

"""

import os
import io
import sys
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append('/home/Daniele/codes/VISSL_postprocessing/utils/')
from plotting.class_colors import CLOUD_CLASS_INFO

# =====================
# CONFIG
# =====================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
epoch = "epoch_800"
crop_sel = "all"


path_to_dir = f"/data1/fig/{RUN_NAME}/{epoch}/{crop_sel}/"

output_path = os.path.join(
    path_to_dir,
    "merged_crops_stats_all_cvc_cot_fractions.csv"
)

cloud_items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])



def print_day_night_numerosity_per_class(df, items, path_to_dir):

    labels_ordered = [lbl for lbl, _ in items]
    short_labels = [info["short"] for _, info in items]
    colors = {lbl: info["color"] for lbl, info in items}

    class_col = "label"

    # --- Identify daytime / nighttime ---
    
    df_daytime = df[df["nighttime"] == 'False']
    df_nighttime = df[df["nighttime"] == 'True']
    #print(df_nighttime)

    # --- Count samples ---
    daytime_counts = df_daytime[class_col].value_counts().reindex(labels_ordered).fillna(0)
    nighttime_counts = df_nighttime[class_col].value_counts().reindex(labels_ordered).fillna(0)
    
    #nighttime_counts = df_nighttime[class_col].value_counts().reindex(labels_ordered).fillna(0)

    # --- Print numerosity ---
    print("Numerosity per cloud class:")
    for lbl in labels_ordered:
        print(
            f"  {lbl}: "
            f"{int(daytime_counts[lbl])} daytime + "
            f"{int(nighttime_counts[lbl])} nighttime"
        )

    # --- Plot ---
    x = np.arange(len(labels_ordered))
    width = 0.7

    plt.figure(figsize=(4, 2.5))

    # Daytime bars (bottom)
    plt.bar(
        x,
        daytime_counts.values * 1e-3,  # Scale to thousands
        width=width,
        color=[colors[lbl] for lbl in labels_ordered],
        label="Daytime"
    )

    # Nighttime bars (stacked, hatched)
    plt.bar(
        x,
        nighttime_counts.values * 1e-3,  # Scale to thousands
        width=width,
        bottom=daytime_counts.values * 1e-3,  # Scale to thousands
        color=[colors[lbl] for lbl in labels_ordered],
        hatch="///",
        edgecolor="black",
        linewidth=0.5,
        label="Nighttime"
    )

    # --- Formatting ---
    plt.ylabel("# Samples (10^3)", fontsize=12)
    plt.xticks(x, short_labels, rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("h) Samples per Cloud Class", fontsize=12, fontweight="bold")

    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=10, loc="upper center", framealpha=0.8)

    # --- Save ---
    outpath = os.path.join(path_to_dir, "day_night_numerosity_per_class.png")
    plt.savefig(outpath, dpi=300, transparent=True, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved numerosity plot → {outpath}")



def load_csv(csv_path, random_sample=None):
    if random_sample is not None:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV: {df.shape}")
        crop_indices = df["crop_index"].unique()
        print(f"Found {len(crop_indices)} unique crops")
        crop_indices = np.random.choice(crop_indices, size=random_sample, replace=False)
        df = df[df["crop_index"].isin(crop_indices)]
        print(f"Randomly sampled {random_sample} crops: {df.shape}")
        return df, crop_indices
    else:
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"✅ Loaded CSV: {df.shape}")
        crop_indices = df["crop_index"].unique()
        print(f"Found {len(crop_indices)} unique crops")

    return df, crop_indices




def count_daytime_samples_per_hour(
    df,
    time_col="datetime",
    nighttime_col="nighttime"
):
    """
    Count the number of daytime samples (nighttime == False)
    for each hour of the day.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    time_col : str
        Name of the datetime column.
    nighttime_col : str
        Boolean column indicating nighttime scenes.

    Returns
    -------
    pandas.Series
        Number of daytime samples per hour (0–23).
    """

    # Ensure datetime
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Select daytime samples
    df_daytime = df[df[nighttime_col] == 'False']

    # Count per hour
    hourly_counts = (
        df_daytime[time_col]
        .dt.hour
        .value_counts()
        .sort_index()
        .reindex(range(24), fill_value=0)
    )


    plt.figure(figsize=(5, 2))
    plt.bar(hourly_counts.index, hourly_counts.values)
    plt.xlabel("Hour (UTC)", fontsize=10)
    plt.ylabel("# samples", fontsize=10)
    plt.title("Daytime Samples per Hour", fontsize=10, fontweight="bold")
    plt.xticks(hourly_counts.index, rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    #manuelly set 4 y ticks
    #max_count = hourly_counts.max()
    plt.yticks(np.linspace(0, 6000, num=5))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_dir, "daytime_samples_per_hour.png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

    return hourly_counts



if __name__ == "__main__":
    # Load updated dataframe for plotting
    df_all = pd.read_csv(path_to_dir+'crop_list_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_140207.csv', low_memory=False)
    #remove non valid label -100
    df_all = df_all[df_all['label'] != -100]
    #get crop column from path
    df_all['crop'] = df_all['path'].apply(lambda x: x.split('/')[-1])
   
    df_day, _ = load_csv(output_path, random_sample=None)
    #remove non valid label -100
    df_day = df_day[df_day['label'] != -100]
    #print(df_out['var'].unique())
    #select only rows whee ver == nighttime_scene
    df_day = df_day[df_day['var'] == 'nighttime_scene']
    #rename None column to nighttime
    df_day = df_day.rename(columns={'None': 'nighttime'})
    #merge nighttime colum of df_day with df_all on crop column
    df_all = df_all.merge(df_day[['crop', 'nighttime']], on='crop', how='left')
    #fill nan valu of nighttime with False (nighttime samples)   
    df_all['nighttime'] = df_all['nighttime'].fillna('False')
    print(df_all.columns.tolist())
    #get daytime from crop columns
    df_all['datetime'] = df_all['crop'].apply(lambda x: x.split('_')[0])
   
    df_all['datetime'] = pd.to_datetime(df_all['datetime'], format='%Y-%m-%dT%H:%M:%S')
   
 
    counts = count_daytime_samples_per_hour(df_all)
    print_day_night_numerosity_per_class(df_all, cloud_items, path_to_dir)
        
        

