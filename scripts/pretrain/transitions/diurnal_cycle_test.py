import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cmcrameri.cm as cmc


# ==================================================
# IMPORT PROJECT UTILITIES
# ==================================================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from scripts.pretrain.transitions.data_utils import (
    filter_rows_in_event_window,
    build_event_groups,
    split_by_region,
    compute_variable,
    load_data
)

from scripts.pretrain.transitions.plot_utils import (
    plot_label_freq,
    plot_distribution,

)

from utils.plotting.class_colors import CLOUD_CLASS_INFO


# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"

LAT_DIVISION = 47
REGIONS = ["NORTH", "SOUTH"]

EVENTS = [
    {"name": "ALL"},
    {"name": "PRECIP"},
    {"name": "HAIL"},
    {"name": "MIXED"},
]

EVENT_ORDER = [e["name"] for e in EVENTS]
#EVENT_LABELS = [e["label"] for e in EVENTS]

# What to plot
PLOT_VARIABLE = "hourly_intensities"  # options: "label_freq", "hourly_occurrences", "hourly_intensities"
STAT_VARIABLE = "precipitation" 
# options: "label_freq", "hourly_occurrences", "hourly_intensity", 
# TODO: "bt", "precip", "lightning", cmsaf
PERCENTILE = '99'
VMAX = 1000
COLORBAR_LABEL = "Rain Rate (mm/h)"

#path to PRECIP and HAIL reports csv
REPORTS_CSV_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test/essl_reports"
precip_reports_csv = os.path.join(REPORTS_CSV_PATH, "PRECIP_grouped.csv")
hail_reports_csv = os.path.join(REPORTS_CSV_PATH, "HAIL_grouped.csv")

stat_csv_path = f"{BASE_DIR}/crops_stats_vars-cth-cma-precipitation-euclid_msg_grid_stats-50-99-25-75_frames-1_coords-datetime_{RUN_NAME}_all_0.csv"

cloud_items_ordered = sorted(
    CLOUD_CLASS_INFO.items(),
    key=lambda x: x[1]["order"]
)

labels_ordered = [lbl for lbl, _ in cloud_items_ordered]
short_labels = [info["short"] for _, info in cloud_items_ordered]
colors_ordered = [info["color"] for _, info in cloud_items_ordered]


def plot_multiplot(df, df_reports, variable, labels=None, variable_stat=None):
    nrows = len(REGIONS)
    ncols = len(EVENT_ORDER)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 2 * nrows),
        sharex=False,
        sharey=False
    )

    region_dfs = split_by_region(df, lat_column="lat", LAT_DIVISION=LAT_DIVISION)
    region_dfs_reports = split_by_region(df_reports, lat_column="LATITUDE", LAT_DIVISION=LAT_DIVISION)
    mappable = None

    for r, region in enumerate(REGIONS):
        df_region = region_dfs[region]
        df_region_reports = region_dfs_reports[region]
        # groups = build_event_groups(df_region,
        #                             percentile=PERCENTILE,
        #                             intensity_col="max_intensity",
        #                             vector_col="vector_type")
        # groups_reports = build_event_groups(
        #     df_region_reports,
        #     percentile=PERCENTILE,
        #     intensity_col="intensity",
        #     vector_col="TYPE_EVENT"
        # )   

        for c, event in enumerate(EVENT_ORDER):
            ax = axes[r, c]
            if event == "ALL":
                df_event = df_region
                df_event_reports = df_region_reports
            else:
                df_event = df_region[df_region["storm_type"] == event]
                df_event_reports = df_region_reports[df_region_reports["TYPE_EVENT"] == event]

            if variable == "label_freq":
                data = compute_variable(df_event, variable=variable, labels=labels, variable_stat=variable_stat)
                data = data['mean']
                mappable = plot_label_freq(ax, data, vmax=VMAX, cmap=cmc.batlow)
            else:
                data = compute_variable(df_event, variable=variable, labels=labels, variable_stat=variable_stat)
                print(data)
                if variable == 'hourly_occurrences':
                    ax.bar(
                        data.index,
                        data.values,
                        width=0.8,
                        color='tab:blue',
                        alpha=0.7
                    )
                else:
                    #if event != 'ALL': 
                    ax.boxplot(
                    data,
                    positions=range(24),
                    widths=0.6,
                    showfliers=False
                )
                    #set y ticklabel to fontsize 10
                    ax.tick_params(axis='y', labelsize=8)


            if c == 0 and variable == "label_freq":
                #ax.set_ylabel('Class label', fontsize=14)
                #put tickes in the middle of each label
                ax.set_yticks([i + 0.5 for i in range(len(short_labels))])
                ax.set_yticklabels(short_labels, fontsize=14, rotation=0)
            if c==0 and variable == "hourly_occurrences":
                ax.set_ylabel('Counts', fontsize=14)
                #ax.set_yticklabels( fontsize=12)
            if c==0 and variable == "hourly_intensities":
                ax.set_ylabel(COLORBAR_LABEL, fontsize=12)
                #ax.set_yticklabels(ax.get_yticks(), fontsize=10)
            if c == 0:
                ax.text(
                    -0.4, 0.5, region,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=14,
                    fontweight="bold"
                )

            if r == 0:
                ax.set_title(EVENT_ORDER[c], fontsize=14, fontweight="bold")
                ax.set_xticks(range(0, 25, 3))
                #empty x tick labels
                ax.set_xticklabels([])

            if r == nrows - 1:
                ax.set_xlabel('Hour (UTC)', fontsize=12)
                ax.set_xticks(range(0, 25, 3))
                ax.set_xticklabels(range(0, 25, 3), fontsize=12, rotation=45, ha='right')
    
    # ---- ADD SHARED COLORBAR ----
    if variable == "label_freq" and mappable is not None:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(COLORBAR_LABEL, fontsize=14, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

    
    #plt.tight_layout(rect=[0, 0, 0.9, 1])
    out = f"{BASE_DIR}/multiplot_{variable}_{STAT_VARIABLE}-{PERCENTILE}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {out}")


if __name__ == "__main__":
    #concat precip and hail reports into a single dataframe
    df_reports_precip = pd.read_csv(precip_reports_csv, low_memory=False)
    df_reports_hail = pd.read_csv(hail_reports_csv, low_memory=False)
    #rename column 'precipitation_amount'and max_hail_diameter to 'intensity'
    df_reports_precip = df_reports_precip.rename(columns={"PRECIPITATION_AMOUNT": "intensity"})
    df_reports_hail = df_reports_hail.rename(columns={"MAX_HAIL_DIAMETER": "intensity"})

    df_reports = pd.concat([df_reports_precip, df_reports_hail], ignore_index=True)
    print(df_reports.columns.tolist())

    #df = load_data(BASE_DIR, RUN_NAME)
    feature_space_path = os.path.join(BASE_DIR, f"features_train_test_{RUN_NAME}_2nd_labels.csv") 

    #open df featture csv
    df_features = pd.read_csv(feature_space_path, low_memory=False)
    print(df_features)

    df_test = df_features[df_features["vector_type"] != "TRAIN"]
    #from path column extract basename and store in filename column
    df_test["crop"] = df_test["path"].apply(lambda x: os.path.basename(x))

    if STAT_VARIABLE is not None:
        df_stat = pd.read_csv(stat_csv_path, low_memory=False)
        #change crop column to filename to match df_test
        #select var (keep rows if var == VAR)
        df_stat_var = df_stat[df_stat["var"] == STAT_VARIABLE]
        print(df_stat_var)
        #merge df_stat columns (50, 99, 25, 75, None) to df_test based on filename
        df_test = pd.merge(df_test, df_stat_var[["crop", "var", "50", "99", "25", "75", "None"]], on="crop", how="left")

    print(df_test)

    #df = filter_rows_in_event_window(df)

    #labels = df["label"].unique()
    #labels.sort()

    plot_multiplot(df_test, df_reports, variable=PLOT_VARIABLE, labels=labels_ordered, variable_stat=PERCENTILE)
