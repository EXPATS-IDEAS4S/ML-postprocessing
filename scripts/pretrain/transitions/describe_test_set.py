import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import cartopy.crs as ccrs

# ==================================================
# IMPORT PROJECT UTILITIES
# ==================================================
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from cluster_analysis.test_analysis.utils_func import (
    build_storm_event_groups,
    filter_rows_in_event_window,
    stratifiy_by_latitude,
    build_event_groups,
    compute_temporal_popularity,
    plot_temporal_barplot,
    count_storms_and_crops_from_filenames,
)
from transitions.plot_utils import (
    plot_event_density_map,
    plot_orography_map
)

from transitions.data_utils import (
    update_df_with_bt_stats,
    load_data
)

# ==================================================
# CONFIGURATION
# ==================================================
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
os.makedirs(OUTPUT_PATH, exist_ok=True)
CSV_PATH = os.path.join("/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/storm_trajectories_after_merge.csv")
OROG_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"

PLOT_DIR = os.path.join(OUTPUT_PATH, "test_set_description_plots_no_regions")
os.makedirs(PLOT_DIR, exist_ok=True)

LAT_DIVISION = 47
PERCENTILE = 50
BT_VAR = "IR_108"
BT_STATS = ["p50", "p01"]
LAT_COL = "lat"

MAP_EXTENT = [5, 16, 42, 51.5]  # lon_min, lon_max, lat_min, lat_max

# ==================================================
# EVENT DEFINITIONS
# ==================================================
# EVENTS = [
#     {"name": "ALL",               "label": "All"},
#     {"name": "RAIN_ALL",          "label": "All \n Precip"},
#     {"name": f"RAIN_{PERCENTILE}th", "label": f"Top {100-PERCENTILE}% \n Precip"},
#     {"name": "HAIL_ALL",          "label": "All \n Hail"},
#     {"name": f"HAIL_{PERCENTILE}th", "label": f"Top {100-PERCENTILE}% \n Hail"},
# ]

EVENTS = [
    #{"name": "ALL"}, 
    {"name": "PRECIP"},
    {"name": "HAIL"},
    {"name": "MIXED"},
    #{"name": "UNDEFINED"},
]

EVENT_ORDER = [e["name"] for e in EVENTS]
#EVENT_LABELS = [e.get("label", e["name"]) for e in EVENTS]

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_csv(CSV_PATH, low_memory=False)
print(df.columns.to_list())

#df = df[df["label"] != -100]
#df["max_intensity"] = pd.to_numeric(df["max_intensity"], errors="coerce")



#count per crops and per stroms in each region
stats = count_storms_and_crops_from_filenames(
    base_dir="/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj/nc/1",
    lat_split=47.0)
print("Overall statistics:")
#print(stats["storms_per_type"])
#print(stats["crops_per_storm_type"])
#print(stats["crops_per_crop_type"])
print(stats["overall_storms_per_type"])
print(stats["overall_crops_per_storm_type"])
print(stats["overall_crops_per_crop_type"])



#prepare df for further analysis
df = stats['raw']
print(df)
#save the df in csv if not already saved
output_csv_path_crops_des = os.path.join(PLOT_DIR, "test_set_storms_summary.csv")
if not os.path.exists(output_csv_path_crops_des):
    df.to_csv(output_csv_path_crops_des, index=False)


# # ==================================================
# # ========== 1) BT DISTRIBUTION PLOTS ===============
# # ==================================================
# fig_bt, axes_bt = plt.subplots(
#     2, len(EVENT_ORDER),
#     figsize=(4 * len(EVENT_ORDER), 6),
#     sharey=True
# )

# for row, (region_name, df_region) in enumerate(REGIONS.items()):
#     df_region = filter_rows_in_event_window(df_region)

#     # Compute BT statistics ONCE
#     df_bt = update_df_with_bt_stats(df_region, "path", BT_VAR)

#     # Build event masks
#     event_groups = build_event_groups(df_bt, PERCENTILE)

#     for col, event in enumerate(EVENT_ORDER):
#         ax = axes_bt[row, col]

#         if event not in event_groups:
#             ax.set_visible(False)
#             continue

#         df_event = df_bt.loc[event_groups[event]]

#         # Stack selected percentiles into one column
#         bt_values = pd.concat(
#             [df_event[stat] for stat in BT_STATS],
#             axis=0
#         )

#         plot_bt_boxplot_by_event(
#             df=pd.DataFrame({
#                 "event": event,
#                 "BT": bt_values
#             }),
#             value_col="BT",
#             event_col="event",
#             event_order=[event],
#             ax=ax
#         )

#         if col == 0:
#             ax.text(
#                 -0.25, 0.5, region_name,
#                 transform=ax.transAxes,
#                 rotation=90,
#                 va="center",
#                 ha="right",
#                 fontsize=12,
#                 fontweight="bold"
#             )

# fig_bt.tight_layout()
# fig_bt.savefig(
#     os.path.join(PLOT_DIR, "BT_distribution_by_event_region.png"),
#     dpi=300,
#     bbox_inches="tight"
# )

# ==================================================
# ========== 2) GEOGRAPHICAL DENSITY MAPS ===========
# ==================================================

fig_map, axes_map = plt.subplots(
    1, len(EVENT_ORDER),
    figsize=(1 * len(EVENT_ORDER), 1.5),
    subplot_kw={"projection": ccrs.PlateCarree()}
)
axes_map = axes_map.reshape(1, len(EVENT_ORDER))
#build a ditc with event name and count for each region and event


for col, event in enumerate(EVENT_ORDER):
    print(f"Plotting density map for event: {event}")
    ax = axes_map[0, col]
    #mask = masks[event]
    #print(mask)
    if event == "ALL":
        df_event = df
    else:
        df_event = df.loc[df["storm_type"] == event]

    print(df_event)#.columns.to_list())
    #print(f"Number of storms in event {event}, region {region_name}: {len(df_event)}")
    #print(df_event[["n_precip","n_hail"]].describe())
    
    plot_orography_map(
        OROG_PATH,
        ax=ax,
        var_name="DEM",
        extent=MAP_EXTENT,
        cmap="Greys",
        levels=30,
        alpha=0.3
    )

    plot_event_density_map(
        df_event,
        lat_col="lat",
        lon_col="lon",
        ax=ax,
        extent=MAP_EXTENT,
        cmap=cmc.lajolla,
        levels=10,
        scatter=False
    )

    # North / South divider
    #ax.axhline(LAT_DIVISION, color="yellow", linestyle="--", linewidth=2)

    #if row == 0:
    ax.set_title(EVENT_ORDER[col], fontsize=8, fontweight="bold")

fig_map.savefig(
    os.path.join(PLOT_DIR, "event_density_maps.png"),
    dpi=300,
    bbox_inches="tight"
)

print("Figures saved in:", PLOT_DIR)



# ==================================================
# STRATIFY BY LATITUDE
# ==================================================

#create a datetime (pandas format) column from date (YYYY-MM-DD) and time (HH-MM) columns
df['time'] = df['time'].str.replace("-", ":")
df['datetime'] = pd.to_datetime(df['date'] + "T" + df['time'])

# df_north, df_south = stratifiy_by_latitude(df, LAT_COL, LAT_DIVISION)
# REGIONS = {
#     "North": df_north,
#     "South": df_south
# }

def plot_temporal_multiplot(
    df,
    EVENT_ORDER,
    time_unit="hour"
):
    fig, axes = plt.subplots(
        1, len(EVENT_ORDER),
        figsize=(1.5 * len(EVENT_ORDER), 1.5),
        sharey=True
    )

    #axes = axes.reshape(1, len(EVENT_ORDER))

    #for row, (region_name, df_region) in enumerate(REGIONS.items()):
        #df_region = filter_rows_in_event_window(df_region)
        #groups = build_event_groups(df_region, PERCENTILE)

    for col, event in enumerate(EVENT_ORDER):
        ax = axes[col]
        if event == "ALL":
            df_event = df
        else:
            df_event = df.loc[df["storm_type"] == event]
        freq = compute_temporal_popularity(df_event, time_unit)

        plot_temporal_barplot(ax, freq, time_unit)
        #only 3 y ticks with no labels
        ymin, ymax = ax.get_ylim()
        ticks = np.linspace(ymin, ymax, 3)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=8)

        if col == 0:
            ax.set_ylabel("Rel. Freq.", fontsize=10)
        else:
            ax.set_ylabel("")

        #ax.set_yticklabels(["", "", ""])
        if time_unit == "month":
            ax.set_xlim(3.5, 9.5)  # only April to September
        #remove all y ticks labels except for first column

        #if row == 0:
        ax.set_title(EVENT_ORDER[col], fontsize=10, fontweight="bold")
        ax.set_xticklabels([])
        ax.set_xlabel("")
        #delete x labels and tickes labels
        if time_unit == "month": #only April to September
            ax.set_xticks(range(4,10))
        if time_unit == "year":
            ax.set_xticks(range(2014, 2025, 1))
        if time_unit == "hour":
            ax.set_xticks(range(0,25,3))
        
        #if row == 1:
        #rotate the x tickes labels by 45 degrees
        if time_unit == "month": #only April to September
            ax.set_xticks(range(4,10))
            ax.set_xticklabels([
                "Apr", "May", "Jun",
                "Jul", "Aug", "Sep"
            ], fontsize=10, rotation=45, ha="right")
        if time_unit == "year":
            ax.set_xticks(range(2014, 2025, 1))
            ax.set_xticklabels(['2015', '2015', '2016', '', '2018', '2019', '2020', '', '2022', '2023', '2024'],
                                fontsize=8, rotation=45, ha="right")
        if time_unit == "hour":
            ax.set_xticks(range(0,25,3))
            ax.set_xticklabels([f"{h:02d}" for h in range(0,25,3)], fontsize=8, rotation=45, ha="right")
        
        # if col == 0:
        #     ax.text(
        #         -0.55, 0.5, region_name,
        #         transform=ax.transAxes,
        #         rotation=90,
        #         va="center",
        #         ha="right",
        #         fontsize=12,
        #         fontweight="bold"
        #     )              
                
    plt.tight_layout()
    return fig


fig_hour = plot_temporal_multiplot(
    df,
    EVENT_ORDER,
    time_unit="hour"
)

fig_hour.savefig(
    os.path.join(PLOT_DIR, "event_hourly_popularity.png"),
    dpi=300,
    bbox_inches="tight"
)

fig_month = plot_temporal_multiplot(
    df,
    EVENT_ORDER,
    time_unit="month"
)

fig_month.savefig(
    os.path.join(PLOT_DIR, "event_monthly_popularity.png"),
    dpi=300,
    bbox_inches="tight"
)

fig_year = plot_temporal_multiplot(
    df,
    EVENT_ORDER,
    time_unit="year"
)

fig_year.savefig(
    os.path.join(PLOT_DIR, "event_yearly_popularity.png"),
    dpi=300,
    bbox_inches="tight"
)
