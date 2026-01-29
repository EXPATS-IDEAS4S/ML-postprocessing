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
sys.path.append("/home/Daniele/codes/VISSL_postprocessing/scripts/pretrain/")
from transitions.data_utils import (
    filter_rows_in_event_window,
    build_event_groups,
    split_by_region,
    load_data
)

from transitions.case_study_utils import (
    build_trajectory_id,
    select_extreme_cases,
    show_crop_table_with_gaps,
    plot_feature_space_with_trajectory,
    plot_feature_trajectory,
    build_crop_filename
)


# === CONFIG ===
RUN_NAME = 'dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc'
EVENT_TYPES = ["PRECIP", "HAIL"]
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test"
CROP_BASE_DIR = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma"
SUMMARY_DIR = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma"
LAT_DIV = 47
REGIONS = ["NORTH", "SOUTH"]
EVENT_TYPES = ["PRECIP", "HAIL"]
FEATURE_SPACE = os.path.join(BASE_DIR, "tsne_all_vectors_with_centroids.csv")
TSNE_COLS = ['tsne_dim_1', 'tsne_dim_2']

# === COLOR MAP ===
COLORS_PER_CLASS = {
    0: 'darkgray', 1: 'darkslategrey', 2: 'peru', 3: 'orangered',
    4: 'lightcoral', 5: 'deepskyblue', 6: 'purple', 7: 'lightblue',
    8: 'green', 9: 'goldenrod', 10: 'magenta', 11: 'dodgerblue',
    12: 'darkorange', 13: 'olive', 14: 'crimson'
}

path = f"{BASE_DIR}/features_train_test_{RUN_NAME}.csv"


df = load_data(path, filter_event_window=False)
df = df.rename(columns={"2d_dim_1": "tsne_dim_1", "2d_dim_2": "tsne_dim_2"})
labels_sorted = sorted(df['label'].unique())
#print(df['path'].loc[0])

#extrat filenmae from path and change basename from this 2014-04-04T12:00_44.59_10.95.nc to 2014-04-03T00:00_50.53_12.82_20140403T0000.png
df['crop_filename'] = df.apply(lambda row: build_crop_filename(
    row['path'],
    row['lat_centre'],
    row['lon_centre']
), axis=1)

#print(df['crop_filename'].loc[0])
#print(df)

#open df featture csv
df_features = pd.read_csv(FEATURE_SPACE, low_memory=False)

df_centroids = df_features[
    (df_features["vector_type"] == "CENTROID")
].dropna(subset=TSNE_COLS)
#add label column to centroids df
df_centroids["label"] = labels_sorted[:len(df_centroids)]

df_features = df_features[
    (df_features["vector_type"] != "CENTROID")
].dropna(subset=TSNE_COLS)


#replace feature columns in df with those from df_features based on the vector_type and index
for vtype in df_features["vector_type"].unique():
    mask = df["vector_type"] == vtype
    df_v = df_features[df_features["vector_type"] == vtype]
    df_v = df.loc[mask, :].copy()   # IMPORTANT: keep index
    df.loc[mask, TSNE_COLS] = df_v[TSNE_COLS].to_numpy()




df_train = df[df["vector_type"] == "TRAIN"]

df_regions = split_by_region(df, lat_column="lat_centre", LAT_DIVISION=LAT_DIV)
#print(df)

for region, df_region in zip(REGIONS, df_regions.values()):
    print(f"Processing region: {region}")
    #print(df_region)
    for event in EVENT_TYPES:
        print(f"  Processing event: {event}")
        #select rows where vector_type is event
        df_event = df_region[df_region['vector_type'] == event]
        #include a column with day only
        df_event['date'] = pd.to_datetime(df_event['datetime']).dt.date

        df_event = build_trajectory_id(df_event, time_column="date", lat_column="lat_centre", lon_column="lon_centre")

        df_cases, summaries = select_extreme_cases(df_event, intensity_column="max_intensity", traj_id_column="trajectory_id", n_cases=5, agg="max")
        #print(df_case['datetime'].to_list())
        for summury, df_case in zip(summaries, df_cases):
            print(summury)
            print(df_case)
         
            print(f"📌 {region} – {event}: {df_case['trajectory_id'].iloc[0]} with max intensity {df_case['max_intensity'].iloc[0]}")
            
            
            fig1 = show_crop_table_with_gaps(
                    img_dir=f"{CROP_BASE_DIR}/{event}/images/IR_108/png_vmin-vmax_greyscale_CMA",
                    df_traj=df_case,
                    class_colors=COLORS_PER_CLASS,
                )

            #fig2 = plot_feature_trajectory(df_case)
            fig1.savefig(f"{BASE_DIR}/trajectory_crops/case_{region}_{event}_{df_case['trajectory_id'].iloc[0]}_crops_table.png", dpi=300)
            #fig2.savefig(f"{BASE_DIR}/case_{region}_{event}_features.png", dpi=300)

            fig, ax = plt.subplots(figsize=(8, 6))

            plot_feature_space_with_trajectory(
                ax,
                df_case,
                df_train,
                df_centroids,
                xcol="tsne_dim_1",
                ycol="tsne_dim_2",
                label_col="label",
                time_col="datetime",
                class_colors=COLORS_PER_CLASS,
            )

            out_path = f"{BASE_DIR}/trajectory_feature_space/case_{region}_{event}_{df_case['trajectory_id'].iloc[0]}_feature_space_trajectory.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)