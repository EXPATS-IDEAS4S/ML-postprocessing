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
    load_data
)

from scripts.pretrain.transitions.case_study_utils import (
    build_trajectory_id,
    select_extreme_cases,
    show_crop_table_with_gaps,
    show_crop_table,
    plot_feature_space_with_trajectory,
    plot_feature_trajectory,
    build_crop_filename
)

from utils.plotting.class_colors import CLOUD_CLASS_INFO

# === CONFIG ===
RUN_NAME = 'dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc'
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"

CROP_BASE_DIR = "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj"
PLOT_DIR = f"{BASE_DIR}/case_studies"
os.makedirs(PLOT_DIR, exist_ok=True)

EVENT_TYPES = ["PRECIP", "HAIL", "MIXED"]
LAT_DIV = 47
REGIONS = ["NORTH", "SOUTH"]
N_STORMS = 10
RANDOM_SEED = 42

FEATURE_SPACE = os.path.join(BASE_DIR, "tsne_all_vectors_with_centroids.csv")
NEIGH_CSV = os.path.join(BASE_DIR, "hypersphere_analysis", "test_vectors_local_label_composition.csv")
STATS_CSV = os.path.join(BASE_DIR, "crops_stats_vars-cth-cma-precipitation-euclid_msg_grid_stats-50-99-25-75_frames-1_coords-datetime_dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc_all_0.csv")

TSNE_COLS = ['tsne_dim_1', 'tsne_dim_2']


cloud_items_ordered = sorted(
    CLOUD_CLASS_INFO.items(),
    key=lambda x: x[1]["order"]
)

labels_ordered = [lbl for lbl, _ in cloud_items_ordered]
short_labels = [info["short"] for _, info in cloud_items_ordered]
colors_ordered = [info["color"] for _, info in cloud_items_ordered]
#order colors by keys in CLOUD_CLASS_INFO
colors = []
for idx in range(len(labels_ordered)):
    info = CLOUD_CLASS_INFO[idx]
    colors.append(info["color"])
#colors = [info["color"] for _, info in CLOUD_CLASS_INFO.items()]
print(CLOUD_CLASS_INFO)
print(colors)


path = f"{BASE_DIR}/features_train_test_{RUN_NAME}_2nd_labels.csv"

#open columns: 'path', 'datetime', 'lat', 'lon', 'label', 'distance', 'vector_type', 'label_2nd', 'storm_id', 'storm_type', 'crop_type', 'region'
df = pd.read_csv(path, usecols=[
    'path', 'datetime', 'lat', 'lon', 'label', 
    'distance', 'vector_type', 'label_2nd', 'storm_id', 
    'storm_type', 'crop_type', 'region'
])

#seect only rows with vector_type different then TRAIN
df = df[df['vector_type'] != 'TRAIN']

#create filenmae column from path
df['filename'] = df['path'].apply(lambda x: os.path.basename(x))
#print(df.columns.to_list())

df_tsne = pd.read_csv(FEATURE_SPACE, low_memory=False)
#print(df_tsne)#.columns.to_list())

df_tsne_train = df_tsne[df_tsne['vector_type'] == 'TRAIN']
#print(df_tsne_train)
#remove invalid rows where label is -100
df_tsne_train = df_tsne_train[df_tsne_train['label'] != -100]
#add a column color based on label using colors list
df_tsne_train['color'] = df_tsne_train['label'].apply(lambda x: colors[x])

df_tsne_centroids = df_tsne[df_tsne['vector_type'] == 'CENTROID']
df_tsne_test = df_tsne[(df_tsne['vector_type'] != 'TRAIN') & (df_tsne['vector_type'] != 'CENTROID')]

print(df_tsne_test)

#merge df with df_tsne_test on filename
df = df.merge(
    df_tsne_test[['filename'] + TSNE_COLS],
    on='filename',
    how='left'
)
#print(df.columns.to_list())

df_neigh = pd.read_csv(NEIGH_CSV, low_memory=False)
#remove the columns: 'lat', 'lon', 'storm_type', 'crop_type', 'label', 'label_2nd',
df_neigh = df_neigh.drop(columns=[
    'lat', 'lon', 'storm_type', 'crop_type', 'label', 'label_2nd'
])

#merge df with df_neigh on filename
df = df.merge(
    df_neigh,
    on='filename',
    how='left'
)
print(df.columns.to_list())

df_stats = pd.read_csv(STATS_CSV, low_memory=False)

df['trajectory_id'] = df['filename'].apply(lambda x: x.split('_')[0])
print(df.columns.to_list())

#set random seed
np.random.seed(RANDOM_SEED)

for event in EVENT_TYPES:
    print(f"  Processing event: {event}")
    #select rows where vector_type is event
    df_event = df[df['storm_type'] == event]
    #print(df_event)
    #add column traj_id by applytins split('_')[0] to filename
    
    #include a column with day only
    #df_event['date'] = pd.to_datetime(df_event['datetime']).dt.date
    #group by storm_id and select random N_STORMS storm_id
    selected_storms = df_event['trajectory_id'].drop_duplicates().sample(n=N_STORMS, random_state=RANDOM_SEED, replace=False).to_list()
    #filter df_event by selected_storms
    df_event = df_event[df_event['trajectory_id'].isin(selected_storms)]

    for df_case_idx, df_case in df_event.groupby('trajectory_id'):
        print(f"    Processing storm_id: {df_case_idx}")
        print(df_case)
        
        fig1 = show_crop_table(
                img_dir=f"{CROP_BASE_DIR}/images/IR_108/png_vmin-vmax_greyscale_CMA",
                df_traj=df_case,
                storm_id = df_case_idx,
                time_col="datetime",
                label_col="label",
                class_colors=colors,
                #freq="15min",        # "15min" or "1H"
                cols=8,
                max_rows=12,
            )

        #fig2 = plot_feature_trajectory(df_case)
        fig1.savefig(f"{PLOT_DIR}/case__{event}_{df_case['trajectory_id'].iloc[0]}_crops_table.png", dpi=300)
        #fig2.savefig(f"{BASE_DIR}/case_{region}_{event}_features.png", dpi=300)
        
        
        #built time aligned column in hours from the first timestamp
        # --- trajectory prep ---
        df_case['datetime'] = pd.to_datetime(df_traj[time_col], utc=True).dt.tz_convert(None)
        df_traj = df_traj.sort_values(time_col).copy()

        t0 = df_traj[time_col].iloc[len(df_traj) // 2]  # midpoint reference
        df_traj["t_aligned"] = (
            (df_traj[time_col] - t0).dt.total_seconds() / 3600.0
        )
        
        
        
        fig, ax = plt.subplots(figsize=(10, 10))

        plot_feature_space_with_trajectory(
            ax,
            df_case,
            df_tsne_train,
            df_tsne_centroids,
            xcol="tsne_dim_1",
            ycol="tsne_dim_2",
            label_col="label",
            time_col="datetime",
            class_colors=colors,
        )

        out_path = f"{PLOT_DIR}/case_{event}_{df_case['trajectory_id'].iloc[0]}_feature_space_trajectory.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)

        
        
        #Plot the trajectrpy of the wperc_label_ of each class over time in a line plot over aligned time
        # identify weighted-percentage columns
        wperc_cols = [c for c in df_case.columns if c.startswith("wperc_label_")]

        df_long = (
            df_case
            .sort_values("aligned_time_hours")
            .melt(
                id_vars=["aligned_time_hours"],
                value_vars=wperc_cols,
                var_name="class",
                value_name="wperc"
            )
        )

        # extract class id
        df_long["class"] = df_long["class"].str.replace("wperc_label_", "").astype(int)
