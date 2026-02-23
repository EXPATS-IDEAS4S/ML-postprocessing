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

from scripts.pretrain.transitions.compute_transitions_utils import (
    compute_transitions_and_persistence,
    compute_weighted_transitions_and_persistence,
    compute_transitions_and_persistence_weighted,
    split_traj_by_region,
    check_continuous_timestamps
)
from scripts.pretrain.transitions.plot_transitions_utils import (
    plot_persistence_bar_multiplot,
    plot_persistence_time_multiplot,
    plot_transition_heatmap_multiplot,
    plot_entropy_persistence_multiplot,
    plot_alluvial_multiplot,
    plot_transition_graph_multiplot,
)

from utils.plotting.class_colors import CLOUD_CLASS_INFO

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUTDIR = f"{BASE_DIR}/transitions_plots_no_dominance"
os.makedirs(OUTDIR, exist_ok=True)

LAT_DIVISION = 47
REGIONS = ["NORTH", "SOUTH"]

EVENTS = [
    {"name": "ALL"},
    #{"name": "PRECIP"},
    #{"name": "HAIL"},
    #{"name": "MIXED"},
]

EVENT_ORDER = [e["name"] for e in EVENTS]
#EVENT_LABELS = [e["label"] for e in EVENTS]

VMAX = 1

cloud_items_ordered = sorted(
    CLOUD_CLASS_INFO.items(),
    key=lambda x: x[1]["order"]
)

labels_ordered = [lbl for lbl, _ in cloud_items_ordered]
short_labels = [info["short"] for _, info in cloud_items_ordered]
colors_ordered = [info["color"] for _, info in cloud_items_ordered]

SELECTED_CLASSES = [1,2,4]
selected_labels_ordered = [lbl for lbl in labels_ordered if lbl in SELECTED_CLASSES]
selected_short_labels = [short_labels[i] for i, lbl in enumerate(labels_ordered) if lbl in SELECTED_CLASSES]
selected_colors_ordered = [colors_ordered[i] for i, lbl in enumerate(labels_ordered) if lbl in SELECTED_CLASSES]
print(f"Selected labels: {selected_labels_ordered}")
print(f"Selected short labels: {selected_short_labels}")
print(f"Selected colors: {selected_colors_ordered}")

#function to check if for each storm id rows, the timestamps are all continuous (no missing gaps)



def plot_multiplot_regions_events(
    df,
    plot_func,
    plot_type,
    labels=None,
    vmax=1.0,
    outdir=OUTDIR,
    colors_per_class=None,
    class_names=None,
):
    nrows = 1 #len(REGIONS)
    ncols = len(EVENT_ORDER)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2. * ncols, 2. * nrows),
        sharex=False,
        sharey=False
    )

    # region_dfs = split_traj_by_region(
    #     df,
    #     lat_column="lat",
    #     lat_division=47
    # )

    #for r, region in enumerate(REGIONS):
        #df_region =  #region_dfs[region]

        # groups = build_event_groups(
        #     df_region,
        #     percentile=PERCENTILE,
        #     intensity_col="max_intensity",
        #     vector_col="vector_type"
        # )

    for c, event in enumerate(EVENT_ORDER):
        ax = axes#[c]

        if event == "ALL":
            df_event = df
        else:
            df_event = df[df['storm_type'] == event]

        if len(df_event) < 50:
            ax.axis("off")
            continue

        print(df_event.columns.tolist())

        # Compute transitions without weighted/dominance split
        results = compute_transitions_and_persistence(
                df_event,
                label_col="label",
                time_col="datetime",
                lat_col="lat",
                lon_col="lon",
                labels=labels,
        )
        
        print(results)
        # Get labels from results 
        label_names = results['labels']
        print(label_names)
        
        # No need to handle '_' in class names since we're not splitting by dominance
        # Simply assign colors based on the labels
        #colors_per_class = selected_colors_ordered.copy()
        class_names = selected_short_labels.copy()
        

        # 🔹 delegate plotting
        plot_func(ax, results, label_names, vmax, colors=colors_per_class, class_names=class_names)

        # ---- region label
        # if c == 0:
        #     ax.text(
        #         -0.4, 0.5, region,
        #         transform=ax.transAxes,
        #         rotation=90,
        #         va="center",
        #         ha="right",
        #         fontsize=12,
        #         fontweight="bold"
        #     )

        # ---- column title
        #if r == 0:
        #ax.set_title(EVENT_ORDER[c], fontsize=12, fontweight="bold")
            #ax.set_xticklabels([])
        

    out = f"{outdir}/multiplot_{plot_type}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {out}")


if __name__ == "__main__":

    path = f"{BASE_DIR}/features_train_test_{RUN_NAME}_2nd_labels.csv"
    df = pd.read_csv(path, low_memory=False)
    df_test = df[df["vector_type"] != "TRAIN"]

    df_test['crop'] = df_test['path'].apply(lambda x: os.path.basename(x))
    #extract storm if from crop
    df_test['storm_id'] = df_test['crop'].apply(lambda x: x.split('_')[0])
    #print first crop and strom id to check if they match
    #df = load_data(path)
    print(df_test)
    #labels = np.sort(df["label"].unique())
    df_sum = check_continuous_timestamps(df_test, id_col="storm_id", time_col="datetime", freq='15min')
    print(df_sum)
    #howm many entries have is_continuous == True
    n_continuous = df_sum['is_continuous'].sum()
    print(f"Number of continuous trajectories: {n_continuous} out of {len(df_sum)}")
    #filter out non continuous trajectories
    df_continuous = df_test[df_test['storm_id'].isin(df_sum[df_sum['is_continuous']==True]['storm_id'])]
    print(df_continuous.columns.tolist())

    #open neighborhoods csv
    neigh_csv = os.path.join(
    BASE_DIR, 'hypersphere_analysis',
    "test_vectors_local_label_composition.csv"
    )
    
    df_neighbors = pd.read_csv(neigh_csv)
    print(df_neighbors.columns.tolist())
    #change column filename to crop
    df_neighbors = df_neighbors.rename(columns={"filename": "crop"})
    #delete the following columns  'lat', 'lon', 'storm_type', 'crop_type', 'label', 'label_2nd'
    df_neighbors = df_neighbors.drop(columns=['lat', 'lon', 'storm_type', 'crop_type', 'label', 'label_2nd'])
    #merge df_continuous with df_neighbors on crop
    df_continuous = pd.merge(df_continuous, df_neighbors, on='crop', how='inner')
    print(f"Merged dataframe shape: {df_continuous.shape}")
    #remove all 'dim_' columns
    dim_cols = [c for c in df_continuous.columns if c.startswith("dim_")]
    df_continuous = df_continuous.drop(columns=dim_cols)
    print(df_continuous.columns.tolist())


    # plot_multiplot_regions_events(
    #     df_continuous,
    #     plot_func=plot_persistence_time_multiplot,
    #     plot_type="persistence_time",
    #     labels=selected_labels_ordered,
    #     vmax=10,
    #     colors_per_class=selected_colors_ordered,
    #     class_names = selected_short_labels
    # )
    

    plot_multiplot_regions_events(
        df_continuous,
        plot_func=plot_transition_heatmap_multiplot,
        plot_type="transition_matrix",
        vmax=VMAX,
        labels= selected_labels_ordered,
        colors_per_class=cmc.batlow,
        class_names = selected_short_labels
    )

 
    # plot_multiplot_regions_events(
    #     df_continuous,
    #     plot_func=plot_entropy_persistence_multiplot,
    #     plot_type="persistence_vs_entropy",
    #     labels=labels_ordered,
    #     vmax=None,
    #     colors_per_class=colors_ordered,
    #     class_names = short_labels
    # )



    # plot_multiplot_regions_events(
    #     df_continuous,
    #     plot_func=plot_transition_graph_multiplot,
    #     plot_type="transition_graph",
    #     labels=selected_labels_ordered,
    #     vmax=0.05,
    #     colors_per_class=selected_colors_ordered,
    #     class_names = selected_short_labels
    # )






