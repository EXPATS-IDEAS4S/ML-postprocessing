import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
import cmcrameri.cm as cmc
import numpy as np
from matplotlib.lines import Line2D

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from scripts.pretrain.cluster_analysis.test_analysis.utils_func import (filter_rows_in_event_window, stratifiy_by_latitude, 
                        build_event_groups, print_sample_counts, plot_event_trajectories,
                        compute_class_kde_grid, plot_class_kde_background, plot_feature_space_dots,
                        plot_density_contours, plot_distance_boxplot)

from utils.plotting.class_colors import COLORS_PER_CLASS, CLOUD_CLASS_INFO

# ================================================
# LOAD DATA
# ================================================
run_name = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
output_path = f"/data1/fig/{run_name}/epoch_800/test_traj"
output_test_csv = os.path.join(output_path, f"features_train_test_{run_name}_2nd_labels.csv")

feature_space = os.path.join(output_path, "tsne_all_vectors_with_centroids.csv")

#train_feature_space = os.path.join(output_path, "merged_tsne_crop_list_with_img_path.csv")

plot_dir = os.path.join(output_path, "feature_space_plots_ALL")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(output_test_csv, low_memory=False)
#print(df.columns.tolist())

#df_train = pd.read_csv(train_feature_space, low_memory=False)

PERCENTILE = 50  # Percentile for intensity thresholding
LAT_DIVISION = 47
MIN_SAMPLES = 50  # Minimum samples to plot per class label
MIN_REL_PERCENT = 5.0   # percent
WITHOUT_EXTRAPOLATED = False

percentiles = [50, 90]
linestyles_list = ['solid', 'dashed','dotted', 'dashdot']

# feature columns
#dim_cols = [c for c in df.columns if c.startswith("dim_")]

#feature_cols_2d = ["2d_dim_1", "2d_dim_2"]
tsne_cols = ["tsne_dim_1", "tsne_dim_2"]

#dictionary with group names and colors
groups_dict = [
    {'name' :"ALL"},
    #{'name': "PRECIP"},
    #{'name': "HAIL" },
    #{'name': "MIXED" }
]


cloud_items_ordered = sorted(
    CLOUD_CLASS_INFO.items(),
    key=lambda x: x[1]["order"]
)

labels_ordered = [lbl for lbl, _ in cloud_items_ordered]
short_labels = [info["short"] for _, info in cloud_items_ordered]
colors_ordered = [info["color"] for _, info in cloud_items_ordered]


#rename feature columns to tsne columns
#df = df.rename(columns={"2d_dim_1": "tsne_dim_1", "2d_dim_2": "tsne_dim_2"})
labels_sorted = sorted(CLOUD_CLASS_INFO.keys())#, key=lambda x: CLOUD_CLASS_INFO[x]["order"])
print(labels_sorted)

#open df featture csv
df_features = pd.read_csv(feature_space, low_memory=False)

df_centroids = df_features[
    (df_features["vector_type"] == "CENTROID")
].dropna(subset=tsne_cols)
#add label column to centroids df
df_centroids["label"] = labels_sorted[:len(df_centroids)]

df_features = df_features[
    (df_features["vector_type"] != "CENTROID")
].dropna(subset=tsne_cols)

df_test = df_features[df_features["vector_type"] != "TRAIN"]

df_train = df_features[df_features["vector_type"] == "TRAIN"]
df_train = df_train[df_train["label"] != -100]

print(df_test)
print(df_train)

#extract filenmae in df (from path)
df['filename'] = df['path'].apply(lambda x: os.path.basename(x))

#select in df only the test vector type and attach the correct tsne col from df_test using filename match
df_test_feat = df[df["vector_type"] != "TRAIN"].copy()
print(df_test_feat)

df_test_label = df_test_feat.merge(
    df_test[["filename", "tsne_dim_1", "tsne_dim_2"]],
    on="filename",
    how="left",
)

print(df_test_label)


#if WITHOUT EXTRAPOLATED REMOVE THE rows with crop_tyoe == EXTRAPOLATED
if WITHOUT_EXTRAPOLATED:
    df_test_label = df_test_label[df_test_label["crop_type"] != "extrapolated"]

# #replace feature columns in df with those from df_features based on the vector_type and index
# for vtype in df_test_feat["vector_type"].unique():
#     mask = df_test_feat["vector_type"] == vtype
#     df_v = df_test_feat[mask] 
#     df_test_label.loc[mask, tsne_cols] = df_v.loc[df_test_feat.index[mask], tsne_cols].values


# ================================================
# BUILD GROUP MASKS
# ================================================
cmap = cmc.lisbon
norm = mpl.colors.Normalize(vmin=0, vmax=23)

n_rows = 1
n_cols = len(groups_dict)

fig_fs, axes_fs = plt.subplots(
    n_rows, n_cols,
    figsize=(5 * n_cols, 5 * n_rows),
    sharex=True, sharey=True
)

fig_bar, axes_bar = plt.subplots(
    n_rows, n_cols,
    figsize=(2 * n_cols, 1 * n_rows),
    sharex=True, sharey=True
)


axes_fs = np.atleast_2d(axes_fs)
axes_bar = np.atleast_2d(axes_bar)
#axes_dist = np.atleast_2d(axes_dist)
#axes_2nd = np.atleast_2d(axes_2nd)


#lat_groups = stratifiy_by_latitude(df_test_label, lat_col="lat", LAT_DIVISION=LAT_DIVISION)



#for r, (region, df_lat) in enumerate(zip(["NORTH", "SOUTH"], lat_groups)):

#print(f"\n=== {region} REGION ===")
#df_groups = build_event_groups(df_lat, PERCENTILE)
df_lat = df_test_label.copy()
for c, ginfo in enumerate(groups_dict):
    event_name = ginfo["name"]
    print(f"\n--- Processing group: {event_name} ---")
    if event_name == "ALL":
        df_group = df_lat.copy()
    else:
        df_group = df_lat[df_lat['storm_type'] == event_name].copy()

    #ax_fs = axes_fs[0, c] if region == "NORTH" else axes_fs[1, c]
    #ax_bar = axes_bar[0, c] if region == "NORTH" else axes_bar[1, c]
    #ax_dist = axes_dist[0, c] if region == "NORTH" else axes_dist[1, c]

    # =============================
    # FEATURE SPACE PLOT
    # ============================
    plot_feature_space_dots(
        axes_fs[0, c],
        df_train,
        xcol="tsne_dim_1",
        ycol="tsne_dim_2",
        label_col="label",
        class_colors=CLOUD_CLASS_INFO,
        s=5,
        alpha=0.1,
        rasterized=True,
    )

    #add centroids as markers with the label color
    for _, row in df_centroids.iterrows():
        axes_fs[0, c].scatter(
            row["tsne_dim_1"],
            row["tsne_dim_2"],
            color = CLOUD_CLASS_INFO[int(row["label"])]["color"],
            marker="^",
            s=100,
            edgecolor="black",
            linewidth=1,
            zorder=5
        )

    # ---- test vectors ----
    #df_events = filter_rows_in_event_window(df_group)
    #plot_event_trajectories(ax_fs, df_events, cmap, norm, alpha=0.7, linewidth=0.7)
    print(df_group)
    plot_density_contours(
        axes_fs[0, c],
        df_group,
        xcol="tsne_dim_1",
        ycol="tsne_dim_2",
        percentiles=percentiles,
        bandwidth=0.25,
        color="black",
        linewidths=(2.0, 2.0, 2.0),
        alpha=1.0,
        linestyles_list=linestyles_list
    )

    #remove spines, axes and ticks
    axes_fs[0, c].set_xticks([])
    axes_fs[0, c].set_yticks([])
    axes_fs[0, c].set_xlabel("")
    axes_fs[0, c].set_ylabel("")
    axes_fs[0, c].spines['top'].set_visible(False)
    axes_fs[0, c].spines['right'].set_visible(False)
    axes_fs[0, c].spines['left'].set_visible(False)
    axes_fs[0, c].spines['bottom'].set_visible(False)

    # ----------------------------------
    # Count per class label
    # ----------------------------------

    counts = (
        df_group["label"]
        .astype(int)
        .value_counts()
        .reindex(labels_ordered, fill_value=0)
    )


    #normalize counts to total number of samples in the group
    #total_counts = counts.sum()
    #print(f"Total samples in group {event_name}: {total_counts}")

    #Set count to 0 remove labels with counts below MIN_SAMPLES
    #counts = counts[counts >= MIN_SAMPLES]

    #rel_counts = counts / total_counts * 100  # per 100

    #make bar plot with the normalized counts
    x = np.arange(len(labels_ordered))

    axes_bar[0, c].bar(
        x,
        counts.values * 1e-3,  #scale to thousands
        color=colors_ordered,
        alpha=0.85,
    )



    # ----------------------------------
    # Titles & row labels
    # ----------------------------------
    #if r == 0:
    # axes_fs[0, c].set_title(
    #     ginfo["name"],
    #     fontsize=14,
    #     fontweight="bold"
    # )
    # axes_bar[0, c].set_title(
    #     ginfo["name"],
    #     fontsize=11,
    #     fontweight="bold"
    # )

    #if c == 0:
        # axes_fs[0, c].text(
        #     -0.01, 0.5, region,
        #     transform=ax_fs.transAxes,
        #     fontsize=14,
        #     fontweight="bold",
        #     va="center",
        #     ha="right",
        #     rotation=90
        # ) 
        # #shift this to the left
        # axes_bar[0, c].text(
        #     -0.5, 0.5, region,
        #     transform=axes_bar[0, c].transAxes,
        #     fontsize=12,
        #     fontweight="bold",
        #     va="center",
        #     ha="right",
        #     rotation=90
        # )

    #if r == n_rows - 1:
    axes_bar[0, c].set_xticks(x)
    axes_bar[0, c].set_xticklabels(short_labels, fontsize=11, rotation=45, ha="right")

    #put grid behind bars
    axes_bar[0, c].set_axisbelow(True)
    axes_bar[0, c].yaxis.grid(alpha=0.3)

    

    # ----------------------------------
    # Minimal styling
    # ----------------------------------
    #axes_bar[0, c].set_yscale("log")
    #show customized y tichks for bar plot
    #xticks_bar = [10, 100, 1000, 10000]
    xticks_bar = [0, 5, 10, 15]
    axes_bar[0, c].set_yticks(xticks_bar)

    #ax_bar.set_xticks(labels_sorted)
    #ax_bar.set_xticklabels([], fontsize=11)

    #ax_bar.set_axisbelow(True)
    #ax_bar.tick_params(axis="y", labelsize=10)
    #set y ticks (3 integer ticks)
    #ax_bar.yaxis.set_major_locator(plt.MaxNLocator(4))


#shift this more to the right
fig_bar.text(-0.12, 0.5, "Count (10^3)", va="center",
            rotation="vertical", fontsize=10)


#add legend on the right side of the figure (outside) for the percentile of the contours
#also add the linestyle used for each percentile
legend_handles = [
    Line2D([0], [0], color="black", linestyle="solid", linewidth=2.0),
    Line2D([0], [0], color="black", linestyle="dashed", linewidth=2.0),
    #Line2D([0], [0], color="black", linestyle="dotted", linewidth=2.0),   
]

legend_labels = [
    "50th percentile",
    "90th percentile",
    #"99th percentile"
]

fig_fs.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="center right",
    bbox_to_anchor=(1.2, 0.75),   # push legend outside
    fontsize=10,
    title="Density contours",
    title_fontsize=10,
    frameon=False,
)

#add title to the figure with the group name and event
fig_fs.suptitle(
    f"a)",# Training embedding with \n test vector density contours",
    fontsize=10, fontweight="bold", y=1.03, x=0.82, #shift more to the right
)


#Save figure feature space
if WITHOUT_EXTRAPOLATED:
    outname_fs = f"feature_space_region_event_no_extrapolated.png"
else:
    outname_fs = f"feature_space_region_event.png"
fig_fs.savefig(os.path.join(plot_dir, outname_fs),
            dpi=300, bbox_inches="tight", transparent=True)

print("Saved:", outname_fs)


#save bar and distance figures


fig_bar.subplots_adjust(wspace=0.15, hspace=0.15)

#add title to the figure with the group name and event
fig_bar.suptitle(
    f"b)",# Group population",
    fontsize=10, fontweight="bold", y=1.09
)

if WITHOUT_EXTRAPOLATED:
    outname_bar = "group_population_by_label_latitude_no_extrapolated.png"
else:
    outname_bar = "group_population_by_label_latitude.png"

fig_bar.savefig(
    os.path.join(plot_dir, outname_bar),
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

print("✅ Saved:", outname_bar)


plt.close()