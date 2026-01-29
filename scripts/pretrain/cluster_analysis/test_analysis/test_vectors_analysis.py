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

train_feature_space = os.path.join(output_path, "merged_tsne_crop_list_with_img_path.csv")

plot_dir = os.path.join(output_path, "feature_space_plots")
os.makedirs(plot_dir, exist_ok=True)

df = pd.read_csv(output_test_csv, low_memory=False)
print(df.columns.tolist())

df_train = pd.read_csv(train_feature_space, low_memory=False)

PERCENTILE = 50  # Percentile for intensity thresholding
LAT_DIVISION = 47
MIN_SAMPLES = 50  # Minimum samples to plot per class label
MIN_REL_PERCENT = 5.0   # percent

percentiles = [50, 90]
linestyles_list = ['solid', 'dashed','dotted', 'dashdot']

# feature columns
dim_cols = [c for c in df.columns if c.startswith("dim_")]

feature_cols_2d = ["2d_dim_1", "2d_dim_2"]
tsne_cols = ["tsne_dim_1", "tsne_dim_2"]

#dictionary with group names and colors
groups_dict = [
    {'name' :"ALL"},
    {'name': "PRECIP"},
    {'name': "HAIL" },
    {'name': "MIXED" }
]


cloud_items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
print(cloud_items)

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

df_test_label = df[df["vector_type"] != "TRAIN"]

df_test_feat = df_features[df_features["vector_type"] != "TRAIN"]

print(df_test_label)
print(df_test_feat)


#replace feature columns in df with those from df_features based on the vector_type and index
for vtype in df_test_feat["vector_type"].unique():
    mask = df_test_feat["vector_type"] == vtype
    df_v = df_test_feat[mask] 
    df_test_label.loc[mask, tsne_cols] = df_v.loc[df_test_feat.index[mask], tsne_cols].values

print(df_test_label)
print(df_test_feat)
print(df_train)


#remove rows with label -100 (invalid)
#df = df[df["label"] != -100]

# ================================================
# BUILD GROUP MASKS
# ================================================
cmap = cmc.lisbon
norm = mpl.colors.Normalize(vmin=0, vmax=23)

n_rows = 2
n_cols = len(groups_dict)

fig_fs, axes_fs = plt.subplots(
    n_rows, n_cols,
    figsize=(2.5 * n_cols, 2 * n_rows),
    sharex=True, sharey=True
)

fig_bar, axes_bar = plt.subplots(
    n_rows, n_cols,
    figsize=(2 * n_cols, 1 * n_rows),
    sharex=True, sharey=True
)

# fig_dist, axes_dist = plt.subplots(
#     n_rows, n_cols,
#     figsize=(2 * n_cols, 1 * n_rows),
#     sharex=True, sharey=True
# )

# fig_2nd, axes_2nd = plt.subplots(
#     n_rows, n_cols,
#     figsize=(2 * n_cols, 1 * n_rows),
#     sharex=True, sharey=True
# )

axes_fs = np.atleast_2d(axes_fs)
axes_bar = np.atleast_2d(axes_bar)
#axes_dist = np.atleast_2d(axes_dist)
#axes_2nd = np.atleast_2d(axes_2nd)


lat_groups = stratifiy_by_latitude(df_test_label, lat_col="lat", LAT_DIVISION=LAT_DIVISION)



for r, (region, df_lat) in enumerate(zip(["NORTH", "SOUTH"], lat_groups)):

    print(f"\n=== {region} REGION ===")
    #df_groups = build_event_groups(df_lat, PERCENTILE)

    for c, ginfo in enumerate(groups_dict):
        event_name = ginfo["name"]
        print(f"\n--- Processing group: {event_name} ---")
        if event_name == "ALL":
            df_group = df_lat.copy()
        else:
            df_group = df_lat[df_lat['vector_type'] == event_name].copy()

        ax_fs = axes_fs[0, c] if region == "NORTH" else axes_fs[1, c]
        ax_bar = axes_bar[0, c] if region == "NORTH" else axes_bar[1, c]
        #ax_dist = axes_dist[0, c] if region == "NORTH" else axes_dist[1, c]

        # =============================
        # FEATURE SPACE PLOT
        # ============================
        plot_feature_space_dots(
            ax_fs,
            df_train,
            xcol="tsne_dim_1",
            ycol="tsne_dim_2",
            label_col="label",
            class_colors=CLOUD_CLASS_INFO,
            s=4,
            alpha=0.08,
            rasterized=True,
        )

        #add centroids as markers with the label color
        for _, row in df_centroids.iterrows():
            ax_fs.scatter(
                row["tsne_dim_1"],
                row["tsne_dim_2"],
                color = CLOUD_CLASS_INFO[int(row["label"])]["color"],
                marker="^",
                s=80,
                edgecolor="white",
                linewidth=0.5,
                zorder=5
            )

        # ---- test vectors ----
        #df_events = filter_rows_in_event_window(df_group)
        #plot_event_trajectories(ax_fs, df_events, cmap, norm, alpha=0.7, linewidth=0.7)
        print(df_group)
        plot_density_contours(
            ax_fs,
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
        ax_fs.set_xticks([])
        ax_fs.set_yticks([])
        ax_fs.set_xlabel("")
        ax_fs.set_ylabel("")
        ax_fs.spines['top'].set_visible(False)
        ax_fs.spines['right'].set_visible(False)
        ax_fs.spines['left'].set_visible(False)
        ax_fs.spines['bottom'].set_visible(False)

        # ----------------------------------
        # Count per class label
        # ----------------------------------
        counts = (
            df_group["label"]
            .value_counts()
            .reindex(labels_sorted, fill_value=0)
        )
        print(counts)

        #normalize counts to total number of samples in the group
        #total_counts = counts.sum()
        #print(f"Total samples in group {event_name}: {total_counts}")

        #Set count to 0 remove labels with counts below MIN_SAMPLES
        #counts = counts[counts >= MIN_SAMPLES]

        #rel_counts = counts / total_counts * 100  # per 100

        #make bar plot with the normalized counts
        ax_bar.bar(
            [item[0] for item in cloud_items],
            counts.values,
            color=[item[1]["color"] for item in cloud_items],
            alpha=0.85
        )

        
        # plot_distance_boxplot(
        #     ax_dist,
        #     df_group,
        #     label_col="label",
        #     distance_col="distance",
        #     class_colors=cloud_items,
        #     showfliers=False,
        #     alpha=0.8,
        # )

        # # Absolute counts of (label, label_2nd)
        # counts = (
        #     df_group
        #     .groupby(["label", "label_2nd"])
        #     .size()
        #     .unstack(fill_value=0)
        # )

        # # Enforce consistent order
        # counts_2nd = counts.reindex(index=labels_sorted, columns=labels_sorted, fill_value=0)

        # # Total samples per primary label
        # #total_per_label = counts_2nd.sum(axis=1)

        # #valid_labels = total_per_label[total_per_label >= MIN_SAMPLES].index

        # # rel_freq = (
        # #     counts.loc[valid_labels]
        # #     .div(counts.loc[valid_labels].sum(axis=1), axis=0)
        # #     * 100
        # # )

        # rel_freq = counts_2nd.copy().astype(float)

        # for lbl in labels_sorted:
        #     if counts_2nd[lbl] < MIN_REL_PERCENT:
        #         # Too rare overall → zero out stacked bar
        #         counts_2nd.loc[lbl, :] = 0.0
        #     else:
        #         counts_2nd.loc[lbl, :] = (
        #             counts_2nd.loc[lbl] / counts_2nd.loc[lbl].sum() * 100
        #         )


        # ax_2nd = axes_2nd[0, c] if region == "NORTH" else axes_2nd[1, c]

        # bottom = np.zeros(len(labels_sorted))

        # for lbl_2nd, color_info in cloud_items:
        #     ax_2nd.bar(
        #         labels_sorted,
        #         rel_freq[lbl_2nd].values,
        #         bottom=bottom,
        #         color=color_info["color"],
        #         alpha=0.85
        #     )
        #     bottom += rel_freq[lbl_2nd].values


        # ----------------------------------
        # Titles & row labels
        # ----------------------------------
        if r == 0:
            ax_fs.set_title(
                ginfo["name"],
                fontsize=14,
                fontweight="bold"
            )
            ax_bar.set_title(
                ginfo["name"],
                fontsize=11,
                fontweight="bold"
            )

        if c == 0:
            ax_fs.text(
                -0.01, 0.5, region,
                transform=ax_fs.transAxes,
                fontsize=14,
                fontweight="bold",
                va="center",
                ha="right",
                rotation=90
            ) 
            #shift this to the left
            ax_bar.text(
                -0.5, 0.5, region,
                transform=ax_bar.transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                ha="right",
                rotation=90
            )

            # ax_dist.text(
            #     -0.5, 0.5, region,
            #     transform=ax_dist.transAxes,
            #     fontsize=12,
            #     fontweight="bold",
            #     va="center",
            #     ha="right",
            #     rotation=90
            # )

            # ax_2nd.text(
            #     -0.5, 0.5, region,
            #     transform=ax_2nd.transAxes,
            #     fontsize=12,
            #     fontweight="bold",
            #     va="center",
            #     ha="right",
            #     rotation=90
            # )

        

        # ----------------------------------
        # Minimal styling
        # ----------------------------------
        #ax_bar.set_yscale("log")
        #show customized y tichks for bar plot
        #xticks_bar = [10, 100]
        #ax_bar.set_yticks(xticks_bar)

        ax_bar.set_xticks(labels_sorted)
        ax_bar.set_xticklabels([], fontsize=11)

        #ax_bar.set_axisbelow(True)
        ax_bar.tick_params(axis="y", labelsize=10)
        #set y ticks (3 integer ticks)
        ax_bar.yaxis.set_major_locator(plt.MaxNLocator(4))

        # # boxplot distances
        # ax_dist.set_xticks(labels_sorted)
        # ax_dist.set_xticklabels(labels_sorted, fontsize=11)
        # ax_dist.set_axisbelow(True)
        # ax_dist.tick_params(axis="y", labelsize=11)
        # #set y ticks (3 integer ticks)
        # ax_dist.yaxis.set_major_locator(plt.MaxNLocator(4))

        # # barplot 2nd labels
        # ax_2nd.set_xticks(labels_sorted)
        # ax_2nd.set_xticklabels(labels_sorted, fontsize=11)
        # ax_2nd.set_axisbelow(True)
        # ax_2nd.tick_params(axis="y", labelsize=11)
        # #set y ticks (3 integer ticks)
        # ax_2nd.yaxis.set_major_locator(plt.MaxNLocator(4))


#shift this more to the right
fig_bar.text(0.07, 0.5, "1st Label Freq. (%)", va="center",
            rotation="vertical", fontsize=12)

# #shift this more to the right
# fig_dist.text(0.07, 0.5, "Cosine Similarity", va="center",
#             rotation="vertical", fontsize=12)

# #shift this a bit down
# fig_dist.text(0.5, -0.1, "Class label", ha="center", fontsize=12)


# fig_2nd.text(0.07, 0.5, "2nd Label Freq. (%)", va="center",
#             rotation="vertical", fontsize=12)

# #shift this a bit down
# fig_2nd.text(0.5, -0.1, "Class label", ha="center", fontsize=12)

# # ---- colorbar (hour) ----
# # create ScalarMappable
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# # add colorbar axis: [left, bottom, width, height]
# cax = fig_fs.add_axes([0.92, 0.15, 0.02, 0.7])
# cbar = fig_fs.colorbar(sm, cax=cax)
# cbar.set_label("Hour of day (UTC)", fontsize=14)
# #increase ticks of colorbar 
# cbar.ax.tick_params(labelsize=12)

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
    bbox_to_anchor=(1.03, 0.5),   # push legend outside
    fontsize=10,
    title="Density \n contours",
    title_fontsize=10,
    frameon=False,
)


#Save figure feature space

outname_fs = f"feature_space_region_event.png"
fig_fs.savefig(os.path.join(plot_dir, outname_fs),
            dpi=300, bbox_inches="tight", transparent=True)

print("Saved:", outname_fs)


#save bar and distance figures


fig_bar.subplots_adjust(wspace=0.15, hspace=0.15)

outname_bar = "group_population_by_label_latitude.png"

fig_bar.savefig(
    os.path.join(plot_dir, outname_bar),
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

print("✅ Saved:", outname_bar)



# fig_dist.subplots_adjust(wspace=0.15, hspace=0.15)

# outname_dist = "group_distance_by_label_latitude.png"

# fig_dist.savefig(
#     os.path.join(plot_dir, outname_dist),
#     dpi=300,
#     bbox_inches="tight",
#     transparent=True
# )

# print("✅ Saved:", outname_dist)


# fig_2nd.subplots_adjust(wspace=0.15, hspace=0.15)

# outname_2nd = "group_2nd_label_freq_by_label_latitude.png"

# fig_2nd.savefig(
#     os.path.join(plot_dir, outname_2nd),
#     dpi=300,
#     bbox_inches="tight",
#     transparent=True
# )

# print("✅ Saved:", outname_2nd)

plt.close()