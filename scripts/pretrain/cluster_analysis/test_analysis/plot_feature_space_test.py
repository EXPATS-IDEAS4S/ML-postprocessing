"""Feature-space and storm-aggregate summary plots for test data.

The script produces the baseline figures using all test samples, then a second
set of figures restricted to the test samples belonging to the top-50 unique
rain and hail storms selected from the pathway analysis CSV.
"""

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
pathway_csv = f"{output_path}/pathway_analysis/df_pathways_merged_no_dominance.csv"

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


def normalize_storm_id(series):
    """Convert storm identifiers to canonical form: storm<digits>."""
    s = series.astype(str).str.strip()
    has_prefix = s.str.lower().str.startswith("storm")
    out = pd.Series(index=s.index, dtype=object)

    prefixed = s[has_prefix].str.extract(r"(storm\d+)", expand=False)
    out.loc[has_prefix] = prefixed

    numeric = pd.to_numeric(s[~has_prefix], errors="coerce")
    out.loc[~has_prefix] = np.where(
        numeric.notna(),
        "storm" + numeric.astype("Int64").astype(str),
        pd.NA,
    )
    return out


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


    def select_top_unique_storms(pathway_df, rain_col="precipitation99", hail_col="max_hail_intensity", top_n=50):
        """Return the top-N unique storm_ids for rain and hail, deduplicated across both lists."""
        rain_sorted = pathway_df[pathway_df[rain_col].notna()].copy()
        hail_sorted = pathway_df[pathway_df[hail_col].notna()].copy()

        rain_sorted = rain_sorted.sort_values([rain_col, "pathway_prob"], ascending=[False, False])
        hail_sorted = hail_sorted.sort_values([hail_col, "pathway_prob"], ascending=[False, False])

        rain_unique = rain_sorted.drop_duplicates(subset=["storm_id"]).head(top_n).copy()
        hail_unique = hail_sorted.drop_duplicates(subset=["storm_id"]).head(top_n).copy()

        rain_unique["event_category"] = "RAIN"
        hail_unique["event_category"] = "HAIL"

        selected = pd.concat([rain_unique, hail_unique], ignore_index=True)
        selected["storm_id"] = selected["storm_id"].astype(str)
        selected["storm_id_norm"] = normalize_storm_id(selected["storm_id"])

        # Keep a single record per storm, preserving the strongest event information.
        selected = selected.sort_values(
            ["event_category", rain_col, hail_col, "pathway_prob"],
            ascending=[True, False, False, False],
            na_position="last",
        ).drop_duplicates(subset=["storm_id"], keep="first")

        selected_ids = selected["storm_id_norm"].dropna().astype(str).unique().tolist()
        return selected, sorted(selected_ids)
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

    # Put grid behind bars and use a light-grey guide grid for readability.
    axes_bar[0, c].set_axisbelow(True)
    axes_bar[0, c].grid(color='grey', alpha=0.35, linestyle='--')

    # Keep only essential axes (left and bottom), make them slightly bolder.
    axes_bar[0, c].spines['top'].set_visible(False)
    axes_bar[0, c].spines['right'].set_visible(False)
    axes_bar[0, c].spines['bottom'].set_linewidth(1.5)
    axes_bar[0, c].spines['left'].set_linewidth(1.5)

    # Use smaller, regular-weight tick labels.
    axes_bar[0, c].tick_params(axis='both', which='major', labelsize=11)
    for tick in axes_bar[0, c].get_xticklabels() + axes_bar[0, c].get_yticklabels():
        tick.set_fontweight('normal')

    

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
fig_bar.text(-0.12, 0.5, r"Count ($10^{3}$)", va="center",
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


# ================================================
# MULTIPLOT: FEATURE SPACE + cosine similarity
# ================================================
multiplot_labels = [
    lbl for lbl in labels_ordered
    if CLOUD_CLASS_INFO[lbl]["short"] in {"EC", "DC", "OA"}
]
multiplot_short = [CLOUD_CLASS_INFO[lbl]["short"] for lbl in multiplot_labels]
multiplot_colors = [CLOUD_CLASS_INFO[lbl]["color"] for lbl in multiplot_labels]

distance_train = df[df["vector_type"] == "TRAIN"].copy()
distance_test = df[df["vector_type"] != "TRAIN"].copy()
for frame in (distance_train, distance_test):
    frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")

distance_train = distance_train[(distance_train["distance"] > 0) & (distance_train["label"] != -100)].copy()
distance_test = distance_test[distance_test["distance"] > 0].copy()

feature_x = pd.concat([
    pd.to_numeric(df_train["tsne_dim_1"], errors="coerce"),
    pd.to_numeric(df_centroids["tsne_dim_1"], errors="coerce"),
], ignore_index=True).dropna()
feature_y = pd.concat([
    pd.to_numeric(df_train["tsne_dim_2"], errors="coerce"),
    pd.to_numeric(df_centroids["tsne_dim_2"], errors="coerce"),
], ignore_index=True).dropna()

if len(feature_x) > 0 and len(feature_y) > 0:
    x_pad = (feature_x.max() - feature_x.min()) * 0.05
    y_pad = (feature_y.max() - feature_y.min()) * 0.05
    feature_xlim = (feature_x.min() - x_pad, feature_x.max() + x_pad)
    feature_ylim = (feature_y.min() - y_pad, feature_y.max() + y_pad)
else:
    feature_xlim = None
    feature_ylim = None

distance_bins = np.linspace(0.4, 1.0, 45)

fig_multi, axes_multi = plt.subplots(
    2,
    3,
    figsize=(14.0, 7.8),
    sharex=False,
    sharey=False,
    gridspec_kw={"height_ratios": [1.0, 0.5]},
)
axes_multi = np.atleast_2d(axes_multi)

for col, label in enumerate(multiplot_labels):
    short_name = CLOUD_CLASS_INFO[label]["short"]
    class_color = CLOUD_CLASS_INFO[label]["color"]

    # Top row: feature space with train background, centroids, and class contours.
    ax_fs = axes_multi[0, col]
    plot_feature_space_dots(
        ax_fs,
        df_train,
        xcol="tsne_dim_1",
        ycol="tsne_dim_2",
        label_col="label",
        class_colors=CLOUD_CLASS_INFO,
        s=5,
        alpha=0.1,
        rasterized=True,
    )

    for _, row in df_centroids.iterrows():
        centroid_color = CLOUD_CLASS_INFO[int(row["label"])]
        ax_fs.scatter(
            row["tsne_dim_1"],
            row["tsne_dim_2"],
            color=centroid_color["color"],
            marker="^",
            s=100,
            edgecolor="black",
            linewidth=1,
            zorder=5,
        )

    df_class_test = df_test_label[df_test_label["label"] == label].dropna(subset=tsne_cols).copy()
    plot_density_contours(
        ax_fs,
        df_class_test,
        xcol="tsne_dim_1",
        ycol="tsne_dim_2",
        percentiles=(50, 90),
        bandwidth=0.25,
        color="black",
        linewidths=(2.0, 2.0),
        alpha=1.0,
        linestyles_list=["solid", "dashed"],
    )

    if feature_xlim is not None:
        ax_fs.set_xlim(*feature_xlim)
    if feature_ylim is not None:
        ax_fs.set_ylim(*feature_ylim)

    ax_fs.set_xticks([])
    ax_fs.set_yticks([])
    ax_fs.set_xlabel("")
    ax_fs.set_ylabel("")
    ax_fs.spines["top"].set_visible(False)
    ax_fs.spines["right"].set_visible(False)
    ax_fs.spines["left"].set_visible(False)
    ax_fs.spines["bottom"].set_visible(False)
    ax_fs.set_title(f"{chr(ord('c') + col)}) {short_name} feature space", fontsize=13, fontweight="bold")

    # Bottom row: cosine similarity distributions for train vs test.
    ax_dist = axes_multi[1, col]
    train_vals = pd.to_numeric(
        distance_train[distance_train["label"] == label]["distance"],
        errors="coerce",
    ).dropna()
    test_vals = pd.to_numeric(
        distance_test[distance_test["label"] == label]["distance"],
        errors="coerce",
    ).dropna()

    if len(train_vals) > 0:
        ax_dist.hist(
            train_vals,
            bins=distance_bins,
            histtype="step",
            density=True,
            color=class_color,
            linestyle=":",
            linewidth=2.0,
        )

    if len(test_vals) > 0:
        ax_dist.hist(
            test_vals,
            bins=distance_bins,
            histtype="step",
            density=True,
            color=class_color,
            linestyle="-",
            linewidth=2.8,
        )

    ax_dist.set_xlim(distance_bins.min(), distance_bins.max())
    ax_dist.set_xlabel("cosine similarity", fontsize=13)
    if col == 0:
        ax_dist.set_ylabel("Probability density", fontsize=13)
    else:
        ax_dist.set_ylabel("")
    ax_dist.set_title(f"{chr(ord('f') + col)}) {short_name} cosine similarity", fontsize=13, fontweight="bold")
    ax_dist.grid(color="grey", alpha=0.25, linestyle="--")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.spines["bottom"].set_linewidth(1.5)
    ax_dist.spines["left"].set_linewidth(1.5)
    ax_dist.tick_params(axis="both", which="major", labelsize=13)

line_handles = [
    Line2D([0], [0], color="black", linestyle=":", linewidth=2.0, label="Train"),
    Line2D([0], [0], color="black", linestyle="-", linewidth=2.8, label="Test"),
]
fig_multi.suptitle(
    "",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

fig_multi.subplots_adjust(wspace=0.25, hspace=0.28)

axes_multi[1, -1].legend(
    handles=line_handles,
    loc="upper left",
    fontsize=13,
    frameon=False,
    #title="Train / Test",
    title_fontsize=13,
)

outname_multi = "feature_space_and_distance_multiplot.png"
fig_multi.savefig(
    os.path.join(plot_dir, outname_multi),
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

print("✅ Saved:", outname_multi)


plt.close()


def build_summary_figures(test_df_source, output_suffix, title_suffix):
    """Build the same summary figures using a restricted TEST subset."""
    # -------------------------
    # FEATURE SPACE / BAR FIGURES
    # -------------------------
    fig_fs_local, axes_fs_local = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        sharex=True, sharey=True,
    )
    fig_bar_local, axes_bar_local = plt.subplots(
        n_rows, n_cols,
        figsize=(2 * n_cols, 1 * n_rows),
        sharex=True, sharey=True,
    )
    axes_fs_local = np.atleast_2d(axes_fs_local)
    axes_bar_local = np.atleast_2d(axes_bar_local)

    df_lat_local = test_df_source.copy()
    for c, ginfo in enumerate(groups_dict):
        event_name = ginfo["name"]
        if event_name == "ALL":
            df_group = df_lat_local.copy()
        else:
            df_group = df_lat_local[df_lat_local["storm_type"] == event_name].copy()

        plot_feature_space_dots(
            axes_fs_local[0, c],
            df_train,
            xcol="tsne_dim_1",
            ycol="tsne_dim_2",
            label_col="label",
            class_colors=CLOUD_CLASS_INFO,
            s=10,
            alpha=0.1,
            rasterized=True,
        )

        for _, row in df_centroids.iterrows():
            axes_fs_local[0, c].scatter(
                row["tsne_dim_1"],
                row["tsne_dim_2"],
                color=CLOUD_CLASS_INFO[int(row["label"])] ["color"],
                marker="^",
                s=300,
                edgecolor="black",
                linewidth=1,
                zorder=5,
            )

        plot_density_contours(
            axes_fs_local[0, c],
            df_group,
            xcol="tsne_dim_1",
            ycol="tsne_dim_2",
            percentiles=percentiles,
            bandwidth=0.25,
            color="black",
            linewidths=(3.0, 3.0, 3.0),
            alpha=1.0,
            linestyles_list=linestyles_list,
        )

        axes_fs_local[0, c].set_xticks([])
        axes_fs_local[0, c].set_yticks([])
        axes_fs_local[0, c].set_xlabel("")
        axes_fs_local[0, c].set_ylabel("")
        axes_fs_local[0, c].spines["top"].set_visible(False)
        axes_fs_local[0, c].spines["right"].set_visible(False)
        axes_fs_local[0, c].spines["left"].set_visible(False)
        axes_fs_local[0, c].spines["bottom"].set_visible(False)
        #axes_fs_local[0, c].set_title("a) All samples", fontsize=13, fontweight="bold")

        counts = (
            df_group["label"]
            .astype(int)
            .value_counts()
            .reindex(labels_ordered, fill_value=0)
        )
        x = np.arange(len(labels_ordered))
        axes_bar_local[0, c].bar(
            x,
            counts.values * 1e-3,
            color=colors_ordered,
            alpha=0.85,
        )
        axes_bar_local[0, c].set_xticks(x)
        axes_bar_local[0, c].set_xticklabels(short_labels, fontsize=11, rotation=45, ha="right")
        axes_bar_local[0, c].set_axisbelow(True)
        axes_bar_local[0, c].grid(color="grey", alpha=0.35, linestyle="--")
        axes_bar_local[0, c].spines["top"].set_visible(False)
        axes_bar_local[0, c].spines["right"].set_visible(False)
        axes_bar_local[0, c].spines["bottom"].set_linewidth(1.5)
        axes_bar_local[0, c].spines["left"].set_linewidth(1.5)
        axes_bar_local[0, c].tick_params(axis="both", which="major", labelsize=11)
        for tick in axes_bar_local[0, c].get_xticklabels() + axes_bar_local[0, c].get_yticklabels():
            tick.set_fontweight("normal")
        axes_bar_local[0, c].set_yticks([0, 5, 10, 15])

    fig_bar_local.text(-0.12, 0.5, r"Count ($10^{3}$)", va="center", rotation="vertical", fontsize=10)
    fig_fs_local.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="solid", linewidth=2.0),
            Line2D([0], [0], color="black", linestyle="dashed", linewidth=2.0),
        ],
        labels=["50th percentile", "90th percentile"],
        loc="center right",
        bbox_to_anchor=(1.2, 0.75),
        fontsize=11,
        title="Density contours",
        title_fontsize=11,
        frameon=False,
    )
    fig_fs_local.tight_layout()
    fig_fs_local.savefig(
        os.path.join(plot_dir, f"feature_space_region_event_{output_suffix}.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    fig_bar_local.subplots_adjust(wspace=0.15, hspace=0.15)
    fig_bar_local.tight_layout()
    fig_bar_local.savefig(
        os.path.join(plot_dir, f"group_population_by_label_latitude_{output_suffix}.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    # -------------------------
    # MULTIPLOT FIGURE
    # -------------------------
    distance_test_local = test_df_source.copy()
    for frame in (distance_train, distance_test_local):
        frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce")
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce")

    distance_test_local = distance_test_local[distance_test_local["distance"] > 0].copy()

    feature_x = pd.concat([
        pd.to_numeric(df_train["tsne_dim_1"], errors="coerce"),
        pd.to_numeric(df_centroids["tsne_dim_1"], errors="coerce"),
    ], ignore_index=True).dropna()
    feature_y = pd.concat([
        pd.to_numeric(df_train["tsne_dim_2"], errors="coerce"),
        pd.to_numeric(df_centroids["tsne_dim_2"], errors="coerce"),
    ], ignore_index=True).dropna()

    if len(feature_x) > 0 and len(feature_y) > 0:
        x_pad = (feature_x.max() - feature_x.min()) * 0.05
        y_pad = (feature_y.max() - feature_y.min()) * 0.05
        feature_xlim = (feature_x.min() - x_pad, feature_x.max() + x_pad)
        feature_ylim = (feature_y.min() - y_pad, feature_y.max() + y_pad)
    else:
        feature_xlim = None
        feature_ylim = None

    distance_bins = np.linspace(0.4, 1.0, 45)
    fig_multi_local, axes_multi_local = plt.subplots(
        2,
        3,
        figsize=(13.4, 7.0),
        sharex=False,
        sharey=False,
        gridspec_kw={"height_ratios": [0.85, 0.38]},
    )
    axes_multi_local = np.atleast_2d(axes_multi_local)

    for col, label in enumerate(multiplot_labels):
        short_name = CLOUD_CLASS_INFO[label]["short"]
        class_color = CLOUD_CLASS_INFO[label]["color"]

        ax_fs = axes_multi_local[0, col]
        plot_feature_space_dots(
            ax_fs,
            df_train,
            xcol="tsne_dim_1",
            ycol="tsne_dim_2",
            label_col="label",
            class_colors=CLOUD_CLASS_INFO,
            s=5,
            alpha=0.1,
            rasterized=True,
        )

        for _, row in df_centroids.iterrows():
            centroid_color = CLOUD_CLASS_INFO[int(row["label"])]
            ax_fs.scatter(
                row["tsne_dim_1"],
                row["tsne_dim_2"],
                color=centroid_color["color"],
                marker="^",
                s=100,
                edgecolor="black",
                linewidth=1,
                zorder=5,
            )

        df_class_test = distance_test_local[distance_test_local["label"] == label].dropna(subset=tsne_cols).copy()
        plot_density_contours(
            ax_fs,
            df_class_test,
            xcol="tsne_dim_1",
            ycol="tsne_dim_2",
            percentiles=(50, 90),
            bandwidth=0.25,
            color="black",
            linewidths=(2.6, 2.6),
            alpha=1.0,
            linestyles_list=["solid", "dashed"],
        )

        if feature_xlim is not None:
            ax_fs.set_xlim(*feature_xlim)
        if feature_ylim is not None:
            ax_fs.set_ylim(*feature_ylim)
        ax_fs.set_xticks([])
        ax_fs.set_yticks([])
        ax_fs.set_xlabel("")
        ax_fs.set_ylabel("")
        ax_fs.spines["top"].set_visible(False)
        ax_fs.spines["right"].set_visible(False)
        ax_fs.spines["left"].set_visible(False)
        ax_fs.spines["bottom"].set_visible(False)
        ax_fs.set_title(f"{chr(ord('c') + col)}) {short_name}", fontsize=13, fontweight="bold")

        ax_dist = axes_multi_local[1, col]
        train_vals = pd.to_numeric(
            distance_train[distance_train["label"] == label]["distance"],
            errors="coerce",
        ).dropna()
        test_vals = pd.to_numeric(
            distance_test_local[distance_test_local["label"] == label]["distance"],
            errors="coerce",
        ).dropna()

        if len(train_vals) > 0:
            ax_dist.hist(
                train_vals,
                bins=distance_bins,
                histtype="step",
                density=True,
                color=class_color,
                linestyle=":",
                linewidth=2.0,
            )
        if len(test_vals) > 0:
            ax_dist.hist(
                test_vals,
                bins=distance_bins,
                histtype="step",
                density=True,
                color=class_color,
                linestyle="-",
                linewidth=2.8,
            )

        ax_dist.set_xlim(distance_bins.min(), distance_bins.max())
        ax_dist.set_xlabel("cosine similarity", fontsize=13)
        if col == 0:
            ax_dist.set_ylabel("Probability density", fontsize=13)
        else:
            ax_dist.set_ylabel("")
        ax_dist.set_title(f"{chr(ord('f') + col)}) {short_name} cosine similarity", fontsize=13, fontweight="bold")
        ax_dist.grid(color="grey", alpha=0.25, linestyle="--")
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["right"].set_visible(False)
        ax_dist.spines["bottom"].set_linewidth(1.5)
        ax_dist.spines["left"].set_linewidth(1.5)
        ax_dist.tick_params(axis="both", which="major", labelsize=13)

    fig_multi_local.subplots_adjust(wspace=0.18, hspace=0.20)
    axes_multi_local[1, -1].legend(
        handles=[
            Line2D([0], [0], color="black", linestyle=":", linewidth=2.0, label="Train"),
            Line2D([0], [0], color="black", linestyle="-", linewidth=2.8, label="Test"),
        ],
        loc="upper left",
        fontsize=13,
        frameon=False,
        title_fontsize=13,
    )
    fig_multi_local.savefig(
        os.path.join(plot_dir, f"feature_space_and_distance_multiplot_{output_suffix}.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )


build_summary_figures(
    df_test_label,
    output_suffix="all_test_samples",
    title_suffix="All Test Samples",
)