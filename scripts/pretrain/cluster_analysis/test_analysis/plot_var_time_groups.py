import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO

# -------------------------
# USER SETTINGS
# -------------------------
run_name = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
base_dir = f"/data1/fig/{run_name}/epoch_800/test_traj"
csv_path = f"{base_dir}/hypersphere_analysis"
csv_file = "test_vectors_neigh_with_stats.csv"

csv_tnse_file = "tsne_all_vectors_with_centroids.csv"

physical_vars = ["precipitation99", "euclid_msg_grid"]

var_names = {
    "precipitation99": "Rain Rate P99 (mm/h)", 
    "euclid_msg_grid": "Lightning Counts"
}
#set the xlim for each variable in a dict
var_xlim = {
    "precipitation99": (0, 70), 
    "euclid_msg_grid": (0, 13000)
}


wperc_cols = [
    'wperc_label_0', 'wperc_label_1', 'wperc_label_2',
    'wperc_label_3', 'wperc_label_4', 'wperc_label_5',
    'wperc_label_6'
]

# grayscale styles
styles = {
    #"all": dict(color="black", alpha=0.8, lw=2),
    "max_wperc>=75%": dict(alpha=0.5, lw=2, ls='-'),
    "max_wperc<75%": dict(alpha=0.9, lw=3, ls=':'),
}

labels_ordered = sorted(
    CLOUD_CLASS_INFO.keys(),
    key=lambda x: CLOUD_CLASS_INFO[x]["order"]
)
print(labels_ordered)

colors = {l: CLOUD_CLASS_INFO[l]["color"] for l in labels_ordered}
short = {l: CLOUD_CLASS_INFO[l]["short"] for l in labels_ordered}
print(colors)
print(short)

interesting_labels = [1, 2, 4]   # <-- your 3 labels
INTERESTING_CLASSES = ["EC", "DC", "OA"]
LABEL_NAME_MAP = {
    "EC": "Early Convection",
    "DC": "Deep Convection",
    "OA": "Overcast Anvil"
}
#get labels ordered interesting from short dict {3: 'CS', 0: 'SC', 6: 'MC1', 5: 'MC2', 1: 'EC', 2: 'DC', 4: 'OA'}
label_interesting_short = [
    key for key, lbl in short.items()
    if lbl in INTERESTING_CLASSES
]
colors_interesting = {k: colors[k] for k in label_interesting_short}
print(label_interesting_short)
print(colors_interesting)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(csv_path + "/" + csv_file)
df["storm_id"] = df["filename"].str.split("_").str[0]
df["datetime"] = pd.to_datetime(df["filename"].str.split("_").str[1], format="%Y-%m-%dT%H-%M")
print(df.columns.tolist()) 

#extract datetime from filename column and create new column 'datetime' in df
aligned_times = []

for storm_id, g in df.groupby("storm_id"):
    g = g.sort_values("datetime")
    t0 = g["datetime"].iloc[0]
    t1 = g["datetime"].iloc[-1]
    t_center = t0 + (t1 - t0) / 2

    aligned = (g["datetime"] - t_center).dt.total_seconds() / 3600.0
    aligned_times.append(aligned)

df["t_aligned"] = pd.concat(aligned_times).sort_index()

BIN_HOURS = 1.0
df["t_bin"] = (df["t_aligned"] / BIN_HOURS).round() * BIN_HOURS

#get max and min t_bin
t_bin_min = df["t_bin"].min()
t_bin_max = df["t_bin"].max()
print(f"t_bin_min: {t_bin_min}, t_bin_max: {t_bin_max}")

#create a new column 'time_group' based on t_bin values
def assign_time_group(t_bin):
    if t_bin < -2:
        return "pre-2h"
    elif -2 <= t_bin <= 2:
        return "center"
    else:  # t_bin > 2
        return "post-2h"

df["time_group"] = df["t_bin"].apply(assign_time_group)

df_tsne = pd.read_csv(base_dir + "/" + csv_tnse_file)

df_tsne_centroids = df_tsne[df_tsne["vector_type"] == "CENTROID"]
#df_tsne_test = df_tsne[(df_tsne["vector_type"] != "TRAIN ") & (df_tsne["vector_type"] != "CENTROID")]
df_tsne_train = df_tsne[df_tsne["vector_type"] == "TRAIN"]


def plot_tsne_density_percentile(
    ax,
    x,
    y,
    percentile=75,
    gridsize=200,
    **kwargs
):
    if len(x) < 10:
        return

    values = np.vstack([x, y])
    kde = gaussian_kde(values)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.mgrid[xmin:xmax:gridsize*1j,
                      ymin:ymax:gridsize*1j]

    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid).reshape(xx.shape)

    # --- compute probability mass ---
    z_flat = zz.ravel()
    idx = np.argsort(z_flat)[::-1]
    z_sorted = z_flat[idx]

    cumsum = np.cumsum(z_sorted)
    cumsum /= cumsum[-1]

    # density threshold at desired percentile
    z_thresh = z_sorted[np.searchsorted(cumsum, percentile / 100.0)]

    # plot single contour
    ax.contour(
        xx, yy, zz,
        levels=[z_thresh],
        **kwargs
    )



# -------------------------
# SETUP FIGURE
# -------------------------
n_labels = len(interesting_labels)
fig, axes = plt.subplots(
    nrows=4,
    ncols=n_labels,
    figsize=(3 * n_labels, 10),
    sharey="row",
    gridspec_kw={"height_ratios": [1.2, 0.75, 0.75, 0.75]}
)
#reduce space from first row and second row and increase space between second and third row
plt.subplots_adjust(hspace=0.45, wspace=0.05)

delta = 0.04  # figure fraction

for ax in axes[0, :]:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 - delta, pos.width, pos.height])


if n_labels == 1:
    axes = axes.reshape(2, 1)

#define groups based on time_group column
time_groups = ["pre-2h", "center", "post-2h"]

# -------------------------
# LOOP OVER LABELS
# -------------------------
for col, time_group in enumerate(time_groups):

    df_time = df[df["time_group"] == time_group].copy()
    if df_time.empty:
        continue

    # =========================
    # ROW 0 — TSNE FEATURE SPACE
    # =========================
    ax_tsne = axes[0, col]

    # --- background: TRAIN vectors ---
    for lbl in labels_ordered:
        df_bg = df_tsne_train[df_tsne_train["label"] == lbl]
        ax_tsne.scatter(
            df_bg["tsne_dim_1"],
            df_bg["tsne_dim_2"],
            s=3,
            color=colors[lbl],
            alpha=0.04,
            rasterized=True
        )

        # --- centroid ---
        df_cent = df_tsne_centroids[df_tsne_centroids["label"] == lbl]
        ax_tsne.scatter(
            df_cent["tsne_dim_1"],
            df_cent["tsne_dim_2"],
            marker="^",
            s=120,
            color=colors[lbl],
            edgecolor="black",
            zorder=5,
        )
    #increase slightly ylim of ax_tsne (only top value)
    ylim = ax_tsne.get_ylim()
    ax_tsne.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.005)

    # --- density contours ---
    plot_tsne_density_percentile(
        ax_tsne,
        df_time["tsne_dim_1"].values,
        df_time["tsne_dim_2"].values,
        percentile=50,
        colors="black",
        linewidths=2,
        linestyles="-"
    )

    plot_tsne_density_percentile(
        ax_tsne,
        df_time["tsne_dim_1"].values,
        df_time["tsne_dim_2"].values,
        percentile=90,
        colors="black",
        linewidths=2,
        linestyles="--"
    )

    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])
    if col == 0:
        ax_tsne.set_title(f"{time_group} \n a)", fontsize=12, fontweight="bold")
    elif col == 1:
        ax_tsne.set_title(f"{time_group} \n b)", fontsize=12, fontweight="bold")
    elif col == 2:
        ax_tsne.set_title(f"{time_group} \n c) ", fontsize=12, fontweight="bold")

    # -----------------------------------
    # BARPLOT: counts by class & dominance
    # -----------------------------------

    ax_count = axes[1, col]

    # compute dominant wperc
    df_time["max_wperc"] = df_time[wperc_cols].max(axis=1)
    median_wperc = 75

    df_high = df_time[df_time["max_wperc"] >= median_wperc]
    df_low  = df_time[df_time["max_wperc"] <  median_wperc]


    # bar positions
    #x = np.arange(len(label_interesting_short))
    bar_width = 0.35

    for l, label in enumerate(interesting_labels):
        # class-specific color
        cls_color = colors_interesting[label]

        # total counts (not hourly anymore)
        df_high_lbl = df_high[df_high["label"] == label]
        df_low_lbl  = df_low[df_low["label"] == label]
        count_high = len(df_high_lbl)
        count_low  = len(df_low_lbl)

        # plot bars
        ax_count.bar(
            l - bar_width / 2 ,
            count_high,
            width=bar_width,
            color=cls_color,
            alpha=styles["max_wperc>=75%"]["alpha"],
            edgecolor="black",
            linewidth=1.2,
            label="max_wperc ≥ 75%" if col == 0 else None,
        )

        ax_count.bar(
            l + bar_width / 2,
            count_low,
            width=bar_width,
            color=cls_color,
            alpha=styles["max_wperc<75%"]["alpha"],
            hatch="///",
            edgecolor="black",
            linewidth=1.2,
            label="max_wperc < 75%" if col == 0 else None,
        )

    # axis formatting
    ax_count.set_yscale("log")
    #ax_count.set_ylim(5, None)
    ax_count.set_yticks([10, 100, 1000])
    ax_count.set_yticklabels(["10", "100", "1000"])

    ax_count.grid(True, axis="y", which="both", ls="--", alpha=0.4)
    ax_count.tick_params(axis="both", labelsize=10)

    if col == 0:
        ax_count.set_ylabel("Counts", fontsize=12)
        ax_count.set_title("d)", fontsize=12, fontweight="bold")
    elif col == 1:
        ax_count.set_title("e)", fontsize=12, fontweight="bold")
    elif col == 2:
        ax_count.set_title("f)", fontsize=12, fontweight="bold")

    ax_count.set_xticks([l for l in range(len(label_interesting_short))])
    ax_count.set_xticklabels(INTERESTING_CLASSES, fontsize=12)

    #legend customized for the box plots
    if col == 0:
        from matplotlib.patches import Rectangle

        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='grey', alpha=0.5, edgecolor='black', linewidth=1.2, label='dominance>=75%'),
            Rectangle((0, 0), 1, 1, facecolor='grey', alpha=0.9, edgecolor='black', linewidth=1.2, hatch='///', label='dominance<75%')
        ]

        ax_count.legend(
            handles=legend_elements,
            title="",
            frameon=False,
            fontsize=8
        )


    for row, var in enumerate(physical_vars, start=2):
        ax = axes[row, col]


        data = df_time[var].dropna()

        ax.hist(
            data,
            bins=30,
            #density=True,
            histtype="step",
            color='darkblue',
        )

        if col == 0:
            ax.set_ylabel("Counts", fontsize=12)
            #ax.set_title("d)", fontsize=12, fontweight="bold")
        # elif col == 1:
        #     ax.set_title("e)", fontsize=12, fontweight="bold")
        # elif col == 2:
        #     ax.set_title("f)", fontsize=12, fontweight="bold")
        ax.set_xlabel(var_names.get(var, var), fontsize=12)
        ax.set_yscale("log")
        #set manuely ticks label of ry axis
        yticks = [10, 100, 1000]#, 10000]
        ax.set_yticks(yticks)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=10)
        #set xlim based on var_xlim dict
        #if var in var_xlim:
        #    ax.set_xlim(var_xlim[var])


# -------------------------
# LEGEND (only once) sing lines
# ------------------------

# legend_elements = [
#     Line2D([0], [0], color='black', lw=2, ls='-', alpha=0.5, label='max_wperc>=75%'),
#     Line2D([0], [0], color='black', lw=3, ls=':', alpha=0.9, label='max_wperc<75%')
# ]

# axes[3, 2].legend(
#     handles=legend_elements,
#     title="",
#     frameon=False,
#     fontsize=10
# )

# plt.suptitle(
#     "Physical variable distributions\n"
#     "Split by weighted label dominance",
#     fontsize=14, y=1.02
# )

plt.savefig(
    csv_path + "/var_distributions_by_time_groups.png",
    dpi=150,
    bbox_inches="tight"
)
