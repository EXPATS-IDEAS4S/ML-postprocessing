import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import cmcrameri.cm as cmc

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO

# -------------------------
# USER SETTINGS
# -------------------------
run_name = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
base_dir = f"/data1/fig/{run_name}/epoch_800/test_traj"
csv_path = f"{base_dir}/hypersphere_analysis"
csv_file = "test_vectors_neigh_with_stats.csv"
cot_file = f"/data1/fig/{run_name}/epoch_800/all/merged_crops_stats_all_cvc_cot_fractions.csv"
csv_tnse_file = "tsne_all_vectors_with_centroids.csv"


physical_vars_distr = ["cma"]
physical_var_scatter = ["cth_very_high", 'cot_thick' ]

var_names = {
    #"precipitation99": "Rain Rate P99 (mm/h)", 
    #"euclid_msg_grid": "Lightning Counts"
    "cma": "CC",
    "cth50": "CTH Median (km)",
    "cth99": "CTH 99th Percentile (km)",
    "cot_thick": "COT 30+",
    "cth_very_high": "CTH 10km+",
}
#set the xlim for each variable in a dict
var_xlim = {
    "precipitation99": (0, 70), 
    "euclid_msg_grid": (0, 13000),
    "cma": (0, 1.0),
    "cth50": (0, 13),
    "cth99": (0, 15),
    "cot_thick": (0, 0.75),
    "cth_very_high": (0, 1),
}

scale_factor_var = {
    "precipitation99": 1.0, 
    "euclid_msg_grid": 1.0,
    "cma": 1.0,
    "cth50": 0.001,
    "cth99": 0.001,
    "cot_thick": 1.0,
    "cth_very_high": 1.0
}

letters_title = {
    "cma": ["g)", "h)", "i)"],
    "cth50": ["j)", "k)", "l)"],
    #"cth99": ["j)", "k)", "l)"],
    "cot_thick": ["m)", "n)", "o)"],
    "cth_very_high": ["p)", "q)", "r)"],
}


wperc_cols = [
    'wperc_label_0', 'wperc_label_1', 'wperc_label_2',
    'wperc_label_3', 'wperc_label_4', 'wperc_label_5',
    'wperc_label_6'
]

# grayscale styles
styles = {
    #"all": dict(color="black", alpha=0.8, lw=2),
    "dominance>=75%": dict(alpha=0.5, lw=2, ls='-'),
    "dominance<75%": dict(alpha=0.9, lw=3, ls=':'),
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
# print(df.columns.tolist()) 
# print(df)

# df_cot = pd.read_csv(cot_file)
# print(df_cot.columns.tolist())
# #rename column 'crop' in df_cot to 'filename' to merge with df
# df_cot = df_cot.rename(columns={"crop": "filename"})
# #select only row with var == 'cot_thick' and rename column 'None' to 'cot_thick'
# df_cot = df_cot[df_cot["var"] == "cot_thick"].copy()
# df_cot = df_cot.rename(columns={"None": "cot_thick"})
# print(df_cot.columns.tolist())
# print(df_cot)
# exit()

#extract datetime from filename column and create new column 'datetime' in df
print(df['filename'].head())
df['datetime'] = pd.to_datetime(df['filename'].str.split('_').str[1], format='%Y-%m-%dT%H-%M')
#get hour column from datetime
df['hour'] = df['datetime'].dt.hour
print(df.head())

# #check how many times the entry hour concide with the max perc_hour_* columns
# match_count = 0
# for idx, row in df.iterrows():
#     hour = row['hour']
#     max_hour_cols = ['perc_hour_0', 'perc_hour_1', 'perc_hour_2',
#                      'perc_hour_3', 'perc_hour_4', 'perc_hour_5',
#                      'perc_hour_6', 'perc_hour_7', 'perc_hour_8',
#                      'perc_hour_9', 'perc_hour_10', 'perc_hour_11',
#                      'perc_hour_12', 'perc_hour_13', 'perc_hour_14',
#                      'perc_hour_15', 'perc_hour_16', 'perc_hour_17',
#                      'perc_hour_18', 'perc_hour_19', 'perc_hour_20',
#                      'perc_hour_21', 'perc_hour_22', 'perc_hour_23']
#     match = False                 
#     for col in max_hour_cols: 
#         if row[col] == hour:
#             match = True
#             break
#     if not match:
#         print(f"Hour mismatch for index {idx}, hour: {hour}, row values: {row[max_hour_cols].values}")
#     else:
#         match_count += 1

# print(f"Total matches: {match_count}")

# exit()


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
    nrows=5,
    ncols=n_labels,
    figsize=(3 * n_labels, 13),
    sharey="row",
    gridspec_kw={"height_ratios": [1.2, 0.75, 0.75, 0.9, 0.9]}
)
#reduce space from first row and second row and increase space between second and third row
plt.subplots_adjust(hspace=0.55, wspace=0.1)

delta = 0.04  # figure fraction

for ax in axes[0, :]:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 - delta, pos.width, pos.height])


if n_labels == 1:
    axes = axes.reshape(5, 1)

# -------------------------
# PREPASS: GLOBAL DENSITY SCALE
# -------------------------
bins = 25
y_var = physical_var_scatter[0]
x_var = physical_var_scatter[1]
x_range = var_xlim.get(x_var, None)
y_range = var_xlim.get(y_var, None)

global_vmax = 0.0
for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue

    df_lab["max_wperc"] = df_lab[wperc_cols].max(axis=1)
    df_high = df_lab[df_lab["max_wperc"] >= 75]
    df_low = df_lab[df_lab["max_wperc"] < 75]

    x_high = df_high[x_var] * scale_factor_var.get(x_var, 1.0)
    y_high = df_high[y_var] * scale_factor_var.get(y_var, 1.0)
    x_low = df_low[x_var] * scale_factor_var.get(x_var, 1.0)
    y_low = df_low[y_var] * scale_factor_var.get(y_var, 1.0)

    xr = x_range if x_range is not None else (x_high.min(), x_high.max())
    yr = y_range if y_range is not None else (y_high.min(), y_high.max())

    h_high, _, _ = np.histogram2d(
        x_high, y_high, bins=bins, range=[xr, yr], density=True
    )
    h_low, _, _ = np.histogram2d(
        x_low, y_low, bins=bins, range=[xr, yr], density=True
    )
    global_vmax = max(global_vmax, np.nanmax(h_high), np.nanmax(h_low))

if global_vmax == 0:
    global_vmax = 1.0

cmap = cmc.lipari
cmap.set_bad(color="white", alpha=0.0)

# -------------------------
# LOOP OVER LABELS
# -------------------------
last_mappable = None
for col, label in enumerate(interesting_labels):
    color = colors[label]


    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue

    # compute dominant wperc
    df_lab["max_wperc"] = df_lab[wperc_cols].max(axis=1)
    print(df_lab["max_wperc"].median())
    median_wperc = 75# df_lab["max_wperc"].median()


    df_all = df_lab
    df_high = df_lab[df_lab["max_wperc"] >= median_wperc]
    df_low  = df_lab[df_lab["max_wperc"] <  median_wperc]


    # =========================
    # ROW 0 — TSNE FEATURE SPACE
    # =========================
    ax_tsne = axes[0, col]

    # --- background: TRAIN vectors ---
    for lbl in labels_ordered:
        if lbl == label:
            alpha = 0.5
        else:
            alpha = 0.03
        df_bg = df_tsne_train[df_tsne_train["label"] == lbl]
        ax_tsne.scatter(
            df_bg["tsne_dim_1"],
            df_bg["tsne_dim_2"],
            s=3,
            color=colors[lbl],
            alpha=alpha,
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
        df_high["tsne_dim_1"].values,
        df_high["tsne_dim_2"].values,
        percentile=75,
        colors="black",
        linewidths=2,
        linestyles="-"
    )

    plot_tsne_density_percentile(
        ax_tsne,
        df_low["tsne_dim_1"].values,
        df_low["tsne_dim_2"].values,
        percentile=75,
        colors="black",
        linewidths=2,
        linestyles=":"
    )


    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])
    if col == 0:
        ax_tsne.set_title(f"{LABEL_NAME_MAP[short[label]]} \n  a)", fontsize=12, fontweight="bold")
    elif col == 1:
        ax_tsne.set_title(f"{LABEL_NAME_MAP[short[label]]} \n b)", fontsize=12, fontweight="bold")
    elif col == 2:
        ax_tsne.set_title(f"{LABEL_NAME_MAP[short[label]]} \n c) ", fontsize=12, fontweight="bold")

    ax_hour = axes[1, col]
    #plot hourly count distribution for df_high and df_low
    for key, subset in zip(
        ["dominance>=75%", "dominance<75%"],
        [df_high, df_low]
    ):
        hour_counts = subset['hour'].value_counts().sort_index()
        hours = np.arange(0, 24)
        counts = [hour_counts.get(h, 0) for h in hours]

        ax_hour.plot(
            hours,
            counts,
            color=colors[label],
            label=key,
            **styles[key]
        )
    ax_hour.set_xlabel("Hour (UTC)", fontsize=12)
    if col == 0:
        ax_hour.set_ylabel("Counts", fontsize=12)# if col == 0 else None)
    ax_hour.set_xticks(np.arange(0, 25, 3))
    ax_hour.grid(True, which="both", ls="--", alpha=0.5)
    ax_hour.tick_params(axis="both", which="major", labelsize=10)
    ax_hour.set_yscale("log")
    #set manuely ticks label of ry axis
    yticks = [10, 100, 1000]
    ax_hour.set_yticks(yticks)
    if col == 0:
        ax_hour.set_title(f"d)", fontsize=12, fontweight="bold")
    elif col == 1:
        ax_hour.set_title(f"e)", fontsize=12, fontweight="bold")
    elif col == 2:
        ax_hour.set_title(f"f)", fontsize=12, fontweight="bold")
    
    #prepare letters for title starting from g) and 3 each var

    for row, var in enumerate(physical_vars_distr, start=2):
        ax = axes[row, col]
        letters = letters_title[var]
        var_name = var_names.get(var, var)

        # plot distributions
        for key, subset in zip(
            ["dominance>=75%", "dominance<75%"],
            [df_high, df_low]
        ):
            data = subset[var].dropna()
            if len(data) < 5:
                continue

            ax.hist(
                data * scale_factor_var.get(var, 1.0),
                bins=30,
                #density=True,
                histtype="step",
                **styles[key],
                color=colors[label],
                label=key if (row == 0 and col == 0) else None
            )

        
        if col == 0:
            ax.set_title(f"{letters[0]}", fontsize=12, fontweight="bold")
        elif col == 1:
            ax.set_title(f"{letters[1]}", fontsize=12, fontweight="bold")
        elif col == 2:
            ax.set_title(f"{letters[2]}", fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Counts", fontsize=12)

        #set ticks label uisg xlim from var_xlim dict if var in var_xlim,
        #plot 5 ticks evenly spaced between var_xlim[var][0] and var_xlim[var][1]
        xmin, xmax = var_xlim[var]
        ax.set_xlim(xmin, xmax)
        #ax.set_xticks(np.linspace(xmin, xmax, 5))
        ticks = np.linspace(xmin, xmax, 6)
        #remove first tick
        ticks = ticks[1:]
        ax.set_xticks(np.linspace(xmin, xmax, 5))
        ax.set_xticks(ticks)
        if var == "cma" or var == "cot_thick":
            ax.set_xticklabels([f"{t:.2f}" for t in ticks])
        else:
            #integer ticks
            ax.set_xticklabels([f"{int(t)}" for t in ticks])
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

    # =========================
    # LAST TWO ROWS — SCATTER PLOTS (HIGH/LOW)
    # =========================
    ax_scatter_high = axes[-2, col]
    ax_scatter_low = axes[-1, col]

    x_high = df_high[x_var] * scale_factor_var.get(x_var, 1.0)
    y_high = df_high[y_var] * scale_factor_var.get(y_var, 1.0)
    x_low = df_low[x_var] * scale_factor_var.get(x_var, 1.0)
    y_low = df_low[y_var] * scale_factor_var.get(y_var, 1.0)

    xr = x_range if x_range is not None else (x_high.min(), x_high.max())
    yr = y_range if y_range is not None else (y_high.min(), y_high.max())

    h_high, xedges, yedges = np.histogram2d(
        x_high, y_high, bins=bins, range=[xr, yr], density=True
    )
    h_low, _, _ = np.histogram2d(
        x_low, y_low, bins=bins, range=[xr, yr], density=True
    )

    h_high = np.ma.masked_where(h_high == 0, h_high)
    h_low = np.ma.masked_where(h_low == 0, h_low)

    mappable = ax_scatter_high.pcolormesh(
        xedges,
        yedges,
        h_high.T,
        cmap=cmap,
        vmin=0,
        vmax=15,
        shading="auto",
    )
    ax_scatter_low.pcolormesh(
        xedges,
        yedges,
        h_low.T,
        cmap=cmap,
        vmin=0,
        vmax=15,
        shading="auto",
    )
    last_mappable = mappable

    # mask_high = np.isfinite(x_high) & np.isfinite(y_high)
    # xh = x_high[mask_high]
    # yh = y_high[mask_high]
    # if xh.size > 1 and np.unique(xh).size > 1:
    #     coef_high = np.polyfit(xh, yh, 1)
    #     x_line = np.linspace(xr[0], xr[1], 200)
    #     y_line = coef_high[0] * x_line + coef_high[1]
    #     ax_scatter_high.plot(x_line, y_line, color="red", lw=2.0, zorder=5, ls="--")

    # mask_low = np.isfinite(x_low) & np.isfinite(y_low)
    # xl = x_low[mask_low]
    # yl = y_low[mask_low]
    # if xl.size > 1 and np.unique(xl).size > 1:
    #     coef_low = np.polyfit(xl, yl, 1)
    #     x_line = np.linspace(xr[0], xr[1], 200)
    #     y_line = coef_low[0] * x_line + coef_low[1]
    #     ax_scatter_low.plot(x_line, y_line, color="red", lw=2.0, zorder=5, ls="--")

    for ax_scatter in (ax_scatter_high, ax_scatter_low):
        if x_var in var_xlim:
            ax_scatter.set_xlim(*var_xlim[x_var])
        if y_var in var_xlim:
            ax_scatter.set_ylim(*var_xlim[y_var])
        ax_scatter.set_xlabel(var_names.get(x_var, x_var), fontsize=12)
        ax_scatter.grid(True, which="both", ls="--", alpha=0.5)
        ax_scatter.tick_params(axis="both", which="major", labelsize=10)


    if col == 0:
        ax_scatter_high.set_ylabel(
            f"{var_names.get(y_var, y_var)}",
            fontsize=12,
            #fontweight="bold"
        )
        ax_scatter_low.set_ylabel(
            f"{var_names.get(y_var, y_var)}",
            fontsize=12,
            #fontweight="bold"
        )

    if col == 0:
        ax_scatter_high.set_title("j) High", fontsize=12, fontweight="bold")
        ax_scatter_low.set_title("m) Low", fontsize=12, fontweight="bold")
    elif col == 1:
        ax_scatter_high.set_title("k) High", fontsize=12, fontweight="bold")
        ax_scatter_low.set_title("n) Low", fontsize=12, fontweight="bold")
    elif col == 2:
        ax_scatter_high.set_title("l) High", fontsize=12, fontweight="bold")
        ax_scatter_low.set_title("o) Low", fontsize=12, fontweight="bold")

if last_mappable is not None:
    cbar = fig.colorbar(
        last_mappable,
        ax=axes[-2:, :],
        fraction=0.046,
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Counts", fontsize=9)

        
# -------------------------
# LEGEND (only once) sing lines
# ------------------------

legend_elements = [
    Line2D([0], [0], color='black', lw=2, ls='-', alpha=0.5, label='dom>=75%'),
    Line2D([0], [0], color='black', lw=3, ls=':', alpha=0.9, label='dom<75%')
]

#add box around legend
axes[2, 2].legend(
    handles=legend_elements,
    title="",
    frameon=True,
    fontsize=10,
    loc='upper left'
)

# plt.suptitle(
#     "Physical variable distributions\n"
#     "Split by weighted label dominance",
#     fontsize=14, y=1.02
# )
#add physical variable names to filename
plt.savefig(
    csv_path + "/var_distributions_by_label_dominance_with_scatterplot" + ".png",
    dpi=150,
    bbox_inches="tight"
)
