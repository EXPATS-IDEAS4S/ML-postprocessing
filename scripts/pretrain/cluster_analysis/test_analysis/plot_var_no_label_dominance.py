import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import cmcrameri.cm as cmc
import seaborn as sns

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

# styles for all data (no dominance split)
styles = {
    "all": dict(alpha=0.8, lw=2, ls='-'),
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
#get shprt names ordered (all labels)
short_ordered = [short[key] for key in labels_ordered]


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
fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(
    nrows=3,
    ncols=3,
    height_ratios=[1, 1, 0.75],
    hspace=0.45,
    wspace=0.3
)

# Define axes
# Row 0: Feature space, hourly, CC histogram
ax_tsne = fig.add_subplot(gs[0, 0])  # Feature space
ax_hour = fig.add_subplot(gs[0, 1])  # Hourly distribution
ax_cc = fig.add_subplot(gs[0, 2])     # CC histogram
# Row 1: Scatter plots
ax_scatter = [fig.add_subplot(gs[1, i]) for i in range(3)]  # Scatter plots
# Row 2: Dominance distributions (one per class)
ax_dom = [fig.add_subplot(gs[2, i]) for i in range(3)]  # Dominance distributions

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

    x_data = df_lab[x_var] * scale_factor_var.get(x_var, 1.0)
    y_data = df_lab[y_var] * scale_factor_var.get(y_var, 1.0)

    xr = x_range if x_range is not None else (x_data.min(), x_data.max())
    yr = y_range if y_range is not None else (y_data.min(), y_data.max())

    h_data, _, _ = np.histogram2d(
        x_data, y_data, bins=bins, range=[xr, yr], density=True
    )
    global_vmax = max(global_vmax, np.nanmax(h_data))

if global_vmax == 0:
    global_vmax = 1.0

cmap = cmc.lipari
cmap.set_bad(color="white", alpha=0.0)

# ========================================================== 
# ROW 0: DOMINANCE DISTRIBUTION BOXPLOTS
# ==========================================================
# Melt wperc columns for boxplot visualization
df_melted = df[df['label'].isin(interesting_labels)].melt(
    id_vars=["filename", "label"],
    value_vars=wperc_cols,
    var_name="neighbor_label",
    value_name="probability"
)
print(df_melted['label'].value_counts())


# Create palette for all labels
palette = {
    f"wperc_label_{k}": v
    for k, v in colors.items()
}

for col, label in enumerate(interesting_labels):
    class_name = short[label]  # Convert numeric label to class name (e.g., 1 -> "EC")
    df_cls = df_melted[df_melted['label'] == label]
    print(df_cls.head())
    
    # Create order based on labels_ordered
    order = [f"wperc_label_{l}" for l in labels_ordered]
    
    sns.boxplot(
        data=df_cls,
        x="neighbor_label",
        y="probability",
        hue="neighbor_label",
        ax=ax_dom[col],
        palette=palette,
        legend=False,
        showfliers=False,
        width=0.6,
        order=order
    )
    
    ax_dom[col].set_xlabel("", fontsize=11)
    if col == 0:
        ax_dom[col].set_ylabel("Dominance (%)", fontsize=11, labelpad=10)
    # Add letter title (g, h, i) along with class name
    letters = ["g)", "h)", "i)"]
    ax_dom[col].set_title(
        f"{letters[col]} {LABEL_NAME_MAP.get(class_name, class_name)}", 
        fontsize=12, 
        fontweight="bold"
    )
    # Set x-axis labels in short_ordered order
    ax_dom[col].set_xticks(np.arange(len(labels_ordered)))
    ax_dom[col].set_xticklabels(
        [s.upper() for s in short_ordered],
        rotation=45,
        fontsize=10
    )
    # ax_dom[col].axhline(
    #     75, 
    #     color="red", 
    #     linestyle="--", 
    #     linewidth=1.5, 
    #     alpha=0.7
    # )
    ax_dom[col].grid(True, axis="y", alpha=0.3)

# -------------------------
# LOOP OVER LABELS TO COLLECT DATA AND CREATE PLOTS
# -------------------------

# ==========================================================
# PLOT 1: FEATURE SPACE WITH ALL CONTOURS
# ==========================================================

# --- background: TRAIN vectors ---
for lbl in labels_ordered:
    alpha = 0.03 if lbl not in interesting_labels else 0.2
    df_bg = df_tsne_train[df_tsne_train["label"] == lbl]
    ax_tsne.scatter(
        df_bg["tsne_dim_1"],
        df_bg["tsne_dim_2"],
        s=1,
        color=colors[lbl],
        alpha=0.01,
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

# --- density contours for each class ---
for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue
    
    plot_tsne_density_percentile(
        ax_tsne,
        df_lab["tsne_dim_1"].values,
        df_lab["tsne_dim_2"].values,
        percentile=75,
        colors= colors[label],
        linewidths=2.5,
        linestyles="-"
    )

ylim = ax_tsne.get_ylim()
#shift the plot mre towards the bottom
ax_tsne.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.1)
ax_tsne.set_xticks([])
ax_tsne.set_yticks([])
ax_tsne.set_title("a) Feature Space", fontsize=13, fontweight="bold")

# ==========================================================
# PLOT 2: HOURLY OCCURRENCE WITH ALL CLASSES
# ==========================================================

for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue
    
    hour_counts = df_lab['hour'].value_counts().sort_index()
    hours = np.arange(0, 24)
    counts = [hour_counts.get(h, 0) for h in hours]

    ax_hour.plot(
        hours,
        counts,
        color=colors[label],
        label=LABEL_NAME_MAP[short[label]],
        **styles["all"]
    )

ax_hour.set_xlabel("Hour (UTC)", fontsize=12)
ax_hour.set_ylabel("Counts", fontsize=12)
ax_hour.set_xticks(np.arange(0, 25, 3))
ax_hour.grid(True, which="both", ls="--", alpha=0.5)
ax_hour.tick_params(axis="both", which="major", labelsize=10)
ax_hour.set_yscale("log")
ax_hour.set_yticks([10, 100, 1000])
ax_hour.set_ylim(5, None)
#ax_hour.legend(fontsize=9, loc="upper right")
ax_hour.set_title("b) Hourly Occurrence", fontsize=13, fontweight="bold")

# ==========================================================
# PLOT 3: CC HISTOGRAM WITH ALL CLASSES
# ==========================================================

var = "cma"
for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue
    
    data = df_lab[var].dropna()
    if len(data) >= 5:
        ax_cc.hist(
            data * scale_factor_var.get(var, 1.0),
            bins=30,
            histtype="step",
            alpha=0.8,
            lw=2,
            color=colors[label],
            label=LABEL_NAME_MAP[short[label]]
        )

xmin, xmax = var_xlim[var]
ax_cc.set_xlim(xmin, xmax)
ticks = np.linspace(xmin, xmax, 6)[1:]
ax_cc.set_xticks(ticks)
ax_cc.set_xticklabels([f"{t:.2f}" for t in ticks])
ax_cc.set_xlabel(var_names.get(var, var), fontsize=12)
ax_cc.set_ylabel("Counts", fontsize=12)
ax_cc.set_yscale("log")
ax_cc.set_yticks([10, 100, 1000])
ax_cc.set_ylim(5, None)
ax_cc.grid(True, which="both", ls="--", alpha=0.5)
ax_cc.tick_params(axis="both", which="major", labelsize=10)
#ax_cc.legend(fontsize=9, loc="upper right")
ax_cc.set_title("c) Cloud Cover Distribution", fontsize=13, fontweight="bold")

# ==========================================================
# PLOT 4-6: SCATTER PLOTS (ONE PER CLASS)
# ==========================================================

last_mappable = None
titles = ["d) Early Convection", "e) Deep Convection", "f) Overcast Anvil"]

for col, label in enumerate(interesting_labels):
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue

    x_data = df_lab[x_var] * scale_factor_var.get(x_var, 1.0)
    y_data = df_lab[y_var] * scale_factor_var.get(y_var, 1.0)

    xr = x_range if x_range is not None else (x_data.min(), x_data.max())
    yr = y_range if y_range is not None else (y_data.min(), y_data.max())

    h_data, xedges, yedges = np.histogram2d(
        x_data, y_data, bins=bins, range=[xr, yr], density=True
    )

    h_data = np.ma.masked_where(h_data == 0, h_data)

    mappable = ax_scatter[col].pcolormesh(
        xedges,
        yedges,
        h_data.T,
        cmap=cmap,
        vmin=0,
        vmax=15,
        shading="auto",
    )
    last_mappable = mappable

    if x_var in var_xlim:
        ax_scatter[col].set_xlim(*var_xlim[x_var])
    if y_var in var_xlim:
        ax_scatter[col].set_ylim(*var_xlim[y_var])
    ax_scatter[col].set_xlabel(var_names.get(x_var, x_var), fontsize=12)
    ax_scatter[col].grid(True, which="both", ls="--", alpha=0.5)
    ax_scatter[col].tick_params(axis="both", which="major", labelsize=10)
    
    if col == 0:
        ax_scatter[col].set_ylabel(
            f"{var_names.get(y_var, y_var)}",
            fontsize=12,
        )
    
    ax_scatter[col].set_title(titles[col], fontsize=13, fontweight="bold")


# Add colorbar for scatter plots
if last_mappable is not None:
    cbar = fig.colorbar(
        last_mappable,
        ax=ax_scatter,
        fraction=0.046,
        pad=0.02,
        orientation='vertical'
    )
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Density", fontsize=11)

# Save figure
plt.savefig(
    csv_path + "/var_distributions_by_label_no_dominance_redesigned.png",
    dpi=150,
    bbox_inches="tight"
)
