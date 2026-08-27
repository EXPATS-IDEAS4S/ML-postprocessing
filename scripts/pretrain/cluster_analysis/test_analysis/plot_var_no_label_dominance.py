"""
Multi-panel comparison of TEST vs TRAIN physical-variable behavior.

Panels:
1) Hourly occurrence curves (TEST solid, TRAIN dashed) for EC/DC/OA.
2) CC distributions (log-scale histogram) for EC/DC/OA.
3) New row: CTH median vs COT median, shown as TEST 2D histogram per class,
   with TRAIN density contour overlaid for visual comparison.
4) Existing row: CTH10+ vs COT30+ TEST 2D histogram per class.

Each 2D-hist row has its own colorbar with an explicit label describing what the
colors represent: bin frequency as percentage of class samples.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
embedding_vectors_file = f"{base_dir}/features_train_test_{run_name}_2nd_labels.csv"
cot_file = f"/data1/fig/{run_name}/epoch_800/all/merged_crops_stats_all_cvc_cot_fractions.csv"
cot_percentiles_file = f"/data1/fig/{run_name}/epoch_800/test_traj/merged_crops_stats_cot_percentiles.csv"
csv_tnse_file = "tsne_all_vectors_with_centroids.csv"
train_cvc_file = f"/data1/fig/{run_name}/epoch_800/all/merged_crops_stats_all_cvc.csv"
train_fraction_file = f"/data1/fig/{run_name}/epoch_800/all/merged_crops_stats_all_cvc_cot_cth_fractions.csv"


physical_vars_distr = ["cma"]
physical_var_scatter = ["cth_very_high", 'cot_thick' ]
LOCAL_TRAIN_K = 10

var_names = {
    #"precipitation99": "Rain Rate P99 (mm/h)", 
    #"euclid_msg_grid": "Lightning Counts"
    "cma": "CC (%)",
    "cth50": "CTH Median (km)",
    "cth99": "CTH 99th Percentile (km)",
    "cot50": "COT Median",
    "cot_thick": "COT 30+ (%)",
    "cth_very_high": "CTH 10km+ (%)",
}
#set the xlim for each variable in a dict
var_xlim = {
    "precipitation99": (0, 70), 
    "euclid_msg_grid": (0, 13000),
    "cma": (0, 100.0),
    "cth50": (0, 13),
    "cth99": (0, 15),
    "cot50": (0, 150),
    "cot_thick": (0, 75),
    "cth_very_high": (0, 100),
}

scale_factor_var = {
    "precipitation99": 1.0, 
    "euclid_msg_grid": 1.0,
    "cma": 100.0,
    "cth50": 0.001,
    "cth99": 0.001,
    "cot50": 1.0,
    "cot_thick": 100.0,
    "cth_very_high": 100.0
}

AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 16
CBAR_TICK_FONTSIZE = 13
CBAR_LABEL_FONTSIZE = 14

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

# Attach COT median (P50) from precomputed percentile file.
df_cot_pct = pd.read_csv(cot_percentiles_file, low_memory=False)
if {"var", "50", "crop"}.issubset(df_cot_pct.columns):
    df_cot_pct = df_cot_pct[df_cot_pct["var"] == "cot_percentiles"].copy()
    df_cot_pct["filename"] = df_cot_pct["crop"].astype(str).str.split("/").str[-1]
    df_cot_pct["cot50"] = pd.to_numeric(df_cot_pct["50"], errors="coerce")
    df = df.merge(
        df_cot_pct[["filename", "cot50"]].drop_duplicates(subset=["filename"]),
        on="filename",
        how="left",
    )
else:
    print("Warning: cot percentile file missing expected columns (var, 50, crop)")

# Training set: use cma rows from merged all-cvc table
df_train_raw = pd.read_csv(train_cvc_file, low_memory=False)
if not {"label", "time", "var", "None"}.issubset(df_train_raw.columns):
    raise ValueError(
        "Training file missing required columns. Expected: label, time, var, None"
    )

df_train = df_train_raw[df_train_raw["var"] == "cma"].copy()
df_train = df_train.rename(columns={"None": "cma"})
df_train["datetime"] = pd.to_datetime(df_train["time"], errors="coerce")
df_train["hour"] = df_train["datetime"].dt.hour
df_train["label"] = pd.to_numeric(df_train["label"], errors="coerce")
df_train["cma"] = pd.to_numeric(df_train["cma"], errors="coerce")
df_train = df_train.dropna(subset=["label", "hour", "cma"])
df_train["label"] = df_train["label"].astype(int)

print(f"Loaded training cma rows: {len(df_train)}")

# Build TRAIN median scatter data (cth50 and cot50) from long-format table.
df_train_median_raw = df_train_raw[df_train_raw["var"].isin(["cth", "cot"])].copy()
df_train_median_raw["value50"] = pd.to_numeric(df_train_median_raw["50"], errors="coerce")
df_train_median = (
    df_train_median_raw
    .pivot_table(
        index=["crop", "time", "label"],
        columns="var",
        values="value50",
        aggfunc="first",
    )
    .reset_index()
)
df_train_median = df_train_median.rename(columns={"cth": "cth50", "cot": "cot50"})
df_train_median["label"] = pd.to_numeric(df_train_median["label"], errors="coerce")
df_train_median["cth50"] = pd.to_numeric(df_train_median["cth50"], errors="coerce")
df_train_median["cot50"] = pd.to_numeric(df_train_median["cot50"], errors="coerce")
df_train_median = df_train_median.dropna(subset=["label", "cth50", "cot50"])
df_train_median["label"] = df_train_median["label"].astype(int)

# Build TRAIN fraction scatter data (cth_very_high and cot_thick) from long-format table.
df_train_fraction_raw = pd.read_csv(train_fraction_file, low_memory=False)
if not {"label", "time", "var", "None"}.issubset(df_train_fraction_raw.columns):
    raise ValueError(
        "Training fraction file missing required columns. Expected: label, time, var, None"
    )

df_train_fraction_raw = df_train_fraction_raw[df_train_fraction_raw["var"].isin(["cth_very_high", "cot_thick"])].copy()
df_train_fraction_raw["value"] = pd.to_numeric(df_train_fraction_raw["None"], errors="coerce")
df_train_fraction = (
    df_train_fraction_raw
    .pivot_table(
        index=["crop", "time", "label"],
        columns="var",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)
df_train_fraction["label"] = pd.to_numeric(df_train_fraction["label"], errors="coerce")
df_train_fraction["cth_very_high"] = pd.to_numeric(df_train_fraction["cth_very_high"], errors="coerce")
df_train_fraction["cot_thick"] = pd.to_numeric(df_train_fraction["cot_thick"], errors="coerce")
df_train_fraction = df_train_fraction.dropna(subset=["label", "cth_very_high", "cot_thick"])
df_train_fraction["label"] = df_train_fraction["label"].astype(int)


def apply_axis_style(ax, tick_size=12):
    """Apply a clean visual style: only left/bottom spines and light grid."""
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.grid(color='grey', alpha=0.35, linestyle='--')


def relative_hist2d(x_data, y_data, bins, xr, yr):
    """Return 2D histogram as percentage of class samples per bin."""
    h_counts, xedges, yedges = np.histogram2d(
        x_data,
        y_data,
        bins=bins,
        range=[xr, yr],
        density=False,
    )
    if h_counts.sum() > 0:
        h_rel = (h_counts / h_counts.sum()) * 100.0
    else:
        h_rel = h_counts
    return h_rel, xedges, yedges


def plot_mean_cross(ax, x_value, y_value, color, size=320, line_width=3.0, marker="x"):
    """Plot a large mean cross if both coordinates are finite and positive when needed."""
    if not (np.isfinite(x_value) and np.isfinite(y_value)):
        return
    if x_value <= 0:
        return
    ax.scatter(
        [x_value],
        [y_value],
        marker=marker,
        s=size,
        c=color,
        linewidths=line_width,
        zorder=20,
    )


def print_mean_std_summary(
    title,
    test_df,
    train_df_cma,
    train_df_median,
    train_df_fraction,
    labels_to_report,
):
    """Print overall and per-class mean/std for plotted TEST vs TRAIN variables."""
    specs = [
        ("cma", test_df, train_df_cma, scale_factor_var.get("cma", 1.0)),
        ("cot50", test_df, train_df_median, scale_factor_var.get("cot50", 1.0)),
        ("cth50", test_df, train_df_median, scale_factor_var.get("cth50", 1.0)),
        ("cot_thick", test_df, train_df_fraction, scale_factor_var.get("cot_thick", 1.0)),
        ("cth_very_high", test_df, train_df_fraction, scale_factor_var.get("cth_very_high", 1.0)),
    ]

    print("\n" + "=" * 70)
    print(f"{title} - mean/std summary (plot units)")
    print("=" * 70)

    def fmt(arr):
        if len(arr) == 0:
            return "N=0"
        return f"N={len(arr)}, mean={arr.mean():.4f}, std={arr.std():.4f}"

    for var, test_source, train_source, scale in specs:
        print(f"\n[{var_names.get(var, var)}] ({var})")

        test_all = pd.to_numeric(test_source[var], errors="coerce") * scale
        train_all = pd.to_numeric(train_source[var], errors="coerce") * scale
        test_all = test_all[np.isfinite(test_all)]
        train_all = train_all[np.isfinite(train_all)]

        print(f"  TEST  overall: {fmt(test_all)}")
        print(f"  TRAIN overall: {fmt(train_all)}")
        print("  Per class:")

        for lbl in labels_to_report:
            class_name = LABEL_NAME_MAP.get(short[lbl], short[lbl])

            test_class = pd.to_numeric(
                test_source.loc[test_source["label"] == lbl, var],
                errors="coerce",
            ) * scale
            train_class = pd.to_numeric(
                train_source.loc[train_source["label"] == lbl, var],
                errors="coerce",
            ) * scale

            test_class = test_class[np.isfinite(test_class)]
            train_class = train_class[np.isfinite(train_class)]

            print(f"    {short[lbl]} ({class_name})")
            print(f"      TEST : {fmt(test_class)}")
            print(f"      TRAIN: {fmt(train_class)}")


print_mean_std_summary(
    title="Global TRAIN vs TEST",
    test_df=df,
    train_df_cma=df_train,
    train_df_median=df_train_median,
    train_df_fraction=df_train_fraction,
    labels_to_report=interesting_labels,
)

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

# Embedding-space vectors (dim_1..dim_128) for robust cosine-KNN local selection.
df_embeddings = pd.read_csv(embedding_vectors_file, low_memory=False)


def kde_mass_thresholds(density, masses=(50, 90), grid_x=None, grid_y=None):
    density = np.asarray(density)
    valid = density[np.isfinite(density) & (density > 0)]
    if valid.size == 0:
        return np.asarray([])

    if grid_x is not None and grid_y is not None and len(grid_x) > 1 and len(grid_y) > 1:
        dx = float(np.mean(np.diff(np.asarray(grid_x))))
        dy = float(np.mean(np.diff(np.asarray(grid_y))))
        cell_area = abs(dx * dy)
    else:
        cell_area = 1.0

    sorted_density = np.sort(valid)[::-1]
    cumulative_mass = np.cumsum(sorted_density * cell_area)
    total_mass = cumulative_mass[-1]
    if total_mass <= 0:
        return np.asarray([])

    thresholds = []
    for mass in masses:
        target_mass = float(mass) / 100.0 * total_mass
        idx = np.searchsorted(cumulative_mass, target_mass, side="left")
        idx = min(idx, sorted_density.size - 1)
        thresholds.append(sorted_density[idx])

    return np.asarray(thresholds)


def plot_density_contours(
    ax,
    x,
    y,
    percentiles=(75, 90, 99),
    gridsize=200,
    color="green",
    linewidths=1.4,
    linestyles="-",
    bandwidth=0.25,
):
    if len(x) < 10:
        return

    values = np.vstack([x, y])
    kde = gaussian_kde(values, bw_method=bandwidth)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.mgrid[xmin:xmax:gridsize*1j,
                      ymin:ymax:gridsize*1j]

    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid).reshape(xx.shape)

    levels = kde_mass_thresholds(zz, masses=percentiles, grid_x=xx[:, 0], grid_y=yy[0, :])
    if levels.size == 0:
        return

    order = np.argsort(levels)
    levels = levels[order]

    if isinstance(linestyles, (list, tuple, np.ndarray)) and len(levels) <= len(linestyles):
        contour_linestyles = [linestyles[i] for i in order]
    else:
        contour_linestyles = linestyles

    ax.contour(
        xx,
        yy,
        zz,
        levels=levels,
        colors=color,
        linewidths=linewidths,
        linestyles=contour_linestyles,
    )


def _basename(series):
    return series.astype(str).str.split("/").str[-1]


def select_local_train_filenames_cosine(embedding_df, test_df, labels_to_use, k_neighbors=5, chunk_size=512):
    """Select class-wise TRAIN filenames via cosine-KNN on normalized dim_1..dim_128 vectors."""
    emb = embedding_df.copy()
    if "filename" not in emb.columns:
        if "path" in emb.columns:
            emb["filename"] = _basename(emb["path"])
        else:
            raise ValueError("Embedding table missing both 'filename' and 'path' columns")

    emb["label"] = pd.to_numeric(emb["label"], errors="coerce")

    dim_cols = [c for c in emb.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("No embedding columns dim_* found for cosine-KNN selection")

    # Keep only test samples that belong to the current plotting test set.
    test_lookup = test_df[["filename", "label"]].copy()
    test_lookup["label"] = pd.to_numeric(test_lookup["label"], errors="coerce")
    test_lookup = test_lookup.dropna(subset=["filename", "label"]).drop_duplicates()

    emb_train = emb[emb["vector_type"] == "TRAIN"].copy()
    emb_test_all = emb[emb["vector_type"] != "TRAIN"].copy()
    emb_test = emb_test_all.merge(test_lookup, on=["filename", "label"], how="inner")

    selected = set()

    for lbl in labels_to_use:
        train_lbl = emb_train[emb_train["label"] == lbl].copy()
        test_lbl = emb_test[emb_test["label"] == lbl].copy()
        if train_lbl.empty or test_lbl.empty:
            continue

        train_vec = train_lbl[dim_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        test_vec = test_lbl[dim_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        finite_train = np.isfinite(train_vec).all(axis=1)
        finite_test = np.isfinite(test_vec).all(axis=1)
        train_vec = train_vec[finite_train]
        test_vec = test_vec[finite_test]
        train_names = train_lbl.loc[finite_train, "filename"].astype(str).to_numpy()

        if train_vec.shape[0] == 0 or test_vec.shape[0] == 0:
            continue

        # L2-normalize vectors so dot product equals cosine similarity.
        train_norm = np.linalg.norm(train_vec, axis=1, keepdims=True)
        test_norm = np.linalg.norm(test_vec, axis=1, keepdims=True)
        valid_train = (train_norm[:, 0] > 0)
        valid_test = (test_norm[:, 0] > 0)

        train_vec = train_vec[valid_train] / train_norm[valid_train]
        test_vec = test_vec[valid_test] / test_norm[valid_test]
        train_names = train_names[valid_train]

        if train_vec.shape[0] == 0 or test_vec.shape[0] == 0:
            continue

        k_eff = min(k_neighbors, train_vec.shape[0])
        for start in range(0, test_vec.shape[0], chunk_size):
            stop = min(start + chunk_size, test_vec.shape[0])
            # Cosine similarity because vectors are normalized.
            sim = test_vec[start:stop] @ train_vec.T
            idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
            selected.update(train_names[idx.ravel()].tolist())

    return selected



# -------------------------
# SETUP FIGURE
# -------------------------
n_labels = len(interesting_labels)
fig = plt.figure(figsize=(12, 10.5))
gs = fig.add_gridspec(
    nrows=3,
    ncols=3,
    height_ratios=[1.0, 1.0, 1.0],
    hspace=0.45,
    wspace=0.3
)

# Define axes
# Row 0: hourly (spans 2 columns), CC histogram
# ax_tsne = fig.add_subplot(gs[0, 0])  # Feature space - REMOVED
ax_hour = fig.add_subplot(gs[0, 0:2])  # Hourly distribution (extended to 2 columns)
ax_cc = fig.add_subplot(gs[0, 2])     # CC histogram
# Row 1: Median scatter row (CTH50 vs COT50)
ax_scatter_median = [fig.add_subplot(gs[1, i]) for i in range(3)]
# Row 2: Existing threshold scatter row (CTH10+ vs COT30+)
ax_scatter = [fig.add_subplot(gs[2, i]) for i in range(3)]
# # Row 2: Dominance distributions (one per class) - COMMENTED OUT
# ax_dom = [fig.add_subplot(gs[2, i]) for i in range(3)]  # Dominance distributions

# -------------------------
# PREPASS: GLOBAL DENSITY SCALE
# -------------------------
bins = 25

# Row 2 variables (new): CTH50 vs COT50
y_var_median = "cth50"
x_var_median = "cot50"
x_range_median = var_xlim.get(x_var_median, None)
y_range_median = var_xlim.get(y_var_median, None)

# Row 3 variables (existing): CTH10+ vs COT30+
y_var = physical_var_scatter[0]
x_var = physical_var_scatter[1]
x_range = var_xlim.get(x_var, None)
y_range = var_xlim.get(y_var, None)

global_vmax_median = 0.0
for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue

    x_data = pd.to_numeric(df_lab[x_var_median], errors="coerce") * scale_factor_var.get(x_var_median, 1.0)
    y_data = pd.to_numeric(df_lab[y_var_median], errors="coerce") * scale_factor_var.get(y_var_median, 1.0)
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    if len(x_data) == 0:
        continue

    xr = x_range_median if x_range_median is not None else (x_data.min(), x_data.max())
    yr = y_range_median if y_range_median is not None else (y_data.min(), y_data.max())

    h_data, _, _ = relative_hist2d(x_data, y_data, bins=bins, xr=xr, yr=yr)
    global_vmax_median = max(global_vmax_median, np.nanmax(h_data))

if global_vmax_median == 0:
    global_vmax_median = 1.0

global_vmax = 0.0
for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue

    x_data = df_lab[x_var] * scale_factor_var.get(x_var, 1.0)
    y_data = df_lab[y_var] * scale_factor_var.get(y_var, 1.0)

    xr = x_range if x_range is not None else (x_data.min(), x_data.max())
    yr = y_range if y_range is not None else (y_data.min(), y_data.max())

    h_data, _, _ = relative_hist2d(x_data, y_data, bins=bins, xr=xr, yr=yr)
    global_vmax = max(global_vmax, np.nanmax(h_data))

if global_vmax == 0:
    global_vmax = 1.0

cmap = cmc.lipari
cmap.set_bad(color="white", alpha=0.0)

# # ========================================================== 
# # ROW 2: DOMINANCE DISTRIBUTION BOXPLOTS - COMMENTED OUT
# # ==========================================================
# # Melt wperc columns for boxplot visualization
# df_melted = df[df['label'].isin(interesting_labels)].melt(
#     id_vars=["filename", "label"],
#     value_vars=wperc_cols,
#     var_name="neighbor_label",
#     value_name="probability"
# )
# print(df_melted['label'].value_counts())


# # Create palette for all labels
# palette = {
#     f"wperc_label_{k}": v
#     for k, v in colors.items()
# }

# for col, label in enumerate(interesting_labels):
#     class_name = short[label]  # Convert numeric label to class name (e.g., 1 -> "EC")
#     df_cls = df_melted[df_melted['label'] == label]
#     print(df_cls.head())
#     
#     # Create order based on labels_ordered
#     order = [f"wperc_label_{l}" for l in labels_ordered]
#     
#     sns.boxplot(
#         data=df_cls,
#         x="neighbor_label",
#         y="probability",
#         hue="neighbor_label",
#         ax=ax_dom[col],
#         palette=palette,
#         legend=False,
#         showfliers=False,
#         width=0.6,
#         order=order
#     )
#     
#     ax_dom[col].set_xlabel("", fontsize=11)
#     if col == 0:
#         ax_dom[col].set_ylabel("Dominance (%)", fontsize=11, labelpad=10)
#     # Add letter title (g, h, i) along with class name
#     letters = ["g)", "h)", "i)"]
#     ax_dom[col].set_title(
#         f"{letters[col]} {LABEL_NAME_MAP.get(class_name, class_name)}", 
#         fontsize=12, 
#         fontweight="bold"
#     )
#     # Set x-axis labels in short_ordered order
#     ax_dom[col].set_xticks(np.arange(len(labels_ordered)))
#     ax_dom[col].set_xticklabels(
#         [s.upper() for s in short_ordered],
#         rotation=45,
#         fontsize=10
#     )
#     # ax_dom[col].axhline(
#     #     75, 
#     #     color="red", 
#     #     linestyle="--", 
#     #     linewidth=1.5, 
#     #     alpha=0.7
#     # )
#     ax_dom[col].grid(True, axis="y", alpha=0.3)

# -------------------------
# LOOP OVER LABELS TO COLLECT DATA AND CREATE PLOTS
# -------------------------

# # ==========================================================
# # PLOT 1: FEATURE SPACE WITH ALL CONTOURS - REMOVED
# # ==========================================================

# # --- background: TRAIN vectors ---
# for lbl in labels_ordered:
#     alpha = 0.03 if lbl not in interesting_labels else 0.2
#     df_bg = df_tsne_train[df_tsne_train["label"] == lbl]
#     ax_tsne.scatter(
#         df_bg["tsne_dim_1"],
#         df_bg["tsne_dim_2"],
#         s=1,
#         color=colors[lbl],
#         alpha=0.01,
#         rasterized=True
#     )

#     # --- centroid ---
#     df_cent = df_tsne_centroids[df_tsne_centroids["label"] == lbl]
#     ax_tsne.scatter(
#         df_cent["tsne_dim_1"],
#         df_cent["tsne_dim_2"],
#         marker="^",
#         s=120,
#         color=colors[lbl],
#         edgecolor="black",
#         zorder=5,
#     )

# # --- density contours for each class ---
# for label in interesting_labels:
#     df_lab = df[df["label"] == label].copy()
#     if df_lab.empty:
#         continue
#     
#     plot_density_contours(
#         ax_tsne,
#         df_lab["tsne_dim_1"].values,
#         df_lab["tsne_dim_2"].values,
#         percentiles=(50, 65, 80, 90),
#         color=colors[label],
#         linewidths=2.5,
#         linestyles="-"
#     )

# ylim = ax_tsne.get_ylim()
# #shift the plot mre towards the bottom
# ax_tsne.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.1)
# ax_tsne.set_xticks([])
# ax_tsne.set_yticks([])
# ax_tsne.set_title("a) Feature Space", fontsize=13, fontweight="bold")

# ==========================================================
# PLOT 2: HOURLY OCCURRENCE WITH ALL CLASSES
# ==========================================================

for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    df_train_lab = df_train[df_train["label"] == label].copy()
    if df_lab.empty and df_train_lab.empty:
        continue

    hours = np.arange(0, 24)

    if not df_lab.empty:
        hour_counts_test = df_lab['hour'].value_counts().sort_index()
        counts_test = [hour_counts_test.get(h, 0) for h in hours]
        ax_hour.plot(
            hours,
            counts_test,
            color=colors[label],
            label=LABEL_NAME_MAP[short[label]],
            **styles["all"]
        )

    if not df_train_lab.empty:
        hour_counts_train = df_train_lab['hour'].value_counts().sort_index()
        counts_train = [hour_counts_train.get(h, 0) for h in hours]
        ax_hour.plot(
            hours,
            counts_train,
            color=colors[label],
            alpha=0.95,
            lw=2,
            ls='--',
        )

ax_hour.set_xlabel("Hour (UTC)", fontsize=AXIS_LABEL_FONTSIZE)
ax_hour.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
ax_hour.set_xticks(np.arange(0, 25, 3))
ax_hour.set_yscale("log")
ax_hour.set_yticks([10, 100, 1000])
ax_hour.set_ylim(5, None)
apply_axis_style(ax_hour, tick_size=TICK_LABEL_FONTSIZE)

ax_hour.set_title("a) Hourly Occurrence", fontsize=TITLE_FONTSIZE, fontweight="bold")

# ==========================================================
# PLOT 3: CC HISTOGRAM WITH ALL CLASSES
# ==========================================================

var = "cma"
for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    df_train_lab = df_train[df_train["label"] == label].copy()
    if df_lab.empty and df_train_lab.empty:
        continue

    data_test = df_lab[var].dropna()
    if len(data_test) >= 5:
        ax_cc.hist(
            data_test * scale_factor_var.get(var, 1.0),
            bins=30,
            histtype="step",
            alpha=0.8,
            lw=2,
            color=colors[label],
            label=LABEL_NAME_MAP[short[label]]
        )

    data_train = df_train_lab[var].dropna()
    if len(data_train) >= 5:
        ax_cc.hist(
            data_train * scale_factor_var.get(var, 1.0),
            bins=30,
            histtype="step",
            alpha=0.95,
            lw=2,
            ls='--',
            color=colors[label],
        )

xmin, xmax = var_xlim[var]
ax_cc.set_xlim(xmin, xmax)
# Custom tick labels for CC distribution: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
cc_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) * scale_factor_var.get(var, 1.0)
ax_cc.set_xticks(cc_ticks)
ax_cc.set_xticklabels([f"{t:.0f}" for t in cc_ticks])
ax_cc.set_xlabel(var_names.get(var, var), fontsize=AXIS_LABEL_FONTSIZE)
ax_cc.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
ax_cc.set_yscale("log")
ax_cc.set_yticks([10, 100, 1000])
ax_cc.set_ylim(5, None)
apply_axis_style(ax_cc, tick_size=TICK_LABEL_FONTSIZE)
#ax_cc.legend(fontsize=9, loc="upper right")
ax_cc.set_title("b) CC Distribution", fontsize=TITLE_FONTSIZE, fontweight="bold")

# ==========================================================
# PLOT 4-6: MEDIAN SCATTER PLOTS (CTH50 vs COT50)
# ==========================================================

last_mappable_median = None
titles_median = ["c) Early Convection", "d) Deep Convection", "e) Overcast Anvil"]

for col, label in enumerate(interesting_labels):
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue

    x_data = pd.to_numeric(df_lab[x_var_median], errors="coerce") * scale_factor_var.get(x_var_median, 1.0)
    y_data = pd.to_numeric(df_lab[y_var_median], errors="coerce") * scale_factor_var.get(y_var_median, 1.0)
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    if len(x_data) == 0:
        continue

    y_min = float(y_data.min())
    y_max = float(y_data.max())
    yr = y_range_median if y_range_median is not None else (y_min, y_max)

    x_positive = x_data[x_data > 0]
    if len(x_positive) == 0:
        continue

    x_min = 1.0
    x_max = max(float(x_positive.max()) * 10.0, 100.0)
    xr = (x_min, x_max)
    x_bins = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)

    h_data, xedges, yedges = relative_hist2d(x_data, y_data, bins=[x_bins, bins], xr=xr, yr=yr)
    h_data = np.ma.masked_where(h_data == 0, h_data)

    mappable = ax_scatter_median[col].pcolormesh(
        xedges,
        yedges,
        h_data.T,
        cmap=cmap,
        vmin=0,
        vmax=global_vmax_median,
        shading="auto",
    )
    last_mappable_median = mappable

    # Overlay TRAIN density contour for comparison.
    df_train_lab = df_train_median[df_train_median["label"] == label]
    if not df_train_lab.empty:
        x_train = pd.to_numeric(df_train_lab[x_var_median], errors="coerce") * scale_factor_var.get(x_var_median, 1.0)
        y_train = pd.to_numeric(df_train_lab[y_var_median], errors="coerce") * scale_factor_var.get(y_var_median, 1.0)
        valid_train = np.isfinite(x_train) & np.isfinite(y_train)
        x_train = x_train[valid_train]
        y_train = y_train[valid_train]

        if len(x_train) >= 10:
            plot_density_contours(
                ax_scatter_median[col],
                x_train.values,
                y_train.values,
                percentiles=(50, 75, 90, 99),
                color='turquoise',
                linewidths=1.6,
                linestyles='-',
            )

        if len(x_train) > 0:
            plot_mean_cross(
                ax_scatter_median[col],
                float(np.nanmean(x_train)),
                float(np.nanmean(y_train)),
                color="turquoise",
                size=360,
                line_width=3.2,
            )

    plot_mean_cross(
        ax_scatter_median[col],
        float(np.nanmean(x_data)),
        float(np.nanmean(y_data)),
        color="gold",
        size=360,
        line_width=3.2,
    )

    ax_scatter_median[col].set_xscale("log")
    ax_scatter_median[col].set_xlim(x_min, x_max)
    if y_var_median in var_xlim:
        ax_scatter_median[col].set_ylim(*var_xlim[y_var_median])
        ax_scatter_median[col].set_yticks([0, 2.5, 5, 7.5, 10, 12.5])
        ax_scatter_median[col].set_yticklabels(["0", "2.5", "5", "7.5", "10", "12.5"])
    ax_scatter_median[col].set_xlabel(var_names.get(x_var_median, x_var_median), fontsize=AXIS_LABEL_FONTSIZE)

    if col == 0:
        ax_scatter_median[col].set_ylabel(var_names.get(y_var_median, y_var_median), fontsize=AXIS_LABEL_FONTSIZE)

    tick_values = np.array([1.0, 10.0, 100.0])
    ax_scatter_median[col].set_xticks(tick_values)
    ax_scatter_median[col].set_xlim(1,150)
    ax_scatter_median[col].set_xticklabels([f"{v:g}" for v in tick_values])

    apply_axis_style(ax_scatter_median[col], tick_size=TICK_LABEL_FONTSIZE)
    ax_scatter_median[col].set_title(titles_median[col], fontsize=TITLE_FONTSIZE, fontweight="bold")

# ==========================================================
# PLOT 7-9: SCATTER PLOTS (CTH10+ vs COT30+, ONE PER CLASS)
# ==========================================================

last_mappable = None
titles = ["f) Early Convection", "g) Deep Convection", "h) Overcast Anvil"]

for col, label in enumerate(interesting_labels):
    df_lab = df[df["label"] == label].copy()
    df_train_lab = df_train_fraction[df_train_fraction["label"] == label].copy()
    if df_lab.empty:
        continue

    x_data = df_lab[x_var] * scale_factor_var.get(x_var, 1.0)
    y_data = df_lab[y_var] * scale_factor_var.get(y_var, 1.0)

    xr = x_range if x_range is not None else (x_data.min(), x_data.max())
    yr = y_range if y_range is not None else (y_data.min(), y_data.max())

    h_data, xedges, yedges = relative_hist2d(x_data, y_data, bins=bins, xr=xr, yr=yr)

    h_data = np.ma.masked_where(h_data == 0, h_data)

    mappable = ax_scatter[col].pcolormesh(
        xedges,
        yedges,
        h_data.T,
        cmap=cmap,
        vmin=0,
        vmax=global_vmax,
        shading="auto",
    )
    last_mappable = mappable

    if x_var in var_xlim:
        ax_scatter[col].set_xlim(*var_xlim[x_var])
    if y_var in var_xlim:
        ax_scatter[col].set_ylim(*var_xlim[y_var])
    ax_scatter[col].set_xlabel(var_names.get(x_var, x_var), fontsize=AXIS_LABEL_FONTSIZE)
    
    if col == 0:
        ax_scatter[col].set_ylabel(
            f"{var_names.get(y_var, y_var)}",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

    if not df_train_lab.empty:
        x_train = pd.to_numeric(df_train_lab[x_var], errors="coerce") * scale_factor_var.get(x_var, 1.0)
        y_train = pd.to_numeric(df_train_lab[y_var], errors="coerce") * scale_factor_var.get(y_var, 1.0)
        valid_train = np.isfinite(x_train) & np.isfinite(y_train)
        x_train = x_train[valid_train]
        y_train = y_train[valid_train]

        if len(x_train) >= 10:
            plot_density_contours(
                ax_scatter[col],
                x_train.values,
                y_train.values,
                percentiles=(50, 75, 90, 99),
                color='turquoise',
                linewidths=1.6,
                linestyles='-',
            )

            if len(x_train) > 0:
                plot_mean_cross(
                    ax_scatter[col],
                    float(np.nanmean(x_train)),
                    float(np.nanmean(y_train)),
                    color="turquoise",
                    size=360,
                    line_width=3.2,
                )

        plot_mean_cross(
            ax_scatter[col],
            float(np.nanmean(x_data)),
            float(np.nanmean(y_data)),
            color="gold",
            size=360,
            line_width=3.2,
        )

    # Set x-axis labels
    ax_scatter[col].set_xlabel(var_names.get(x_var, x_var), fontsize=AXIS_LABEL_FONTSIZE)

    apply_axis_style(ax_scatter[col], tick_size=TICK_LABEL_FONTSIZE)
    
    ax_scatter[col].set_title(titles[col], fontsize=TITLE_FONTSIZE, fontweight="bold")


# Create figure-level legends on the right side (outside plot frame, closer to plots)
# Legend 1: Line style and Classes (for top row) - moved up and left
style_handles = [
    Line2D([0], [0], color='black', lw=2, ls='-', label='Test'),
    Line2D([0], [0], color='black', lw=2, ls='--', label='Train'),
]
class_handles = [
    Patch(facecolor=colors[label], label=LABEL_NAME_MAP[short[label]])
    for label in interesting_labels
]
legend_handles = style_handles + class_handles
legend_labels = [h.get_label() for h in legend_handles]

# Position legend outside on the right, above scatter rows (moved up and left)
fig.legend(
    handles=legend_handles,
    fontsize=11,
    loc="center right",
    bbox_to_anchor=(1.18, 0.76),
    frameon=True,
    title="Line style & Classes",
    title_fontsize=11,
)

# Legend 2: Scatter overlays (for scatter rows) with properly colored crosses
# Aligned with end of third row and moved left
overlay_handles = [
    Line2D([0], [0], color='turquoise', lw=2, label='Train density contour'),
    Line2D([0], [0], marker='x', color='turquoise', markerfacecolor='turquoise', markersize=10, markeredgewidth=2.5, label='Train mean', linestyle='None'),
    Line2D([0], [0], marker='x', color='gold', markerfacecolor='gold', markersize=10, markeredgewidth=2.5, label='Test mean', linestyle='None'),
]
fig.legend(
    handles=overlay_handles,
    fontsize=11,
    loc="center right",
    bbox_to_anchor=(1.18, 0.15),
    frameon=True,
    title="Scatter overlays",
    title_fontsize=11,
)

# Save figure
plt.savefig(
    csv_path + "/var_distributions_by_label_no_dominance_redesigned.png",
    dpi=150,
    bbox_inches="tight"
)


# -------------------------
# LOCAL-TRAIN VERSION (KNN in feature space)
# -------------------------
local_train_filenames = select_local_train_filenames_cosine(
    embedding_df=df_embeddings,
    test_df=df,
    labels_to_use=interesting_labels,
    k_neighbors=LOCAL_TRAIN_K,
)

df_train_local = df_train.copy()
if "crop" in df_train_local.columns:
    df_train_local["filename"] = _basename(df_train_local["crop"])
else:
    df_train_local["filename"] = ""
df_train_local = df_train_local[df_train_local["filename"].isin(local_train_filenames)].copy()

df_train_median_local = df_train_median.copy()
if "crop" in df_train_median_local.columns:
    df_train_median_local["filename"] = _basename(df_train_median_local["crop"])
else:
    df_train_median_local["filename"] = ""
df_train_median_local = df_train_median_local[
    df_train_median_local["filename"].isin(local_train_filenames)
].copy()

df_train_fraction_local = df_train_fraction.copy()
if "crop" in df_train_fraction_local.columns:
    df_train_fraction_local["filename"] = _basename(df_train_fraction_local["crop"])
else:
    df_train_fraction_local["filename"] = ""
df_train_fraction_local = df_train_fraction_local[
    df_train_fraction_local["filename"].isin(local_train_filenames)
].copy()

print(
    f"Local train subset (k={LOCAL_TRAIN_K}): "
    f"cma={len(df_train_local)}, median={len(df_train_median_local)}, fraction={len(df_train_fraction_local)}"
)

print_mean_std_summary(
    title=f"Local TRAIN (k={LOCAL_TRAIN_K}) vs TEST",
    test_df=df,
    train_df_cma=df_train_local,
    train_df_median=df_train_median_local,
    train_df_fraction=df_train_fraction_local,
    labels_to_report=interesting_labels,
)

fig_local = plt.figure(figsize=(12, 10.5))
gs_local = fig_local.add_gridspec(
    nrows=3,
    ncols=3,
    height_ratios=[1.0, 1.0, 1.0],
    hspace=0.45,
    wspace=0.3,
)

ax_hour_local = fig_local.add_subplot(gs_local[0, 0:2])
ax_cc_local = fig_local.add_subplot(gs_local[0, 2])
ax_scatter_median_local = [fig_local.add_subplot(gs_local[1, i]) for i in range(3)]
ax_scatter_local = [fig_local.add_subplot(gs_local[2, i]) for i in range(3)]

# Hourly occurrence
for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    df_train_lab = df_train_local[df_train_local["label"] == label].copy()
    if df_lab.empty and df_train_lab.empty:
        continue

    hours = np.arange(0, 24)
    if not df_lab.empty:
        hour_counts_test = df_lab["hour"].value_counts().sort_index()
        counts_test = [hour_counts_test.get(h, 0) for h in hours]
        ax_hour_local.plot(
            hours,
            counts_test,
            color=colors[label],
            label=LABEL_NAME_MAP[short[label]],
            **styles["all"],
        )

    if not df_train_lab.empty:
        hour_counts_train = df_train_lab["hour"].value_counts().sort_index()
        counts_train = [hour_counts_train.get(h, 0) for h in hours]
        ax_hour_local.plot(
            hours,
            counts_train,
            color=colors[label],
            alpha=0.95,
            lw=2,
            ls="--",
        )

ax_hour_local.set_xlabel("Hour (UTC)", fontsize=AXIS_LABEL_FONTSIZE)
ax_hour_local.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
ax_hour_local.set_xticks(np.arange(0, 25, 3))
ax_hour_local.set_yscale("log")
ax_hour_local.set_yticks([10, 100, 1000])
ax_hour_local.set_ylim(5, None)
apply_axis_style(ax_hour_local, tick_size=TICK_LABEL_FONTSIZE)
ax_hour_local.set_title(f"a) Hourly Occurrence", fontsize=TITLE_FONTSIZE, fontweight="bold")

# CC distribution
for label in interesting_labels:
    df_lab = df[df["label"] == label].copy()
    df_train_lab = df_train_local[df_train_local["label"] == label].copy()
    if df_lab.empty and df_train_lab.empty:
        continue

    data_test = df_lab[var].dropna()
    if len(data_test) >= 5:
        ax_cc_local.hist(
            data_test * scale_factor_var.get(var, 1.0),
            bins=30,
            histtype="step",
            alpha=0.8,
            lw=2,
            color=colors[label],
        )

    data_train = df_train_lab[var].dropna()
    if len(data_train) >= 5:
        ax_cc_local.hist(
            data_train * scale_factor_var.get(var, 1.0),
            bins=30,
            histtype="step",
            alpha=0.95,
            lw=2,
            ls="--",
            color=colors[label],
        )

ax_cc_local.set_xlim(*var_xlim[var])
ax_cc_local.set_xticks(cc_ticks)
ax_cc_local.set_xticklabels([f"{t:.1f}" for t in cc_ticks])
ax_cc_local.set_xlabel(var_names.get(var, var), fontsize=AXIS_LABEL_FONTSIZE)
ax_cc_local.set_ylabel("Counts", fontsize=AXIS_LABEL_FONTSIZE)
ax_cc_local.set_yscale("log")
ax_cc_local.set_yticks([10, 100, 1000])
ax_cc_local.set_ylim(5, None)
apply_axis_style(ax_cc_local, tick_size=TICK_LABEL_FONTSIZE)
ax_cc_local.set_title(f"b) CC Distribution", fontsize=TITLE_FONTSIZE, fontweight="bold")

# Median row
titles_median_local = [
    "c) Early Convection",
    "d) Deep Convection",
    "e) Overcast Anvil",
]
for col, label in enumerate(interesting_labels):
    df_lab = df[df["label"] == label].copy()
    if df_lab.empty:
        continue

    x_data = pd.to_numeric(df_lab[x_var_median], errors="coerce") * scale_factor_var.get(x_var_median, 1.0)
    y_data = pd.to_numeric(df_lab[y_var_median], errors="coerce") * scale_factor_var.get(y_var_median, 1.0)
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    if len(x_data) == 0:
        continue

    y_min = float(y_data.min())
    y_max = float(y_data.max())
    yr = y_range_median if y_range_median is not None else (y_min, y_max)
    x_positive = x_data[x_data > 0]
    if len(x_positive) == 0:
        continue

    x_min = 1.0
    x_max = max(float(x_positive.max()) * 10.0, 100.0)
    xr = (x_min, x_max)
    x_bins = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)
    h_data, xedges, yedges = relative_hist2d(x_data, y_data, bins=[x_bins, bins], xr=xr, yr=yr)
    h_data = np.ma.masked_where(h_data == 0, h_data)

    ax_scatter_median_local[col].pcolormesh(
        xedges,
        yedges,
        h_data.T,
        cmap=cmap,
        vmin=0,
        vmax=global_vmax_median,
        shading="auto",
    )

    df_train_lab = df_train_median_local[df_train_median_local["label"] == label]
    if not df_train_lab.empty:
        x_train = pd.to_numeric(df_train_lab[x_var_median], errors="coerce") * scale_factor_var.get(x_var_median, 1.0)
        y_train = pd.to_numeric(df_train_lab[y_var_median], errors="coerce") * scale_factor_var.get(y_var_median, 1.0)
        valid_train = np.isfinite(x_train) & np.isfinite(y_train)
        x_train = x_train[valid_train]
        y_train = y_train[valid_train]

        if len(x_train) >= 10:
            plot_density_contours(
                ax_scatter_median_local[col],
                x_train.values,
                y_train.values,
                percentiles=(50, 75, 90, 99),
                color="turquoise",
                linewidths=1.6,
                linestyles="-",
            )

        if len(x_train) > 0:
            plot_mean_cross(
                ax_scatter_median_local[col],
                float(np.nanmean(x_train)),
                float(np.nanmean(y_train)),
                color="turquoise",
                size=360,
                line_width=3.2,
            )

    plot_mean_cross(
        ax_scatter_median_local[col],
        float(np.nanmean(x_data)),
        float(np.nanmean(y_data)),
        color="gold",
        size=360,
        line_width=3.2,
    )

    ax_scatter_median_local[col].set_xscale("log")
    ax_scatter_median_local[col].set_xlim(1, 150)
    if y_var_median in var_xlim:
        ax_scatter_median_local[col].set_ylim(*var_xlim[y_var_median])
        ax_scatter_median_local[col].set_yticks([0, 2.5, 5, 7.5, 10, 12.5])
        ax_scatter_median_local[col].set_yticklabels(["0", "2.5", "5", "7.5", "10", "12.5"])
    ax_scatter_median_local[col].set_xlabel(var_names.get(x_var_median, x_var_median), fontsize=AXIS_LABEL_FONTSIZE)
    if col == 0:
        ax_scatter_median_local[col].set_ylabel(var_names.get(y_var_median, y_var_median), fontsize=AXIS_LABEL_FONTSIZE)
    tick_values = np.array([1.0, 10.0, 100.0])
    ax_scatter_median_local[col].set_xticks(tick_values)
    ax_scatter_median_local[col].set_xticklabels([f"{v:g}" for v in tick_values])
    apply_axis_style(ax_scatter_median_local[col], tick_size=TICK_LABEL_FONTSIZE)
    ax_scatter_median_local[col].set_title(titles_median_local[col], fontsize=TITLE_FONTSIZE, fontweight="bold")

# Fraction row
titles_local = ["f) Early Convection", "g) Deep Convection", "h) Deep Anvil"]
for col, label in enumerate(interesting_labels):
    df_lab = df[df["label"] == label].copy()
    df_train_lab = df_train_fraction_local[df_train_fraction_local["label"] == label].copy()
    if df_lab.empty:
        continue

    x_data = df_lab[x_var] * scale_factor_var.get(x_var, 1.0)
    y_data = df_lab[y_var] * scale_factor_var.get(y_var, 1.0)
    xr = x_range if x_range is not None else (x_data.min(), x_data.max())
    yr = y_range if y_range is not None else (y_data.min(), y_data.max())
    h_data, xedges, yedges = relative_hist2d(x_data, y_data, bins=bins, xr=xr, yr=yr)
    h_data = np.ma.masked_where(h_data == 0, h_data)

    ax_scatter_local[col].pcolormesh(
        xedges,
        yedges,
        h_data.T,
        cmap=cmap,
        vmin=0,
        vmax=global_vmax,
        shading="auto",
    )

    if x_var in var_xlim:
        ax_scatter_local[col].set_xlim(*var_xlim[x_var])
    if y_var in var_xlim:
        ax_scatter_local[col].set_ylim(*var_xlim[y_var])
    ax_scatter_local[col].set_xlabel(var_names.get(x_var, x_var), fontsize=AXIS_LABEL_FONTSIZE)
    if col == 0:
        ax_scatter_local[col].set_ylabel(var_names.get(y_var, y_var), fontsize=AXIS_LABEL_FONTSIZE)

    if not df_train_lab.empty:
        x_train = pd.to_numeric(df_train_lab[x_var], errors="coerce") * scale_factor_var.get(x_var, 1.0)
        y_train = pd.to_numeric(df_train_lab[y_var], errors="coerce") * scale_factor_var.get(y_var, 1.0)
        valid_train = np.isfinite(x_train) & np.isfinite(y_train)
        x_train = x_train[valid_train]
        y_train = y_train[valid_train]

        if len(x_train) >= 10:
            plot_density_contours(
                ax_scatter_local[col],
                x_train.values,
                y_train.values,
                percentiles=(50, 75, 90, 99),
                color="turquoise",
                linewidths=1.6,
                linestyles="-",
            )
            if len(x_train) > 0:
                plot_mean_cross(
                    ax_scatter_local[col],
                    float(np.nanmean(x_train)),
                    float(np.nanmean(y_train)),
                    color="turquoise",
                    size=360,
                    line_width=3.2,
                )

        plot_mean_cross(
            ax_scatter_local[col],
            float(np.nanmean(x_data)),
            float(np.nanmean(y_data)),
            color="gold",
            size=360,
            line_width=3.2,
        )

    apply_axis_style(ax_scatter_local[col], tick_size=TICK_LABEL_FONTSIZE)
    ax_scatter_local[col].set_title(titles_local[col], fontsize=TITLE_FONTSIZE, fontweight="bold")

style_handles_local = [
    Line2D([0], [0], color="black", lw=2, ls="-", label="Test"),
    Line2D([0], [0], color="black", lw=2, ls="--", label=f"Train (local k={LOCAL_TRAIN_K})"),
]
class_handles_local = [
    Patch(facecolor=colors[label], label=LABEL_NAME_MAP[short[label]])
    for label in interesting_labels
]
fig_local.legend(
    handles=style_handles_local + class_handles_local,
    fontsize=11,
    loc="center right",
    bbox_to_anchor=(1.08, 0.76),
    frameon=True,
    title="Line style & Classes",
    title_fontsize=11,
)

overlay_handles_local = [
    Line2D([0], [0], color="turquoise", lw=2, label="Train density contour"),
    Line2D([0], [0], marker="x", color="turquoise", markerfacecolor="turquoise", markersize=10, markeredgewidth=2.5, label="Train mean", linestyle="None"),
    Line2D([0], [0], marker="x", color="gold", markerfacecolor="gold", markersize=10, markeredgewidth=2.5, label="Test mean", linestyle="None"),
]
fig_local.legend(
    handles=overlay_handles_local,
    fontsize=11,
    loc="center right",
    bbox_to_anchor=(1.18, 0.15),
    frameon=True,
    title="Scatter overlays",
    title_fontsize=11,
)

fig_local.savefig(
    csv_path + f"/var_distributions_by_label_no_dominance_redesigned_local_k{LOCAL_TRAIN_K}.png",
    dpi=150,
    bbox_inches="tight",
)
