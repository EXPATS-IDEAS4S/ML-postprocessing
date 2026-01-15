#!/usr/bin/env python3
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator

# === CONFIG ===
RUN_NAME = "dcv2_resnet_k8_ir108_100x100_2013-2020_1xrandomcrops_1xtimestamp_cma_nc_convective"
path_to_dir = f"/data1/fig/{RUN_NAME}/epoch_800/closest/"
merged_path = os.path.join(path_to_dir, "merged_crops_stats_cvc_imergtime_closest_1000.csv")

# === LOAD DATA ===
df = pd.read_csv(merged_path)
print(f"✅ Loaded dataframe: {merged_path} ({df.shape})")
#print var unique values in 'var' column
print("Unique variables in 'var' column:", df['var'].unique())


# === COLOR MAP ===
COLORS_PER_CLASS = {
    '0': 'darkgray', '1': 'darkslategrey', '2': 'peru', '3': 'orangered',
    '4': 'lightcoral', '5': 'deepskyblue', '6': 'purple', '7': 'lightblue',
    '8': 'green', '9': 'goldenrod', '10': 'magenta', '11': 'dodgerblue',
    '12': 'darkorange', '13': 'olive', '14': 'crimson'
}

# === VARIABLES === TODO: add offset and mult factor to adjust units
VARIABLES = {
    "cth": {"label": "Cloud Top Height (km)", "vmin": 8.5, "vmax": 12.5, "logscale": False, "offset": 0, "mult": 0.001},
    "cot": {"label": "Cloud Optical Thickness", "vmin": 1, "vmax": 150, "logscale": True, "offset": 0, "mult": 1},
    "cma": {"label": "Cloud Cover (%)", "vmin": 10, "vmax": 100, "logscale": False, "offset": 0, "mult": 100},
    "ccv": {"label": "Convective Cloud Cover (%)", "vmin": 0.5, "vmax": 70, "logscale": True, "offset": 0, "mult": 1},
    #"precipitation": {"label": "Total Precipitation (mm)", "vmin": 5, "vmax": 5000, "logscale": True, "offset": 0, "mult": 0.5},
    "precipitation": {"label": "Rain Rate (mm/h)", "vmin": 2, "vmax": 12, "logscale": True, "offset": 0, "mult": 1},
    "euclid_msg_grid": {"label": "Total Lightning Count", "vmin": 2, "vmax": 160, "logscale": True, "offset": 0, "mult": 1},
}

PERCENTILE_VARS = ["cth", "cot", "precipitation"]
CATEGORICAL_VARS = ["cma", "euclid_msg_grid", "ccv"]
PERCENTILE_COLS =  ["99"] #["25", "50", "75", "99"]
CATEGORICAL_COL = "None"

# === VARIABLES TO COMPARE ===
VAR_X = "ccv"  # categorical
VAR_Y = "cth"  # percentile variable

#subdir for output plots based on variables
subdir = f"scatter_{VAR_X}_vs_{VAR_Y}"
path_to_dir = os.path.join(path_to_dir, subdir)
os.makedirs(path_to_dir, exist_ok=True)

# === MARKERS ===
MARKERS = {"25": "o", "50": "s", "75": "^", "99": "D", "None": "o"}

# === DATA EXTRACTION ===
def extract_per_label(df, var_name):
    """Return dataframe with mean per label for each percentile (if continuous) or one value (if categorical)."""
    subset = df[df["var"] == var_name].copy()
    if var_name in PERCENTILE_VARS:
        out = subset.groupby("label")[PERCENTILE_COLS].mean().reset_index()
    elif var_name in CATEGORICAL_VARS:
        subset[CATEGORICAL_COL] = pd.to_numeric(subset[CATEGORICAL_COL], errors="coerce")
        out = subset.groupby("label")[CATEGORICAL_COL].mean().reset_index()
    else:
        raise ValueError(f"Unknown variable type: {var_name}")
    
    #apply offset and mult factor
    offset = VARIABLES[var_name]["offset"]
    mult = VARIABLES[var_name]["mult"]
    if var_name in PERCENTILE_VARS:
        for p in PERCENTILE_COLS:
            out[p] = out[p] * mult + offset
    else:
        out[CATEGORICAL_COL] = out[CATEGORICAL_COL] * mult + offset

    return out


mean_x = extract_per_label(df, VAR_X)
mean_y = extract_per_label(df, VAR_Y)
print(f"Extracted mean data for VAR_X='{VAR_X}' and VAR_Y='{VAR_Y}'")
print(f"Mean X shape: {mean_x.shape}, Mean Y shape: {mean_y.shape}")


# === PLOT FUNCTION ===
def plot_scatter(mean_x, mean_y, var_x, var_y, output_path, separate_label=None, transparent=False):
    fig, ax = plt.subplots(figsize=(5, 4))

    settings_x = VARIABLES[var_x]
    settings_y = VARIABLES[var_y]

    labels_common = sorted(set(mean_x["label"]).intersection(mean_y["label"]))

    for lbl in labels_common:
        label = str(int(lbl))
        if separate_label and label != separate_label:
            continue

        color = COLORS_PER_CLASS.get(label, "black")

        row_x = mean_x[mean_x["label"] == lbl]
        row_y = mean_y[mean_y["label"] == lbl]

        if var_x in PERCENTILE_VARS:
            x_vals = {p: row_x[p].values[0] for p in PERCENTILE_COLS}
        else:
            x_vals = {"None": row_x[CATEGORICAL_COL].values[0]}

        if var_y in PERCENTILE_VARS:
            y_vals = {p: row_y[p].values[0] for p in PERCENTILE_COLS}
        else:
            y_vals = {"None": row_y[CATEGORICAL_COL].values[0]}

        # Combine percentiles for plotting
        for p_x, x_v in x_vals.items():
            for p_y, y_v in y_vals.items():
                #print(f"Plotting label {label}, var_x={var_x}({p_x}): {x_v}, var_y={var_y}({p_y}): {y_v}")
                p_marker = MARKERS.get(p_y if var_y in PERCENTILE_VARS else p_x, "o")
                #print(x_v, y_v, color, p_marker)
                ax.scatter(
                    x_v,
                    y_v,
                    color=color,
                    marker=p_marker,
                    s=150,
                    edgecolors="black",
                    linewidth=0.8,
                    alpha=0.9,
                )

    # === LEGENDS ===
    if separate_label is None:
        label_legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   label=f"Label {lbl}", markersize=10)
            for lbl, color in COLORS_PER_CLASS.items()
            if int(lbl) in labels_common
        ]
        pctl_legend = [
            Line2D([0], [0], marker=MARKERS[p], color="black", linestyle="None",
                   label=f"p{p}", markersize=9)
            for p in PERCENTILE_COLS
        ]
        legend1 = ax.legend(handles=label_legend, title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
        ax.add_artist(legend1)
        if (var_x in PERCENTILE_VARS) or (var_y in PERCENTILE_VARS):
            ax.legend(handles=pctl_legend, title="Percentiles", bbox_to_anchor=(1.05, 0.35), loc="upper left", frameon=False)

    # === AXIS STYLE ===
    ax.set_xlabel(f'Mean {settings_x["label"]}', fontsize=14)
    ax.set_ylabel(f'Mean {settings_y["label"]}', fontsize=14)

    # set limits
    ax.set_xlim(settings_x["vmin"], settings_x["vmax"])
    ax.set_ylim(settings_y["vmin"], settings_y["vmax"])

    # set scales first
    if settings_x["logscale"]:
        ax.set_xscale("log")

    if settings_y["logscale"]:
        ax.set_yscale("log")
    

    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(
        f" Label {separate_label}" if separate_label else "",
        fontsize=14, fontweight="bold"
    )

    fig.patch.set_alpha(0.0 if transparent else 1.0)
    ax.set_facecolor("none" if transparent else "white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=transparent)
    plt.close(fig)

# === MAIN SCATTER ===
main_out = os.path.join(path_to_dir, f"scatter_mean_{VAR_X}_vs_{VAR_Y}_all.png")
plot_scatter(mean_x, mean_y, VAR_X, VAR_Y, main_out, transparent=False)

# === PER LABEL ===
for lbl in sorted(set(mean_x["label"]).intersection(mean_y["label"])):
    lbl_str = str(int(lbl))
    lbl_out = os.path.join(path_to_dir, f"scatter_mean_{VAR_X}_vs_{VAR_Y}_label_{lbl_str}.png")
    plot_scatter(mean_x, mean_y, VAR_X, VAR_Y, lbl_out, separate_label=lbl_str, transparent=True)
    print(f"💾 Saved per-label scatterplot for label {lbl_str}")

print("✅ All adaptive scatterplots saved successfully.")
