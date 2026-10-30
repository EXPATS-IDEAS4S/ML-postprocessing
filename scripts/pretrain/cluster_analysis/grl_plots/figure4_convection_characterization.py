"""
This figure has the goal to demonstrate that we can identify diverse stages of convection 
 over the Alps. It works on all videos of the training set.
 It will show the graduality of changes (growth, stationarity and decay) between convective 
classes and the temporal variability over the duration of the video of the convective class properties. 
The plot has 5 columns (classes 6, 5, 9, 2, 3). The top rows show the temporal evolution of:
- row a: temporal evolution of CTH
- row b: temporal evolution of cloud cover
- row c: temporal evolution of precipitation mean
- row d: temporal evolution of CTH10+ area fraction
Bottom row (e): scatter plots of class gradient percentiles (25th, 50th, 75th, 95th) for cloud cover
vs CTH, lightning vs precipitation, cloud cover vs precipitation extent and COT30+ vs CTH10+.
The remaining gradient scatter plots (CTH10+ vs COT30+, CTH vs COT and
lightning vs precipitation p95) are in a separate figure.

class_colors = {
    # Decaying daytime
    0: "#2CA25F",
    8: "#006D2C",

    # Decaying nighttime
    3: "#807DBA",
    9: "#54278F",

    # Growing convection
    5: "#FDBE85",
    6: "#F16913",
    7: "#B30000",
}

"""



import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.class_colors import colors_per_class_codes_grl
from utils.plotting.plot_class_analysis import style_axis
# load video summary csv file path from config
REPO_ROOT = Path("/home/claudia/codes/ML_postprocessing")
CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"
config = load_config(str(CONFIG_PATH))

# create output directory if it doesn't exist
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MARKER_SIZE = 300
MARKER_EDGE_COLOR = "grey"
MARKER_EDGE_WIDTH = 1.4
PERCENTILE_ERRORBAR_COLOR = (0.78, 0.78, 0.78, 0.45)
CMA_GRADIENT_LIMIT = 0.2
CTH_MEAN_LIMIT = 8000
CTH_GRADIENT_LIMIT = 800
ZERO_AXIS_LINEWIDTH = 2.2
AXIS_LABEL_FONTSIZE = 23
TITLE_FONTSIZE = 23
TICK_LABEL_FONTSIZE = 23
FONT_SIZE_TEXT = 23 
PANEL_LABEL_FONTSIZE = 23
SECTION_TITLE_OFFSET = 0.045
B_TITLE_GAP_FRACTION = 0.38

LEGEND_FONTSIZE = 18
SCATTER_LABEL_FONTSIZE = 19
SCATTER_TICK_FONTSIZE = 18
WIND_DIRECTION_X_LIMITS = (100, 310)
TIME_SERIES_LINEWIDTH = 4
TIME_SERIES_PERCENTILE_LINEWIDTH = 2.2
TIME_SERIES_MARKER_SIZE = 10
TIME_SERIES_PERCENTILES_TO_PLOT = ["25", "50", "75"]
MAIN_ROW_HSPACE = 0.35
LOWER_TIME_SERIES_ROW_SHIFT = 0.04
GRADIENT_PERCENTILE_MARKERS = {0.05: "^", 0.25: "o", 0.5: "s", 0.75: "X", 0.95: "D"}
# (x column, y column, x label, y label, title); gradients are per hour in the summary CSV
GRADIENT_SCATTER_PANELS = [
    ("cma_gradient", "cth_gradient", "Cloud cover gradient \n (fraction/h)", "CTH gradient (m/h)", "Cloud cover vs \n CTH"),
    ("cot30plus_gradient", "cth10plus_gradient", "COT30+ gradient", "CTH10+ gradient", "COT30+ vs \n CTH10+"),
    ("cma_gradient", "prec_fraction_gradient", "Cloud cover gradient \n (fraction/h)", "Precip. extent gradient \n (fraction/h)", "Cloud cover vs \n precip. extent"),
    ("euclid_msg_grid_gradient", "precipitation_gradient", "Lightning gradient \n (counts/h)", "Precipitation gradient \n (mm/h)", "Lightning vs \n precipitation"),
]
EXTRA_GRADIENT_SCATTER_PANELS = [    ("cth10plus_gradient", "cot30plus_gradient", "CTH10+ gradient", "COT30+ gradient", "CTH10+ vs COT30+"),
    ("cth_gradient", "cot_gradient", "CTH gradient (m/h)", "COT gradient (COT/h)", "CTH vs COT"),
    ("euclid_msg_grid_gradient", "precipitation_p95_gradient", "Lightning gradient \n (counts/h)", "Precipitation p95 gradient (mm/h)", "Lightning vs precipitation p95"),
]
SCATTER_ROW_HEIGHT_INCHES = 5.5
TIME_SERIES_ROW_HEIGHT_INCHES = 5.0
# (name, classes, color) for the fit through the group-mean P25/P50/P75/P95 points
FIT_GROUPS = [
    ("Growing (5, 6)", [5, 6], "#F16913"),
    ("Decaying (9, 2, 3)", [9, 2, 3], "#54278F"),
]
FIT_DEGREE = 1  # 1 = linear, 2 = quadratic



# read columns for first scatter plot
scatter_columns = [
    "cth_mean",
    "cth_std", 
    "cma_mean",
    "cma_std",
    "cot_mean",
    "cot_std",
    "precipitation_mean",
    "precipitation_std",
    "euclid_msg_grid_mean",
    "euclid_msg_grid_std",
    "cth10plus_mean",
    "cth10plus_std",
    "cot30plus_mean",
    "cot30plus_std" 
]

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILES = {
    "training_video_summary": config["output_files"]["training_video_summary_temporal"],
    "training_csv_cth": config["output_files"]["training_csv_cth"],
    "training_csv_cot": config["output_files"]["training_csv_cot"],
    "training_csv_euclid_msg_grid": config["output_files"]["training_csv_euclid_msg_grid"],
    "training_csv_cth10plus": config["output_files"]["training_csv_cth10plus"],
    "training_csv_cot30plus": config["output_files"]["training_csv_cot30plus"],
    "training_csv_cma": config["output_files"]["training_csv_cma"],
    "training_csv_precipitation": config["output_files"]["training_csv_precipitation"],
    "training_distances": config["output_files"]["training_features_distances_centroid"],

}
N_CLASSES = 10
N_FRAMES = 8
TARGET_DISTANCE = 1
CHECK_TARGET_DISTANCE = 0
N_SAMPLES_PER_CLASS = 2000
PERCENTILE_COLUMNS = ["25", "50", "75", "95"]
SLOPE_DECIMALS = 2
CLASS_TO_PLOT = {
    "Class 6": [6],
    "Class 5": [5],
    "Class 9": [9],
    "Class 2": [2],
    "Class 3": [3],
}
CLASSES_IN_LEGEND = [6, 5, 9, 2, 3]
PLOT_SPECS = {
    "cth": {
        "csv_key": "training_csv_cth",
        "mode": "percentiles",
        "summary_column": "50",
        "ylabel_abs": "Cloud-top height (m)",
        "ylabel_delta": "$\\Delta$Cloud-top\nheight (m)",
        "slope_unit": "m/frame",
    },
    "cot": {
        "csv_key": "training_csv_cot",
        "mode": "percentiles",
        "ylabel_abs": "Cloud optical thickness",
        "ylabel_delta": r"$\Delta$COT",
        "slope_unit": "COT/frame",
    },
    "euclid_msg_grid": {
        "csv_key": "training_csv_euclid_msg_grid",
        "mode": "single_column",
        "value_column": "None",
        "scale": 1,
        "ylabel_abs": "Lightning counts",
        "ylabel_delta": r"$\Delta$Lightning counts",
        "slope_unit": "counts/frame",
    },
    "cth10plus": {
        "csv_key": "training_csv_cth10plus",
        "mode": "single_column",
        "value_column": "fraction",
        "scale": 100,
        "ylabel_abs": "CTH10+ area fraction (%)",
        "ylabel_delta": r"$\Delta$CTH10+ area fraction (%)",
        "slope_unit": "%/frame",
    },
    "cot30plus": {
        "csv_key": "training_csv_cot30plus",
        "mode": "single_column",
        "value_column": "fraction",
        "scale": 100,
        "ylabel_abs": "COT30+ area fraction (%)",
        "ylabel_delta": r"$\Delta$COT30+ area fraction (%)",
        "slope_unit": "%/frame",
    },
    "cma": {
        "csv_key": "training_csv_cma",
        "mode": "single_column",
        "value_column": "None",
        "scale": 100,
        "ylabel_abs": "Cloud cover (%)",
        "ylabel_delta": r"$\Delta$Cloud cover (%)",
        "slope_unit": "%/frame",
    },
    "precipitation": {
        "csv_key": "training_csv_precipitation",
        "mode": "percentiles",
        "summary_column": "sum[mm]",
        "ylabel_abs": "Cumulative Precipitation (mm)",
    "ylabel_delta": "$\\Delta$Cumulative\nPrecipitation (mm)",
        "slope_unit": "mm/frame",
    },
}

# select the variables to plot in the figure 4
PLOTS_SPEC_SELECTED = {
    "cth": PLOT_SPECS["cth"],
    "cma": PLOT_SPECS["cma"],
    "precipitation": PLOT_SPECS["precipitation"],
    "cth10plus": PLOT_SPECS["cth10plus"],
}


def clean_variable_df(df):
    """
    Clean the variable dataframe by removing rows with label -100 and converting the label column to numeric.
    """
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df[df["label"] != -100]

def load_variable_dataframe(csv_path):
    return pd.read_csv(csv_path)


def extract_crop_name(path_value):
    return os.path.basename(path_value)


def select_samples_closest_to_target_distance(df, target_distance=TARGET_DISTANCE, n_samples=N_SAMPLES_PER_CLASS):
    """
    Select the n_samples rows from the dataframe df that have the closest distance to the target_distance.
    """
    distance_to_target = (df["distance"] - target_distance).abs()
    return df.loc[distance_to_target.nsmallest(n_samples).index]


def load_all_datasets(video_stats_df):
    """Load the time-series CSVs for the plotted variables using all samples (no distance selection)."""
    variable_dfs = {}
    for variable_name, plot_spec in PLOT_SPECS.items():
        variable_df = load_variable_dataframe(CSV_FILES[plot_spec["csv_key"]])
        variable_df = clean_variable_df(variable_df)
        if variable_name == "euclid_msg_grid":
            lightning_per_crop = pd.to_numeric(variable_df[plot_spec["value_column"]], errors="coerce").groupby(variable_df["crop"]).transform("sum")
            variable_df = variable_df[lightning_per_crop > 0]
        variable_dfs[variable_name] = variable_df

    return video_stats_df, variable_dfs


def plot_gradient_percentile_scatter_row(fig, grid, video_stats_plot_df, panels=None, label_prefix="a"):
    """Scatter of class gradient percentiles (25, 50, 75, 95) for each pair in panels."""
    panels = GRADIENT_SCATTER_PANELS if panels is None else panels
    class_labels = [label for labels in CLASS_TO_PLOT.values() for label in labels]
    class_df = video_stats_plot_df[video_stats_plot_df["label"].isin(class_labels)]

    for panel_index, (x_column, y_column, x_label, y_label, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, panel_index])
        panel_df = class_df
        if x_column == "euclid_msg_grid_gradient":
            panel_df = class_df[class_df["euclid_msg_grid_mean"] > 0]
        percentile_frames = {}
        for percentile, marker in GRADIENT_PERCENTILE_MARKERS.items():
            percentile_values = panel_df.groupby("label")[[x_column, y_column]].quantile(percentile)
            percentile_frames[percentile] = percentile_values
            for class_label, row in percentile_values.iterrows():
                ax.scatter(
                    row[x_column],
                    row[y_column],
                    color=colors_per_class_codes_grl[str(int(class_label))],
                    marker=marker,
                    edgecolor="black",
                    s=MARKER_SIZE,
                    linewidth=1.2,
                    zorder=3,
                )
        for group_name, group_labels, group_color in FIT_GROUPS:
            mean_points = np.array([
                [
                    frame.reindex(group_labels)[x_column].mean(),
                    frame.reindex(group_labels)[y_column].mean(),
                ]
                for frame in percentile_frames.values()
            ])
            if not np.isfinite(mean_points).all():
                continue
            coefficients = np.polyfit(mean_points[:, 0], mean_points[:, 1], FIT_DEGREE)
            x_fit = np.linspace(mean_points[:, 0].min(), mean_points[:, 0].max(), 200)
            ax.plot(
                x_fit, np.polyval(coefficients, x_fit), color=group_color, linewidth=3,
                zorder=2,
            )
            print(f"{title} | {group_name}: degree-{FIT_DEGREE} coefficients (highest first) = {coefficients}")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_box_aspect(1)
        if x_column == "euclid_msg_grid_gradient":
            # symlog because lightning gradients can be negative
            ax.set_xscale("symlog", linthresh=1)
            # keep only 0 and sparse powers of ten so the labels do not overlap
            x_min, x_max = ax.get_xlim()
            n_decades = int(np.ceil(np.log10(max(abs(x_min), abs(x_max), 10))))
            step = 1 if n_decades <= 3 else 2
            powers = [10.0 ** k for k in range(0, n_decades + 1, step)]
            candidate_ticks = [0.0] + powers + [-p for p in powers]
            ax.set_xticks([t for t in candidate_ticks if x_min <= t <= x_max])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel(x_label, fontsize=SCATTER_LABEL_FONTSIZE)
        ax.set_ylabel(y_label, fontsize=SCATTER_LABEL_FONTSIZE)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.grid(alpha=0.2)
        style_axis(ax)
        ax.tick_params(axis="both", labelsize=SCATTER_TICK_FONTSIZE)
        add_panel_label(ax, f"{label_prefix}{panel_index + 1}")


def add_gradient_percentile_legend(fig, top=0.95):
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markerfacecolor="0.5",
            markeredgecolor="black",
            markersize=14,
            label=f"{int(percentile * 100)}th",
        )
        for percentile, marker in GRADIENT_PERCENTILE_MARKERS.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.865, top),
        ncol=1,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handletextpad=0.6,
    )
    fit_handles = [
        Line2D([0], [0], color=group_color, linewidth=3, label=group_name)
        for group_name, _, group_color in FIT_GROUPS
    ]
    # each legend entry takes roughly 0.4 in
    fit_top = top - (len(handles) * 0.4 + 0.3) / fig.get_figheight()
    fig.legend(
        handles=fit_handles,
        loc="upper left",
        bbox_to_anchor=(0.865, fit_top),
        ncol=1,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handletextpad=0.6,
    )


def create_extra_scatter_figure(video_stats_plot_df, output_filename):
    print(f"Plotting {output_filename}...")
    fig = plt.figure(figsize=(25, SCATTER_ROW_HEIGHT_INCHES + 2.0))
    grid = fig.add_gridspec(
        1, len(EXTRA_GRADIENT_SCATTER_PANELS), left=0.08, right=0.84,
        top=0.88, bottom=0.18, wspace=0.55,
    )
    plot_gradient_percentile_scatter_row(
        fig, grid, video_stats_plot_df, panels=EXTRA_GRADIENT_SCATTER_PANELS, label_prefix=""
    )
    add_gradient_percentile_legend(fig)
    add_class_legend(fig)
    plt.savefig(OUTPUT_DIR / output_filename, dpi=300)
    plt.close(fig)


def create_finite_difference_figure(variable_dfs, output_filename):
    """Same time-series rows as figure 4 (delta from frame 0, slope annotations) for all variables."""
    print(f"Plotting {output_filename}...")
    n_rows = len(PLOT_SPECS)
    n_columns = len(CLASS_TO_PLOT)
    fig = plt.figure(figsize=(25, TIME_SERIES_ROW_HEIGHT_INCHES * n_rows + 1.5))
    grid = fig.add_gridspec(
        n_rows, n_columns, left=0.08, right=0.84, top=0.96, bottom=0.05,
        hspace=MAIN_ROW_HSPACE, wspace=0.12,
    )
    axes = np.empty((n_rows, n_columns), dtype=object)
    for row in range(n_rows):
        for column in range(n_columns):
            axes[row, column] = fig.add_subplot(
                grid[row, column], sharey=axes[row, 0] if column else None
            )

    for i, (variable_name, plot_spec) in enumerate(PLOT_SPECS.items()):
        plot_time_series_row(
            axes[i, :],
            variable_dfs[variable_name],
            class_2_plot=CLASS_TO_PLOT,
            variable_name=variable_name,
            plot_spec=plot_spec,
            show_xlabel=i == n_rows - 1,
            row_index=i,
        )
        for ax in axes[i, 1:]:
            ax.tick_params(axis="y", labelleft=False)

    add_class_legend(fig)
    add_percentile_legend(fig, frameon=False)
    plt.savefig(OUTPUT_DIR / output_filename, dpi=300)
    plt.close(fig)


def create_figure(video_stats_plot_df, variable_dfs, output_filename):
    print(f"Plotting {output_filename}...")
    n_rows = len(PLOTS_SPEC_SELECTED)
    n_columns = len(CLASS_TO_PLOT)
    time_series_height = TIME_SERIES_ROW_HEIGHT_INCHES * n_rows
    top_margin, row_gap, bottom_margin = 1.0, 1.6, 1.2
    fig_height = top_margin + time_series_height + row_gap + SCATTER_ROW_HEIGHT_INCHES + bottom_margin
    fig = plt.figure(figsize=(25, fig_height))

    time_series_top = 1 - top_margin / fig_height
    time_series_bottom = time_series_top - time_series_height / fig_height
    time_series_grid = fig.add_gridspec(
        n_rows, n_columns, left=0.08, right=0.84,
        top=time_series_top, bottom=time_series_bottom,
        hspace=MAIN_ROW_HSPACE, wspace=0.12,
    )

    scatter_top = time_series_bottom - row_gap / fig_height
    scatter_bottom = scatter_top - SCATTER_ROW_HEIGHT_INCHES / fig_height
    scatter_grid = fig.add_gridspec(
        1, len(GRADIENT_SCATTER_PANELS), left=0.08, right=0.84,
        top=scatter_top, bottom=scatter_bottom, wspace=0.6,
    )
    plot_gradient_percentile_scatter_row(
        fig, scatter_grid, video_stats_plot_df, label_prefix=chr(ord("a") + n_rows)
    )
    axes = np.empty((n_rows, n_columns), dtype=object)
    for row in range(n_rows):
        for column in range(n_columns):
            axes[row, column] = fig.add_subplot(
                time_series_grid[row, column], sharey=axes[row, 0] if column else None
            )

    for i, (variable_name, plot_spec) in enumerate(PLOTS_SPEC_SELECTED.items()):
        plot_time_series_row(axes[i, :],
                            variable_dfs[variable_name],
                            class_2_plot=CLASS_TO_PLOT,
                            variable_name=variable_name,
                            plot_spec=plot_spec,
                            show_xlabel=i == len(PLOTS_SPEC_SELECTED) - 1,
                            row_index=i)
        for ax in axes[i, 1:]:
            ax.tick_params(axis="y", labelleft=False)

    add_gradient_percentile_legend(fig, top=scatter_top)
    add_class_legend(fig)
    add_percentile_legend(fig, frameon=False)
    plt.savefig(OUTPUT_DIR / output_filename, dpi=300)
    plt.close(fig)


def get_time_series_value_column(plot_spec):
    if plot_spec["mode"] == "percentiles":
        return plot_spec.get("summary_column", "50")
    return plot_spec["value_column"]


def get_available_time_series_percentiles(df):
    return [column for column in TIME_SERIES_PERCENTILES_TO_PLOT if column in df.columns]


def compute_frame_index_percentiles(class_df, value_column, scale):
    percentiles = [int(column) / 100 for column in TIME_SERIES_PERCENTILES_TO_PLOT]
    class_time_series = (
        class_df.groupby("frame_index")[value_column]
        .quantile(percentiles)
        .unstack()
        .reset_index()
    )
    class_time_series = class_time_series.rename(
        columns={percentile: str(int(percentile * 100)) for percentile in percentiles}
    )
    percentile_columns = get_available_time_series_percentiles(class_time_series)
    class_time_series[percentile_columns] = class_time_series[percentile_columns] * scale
    return class_time_series


def prepare_class_time_series(variable_df, plot_spec):
    variable_df = variable_df.copy()
    variable_df["time"] = pd.to_datetime(variable_df["time"], errors="coerce")
    required_columns = ["crop", "label", "time"]
    value_column = get_time_series_value_column(plot_spec)

    if plot_spec["mode"] == "percentiles":
        for column in PERCENTILE_COLUMNS:
            variable_df[column] = pd.to_numeric(variable_df[column], errors="coerce")
        if value_column not in PERCENTILE_COLUMNS:
            variable_df[value_column] = pd.to_numeric(variable_df[value_column], errors="coerce")
        required_columns.append(value_column)
    else:
        variable_df[value_column] = pd.to_numeric(variable_df[value_column], errors="coerce")
        required_columns.append(value_column)

    variable_df = variable_df.dropna(subset=required_columns)
    variable_df = variable_df[variable_df["label"] != -100]

    class_time_series_by_label = {}
    for class_label in sorted(variable_df["label"].dropna().astype(int).unique()):
        class_df = variable_df[variable_df["label"] == class_label].copy()
        class_df = class_df.sort_values(["crop", "time"])
        class_df["frame_index"] = class_df.groupby("crop").cumcount()
        class_df = class_df[class_df["frame_index"] < N_FRAMES]

        complete_crops = class_df.groupby("crop")["frame_index"].nunique()
        complete_crops = complete_crops[complete_crops == N_FRAMES].index
        class_df = class_df[class_df["crop"].isin(complete_crops)]
        if class_df.empty:
            continue

        scale = plot_spec.get("scale", 1)
        class_time_series = compute_frame_index_percentiles(class_df, value_column, scale)

        class_time_series_by_label[class_label] = class_time_series

    return class_time_series_by_label


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
    )


def plot_variable_time_series_on_axes(
    axes,
    variable_name,
    plot_spec,
    class_time_series_by_label,
    show_xlabel=False,
    row_index=0,
):
    frames = np.arange(N_FRAMES)

    for column_index, (ax, (group_name, class_labels)) in enumerate(zip(axes, CLASS_TO_PLOT.items())):
        add_panel_label(ax, f"{chr(ord('a') + row_index)}{column_index + 1}")
        if row_index == 0:
            ax.set_title(group_name, fontsize=TITLE_FONTSIZE, fontweight="bold")
        plotted_classes = [
            class_label for class_label in class_labels if class_label in class_time_series_by_label
        ]

        if not plotted_classes:
            ax.text(
                0.5,
                0.5,
                f"No classes available for {group_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(group_name.title(), fontsize=TITLE_FONTSIZE)
            style_axis(ax)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            continue

        annotation_entries = []

        for class_label in plotted_classes:
            series_df = class_time_series_by_label[class_label]
            percentile_columns = get_available_time_series_percentiles(series_df)
            if percentile_columns:
                for percentile_column in percentile_columns:
                    percentile_values = series_df[percentile_column].to_numpy()
                    percentile_delta = percentile_values - percentile_values[0]
                    if percentile_column == "50":
                        linestyle = "-"
                    elif percentile_column == "25":
                        linestyle = ":"
                    else:
                        linestyle = "--"
                    marker = "o" if percentile_column == "50" else None
                    ax.plot(
                        frames,
                        percentile_delta,
                        color=colors_per_class_codes_grl[str(class_label)],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=TIME_SERIES_LINEWIDTH if percentile_column == "50" else TIME_SERIES_PERCENTILE_LINEWIDTH,
                        markersize=TIME_SERIES_MARKER_SIZE,
                        alpha=1.0 if percentile_column == "50" else 0.75,
                        label=f"Class {class_label}",
                    )
                central_series = series_df["50"].to_numpy()
            else:
                mean = series_df["mean"].to_numpy()
                delta = mean - mean[0]
                ax.plot(
                    frames,
                    delta,
                    color=colors_per_class_codes_grl[str(class_label)],
                    marker="o",
                    linewidth=TIME_SERIES_LINEWIDTH,
                    markersize=TIME_SERIES_MARKER_SIZE,
                    label=f"Class {class_label}",
                )
                central_series = mean

            slope = np.polyfit(frames, central_series, 1)[0]
            annotation_entries.append(
                {
                    "text": f"{slope:+.{SLOPE_DECIMALS}f}",
                    "color": colors_per_class_codes_grl[str(class_label)],
                }
            )

        ax.axhline(0, color="0.35", linewidth=1)
        if annotation_entries:
            bbox_style = {"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0}
            x_positions = np.linspace(0.2, 0.8, num=len(annotation_entries))
            for x_position, entry in zip(x_positions, annotation_entries):
                ax.text(
                    x_position,
                    0.06,
                    entry["text"],
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=21,
                    color=entry["color"],
                    fontweight="bold",
                    bbox=bbox_style,
                )
        if column_index == 0:
            ax.set_ylabel(plot_spec["ylabel_delta"], fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")
        if show_xlabel:
            ax.set_xlabel("Frame index", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_xlabel("")
        if column_index == len(CLASS_TO_PLOT) - 1:
            ax.text(
                0.98,
                0.98,
                f"gradient units: \n {plot_spec['slope_unit']}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=18,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
            )
        ax.grid(axis="y", alpha=0.2)
        style_axis(ax)
        ax.set_xticks(frames)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

def add_class_legend(fig):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors_per_class_codes_grl[str(label)],
            markeredgecolor=colors_per_class_codes_grl[str(label)],
            markersize=16,
            label=f"Class {label}",
        )
        for label in CLASSES_IN_LEGEND
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.865, 0.5),
        ncol=1,
        frameon=False,
        fontsize=23,
        handletextpad=0.4,
        columnspacing=0.8,
    )


def add_percentile_legend(fig, frameon=False):
    handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=":",
            linewidth=TIME_SERIES_PERCENTILE_LINEWIDTH,
            label="25th percentile",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="-",
            marker="o",
            linewidth=TIME_SERIES_LINEWIDTH,
            markersize=TIME_SERIES_MARKER_SIZE,
            label="50th percentile",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="--",
            linewidth=TIME_SERIES_PERCENTILE_LINEWIDTH,
            label="75th percentile",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.865, 0.33),
        ncol=1,
        frameon=frameon,
        fontsize=LEGEND_FONTSIZE,
        handletextpad=0.6,
        columnspacing=0.8,
    )


def tighten_lower_time_series_spacing(axes, shift=LOWER_TIME_SERIES_ROW_SHIFT):
    for row_index in range(2, axes.shape[0]):
        row_shift = shift * (row_index - 1)
        for ax in axes[row_index, :]:
            position = ax.get_position()
            ax.set_position([
                position.x0,
                position.y0 + row_shift,
                position.width,
                position.height,
            ])
    
def main():

    print("Loading video summary CSV file...")
    video_stats_df = clean_variable_df(pd.read_csv(CSV_FILES["training_video_summary"]))

    video_stats_plot_df, variable_dfs = load_all_datasets(video_stats_df)
    create_figure(video_stats_plot_df, variable_dfs, "figure4_convection_characterization.png")
    create_extra_scatter_figure(video_stats_plot_df, "figure4_additional_scatter_plots.png")
    create_finite_difference_figure(variable_dfs, "figure4_additional_finite_differences.png")


def plot_scatter_row(fig, axes, video_stats_plot_df, class_2_plot):
    """
    code inspired by test_2d-scatter to plot the scatter plots of 
    - CTH mean vs Cloud Cover
    - Precipitation mean vs EUCLID MSG grid mean
    For each column, the subplot should be organized in 2 scatter plots, one for each pair of variables. Each column
    corresponds to a different convective class (growing, decaying day, decaying night). Each scatter plot should have
    the median of the entire distribution of the class plotted with a larger marker.

    """
    first_group_name = next(iter(class_2_plot))
    shared_top_ax = None
    shared_bottom_ax = None
    selected_labels = [label for class_labels in class_2_plot.values() for label in class_labels]
    selected_df = video_stats_plot_df[video_stats_plot_df["label"].isin(selected_labels)]
    top_values = selected_df["cth_mean"].dropna()
    bottom_values = selected_df.loc[selected_df["euclid_msg_grid_mean"] > 0, "euclid_msg_grid_mean"].dropna()

    for column_index, (parent_ax, (group_name, class_labels)) in enumerate(zip(axes, class_2_plot.items())):
        parent_spec = parent_ax.get_subplotspec()
        parent_ax.remove()
        subgrid = parent_spec.subgridspec(2, 1, hspace=0.45)
        ax_top = fig.add_subplot(subgrid[0, 0], sharey=shared_top_ax)
        ax_bottom = fig.add_subplot(subgrid[1, 0], sharey=shared_bottom_ax)
        add_panel_label(ax_top, f"a{column_index + 1}")
        add_panel_label(ax_bottom, f"a{len(CLASS_TO_PLOT) + column_index + 1}")
        if shared_top_ax is None:
            shared_top_ax = ax_top
        if shared_bottom_ax is None:
            shared_bottom_ax = ax_bottom

        if group_name == first_group_name:
            bbox = ax_top.get_position()
            fig.text(
                bbox.x0,
                bbox.y1 + SECTION_TITLE_OFFSET,
                "a) Convective life cycle",
                fontsize=TITLE_FONTSIZE,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        for class_label in class_labels:
            class_df = video_stats_plot_df[video_stats_plot_df["label"] == class_label]
            if class_df.empty:
                continue
            positive_lightning_df = class_df[class_df["euclid_msg_grid_mean"] > 0]

            ax_top.scatter(
                class_df["cma_mean"] * 100,
                class_df["cth_mean"],
                color=colors_per_class_codes_grl[str(class_label)],
                label=f"Class {class_label}",
                edgecolor=MARKER_EDGE_COLOR,
                s=MARKER_SIZE,
                alpha=0.5,
            )

            if not positive_lightning_df.empty:
                ax_bottom.scatter(
                    positive_lightning_df["precipitation_mean"],
                    positive_lightning_df["euclid_msg_grid_mean"],
                    color=colors_per_class_codes_grl[str(class_label)],
                    label=f"Class {class_label}",
                    edgecolor=MARKER_EDGE_COLOR,
                    s=MARKER_SIZE,
                    alpha=0.5,
                )

        # loop to plot median of the entire distribution of the class for both scatter plots
        for class_label in class_labels:
            median_row = video_stats_plot_df[video_stats_plot_df["label"] == class_label][
                ["cma_mean", "cth_mean"]
            ].median(numeric_only=True)
            ax_top.scatter(
                median_row["cma_mean"] * 100,
                median_row["cth_mean"],
                color=colors_per_class_codes_grl[str(class_label)],
                edgecolor="white",
                marker="X",
                s=MARKER_SIZE * 10,
                linewidth=2.5,
            )

            positive_bottom_df = video_stats_plot_df[
                (video_stats_plot_df["label"] == class_label)
                & (video_stats_plot_df["euclid_msg_grid_mean"] > 0)
            ]
            if not positive_bottom_df.empty:
                bottom_median_row = positive_bottom_df[
                    ["precipitation_mean", "euclid_msg_grid_mean"]
                ].median(numeric_only=True)
                ax_bottom.scatter(
                    bottom_median_row["precipitation_mean"],
                    bottom_median_row["euclid_msg_grid_mean"],
                    color=colors_per_class_codes_grl[str(class_label)],
                    edgecolor="white",
                    marker="X",
                    s=MARKER_SIZE * 10,
                    linewidth=2.5,
                )
        ax_top.set_title(group_name.replace("_", " ").title(), fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")
        if column_index == 0:
            ax_top.set_ylabel("Mean cloud \n top height (m)", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax_top.set_ylabel("")
            ax_top.tick_params(axis="y", labelleft=False)
        ax_top.set_xlabel("Cloud Cover (%)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=14)
        ax_top.set_xlim(0., 100.)
        ax_top.grid(axis="y", alpha=0.2)
        style_axis(ax_top)
        ax_top.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

        if column_index == 0:
            ax_bottom.set_ylabel("Lightning counts", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax_bottom.set_ylabel("")
            ax_bottom.tick_params(axis="y", labelleft=False)
        ax_bottom.set_xlabel("Mean cumulative Precipitation (mm)", fontsize=AXIS_LABEL_FONTSIZE)
        ax_bottom.set_xlim(0.,6000.)
        ax_bottom.set_yscale("log")
        ax_bottom.grid(axis="y", alpha=0.2)
        style_axis(ax_bottom)
        ax_bottom.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

    if shared_top_ax is not None and not top_values.empty:
        top_min = top_values.min()
        top_max = top_values.max()
        top_padding = max((top_max - top_min) * 0.05, 1)
        shared_top_ax.set_ylim(top_min - top_padding, top_max + top_padding)

    if shared_bottom_ax is not None and not bottom_values.empty:
        bottom_min = bottom_values.min()
        bottom_max = bottom_values.max()
        shared_bottom_ax.set_ylim(bottom_min * 0.8, bottom_max * 1.2)




def plot_time_series_row(
    axes,
    variable_df,
    class_2_plot,
    variable_name,
    plot_spec,
    show_xlabel=False,
    row_index=0,
):
    """Plot the temporal delta series for one variable across the three class-group axes."""
    class_time_series_by_label = prepare_class_time_series(variable_df, plot_spec)
    plot_variable_time_series_on_axes(
        axes,
        variable_name,
        plot_spec,
        class_time_series_by_label,
        show_xlabel=show_xlabel,
        row_index=row_index,
    )


if __name__ == "__main__":
    main()


