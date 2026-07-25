"""
This figure demonstrates the physical interpretability of the classes.
It is a multipanel figure with 3 main rows. Each row contains 3 histogram plots corresponding
 to a different set of properties:
- Row 1: Diurnal cycle of cloud classes. First histogram is for clouds with prominence 
of presence during night, second histogram is for clouds with prominence during day, and
 third histogram is for clouds with no strong diurnal preference.
- Row 2: Cloud bulk properties of cloud classs. We have 3 histograms with:
    - cloud cover vs CTH
    - cloud fraction vs precipitation fraction
    - cumulated precipitation vs lightning counts
    
- Row 3: Temporal characteristics of cloud classes. HEre we have 3 histograms with:
    - gradient of COT vs grad CTH
    - grad cloud cover vs grad CTH
    - grad COT30+ vs grad CTH10+

The plot stores the figure in the output directory specified in the config file.
Author: Claudia Acquistapace
Date: 24/07/2026
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.configs import load_config
from scripts.pretrain.cluster_analysis.classes_diurnal_cycle import (
    load_hourly_occurrence,
)
from utils.plotting.class_colors import (
    colors_per_class_codes_grl,
    class_groups_diurnal_cycle,
)
from utils.plotting.plot_class_analysis import plot_hourly_histogram
from scripts.pretrain.cluster_analysis.scatter_video_csv import style_zero_centered_axes, get_axis_values_with_training_percentiles, clean_video_summary_df, plot_class_percentile_errorbar_points, plot_class_errorbar_points
try:
    import cmcrameri.cm as cmc
except ImportError:  # pragma: no cover - optional dependency
    cmc = None


OUTPUT_FILENAME = "figure3_classes_physics_interpretability.png"
N_CLASSES = 10
N_FRAMES = 8
OUTPUT_DPI = 220
TRANSITION_GAMMA = 0.5
FONT_SIZE_TEXT = 23 
TICK_LABEL_SIZE = 23
COMMON_YMAX = 30
DIURNAL_LINEWIDTH = 4.0
FIGURE_LEFT = 0.08
FIGURE_RIGHT = 0.97
FIGURE_BOTTOM = 0.14
FIGURE_TOP = 0.94
ROW_TITLE_OFFSET = 0.025
BULK_ROW_TITLE_OFFSET = 0.04
HOUR_BIN_SIZE = 2
HOUR_BINS = list(range(0, 24, HOUR_BIN_SIZE))
HOUR_BIN_CENTERS = np.array(HOUR_BINS) + HOUR_BIN_SIZE / 2
HOUR_BIN_LABELS = [f"{hour:02d}-{hour + HOUR_BIN_SIZE:02d}" for hour in HOUR_BINS]
SCATTER_MARKER_SIZE = 18
SCATTER_MARKER_AREA = SCATTER_MARKER_SIZE ** 2
SCATTER_MARKER_AREA_LARGE = 1.2 * SCATTER_MARKER_AREA
HIGHLIGHT_MARKER_SIZE = 480
HIGHLIGHT_MARKER_SIZE_LARGE = 580
SCATTER_COLUMNS = [
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
    "cot30plus_std",    
    "prec_fraction_mean", 
    "prec_fraction_std",
]
# load video summary csv file path from config
CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"
config = load_config(str(CONFIG_PATH))
CSV_FILES = {
    "training_video_summary": config["output_files"]["training_video_summary_temporal"],
    "training_distances": config["output_files"]["training_features_distances_centroid"],
}
HIGHLIGHT_CLASSES = {0, 1, 4, 8}
CMA_GRADIENT_LIMIT = 0.2
CTH_MEAN_LIMIT = 8000
CTH_GRADIENT_LIMIT = 800

def style_axis(ax):
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=7, labelsize=TICK_LABEL_SIZE)


def set_hour_bin_axis(ax):
    ax.set_xticks(HOUR_BIN_CENTERS)
    sparse_hour_labels = [label if index % 2 == 0 else "" for index, label in enumerate(HOUR_BIN_LABELS)]
    ax.set_xticklabels(sparse_hour_labels, rotation=45, ha="right")
    ax.set_xlim(HOUR_BINS[0], HOUR_BINS[-1] + HOUR_BIN_SIZE)


def add_row_title(fig, gs, row_index, title, offset=ROW_TITLE_OFFSET):
    row_bbox = gs[row_index, 0].get_position(fig)
    fig.text(
        row_bbox.x0,
        row_bbox.y1 + offset,
        title,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE_TEXT,
        fontweight="bold",
    )


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
        for label in range(N_CLASSES)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05), 
        ncol=N_CLASSES,
        frameon=False,
        fontsize=23,
        handletextpad=0.4,
        columnspacing=1.1,
    )


def plot_class_errorbars(ax, class_means, x_mean, y_mean, x_std, y_std, color="gray"):
    for label in class_means.index:
        xerr = class_means.loc[label, x_std]
        yerr = class_means.loc[label, y_std]
        ax.errorbar(
            class_means.loc[label, x_mean],
            class_means.loc[label, y_mean],
            xerr=0 if not np.isfinite(xerr) else xerr,
            yerr=0 if not np.isfinite(yerr) else yerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=0,
            zorder=2,
        )


def highlight_selected_classes(ax, df, x_column, y_column, size=340):
    selected = df[df.index.isin(HIGHLIGHT_CLASSES)]
    if selected.empty:
        return

    percent_scaled_columns = {"cma_mean", "prec_fraction_mean"}
    x_scale = 100 if x_column in percent_scaled_columns else 1
    y_scale = 100 if y_column in percent_scaled_columns else 1

    highlight = ax.scatter(
        selected[x_column] * x_scale,
        selected[y_column] * y_scale,
        s=size,
        marker="D",
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=4,
    )
    highlight.set_linestyle("--")

def main():

    # load config file  
    config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
    config = load_config(config_path) 

    output_dir = Path(config["output_files"]["figures_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # generate a figure with three main rows each with their title and three histograms
    fig = plt.figure(figsize=(25, 20))
    gs = GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.55)
    fig.subplots_adjust(
        left=FIGURE_LEFT,
        right=FIGURE_RIGHT,
        bottom=FIGURE_BOTTOM,
        top=FIGURE_TOP,
    )

    # Row 1: Diurnal cycle of cloud classes
    row1_axes = [fig.add_subplot(gs[0, index]) for index in range(3)]

    # call function to run plot of diurnal cycle of cloud classes
    plot_first_row_diurnal_cycle(row1_axes, "training", CSV_FILES["training_video_summary"])

    add_row_title(fig, gs, 0, "a) Diurnal cycle of cloud classes")
    # Row 2 and Row 3 will be filled with the other plots (not implemented in this snippet)
    plot_second_row_bulk_properties(fig, gs, config)

    add_row_title(
        fig,
        gs,
        1,
        "b) Bulk properties of cloud classes",
        offset=BULK_ROW_TITLE_OFFSET,
    )

    plot_third_row_temporal_characteristics(fig, gs, config)

    add_row_title(fig, gs, 2, "c) Temporal characteristics of cloud classes")
    add_class_legend(fig)
    fig.savefig(output_dir / OUTPUT_FILENAME, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_first_row_diurnal_cycle(axes, mode, video_summary_csv):
    """
    Plot the diurnal cycle of cloud classes in the first row of the figure.
    The histogram shows the distribution of cloud classes over the 24 hours of the day.
    The x-axis is the hour of the day (0-23), and the y-axis is the count of occurrences.
    Each class is represented by a different color.

    Args:
        axes: matplotlib axes for day, night, and anytime groups
        video_summary_csv: path to the video summary CSV file
    """
    _ = video_summary_csv

    # Load the hourly occurrence DataFrame for the specified mode (training or testing)
    df_grouped = load_hourly_occurrence(mode)

    # Get the hour bin centers for the x-axis
    hours = HOUR_BIN_CENTERS

    group_titles = {
        "day": "1. Daytime classes",
        "night": "2. Nighttime classes",
        "anytime": "3. No strong diurnal preference",
    }
    for ax, (group_name, class_indices) in zip(axes, class_groups_diurnal_cycle.items()):
        plot_single_class_groups(ax, df_grouped, class_indices, hours, group_titles[group_name])


def plot_single_class_groups(ax, df_grouped: pd.DataFrame, group_labels, hours, title: str):
    has_any_label = False

    for label in group_labels:
        if label not in df_grouped.columns:
            continue
        has_any_label = True
        plot_hourly_histogram(
            ax,
            hours,
            df_grouped[label].to_numpy() * 100,  # Convert to percentage
            colors_per_class_codes_grl.get(str(label), None),
            f"Class {label}",
            linewidth=DIURNAL_LINEWIDTH,
        )

    ax.set_xlabel("Hour of the day (hh)", fontsize=FONT_SIZE_TEXT)
    ax.set_ylabel("Occurrence (%)", fontsize=FONT_SIZE_TEXT)
    ax.set_ylim(0, COMMON_YMAX)
    ax.set_title(title, fontsize=FONT_SIZE_TEXT, fontweight="bold", loc="left")
    set_hour_bin_axis(ax)
    style_axis(ax)


def plot_second_row_bulk_properties(fig, gs, config):
    """
    this function should plot 3 scatter plots in the row as they are created in the code scatter_video_csv.py
    using possibly the same functions

    """
    # read the data to use for the scatter plots
    video_stats_df = pd.read_csv(CSV_FILES["training_video_summary"])
    # drop class with label -100
    video_stats_df = clean_video_summary_df(video_stats_df)

    # calculate mean value for each class for each variable and its std
    class_means = video_stats_df.groupby("label")[SCATTER_COLUMNS].mean()

    # create the axes for the 3 scatter plots in the second row
    row2_axes = [fig.add_subplot(gs[1, index]) for index in range(3)]

    # first scatter plot: cloud cover vs CTH
    ax1 = row2_axes[0]
    cot_values = pd.to_numeric(class_means["cot_mean"], errors="coerce")
    cot_cmap = plt.cm.Greys
    scatter_cot = ax1.scatter(
        class_means["cma_mean"]*100,  # convert to percentage
        class_means["cth_mean"],
        c=cot_values,
        cmap=cot_cmap,
        s=SCATTER_MARKER_AREA_LARGE,
        edgecolors=[colors_per_class_codes_grl[str(label)] for label in class_means.index],
        linewidths=2.5, 
        zorder=3,
    )
    plot_class_errorbars(
        ax1,
        class_means,
        "cma_mean",
        "cth_mean",
        "cma_std",
        "cth_std",
    )
    cax = inset_axes(
        ax1,
        width="42%",
        height="5%",
        loc="lower right",
        bbox_to_anchor=(0.0, 0.14, 1.0, 1.0),
        bbox_transform=ax1.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(
        scatter_cot,
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_ticks([5, 10, 15, 20, 25])
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Mean COT", fontsize=18)
    cbar.ax.tick_params(labelsize=18, length=3)
    ax1.set_xlabel("Cloud Cover (%)", fontsize=FONT_SIZE_TEXT)
    ax1.set_ylabel("CTH (m)", fontsize=FONT_SIZE_TEXT)
    ax1.set_title("1.Cloud Cover vs CTH", fontsize=FONT_SIZE_TEXT, fontweight="bold", loc="left")
    ax1.set_xlim(0., 100.)
    highlight_selected_classes(ax1, class_means, "cma_mean", "cth_mean", size=HIGHLIGHT_MARKER_SIZE_LARGE)
    style_axis(ax1) 

    # second scatter plot: cloud cover vs precipitation fraction
    ax2 = row2_axes[1]
    ax2.scatter(
        class_means["cma_mean"]*100, # convert to percentage
        class_means["prec_fraction_mean"]*100, # convert to percentage
        c=[colors_per_class_codes_grl[str(label)] for label in class_means.index],
        s=SCATTER_MARKER_AREA,
        edgecolor="black",
    )
    plot_class_errorbar_points(
        ax2,
        class_means,
        "cma_mean",
        "prec_fraction_mean",
        "cma_std",
        "prec_fraction_std",
        dataset="training",
        marker_size=SCATTER_MARKER_SIZE,
    )
    ax2.set_xlabel("Cloud Cover (%)", fontsize=FONT_SIZE_TEXT)
    ax2.set_ylabel("Precipitation Fraction (%)", fontsize=FONT_SIZE_TEXT)
    ax2.set_title("2.Cloud Cover vs Precipitation Fraction", fontsize=FONT_SIZE_TEXT, fontweight="bold", loc="left")
    ax2.set_xlim(0., 100.)
    highlight_selected_classes(ax2, class_means, "cma_mean", "prec_fraction_mean", size=HIGHLIGHT_MARKER_SIZE)
    style_axis(ax2)


    # third scatter plot: cumulated precipitation vs lightning counts
    ax3 = row2_axes[2]
    ax3.scatter(
        class_means["precipitation_mean"],
        class_means["euclid_msg_grid_mean"],
        c=[colors_per_class_codes_grl[str(label)] for label in class_means.index],
        s=SCATTER_MARKER_AREA,
        edgecolor="black",
    )
    plot_class_errorbar_points(
        ax3,
        class_means,
        "precipitation_mean",
        "euclid_msg_grid_mean",
        "precipitation_std",
        "euclid_msg_grid_std",
        dataset="training",
        marker_size=SCATTER_MARKER_SIZE,
    )
    ax3.set_xlabel("Cumulated Precipitation (mm)", fontsize=FONT_SIZE_TEXT)
    ax3.set_ylabel("Mean lightning Counts (#)", fontsize=FONT_SIZE_TEXT)
    ax3.set_title("3.Cumulated Precipitation vs Lightning Counts", fontsize=FONT_SIZE_TEXT, fontweight="bold", loc="left")
    highlight_selected_classes(ax3, class_means, "precipitation_mean", "euclid_msg_grid_mean", size=HIGHLIGHT_MARKER_SIZE)
    style_axis(ax3)
    return row2_axes


def plot_third_row_temporal_characteristics(fig, gs, config):


   # read the data to use for the scatter plots
    video_stats_df = pd.read_csv(CSV_FILES["training_video_summary"])
    # drop class with label -100
    video_stats_df = clean_video_summary_df(video_stats_df)

    # calculate now mean and std of all columns ending with _gradient for each class
    gradient_columns = [col for col in video_stats_df.columns if col.endswith("_gradient")]
    class_gradients = video_stats_df.groupby("label")[gradient_columns].mean()  
    class_gradients_q25 = video_stats_df.groupby("label")[gradient_columns].quantile(0.25)
    class_gradients_q75 = video_stats_df.groupby("label")[gradient_columns].quantile(0.75)


    # create the axes for the 3 scatter plots in the second row
    row3_axes = [fig.add_subplot(gs[2, index]) for index in range(3)]

    # first scatter plot: cloud cover gradient vs CTH gradient
    ax1 = row3_axes[0]
    plot_class_percentile_errorbar_points( 
        ax1,
        class_gradients,
        class_gradients_q25,
        class_gradients_q75,
        "cma_gradient",
        "cth_gradient",
        dataset="training",
        marker_size=SCATTER_MARKER_SIZE,
    )

    ax1.set_xlabel("Cloud cover fraction gradient (%$\mathrm{h}^{-1}$)", fontsize=FONT_SIZE_TEXT)
    ax1.set_ylabel("CTH gradient (m$\mathrm{h}^{-1}$)", fontsize=FONT_SIZE_TEXT)
    ax1.set_title("1. Cloud cover gradient vs CTH gradient", fontsize=FONT_SIZE_TEXT, fontweight="bold", loc="left")
    highlight_selected_classes(ax1, class_gradients, "cma_gradient", "cth_gradient", size=HIGHLIGHT_MARKER_SIZE)
    style_axis(ax1)
    style_zero_centered_axes(
        ax1,
        get_axis_values_with_training_percentiles(
            class_gradients,
            class_gradients_q25,
            class_gradients_q75,
            None,
            "cma_gradient",
            False,
        ),
        get_axis_values_with_training_percentiles(
            class_gradients,
            class_gradients_q25,
            class_gradients_q75,
            None,
            "cth_gradient",
            False,
        ),
        x_limit_cap=CMA_GRADIENT_LIMIT,
        y_limit_cap=CTH_GRADIENT_LIMIT,
    )
    style_axis(ax1)
    ax1.set_xlim(-0.02, 0.02)
    ax1.set_ylim(-220, 220)

    # second scatter plot: COT gradient vs CTH gradient
    ax2 = row3_axes[1]
    plot_class_percentile_errorbar_points(
        ax2,
        class_gradients,
        class_gradients_q25,
        class_gradients_q75,
        "cot_gradient",
        "cth_gradient",
        dataset="training",
        marker_size=SCATTER_MARKER_SIZE,
    )

    ax2.set_xlabel("COT gradient ($\mathrm{h}^{-1}$)", fontsize=FONT_SIZE_TEXT)
    ax2.set_ylabel("CTH gradient (m$\mathrm{h}^{-1}$)", fontsize=FONT_SIZE_TEXT)
    ax2.set_title("2. COT gradient vs CTH gradient", fontsize=FONT_SIZE_TEXT, fontweight="bold", loc="left")
    highlight_selected_classes(ax2, class_gradients, "cot_gradient", "cth_gradient", size=HIGHLIGHT_MARKER_SIZE)
    style_axis(ax2)
    style_zero_centered_axes(
        ax2,
        get_axis_values_with_training_percentiles(
            class_gradients,
            class_gradients_q25,
            class_gradients_q75,
            None,
            "cot_gradient",
            False,
        ),
        get_axis_values_with_training_percentiles(
            class_gradients,
            class_gradients_q25,
            class_gradients_q75,
            None,
            "cth_gradient",
            False,
        ),
        y_limit_cap=CTH_GRADIENT_LIMIT,
    )
    style_axis(ax2)
    ax2.set_xlim(-4.5, 4.5)
    ax2.set_ylim(-220, 220)

    # third scatter plot: CTH10+ gradient vs COT30+ gradient
    ax3 = row3_axes[2]
    plot_class_percentile_errorbar_points(
        ax3,
        class_gradients,
        class_gradients_q25,
        class_gradients_q75,
        "cot30plus_gradient",
        "cth10plus_gradient",
        dataset="training",
        marker_size=SCATTER_MARKER_SIZE,
    )

    ax3.set_xlabel("COT30+ gradient ($\mathrm{h}^{-1}$)", fontsize=FONT_SIZE_TEXT)
    ax3.set_ylabel("CTH10+ gradient (m$\mathrm{h}^{-1}$)", fontsize=FONT_SIZE_TEXT)
    ax3.set_title("3. CTH10+ gradient vs COT30+ gradient", fontsize=FONT_SIZE_TEXT, fontweight="bold", loc="left")
    highlight_selected_classes(ax3, class_gradients, "cot30plus_gradient", "cth10plus_gradient", size=HIGHLIGHT_MARKER_SIZE)
    style_axis(ax3)
    style_zero_centered_axes(
        ax3,
        get_axis_values_with_training_percentiles(
            class_gradients,
            class_gradients_q25,
            class_gradients_q75,
            None,
            "cot30plus_gradient",
            False,
        ),
        get_axis_values_with_training_percentiles(
            class_gradients,
            class_gradients_q25,
            class_gradients_q75,
            None,
            "cth10plus_gradient",
            False,
        ),
    )
    style_axis(ax3)
    ax3.set_xlim(-0.03, 0.03)
    ax3.set_ylim(-0.018, 0.018)
    return row3_axes



if __name__ == "__main__":
    main()
    