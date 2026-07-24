"""
This code produces the structural layout for Figure 2, with 2 main rows:
- 1st row: a subgrid of 10 rows and 9 columns. Each row starts with a colored
  class box containing the class number, followed by 8 placeholder boxes for
  centroid-video frames ordered in time from left to right.
- 2nd row: a row with 3 plots for class representativity summaries:
    - bar plot for the number of videos per class
        - whisker plot for centroid-distance distributions per class
            Outliers are shown using the default boxplot definition, i.e. points beyond
            $1.5 \times \mathrm{IQR}$ from the first and third quartiles. IQR is the interquartile range, 
            i.e. the difference between the third and first quartiles, that are 75th and 25th percentiles of the distribution, respectively.
    - plot the distributions of the classes in the northern and southern parts of the domain

The current implementation builds the figure structure only, so it can be used
as a scaffold before wiring in the real data products.
"""

from __future__ import annotations
import pandas as pd

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
import numpy as np
try:
    import cmcrameri.cm as cmc
except ImportError:  # pragma: no cover - optional dependency
    cmc = None
BT108_COLORMAP = cmc.romaO_r if cmc is not None else plt.get_cmap("RdBu_r")


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.configs import load_config  # noqa: E402
from utils.plotting.class_colors import colors_per_class1_names  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"
OUTPUT_FILENAME = "figure2_classes_representativity_structure.png"
N_CLASSES = 10
N_FRAMES = 8
OUTPUT_DPI = 220
TRANSITION_GAMMA = 0.5
FONT_SIZE_TEXT = 20 

# load video summary csv file path from config
config = load_config(str(CONFIG_PATH))
CSV_FILES = {
    "training_video_summary": config["output_files"]["training_video_summary"],
    "training_distances": config["output_files"]["training_features_distances_centroid"],
}


def prepare_input_data():
    """
    prepare input data for the figure scaffold. This function reads the csv of the features and provides
    a dataframe with:
    - filename of the video crop
    - label of the video crop
    - distance from centroid of the video crop
    - datetime span by the video crop

    it also creates a separate dataframe with the same columns but containing the info only for the centroids videos, 
    which will be used to plot the frames in the figure scaffold.
    input:
    - feature csv file
    output:
    - dataframe with the columns: crop, label, distance, time_start, time_end
    """
    features_df = pd.read_csv(CSV_FILES["training_distances"])
    video_summary_df = pd.read_csv(CSV_FILES["training_video_summary"])

    features_required_columns = {"path", "label", "distance"}
    video_summary_required_columns = {"crop", "label", "time_start", "time_end", "lat_mid", "lon_mid"}

    missing_feature_columns = features_required_columns.difference(features_df.columns)
    if missing_feature_columns:
        raise KeyError(
            f"Missing columns in training features distances: {sorted(missing_feature_columns)}. "
            f"Available columns: {list(features_df.columns)}"
        )
    missing_summary_columns = video_summary_required_columns.difference(
        video_summary_df.columns
    )
    if missing_summary_columns:
        raise KeyError(
            f"Missing columns in training video summary: {sorted(missing_summary_columns)}. "
            f"Available columns: {list(video_summary_df.columns)}"
        )

    video_summary_df = video_summary_df.drop_duplicates(
        subset=["crop", "label", "time_start", "time_end"]
    ).copy()

    features_df = features_df.copy()
    features_df["crop"] = features_df["path"].map(lambda value: Path(value).name)
    merged_df = features_df.merge(
        video_summary_df[
            ["crop", "label", "time_start", "time_end", "lat_mid", "lon_mid"]
        ],
        on=["crop", "label"],
        how="inner",
        validate="many_to_one",
    )
    if merged_df.empty:
        raise ValueError(
            "The merged training features and training video summary dataframe is empty. "
            "Check that features 'path' filenames match the video summary 'crop' values."
        )

    # create a dataframe with the centroid videos only, by selecting the rows with the minimum distance per class
    centroid_indices = merged_df.groupby("label")["distance"].idxmax()
    centroid_videos_df = merged_df.loc[centroid_indices].sort_values("label")
    if centroid_videos_df.empty:
        raise ValueError("No centroid videos found in training features distances.")

    return merged_df, centroid_videos_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the layout scaffold for Figure 2 classes representativity."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to the configured figures directory.",
    )
    return parser.parse_args()


def resolve_output_path(output_argument: str | None) -> Path:
    if output_argument:
        output_path = Path(output_argument).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    config = load_config(str(CONFIG_PATH))
    output_dir = Path(config["output_files"]["figures_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / OUTPUT_FILENAME

def style_axis(ax):
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=7)


def style_placeholder_axis(ax: plt.Axes, *, facecolor: str = "#f4f4f4") -> None:
    ax.set_facecolor(facecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    # apply a light gray border
    style_axis(ax)



def style_square_axis(ax: plt.Axes, *, facecolor: str = "#f4f4f4") -> None:
    style_placeholder_axis(ax, facecolor=facecolor)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.set_box_aspect(1)


def draw_class_and_frames(parent_spec, centroid_videos_df, fig: plt.Figure) -> None:
    subgrid = parent_spec.subgridspec(
        N_CLASSES,
        N_FRAMES + 1,
        wspace=0.005,
        hspace=0.04,
        width_ratios=[1.0] * (N_FRAMES + 1),
    )
    anchor_ax = None

    # Draw the class boxes in the color of the class
    for class_index in range(N_CLASSES):
        color = colors_per_class1_names.get(str(class_index), "lightgray")

        class_ax = fig.add_subplot(subgrid[class_index, 0])
        if anchor_ax is None:
            anchor_ax = class_ax
        style_square_axis(class_ax, facecolor=color)
        class_ax.spines["right"].set_linewidth(1.5)
        class_ax.spines["top"].set_linewidth(1.5)
        class_ax.text(
            0.5,
            0.5,
            str(class_index),
            ha="center",
            va="center",
            fontsize=FONT_SIZE_TEXT,
            fontweight="bold",
            color="white" if np.mean(mcolors.to_rgb(color)) < 0.45 else "black",
        )

        if class_index == 0:
            class_ax.set_title("Class", fontsize=FONT_SIZE_TEXT, pad=8)

        # draw the frames from the centroids
        centroid_videos_class = centroid_videos_df[
            centroid_videos_df["label"] == class_index
        ]
        for frame_index in range(N_FRAMES):
            frame_ax = fig.add_subplot(subgrid[class_index, frame_index + 1])
            style_square_axis(frame_ax)
            if class_index == 0:
                frame_ax.set_title(f"T{frame_index + 1}", fontsize=FONT_SIZE_TEXT, pad=8)

            if centroid_videos_class.empty:
                frame_ax.text(
                    0.5,
                    0.5,
                    "missing\ncentroid",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )
                continue

            frame_path_column = f"t{frame_index}"
            frame_path = centroid_videos_class.iloc[0].get(frame_path_column)
            if isinstance(frame_path, str) and Path(frame_path).exists():
                frame_ax.imshow(plt.imread(frame_path), aspect="equal")
            else:
                frame_ax.text(
                    0.5,
                    0.5,
                    f"missing\n{frame_path_column}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )

    return anchor_ax


def draw_count_placeholder(ax: plt.Axes, features_df: pd.DataFrame) -> None:
    """Plot normalized class population as percent of the total sample count."""
    class_counts = (
        features_df["label"]
        .value_counts()
        .reindex(range(N_CLASSES), fill_value=0)
        .sort_index()
    )
    total_videos = class_counts.sum()
    normalized_counts = class_counts / total_videos
    bar_colors = [
        colors_per_class1_names.get(str(class_label), "lightgray")
        for class_label in normalized_counts.index
    ]

    ax.bar(
        normalized_counts.index,
        normalized_counts.values * 100,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title(
        "b) Class population",
        fontsize=FONT_SIZE_TEXT,
        pad=8,
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Class", fontsize=FONT_SIZE_TEXT)
    ax.set_ylabel("Percent of total (%)", fontsize=FONT_SIZE_TEXT)
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_xlim(-0.6, N_CLASSES - 0.4)
    ax.set_ylim(0, 25)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TEXT)
    style_axis(ax)


def draw_distance_placeholder(ax: plt.Axes, features_df: pd.DataFrame) -> None:
    """Plot the raw distance distribution per class as boxplots.
    The boxplots are drawn with the default matplotlib definition, i.e. points beyond
    $1.5 \times \mathrm{IQR}$ from the first and third quartiles are 
    considered outliers and plotted as individual points.
    IQR is the interquartile range, i.e. the difference between the third and first quartiles, 
    that are 75th and 25th percentiles of the distribution, respectively.    

    """
    distance_distributions = []
    positions = []
    for class_index in range(N_CLASSES):
        class_distances = features_df.loc[
            features_df["label"] == class_index, "distance"
        ].dropna()
        if class_distances.empty:
            continue
        distance_distributions.append(class_distances.values)
        positions.append(class_index)

    ax.boxplot(
        distance_distributions,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        medianprops=dict(color="red"),
    )
    ax.set_title("c) Class spread", fontsize=FONT_SIZE_TEXT, pad=8, loc="left", fontweight="bold")
    ax.set_xlabel("Class", fontsize=FONT_SIZE_TEXT)
    ax.set_ylabel("Distance distribution", fontsize=FONT_SIZE_TEXT)
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_xlim(-0.6, N_CLASSES - 0.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    style_axis(ax)
    # set font size of x and y tick labels
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TEXT)


def draw_distance_legend(ax: plt.Axes) -> None:
    ax.axis("off")

    y_center = 0.5
    x_low_whisker = 0.08
    x_q1 = 0.24
    x_median = 0.42
    x_q3 = 0.6
    x_high_whisker = 0.76
    box_bottom = y_center - 0.12
    box_top = y_center + 0.12

    ax.plot(
        [x_low_whisker, x_high_whisker],
        [y_center, y_center],
        color="black",
        lw=1.5,
        transform=ax.transAxes,
    )
    ax.plot(
        [x_low_whisker, x_low_whisker],
        [box_bottom, box_top],
        color="black",
        lw=1.5,
        transform=ax.transAxes,
    )
    ax.plot(
        [x_high_whisker, x_high_whisker],
        [box_bottom, box_top],
        color="black",
        lw=1.5,
        transform=ax.transAxes,
    )
    ax.add_patch(
        plt.Rectangle(
            (x_q1, box_bottom),
            x_q3 - x_q1,
            box_top - box_bottom,
            facecolor="lightgray",
            edgecolor="black",
            transform=ax.transAxes,
        )
    )
    ax.plot([x_median, x_median], [box_bottom, box_top], color="red", lw=1.8, transform=ax.transAxes)
    ax.plot(
        [0.02, 0.84],
        [y_center, y_center],
        marker="o",
        linestyle="None",
        color="black",
        markersize=4,
        transform=ax.transAxes,
    )

    legend_lines = [
        (x_low_whisker, "lower\nwhisker"),
        (x_q1, "25th"),
        (x_median, "50th"),
        (x_q3, "75th"),
        (x_high_whisker, "upper\nwhisker"),
    ]

    for x_value, label in legend_lines:
        ax.text(x_value, box_top + 0.08, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=11)

    ax.text(
        x_median,
        box_bottom - 0.12,
        "red line = median",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    ax.text(0.87, y_center, "outlier", transform=ax.transAxes, ha="left", va="center", fontsize=11)

def draw_spatial_placeholder(ax: plt.Axes, fig: plt.Figure, features_df) -> None:
    """"
    code to plot the distributions of the classes in the northern and southern parts of the domain
    plot the amount of videos for each class occurring in the lower part od the domain as barplot. 
    Add to the same barplot but hatched the amount of videos for each class occurring in the upper part of the domain.

    """
    # calculate median lat of the domain
    lat_mid_domain_split = 42+(52-42)/2

    # select all videos with lat_mid above the latitude that splits the domain in two equal parts, and count the amount of videos for each class
    lower_domain_df = features_df[features_df["lat_mid"] < lat_mid_domain_split]
    upper_domain_df = features_df[features_df["lat_mid"] >= lat_mid_domain_split]

    # plot barplot for the lower domain
    lower_class_counts = (
        lower_domain_df["label"]
        .value_counts()
        .reindex(range(N_CLASSES), fill_value=0)
        .sort_index()
    )
    upper_class_counts = (
        upper_domain_df["label"]
        .value_counts()
        .reindex(range(N_CLASSES), fill_value=0)
        .sort_index()
    )
    class_positions = np.arange(N_CLASSES)
    bar_width = 0.38
    bar_colors = [
        colors_per_class1_names.get(str(class_label), "lightgray")
        for class_label in class_positions
    ]

    ax.bar(
        class_positions - bar_width / 2,
        lower_class_counts.values,
        width=bar_width,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        label="Southern",
    )
    ax.bar(
        class_positions + bar_width / 2,
        upper_class_counts.values,
        width=bar_width,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        hatch="//",
        label="Northern",
    )
    ax.set_title(
        "d) Class variability over \n the domain",
        fontsize=FONT_SIZE_TEXT,
        pad=8,
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("Class", fontsize=FONT_SIZE_TEXT)
    ax.set_ylabel("Number of videos", fontsize=FONT_SIZE_TEXT)
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_xlim(-0.6, N_CLASSES - 0.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TEXT)
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="Southern"),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="//",
            label="Northern",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=FONT_SIZE_TEXT - 2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    style_axis(ax)

def build_figure() -> plt.Figure:

    # call function to create input dataset
    features_df, centroid_videos_df = prepare_input_data()

    # Create the main figure and the outer grid for the two main rows
    fig = plt.figure(figsize=(18, 20), constrained_layout=False)
    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[3.0, 1.1],
        hspace=0.18,
    )

    top_row_anchor_ax = draw_class_and_frames(outer[0], centroid_videos_df, fig)

    bottom = outer[1].subgridspec(
        2,
        3,
        height_ratios=[1.0, 0.18],
        hspace=0.45,
        wspace=0.5,
    )

    # Create the three summary plots in the bottom row
    count_ax = fig.add_subplot(bottom[0, 0])
    distance_ax = fig.add_subplot(bottom[0, 1])
    spatial_ax = fig.add_subplot(bottom[0, 2])

    # Draw placeholders for the three summary plots
    draw_count_placeholder(count_ax, features_df)
    draw_distance_placeholder(distance_ax, features_df)
    draw_spatial_placeholder(spatial_ax, fig, features_df)

    # Create empty axes for the two empty helper slots
    empty_cbar_left = fig.add_subplot(bottom[1, 0])
    empty_cbar_middle = fig.add_subplot(bottom[1, 1])
    empty_cbar_left.axis("off")
    empty_cbar_middle.axis("off")

    # set font globally
    plt.rcParams.update({"font.size": 20, "font.family": "sans-serif"})

    # tight layout and save figure
    #fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.94, top=0.94, bottom=0.08)

    distance_bbox = distance_ax.get_position()
    legend_height = 0.05
    legend_bottom = max(0.02, distance_bbox.y0 - 0.09)
    distance_legend_ax = fig.add_axes(
        [distance_bbox.x0, legend_bottom, distance_bbox.width, legend_height]
    )
    draw_distance_legend(distance_legend_ax)

    if top_row_anchor_ax is not None:
        anchor_bbox = top_row_anchor_ax.get_position()
        count_bbox = count_ax.get_position()
        fig.text(
            count_bbox.x0,
            anchor_bbox.y1 + 0.03,
            "a) Class representation",
            fontsize=FONT_SIZE_TEXT,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    return fig


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output)
    fig = build_figure()
    fig.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"Saved figure scaffold to {output_path}")


if __name__ == "__main__":
    main()


