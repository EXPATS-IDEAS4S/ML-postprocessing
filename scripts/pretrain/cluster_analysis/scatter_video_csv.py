"""
This code reads the video summary CSV files and plots class-level scatter plots
of mean video properties.

For each cloud class, the script averages the selected variables over all videos
in the training dataset. By default, it also reads the testing dataset, plots the
same class averages for testing, and draws a dashed arrow from the training point
to the testing point for each class.

The plots include:
    - COT mean vs CTH mean, with class-mean uncertainty from the std columns
    - Cloud Cover mean vs CTH mean, with class-mean uncertainty from the std columns
    - lightning mean vs precipitation mean, with class-mean uncertainty
    - COT30+ mean vs CTH10+ mean, with class-mean uncertainty
    - Cloud Cover gradient mean vs CTH gradient mean, with training 25-75 percentile uncertainty
    - COT gradient mean vs CTH gradient mean, with training 25-75 percentile uncertainty
    - CTH10+ gradient mean vs COT30+ gradient mean, with training 25-75 percentile uncertainty

Markers:
    - circles: training
    - squares with black edge: testing
    - dashed arrows: direction from training to testing for the same class

Outputs are saved in the figures directory specified in process_run_GRL.yaml.

How to run:
    python scatter_video_csv.py
    python scatter_video_csv.py --include-test
    python scatter_video_csv.py --training-only

The default is --include-test. With --training-only, testing points and arrows
are not plotted, and output filenames receive a _training_only suffix.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.plot_class_analysis import plot_hourly_histogram, style_axis as base_style_axis
from utils.plotting.class_colors import colors_per_class1_names

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "training": config["output_files"]["training_video_summary"],
    "testing": config["output_files"]["testing_video_summary"],
}

# create output directory if it doesn't exist
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MARKER_SIZE = 15
TEST_MARKER_SIZE = 18
TEST_MARKER_EDGE_COLOR = "black"
TEST_MARKER_EDGE_WIDTH = 1.4
MARKER_EDGE_COLOR = "black"
MARKER_EDGE_WIDTH = 1.4
CMA_GRADIENT_LIMIT = 50
CTH_MEAN_LIMIT = 8000
CTH_GRADIENT_LIMIT = 200
ZERO_AXIS_LINEWIDTH = 2.2
AXIS_LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot class scatter summaries for training data, optionally with testing data."
    )
    parser.add_argument(
        "--include-test",
        dest="include_test",
        action="store_true",
        default=True,
        help="Plot testing data together with training data. This is the default.",
    )
    parser.add_argument(
        "--training-only",
        dest="include_test",
        action="store_false",
        help="Plot only training data and save figures with a _training_only suffix.",
    )
    return parser.parse_args()


def get_output_path(filename, include_test):
    output_path = OUTPUT_DIR / filename
    if include_test:
        return output_path
    return output_path.with_name(f"{output_path.stem}_training_only{output_path.suffix}")


def get_class_color(label):
    return colors_per_class1_names.get(str(int(label)), None)


def get_class_label(label):
    return int(label)


def style_zero_centered_axes(ax, x_values, y_values, x_limit_cap=None, y_limit_cap=None):
    x_limit = get_zero_centered_axis_limit(x_values, limit_cap=x_limit_cap)
    y_limit = get_zero_centered_axis_limit(y_values, limit_cap=y_limit_cap)

    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=ZERO_AXIS_LINEWIDTH,
        alpha=0.9,
        zorder=1,
    )
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=ZERO_AXIS_LINEWIDTH,
        alpha=0.9,
        zorder=1,
    )


def get_zero_centered_axis_limit(values, limit_cap=None):
    max_abs = np.nanmax(np.abs(values.to_numpy(dtype=float)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    axis_limit = max_abs * 1.1
    if limit_cap is not None:
        axis_limit = min(axis_limit, limit_cap)
    return axis_limit


def get_values_with_uncertainty(values, uncertainty=None):
    values = pd.to_numeric(values, errors="coerce")
    if uncertainty is None:
        return values

    uncertainty = pd.to_numeric(uncertainty, errors="coerce").fillna(0)
    return pd.concat([values - uncertainty, values, values + uncertainty])


def finite_or_zero(value):
    return value if np.isfinite(value) else 0


def get_padded_limits(values, pad_fraction=0.08):
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return 0, 1

    value_min = values.min()
    value_max = values.max()
    if value_min == value_max:
        padding = abs(value_min) * pad_fraction if value_min != 0 else 1.0
    else:
        padding = (value_max - value_min) * pad_fraction
    return value_min - padding, value_max + padding


def set_reference_axis_limits(
    ax,
    reference_values,
    x_column,
    y_column,
    x_uncertainty_column=None,
    y_uncertainty_column=None,
):
    x_values = get_values_with_uncertainty(
        reference_values[x_column],
        reference_values[x_uncertainty_column] if x_uncertainty_column else None,
    )
    y_values = get_values_with_uncertainty(
        reference_values[y_column],
        reference_values[y_uncertainty_column] if y_uncertainty_column else None,
    )
    ax.set_xlim(*get_padded_limits(x_values))
    ax.set_ylim(*get_padded_limits(y_values))


def move_legend_outside(ax):
    ax.legend(
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )


def style_axis(ax):
    base_style_axis(ax)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)


def clean_video_summary_df(video_stats_df):
    video_stats_df["label"] = pd.to_numeric(video_stats_df["label"], errors="coerce")
    return video_stats_df[video_stats_df["label"] != -100]


def plot_training_test_arrows(ax, training_values, test_values, x_column, y_column):
    common_labels = training_values.index.intersection(test_values.index)

    for label in common_labels:
        x_start = training_values.loc[label, x_column]
        y_start = training_values.loc[label, y_column]
        x_end = test_values.loc[label, x_column]
        y_end = test_values.loc[label, y_column]
        if not np.all(np.isfinite([x_start, y_start, x_end, y_end])):
            continue

        ax.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(x_start, y_start),
            arrowprops={
                "arrowstyle": "->",
                "color": get_class_color(label),
                "linestyle": "--",
                "linewidth": 2.0,
                "alpha": 0.9,
                "mutation_scale": 16,
                "shrinkA": 8,
                "shrinkB": 8,
            },
            zorder=3,
        )


def plot_training_test_links(ax, training_values, test_values, x_column, y_column):
    plot_training_test_arrows(
        ax,
        training_values,
        test_values,
        x_column,
        y_column,
    )


def plot_class_errorbar_points(ax, class_means, x_mean, y_mean, x_std, y_std, dataset):
    marker = "o" if dataset == "training" else "s"
    marker_size = MARKER_SIZE if dataset == "training" else TEST_MARKER_SIZE
    linestyle = "-" if dataset == "training" else "--"
    label_suffix = "training" if dataset == "training" else "test"

    for label in class_means.index:
        xerr = finite_or_zero(class_means.loc[label, x_std])
        yerr = finite_or_zero(class_means.loc[label, y_std])
        ax.errorbar(
            class_means.loc[label, x_mean],
            class_means.loc[label, y_mean],
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            markersize=marker_size,
            ecolor="gray",
            elinewidth=1.2,
            linestyle=linestyle,
            color=get_class_color(label),
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=f"Class {get_class_label(label)} ({label_suffix})",
        )


def plot_class_scatter_points(ax, class_values, x_column, y_column, dataset):
    marker = "o" if dataset == "training" else "s"
    marker_size = MARKER_SIZE if dataset == "training" else TEST_MARKER_SIZE
    label_suffix = "training" if dataset == "training" else "test"

    for label in class_values.index:
        ax.scatter(
            class_values.loc[label, x_column],
            class_values.loc[label, y_column],
            s=marker_size ** 2,
            color=get_class_color(label),
            marker=marker,
            edgecolor=MARKER_EDGE_COLOR,
            linewidth=MARKER_EDGE_WIDTH,
            label=f"Class {get_class_label(label)} ({label_suffix})",
        )


def get_percentile_error(mean_value, lower_percentile, upper_percentile):
    if not np.all(np.isfinite([mean_value, lower_percentile, upper_percentile])):
        return None

    lower_error = max(mean_value - lower_percentile, 0)
    upper_error = max(upper_percentile - mean_value, 0)
    return np.array([[lower_error], [upper_error]])


def plot_class_percentile_errorbar_points(
    ax,
    class_means,
    class_q25,
    class_q75,
    x_column,
    y_column,
    dataset,
):
    marker = "o" if dataset == "training" else "s"
    marker_size = MARKER_SIZE if dataset == "training" else TEST_MARKER_SIZE
    label_suffix = "training" if dataset == "training" else "test"

    for label in class_means.index:
        x_value = class_means.loc[label, x_column]
        y_value = class_means.loc[label, y_column]
        xerr = get_percentile_error(
            x_value,
            class_q25.loc[label, x_column],
            class_q75.loc[label, x_column],
        )
        yerr = get_percentile_error(
            y_value,
            class_q25.loc[label, y_column],
            class_q75.loc[label, y_column],
        )

        ax.errorbar(
            x_value,
            y_value,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            markersize=marker_size,
            ecolor="gray",
            elinewidth=1.2,
            capsize=3,
            color=get_class_color(label),
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=f"Class {get_class_label(label)} ({label_suffix})",
        )


def get_axis_values_with_training_percentiles(
    class_gradients,
    class_gradients_q25,
    class_gradients_q75,
    class_gradients_test,
    column,
    include_test,
):
    values = [
        class_gradients[column],
        class_gradients_q25[column],
        class_gradients_q75[column],
    ]
    if include_test:
        values.append(class_gradients_test[column])
    return pd.concat(values)


def main():
    args = parse_args()
    include_test = args.include_test

    # load the crop statistics CSV file
    video_stats_df = pd.read_csv(CSV_FILES["training"])
    video_stats_test_df = pd.read_csv(CSV_FILES["testing"])
    print("Crop statistics CSV files loaded successfully.")

    # count number of samples with label -100
    video_minus100 = video_stats_df[video_stats_df["label"] == -100]

    # print number of samples with label -100
    print(f"Number of samples with label -100: {len(video_minus100)}")
    #print total number of samples in the dataset
    print(f"Total number of samples in the dataset: {len(video_stats_df)}")
    #print percentage of samples with label -100
    print(f"Percentage of samples with label -100: {len(video_minus100)/len(video_stats_df)*100:.2f}%") 

    # drop class with label -100
    video_stats_df = clean_video_summary_df(video_stats_df)
    video_stats_test_df = clean_video_summary_df(video_stats_test_df)

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

    # calculate mean value for each class for each variable and its std
    class_means = video_stats_df.groupby("label")[scatter_columns].mean()
    class_means_test = video_stats_test_df.groupby("label")[scatter_columns].mean()
    class_means_limits = pd.concat([class_means, class_means_test], axis=0)

    # create scatter plot of COT vs CTH means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    if include_test:
        plot_training_test_links(
            ax,
            class_means,
            class_means_test,
            "cot_mean",
            "cth_mean",
        )
    plot_class_errorbar_points(
        ax,
        class_means,
        "cot_mean",
        "cth_mean",
        "cot_std",
        "cth_std",
        dataset="training",
    )
    if include_test:
        plot_class_errorbar_points(
            ax,
            class_means_test,
            "cot_mean",
            "cth_mean",
            "cot_std",
            "cth_std",
            dataset="test",
        )
    ax.set_xlabel("COT Mean", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("CTH Mean [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Scatter Plot of COT vs CTH Means with Uncertainties", fontsize=TITLE_FONTSIZE)
    move_legend_outside(ax)
    style_axis(ax)
    set_reference_axis_limits(
        ax,
        class_means_limits,
        "cot_mean",
        "cth_mean",
        "cot_std",
        "cth_std",
    )
    ax.set_ylim(3500, CTH_MEAN_LIMIT)

    # save figure
    fig_path = get_output_path("scatter_cot_cth_means.png", include_test)
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")

    # create scatter plot of cloud cover vs CTH means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    if include_test:
        plot_training_test_links(
            ax,
            class_means,
            class_means_test,
            "cma_mean",
            "cth_mean",
        )
    plot_class_errorbar_points(
        ax,
        class_means,
        "cma_mean",
        "cth_mean",
        "cma_std",
        "cth_std",
        dataset="training",
    )
    if include_test:
        plot_class_errorbar_points(
            ax,
            class_means_test,
            "cma_mean",
            "cth_mean",
            "cma_std",
            "cth_std",
            dataset="test",
        )
    ax.set_xlabel("Cloud Cover Mean", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("CTH Mean [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Scatter Plot of Cloud Cover vs CTH Means with Uncertainties", fontsize=TITLE_FONTSIZE)
    move_legend_outside(ax)
    style_axis(ax)
    set_reference_axis_limits(
        ax,
        class_means_limits,
        "cma_mean",
        "cth_mean",
        "cma_std",
        "cth_std",
    )
    ax.set_ylim(3500, CTH_MEAN_LIMIT)

    fig_path = get_output_path("scatter_cma_cth_means.png", include_test)
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")
    
    # create scatter plot of lightning and precipitation means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    if include_test:
        plot_training_test_links(
            ax,
            class_means,
            class_means_test,
            "euclid_msg_grid_mean",
            "precipitation_mean",
        )
    plot_class_errorbar_points(
        ax,
        class_means,
        "euclid_msg_grid_mean",
        "precipitation_mean",
        "euclid_msg_grid_std",
        "precipitation_std",
        dataset="training",
    )
    if include_test:
        plot_class_errorbar_points(
            ax,
            class_means_test,
            "euclid_msg_grid_mean",
            "precipitation_mean",
            "euclid_msg_grid_std",
            "precipitation_std",
            dataset="test",
        )
    ax.set_xlabel("Lightning Mean", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Precipitation Mean [mm]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Scatter Plot of Lightning vs Precipitation Means with Uncertainties", fontsize=TITLE_FONTSIZE)
    move_legend_outside(ax)
    style_axis(ax)
    set_reference_axis_limits(
        ax,
        class_means_limits,
        "euclid_msg_grid_mean",
        "precipitation_mean",
        "euclid_msg_grid_std",
        "precipitation_std",
    )

    # save figure
    fig_path = get_output_path("scatter_lightning_precipitation_means.png", include_test)
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")

    # create scatter plot of cot30plus and cth10plus means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    if include_test:
        plot_training_test_links(
            ax,
            class_means,
            class_means_test,
            "cot30plus_mean",
            "cth10plus_mean",
        )
    plot_class_errorbar_points(
        ax,
        class_means,
        "cot30plus_mean",
        "cth10plus_mean",
        "cot30plus_std",
        "cth10plus_std",
        dataset="training",
    )
    if include_test:
        plot_class_errorbar_points(
            ax,
            class_means_test,
            "cot30plus_mean",
            "cth10plus_mean",
            "cot30plus_std",
            "cth10plus_std",
            dataset="test",
        )
    ax.set_xlabel("Mean fraction of pixels with COT > 30", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Mean fraction of pixels with CTH10+ > 10", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Scatter Plot of COT30+ vs CTH10+ Means with Uncertainties", fontsize=TITLE_FONTSIZE)
    move_legend_outside(ax)
    style_axis(ax)
    set_reference_axis_limits(
        ax,
        class_means_limits,
        "cot30plus_mean",
        "cth10plus_mean",
        "cot30plus_std",
        "cth10plus_std",
    )

    # save figure
    fig_path = get_output_path("scatter_cot30plus_cth10plus_means.png", include_test)
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")

    # calculate now mean and std of all columns ending with _gradient for each class
    gradient_columns = [col for col in video_stats_df.columns if col.endswith("_gradient")]
    class_gradients = video_stats_df.groupby("label")[gradient_columns].mean()  
    class_gradients_q25 = video_stats_df.groupby("label")[gradient_columns].quantile(0.25)
    class_gradients_q75 = video_stats_df.groupby("label")[gradient_columns].quantile(0.75)
    class_gradients_test = video_stats_test_df.groupby("label")[gradient_columns].mean()

    # plot scatter plot of cloud cover gradient vs CTH gradient means for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    if include_test:
        plot_training_test_links(
            ax,
            class_gradients,
            class_gradients_test,
            "cma_gradient",
            "cth_gradient",
        )
    plot_class_percentile_errorbar_points(
        ax,
        class_gradients,
        class_gradients_q25,
        class_gradients_q75,
        "cma_gradient",
        "cth_gradient",
        dataset="training",
    )
    if include_test:
        plot_class_scatter_points(
            ax,
            class_gradients_test,
            "cma_gradient",
            "cth_gradient",
            dataset="test",
        )
    ax.set_xlabel("Cloud Cover Gradient Mean", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("CTH Gradient Mean [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Scatter Plot of Cloud Cover vs CTH Gradient Means with Training 25-75% Range", fontsize=TITLE_FONTSIZE)
    move_legend_outside(ax)
    style_axis(ax)
    style_zero_centered_axes(
        ax,
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
    style_axis(ax)
    ax.set_xlim(-0.04, 0.04)
    ax.set_ylim(-CTH_GRADIENT_LIMIT, CTH_GRADIENT_LIMIT)
    
    # save figure
    fig_path = get_output_path("scatter_cma_cth_gradient_means.png", include_test)
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")    

    # plot scatter plot of COT gradient vs CTH gradient means for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    if include_test:
        plot_training_test_links(
            ax,
            class_gradients,
            class_gradients_test,
            "cot_gradient",
            "cth_gradient",
        )
    plot_class_percentile_errorbar_points(
        ax,
        class_gradients,
        class_gradients_q25,
        class_gradients_q75,
        "cot_gradient",
        "cth_gradient",
        dataset="training",
    )
    if include_test:
        plot_class_scatter_points(
            ax,
            class_gradients_test,
            "cot_gradient",
            "cth_gradient",
            dataset="test",
        )
    ax.set_xlabel("COT Gradient Mean", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("CTH Gradient Mean [m]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Scatter Plot of COT vs CTH Gradient Means with Training 25-75% Range", fontsize=TITLE_FONTSIZE)
    move_legend_outside(ax)
    style_axis(ax)
    style_zero_centered_axes(
        ax,
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
    style_axis(ax)
    ax.set_ylim(-CTH_GRADIENT_LIMIT, CTH_GRADIENT_LIMIT)

    # save figure
    fig_path = get_output_path("scatter_cot_cth_gradient_means.png", include_test)
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")


    # plot scatter plot of cth10plus gradient and cot30plus gradient means for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    if include_test:
        plot_training_test_links(
            ax,
            class_gradients,
            class_gradients_test,
            "cth10plus_gradient",
            "cot30plus_gradient",
        )
    plot_class_percentile_errorbar_points(
        ax,
        class_gradients,
        class_gradients_q25,
        class_gradients_q75,
        "cth10plus_gradient",
        "cot30plus_gradient",
        dataset="training",
    )
    if include_test:
        plot_class_scatter_points(
            ax,
            class_gradients_test,
            "cth10plus_gradient",
            "cot30plus_gradient",
            dataset="test",
        )
    ax.set_xlabel("CTH10+ Gradient Mean", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("COT30+ Gradient Mean", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Scatter Plot of CTH10+ vs COT30+ Gradient Means with Training 25-75% Range", fontsize=TITLE_FONTSIZE)
    move_legend_outside(ax)
    style_axis(ax)
    style_zero_centered_axes(
        ax,
        get_axis_values_with_training_percentiles(
            class_gradients,
            class_gradients_q25,
            class_gradients_q75,
            None,
            "cth10plus_gradient",
            False,
        ),
        get_axis_values_with_training_percentiles(
            class_gradients,
            class_gradients_q25,
            class_gradients_q75,
            None,
            "cot30plus_gradient",
            False,
        ),
    )
    style_axis(ax)
    
    # save figure
    fig_path = get_output_path("scatter_cth10plus_cot30plus_gradient_means.png", include_test)
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")    


    # print names of the saved figures
    print(f"Saved scatter plots to: {OUTPUT_DIR}")  
    


if __name__ == "__main__":
    main()
