"""
this code reads the video summaries of the training dataset and plots scatter plots
mean video properties for each variable with their uncertainty (std) for each class
We will plot


- CTh, Cot
- mean lightning, mean precipitation

 and saves the plots in the figures directory specified in the config file.

 """

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.plot_class_analysis import plot_hourly_histogram, style_axis
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
MARKER_SIZE = 12
TEST_MARKER_EDGE_COLOR = "black"
TEST_MARKER_EDGE_WIDTH = 1.4
CMA_GRADIENT_LIMIT = 50


def get_class_color(label):
    return colors_per_class1_names.get(str(int(label)), None)


def get_class_label(label):
    return int(label)


def style_zero_centered_axes(ax, x_values, y_values, x_limit_cap=None):
    x_limit = get_zero_centered_axis_limit(x_values, limit_cap=x_limit_cap)
    y_limit = get_zero_centered_axis_limit(y_values)

    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, zorder=0)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, zorder=0)


def get_zero_centered_axis_limit(values, limit_cap=None):
    max_abs = np.nanmax(np.abs(values.to_numpy(dtype=float)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    axis_limit = max_abs * 1.1
    if limit_cap is not None:
        axis_limit = min(axis_limit, limit_cap)
    return axis_limit


def move_legend_outside(ax):
    ax.legend(
        frameon=False,
        fontsize=10,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )


def clean_video_summary_df(video_stats_df):
    video_stats_df["label"] = pd.to_numeric(video_stats_df["label"], errors="coerce")
    return video_stats_df[video_stats_df["label"] != -100]


def plot_class_errorbar_points(ax, class_means, x_mean, y_mean, x_std, y_std, dataset):
    marker = "o" if dataset == "training" else "s"
    linestyle = "-" if dataset == "training" else "--"
    label_suffix = "training" if dataset == "training" else "test"

    for label in class_means.index:
        ax.errorbar(
            class_means.loc[label, x_mean],
            class_means.loc[label, y_mean],
            xerr=class_means.loc[label, x_std],
            yerr=class_means.loc[label, y_std],
            fmt=marker,
            markersize=MARKER_SIZE,
            ecolor="gray",
            elinewidth=1.2,
            linestyle=linestyle,
            color=get_class_color(label),
            markeredgecolor=TEST_MARKER_EDGE_COLOR if dataset == "test" else None,
            markeredgewidth=TEST_MARKER_EDGE_WIDTH if dataset == "test" else 0,
            label=f"Class {get_class_label(label)} ({label_suffix})",
        )


def plot_class_scatter_points(ax, class_values, x_column, y_column, dataset):
    marker = "o" if dataset == "training" else "s"
    label_suffix = "training" if dataset == "training" else "test"

    for label in class_values.index:
        ax.scatter(
            class_values.loc[label, x_column],
            class_values.loc[label, y_column],
            s=MARKER_SIZE ** 2,
            color=get_class_color(label),
            marker=marker,
            edgecolor=TEST_MARKER_EDGE_COLOR if dataset == "test" else None,
            linewidth=TEST_MARKER_EDGE_WIDTH if dataset == "test" else 0,
            label=f"Class {get_class_label(label)} ({label_suffix})",
        )


def main():



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

    # create scatter plot of COT vs CTH means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_class_errorbar_points(
        ax,
        class_means,
        "cot_mean",
        "cth_mean",
        "cot_std",
        "cth_std",
        dataset="training",
    )
    plot_class_errorbar_points(
        ax,
        class_means_test,
        "cot_mean",
        "cth_mean",
        "cot_std",
        "cth_std",
        dataset="test",
    )
    ax.set_xlabel("COT Mean")
    ax.set_ylabel("CTH Mean")
    ax.set_title("Scatter Plot of COT vs CTH Means with Uncertainties")
    move_legend_outside(ax)
    style_axis(ax)

    # save figure
    fig_path = OUTPUT_DIR / "scatter_cot_cth_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")
    
    # create scatter plot of lightning and precipitation means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_class_errorbar_points(
        ax,
        class_means,
        "euclid_msg_grid_mean",
        "precipitation_mean",
        "euclid_msg_grid_std",
        "precipitation_std",
        dataset="training",
    )
    plot_class_errorbar_points(
        ax,
        class_means_test,
        "euclid_msg_grid_mean",
        "precipitation_mean",
        "euclid_msg_grid_std",
        "precipitation_std",
        dataset="test",
    )
    ax.set_xlabel("Lightning Mean")
    ax.set_ylabel("Precipitation Mean")
    ax.set_title("Scatter Plot of Lightning vs Precipitation Means with Uncertainties")
    move_legend_outside(ax)
    style_axis(ax)

    # save figure
    fig_path = OUTPUT_DIR / "scatter_lightning_precipitation_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")

    # create scatter plot of cot30plus and cth10plus means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_class_errorbar_points(
        ax,
        class_means,
        "cot30plus_mean",
        "cth10plus_mean",
        "cot30plus_std",
        "cth10plus_std",
        dataset="training",
    )
    plot_class_errorbar_points(
        ax,
        class_means_test,
        "cot30plus_mean",
        "cth10plus_mean",
        "cot30plus_std",
        "cth10plus_std",
        dataset="test",
    )
    ax.set_xlabel("Mean fraction of pixels with COT > 30")
    ax.set_ylabel("Mean fraction of pixels with CTH10+ > 10")
    ax.set_title("Scatter Plot of COT30+ vs CTH10+ Means with Uncertainties")
    move_legend_outside(ax)
    style_axis(ax)

    # save figure
    fig_path = OUTPUT_DIR / "scatter_cot30plus_cth10plus_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")

    # calculate now mean and std of all columns ending with _gradient for each class
    gradient_columns = [col for col in video_stats_df.columns if col.endswith("_gradient")]
    class_gradients = video_stats_df.groupby("label")[gradient_columns].mean()  
    class_gradients_test = video_stats_test_df.groupby("label")[gradient_columns].mean()

    # plot scatter plot of CMA gradient vs CTH gradient means for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_class_scatter_points(
        ax,
        class_gradients,
        "cma_gradient",
        "cth_gradient",
        dataset="training",
    )
    plot_class_scatter_points(
        ax,
        class_gradients_test,
        "cma_gradient",
        "cth_gradient",
        dataset="test",
    )
    ax.set_xlabel("CMA Gradient Mean")
    ax.set_ylabel("CTH Gradient Mean")
    ax.set_title("Scatter Plot of CMA vs CTH Gradient Means")
    move_legend_outside(ax)
    style_axis(ax)
    style_zero_centered_axes(
        ax,
        pd.concat([class_gradients["cma_gradient"], class_gradients_test["cma_gradient"]]),
        pd.concat([class_gradients["cth_gradient"], class_gradients_test["cth_gradient"]]),
        x_limit_cap=CMA_GRADIENT_LIMIT,
    )
    
    # save figure
    fig_path = OUTPUT_DIR / "scatter_cma_cth_gradient_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")    

    # plot scatter plot of COT gradient vs CTH gradient means for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_class_scatter_points(
        ax,
        class_gradients,
        "cot_gradient",
        "cth_gradient",
        dataset="training",
    )
    plot_class_scatter_points(
        ax,
        class_gradients_test,
        "cot_gradient",
        "cth_gradient",
        dataset="test",
    )
    ax.set_xlabel("COT Gradient Mean")
    ax.set_ylabel("CTH Gradient Mean")
    ax.set_title("Scatter Plot of COT vs CTH Gradient Means")
    move_legend_outside(ax)
    style_axis(ax)
    style_zero_centered_axes(
        ax,
        pd.concat([class_gradients["cot_gradient"], class_gradients_test["cot_gradient"]]),
        pd.concat([class_gradients["cth_gradient"], class_gradients_test["cth_gradient"]]),
    )

    # save figure
    fig_path = OUTPUT_DIR / "scatter_cot_cth_gradient_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")


    # plot scatter plot of cth10plus gradient and cot30plus gradient means for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_class_scatter_points(
        ax,
        class_gradients,
        "cth10plus_gradient",
        "cot30plus_gradient",
        dataset="training",
    )
    plot_class_scatter_points(
        ax,
        class_gradients_test,
        "cth10plus_gradient",
        "cot30plus_gradient",
        dataset="test",
    )
    ax.set_xlabel("CTH10+ Gradient Mean")
    ax.set_ylabel("COT30+ Gradient Mean")
    ax.set_title("Scatter Plot of CTH10+ vs COT30+ Gradient Means")
    move_legend_outside(ax)
    style_axis(ax)
    style_zero_centered_axes(
        ax,
        pd.concat([class_gradients["cth10plus_gradient"], class_gradients_test["cth10plus_gradient"]]),
        pd.concat([class_gradients["cot30plus_gradient"], class_gradients_test["cot30plus_gradient"]]),
    )
    
    # save figure
    fig_path = OUTPUT_DIR / "scatter_cth10plus_cot30plus_gradient_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")    


    # print names of the saved figures
    print(f"Saved scatter plots to: {OUTPUT_DIR}")  
    


if __name__ == "__main__":
    main()
