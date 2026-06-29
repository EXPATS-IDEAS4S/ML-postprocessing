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


def style_zero_centered_gradient_axes(ax, x_values, y_values):
    values = pd.concat([x_values, y_values])
    max_abs = np.nanmax(np.abs(values.to_numpy(dtype=float)))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    axis_limit = max_abs * 1.1

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, zorder=0)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, zorder=0)


def main():



    # load the crop statistics CSV file
    video_stats_df = pd.read_csv(CSV_FILES["training"])
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
    video_stats_df["label"] = pd.to_numeric(video_stats_df["label"], errors="coerce")
    video_stats_df = video_stats_df[video_stats_df["label"] != -100]

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
    ]

    # calculate mean value for each class for each variable and its std
    class_means = video_stats_df.groupby("label")[scatter_columns].mean()

    # create scatter plot of COT vs CTH means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    for label in class_means.index:
        ax.errorbar(
            class_means.loc[label, "cot_mean"],
            class_means.loc[label, "cth_mean"],
            xerr=class_means.loc[label, "cot_std"],
            yerr=class_means.loc[label, "cth_std"],
            fmt="o",
            markersize=MARKER_SIZE,
            ecolor="gray",
            color=colors_per_class1_names.get(str(label), None),
            label=f"Class {label}",
        )   
    ax.set_xlabel("COT Mean")
    ax.set_ylabel("CTH Mean")
    ax.set_title("Scatter Plot of COT vs CTH Means with Uncertainties")
    ax.legend(frameon=False, fontsize=11, ncol=3)
    style_axis(ax)

    # save figure
    fig_path = OUTPUT_DIR / "scatter_cot_cth_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")
    
    # create scatter plot of lightning and precipitation means with their uncertainties (std) for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    for label in class_means.index:
        ax.errorbar(
            class_means.loc[label, "euclid_msg_grid_mean"],
            class_means.loc[label, "precipitation_mean"],
            xerr=class_means.loc[label, "euclid_msg_grid_std"],
            yerr=class_means.loc[label, "precipitation_std"],
            fmt="o",
            markersize=MARKER_SIZE,
            ecolor="gray",
            color=colors_per_class1_names.get(str(label), None),
            label=f"Class {label}",
        )   
    ax.set_xlabel("Lightning Mean")
    ax.set_ylabel("Precipitation Mean")
    ax.set_title("Scatter Plot of Lightning vs Precipitation Means with Uncertainties")
    ax.legend(frameon=False, fontsize=11, ncol=3)
    style_axis(ax)

    # save figure
    fig_path = OUTPUT_DIR / "scatter_lightning_precipitation_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")

    
    # calculate now mean and std of all columns ending with _gradient for each class
    gradient_columns = [col for col in video_stats_df.columns if col.endswith("_gradient")]
    class_gradients = video_stats_df.groupby("label")[gradient_columns].mean()  

    # plot scatter plot of CMA gradient vs CTH gradient means for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    for label in class_gradients.index:
        ax.scatter(
            class_gradients.loc[label, "cma_gradient"],
            class_gradients.loc[label, "cth_gradient"],
            s=MARKER_SIZE ** 2,
            color=colors_per_class1_names.get(str(label), None),
            label=f"Class {label}",
        )
    ax.set_xlabel("CMA Gradient Mean")
    ax.set_ylabel("CTH Gradient Mean")
    ax.set_title("Scatter Plot of CMA vs CTH Gradient Means")
    ax.legend(frameon=False, fontsize=11, ncol=3)
    style_axis(ax)
    style_zero_centered_gradient_axes(
        ax,
        class_gradients["cma_gradient"],
        class_gradients["cth_gradient"],
    )
    
    # save figure
    fig_path = OUTPUT_DIR / "scatter_cma_cth_gradient_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")    

    # plot scatter plot of COT gradient vs CTH gradient means for each class
    fig, ax = plt.subplots(figsize=(12, 7))
    for label in class_gradients.index: 
        ax.scatter(
            class_gradients.loc[label, "cot_gradient"],
            class_gradients.loc[label, "cth_gradient"],
            s=MARKER_SIZE ** 2,
            color=colors_per_class1_names.get(str(label), None),
            label=f"Class {label}",
        )
    ax.set_xlabel("COT Gradient Mean")
    ax.set_ylabel("CTH Gradient Mean")
    ax.set_title("Scatter Plot of COT vs CTH Gradient Means")
    ax.legend(frameon=False, fontsize=11, ncol=3)
    style_axis(ax)
    style_zero_centered_gradient_axes(
        ax,
        class_gradients["cot_gradient"],
        class_gradients["cth_gradient"],
    )

    # save figure
    fig_path = OUTPUT_DIR / "scatter_cot_cth_gradient_means.png"
    plt.savefig(fig_path, transparent=True, bbox_inches="tight")

    # print names of the saved figures
    print(f"Saved scatter plots to: {OUTPUT_DIR}")  
    


if __name__ == "__main__":
    main()
