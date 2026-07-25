""""
code to test 2d scatter plots for visualization of the convective classes and their properties.
read from video summary CTH adn cloud cover for classes 5, 6, amd 7 and plot
 scatter plots with dots colored with the colors defined in class_colors.py for each class.
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
    "training_video_summary": config["output_files"]["training_video_summary"],
    "training_distances": config["output_files"]["training_features_distances_centroid"],
}


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
AXIS_LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12
WIND_DIRECTION_X_LIMITS = (100, 310)


def main():

    # load the crop statistics CSV file
    video_stats_df = pd.read_csv(CSV_FILES["training_video_summary"])
    video_distances_df = pd.read_csv(CSV_FILES["training_distances"])

    # read column path from video_distances_df and extract the crop name from the path
    video_distances_df["crop"] = video_distances_df["path"].apply(lambda x: os.path.basename(x).split(".")[0])

    # add distance variable from video_distances_df to all the videos in the dataframe
    video_stats_df = video_stats_df.merge(video_distances_df[["crop", "distance"]], on="crop", how="left")

    # drop class with label -100
    video_stats_df = clean_video_summary_df(video_stats_df)

    # select for each class the 1000 closest samples to the centroid based on the distance column
    video_stats_plot_df = video_stats_df.groupby("label").apply(lambda x: x.nsmallest(2000, "distance")).reset_index(drop=True)

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

    #class_2_plot = [5, 6, 7]
    class_2_plot = [3,9]
    #lass_2_plot = [0, 8]
    # plot scatter plots for each pair of variables in scatter_columns
    fig, axes = plt.subplots(1,1, figsize=(15, 15))

    # plot scatter plot for each class with different color and marker
    for class_label in class_2_plot:

        # filter the dataframe for the current class
        class_df = video_stats_plot_df[video_stats_plot_df["label"] == class_label]

        # select closest 1000 samples to the centroid

        # plot scatter plot for the current class
        axes.scatter(
            class_df["cma_mean"],
            class_df["cth_mean"],
            color=colors_per_class1_names[str(class_label)],
            label=f"Class {class_label}",
            edgecolor=MARKER_EDGE_COLOR,
            s=MARKER_SIZE,
            alpha=0.5
        )   
        axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.legend()

    for class_label_1 in class_2_plot:

        # plot position of the median of the entire distribution of the class with a larger marker
        axes.scatter(
            video_stats_df[video_stats_df["label"] == class_label_1]["cma_mean"].median(),
            video_stats_df[video_stats_df["label"] == class_label_1]["cth_mean"].median(),
            color=colors_per_class1_names[str(class_label_1)],
            marker="X",
            s=MARKER_SIZE * 10,
            edgecolor='black',
            linewidth=2.5,
        )   

    # set axis labels and title
    axes.set_ylabel("CTH Mean", fontsize=AXIS_LABEL_FONTSIZE)
    axes.set_xlabel("Cloud Cover", fontsize=AXIS_LABEL_FONTSIZE)
    axes.set_title(f"Scatter plot of CTH Mean vs Cloud Cover for classes {class_2_plot}", fontsize=TITLE_FONTSIZE) 
    
    # save the figure
    output_file = OUTPUT_DIR / f"scatter_cth_mean_cloud_cover_classes_{class_2_plot}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)  


    fig, axes = plt.subplots(1,1, figsize=(15, 15))

    # plot scatter plot for each class with different color and marker
    for class_label in class_2_plot:

        # filter the dataframe for the current class
        class_df = video_stats_plot_df[video_stats_plot_df["label"] == class_label]

        # select closest 1000 samples to the centroid

        # plot scatter plot for the current class
        axes.scatter(
            class_df["euclid_msg_grid_mean"],
            class_df["precipitation_mean"],
            color=colors_per_class1_names[str(class_label)],
            label=f"Class {class_label}",
            edgecolor=MARKER_EDGE_COLOR,
            s=MARKER_SIZE,
            alpha=0.5
        )   
        axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.legend()

    for class_label_1 in class_2_plot:

        # plot position of the median of the entire distribution of the class with a larger marker
        axes.scatter(
            video_stats_df[video_stats_df["label"] == class_label_1]["euclid_msg_grid_mean"].median(),
            video_stats_df[video_stats_df["label"] == class_label_1]["precipitation_mean"].median(),
            color=colors_per_class1_names[str(class_label_1)],
            marker="X",
            s=MARKER_SIZE * 10,
            edgecolor='black',
            linewidth=2.5,
        )   

    # set axis labels and title
    axes.set_ylabel("Precipitation Mean", fontsize=AXIS_LABEL_FONTSIZE)
    axes.set_xlabel("Euclid MSG Grid Mean", fontsize=AXIS_LABEL_FONTSIZE)
    axes.set_title(f"Scatter plot of Euclid MSG Grid Mean vs Precipitation Mean for classes {class_2_plot}", fontsize=TITLE_FONTSIZE) 
    
    # save the figure
    output_file = OUTPUT_DIR / f"scatter_euclid_msg_grid_mean_precipitation_mean_classes_{class_2_plot}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)  

def clean_video_summary_df(video_stats_df):
    video_stats_df["label"] = pd.to_numeric(video_stats_df["label"], errors="coerce")
    return video_stats_df[video_stats_df["label"] != -100]

if __name__ == "__main__":

    main()
