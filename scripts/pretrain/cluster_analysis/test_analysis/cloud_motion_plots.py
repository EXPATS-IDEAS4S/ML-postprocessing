"""
Code to plot cloud motion analysis results for the different classes
The outputs are:
mean_dx_pixels_per_frame      average x displacement
mean_dy_pixels_per_frame      average y displacement
mean_speed_pixels_per_frame   magnitude of average displacement vector
mean_speed_kmh                converted physical speed
mean_direction_to_deg         direction cloud is moving toward
mean_direction_from_deg       opposite direction, like wind-from direction
n_pairs_used                  number of valid frame pairs, usually 7
mean_response                 phase-correlation confidence/quality

after selecting only mean responses >=0.4, we derive. for each class:

- distribution of mean speeds km/h
- distribution of mean directions deg to
- distribution of mean directions deg from
"""



import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    "cloud_motion_train": config["output_files"]["training_cloud_motion"],
    "cloud_motion_test": config["output_files"]["testing_cloud_motion"],
}

# create output directory if it doesn't exist
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_cloud_motion = CSV_FILES["cloud_motion_train"]
csv_cloud_motion_test = CSV_FILES["cloud_motion_test"]

RESPONSE_THRESHOLD = 0.4
SPEED_XMAX_KMH = 150
SPEED_BINS = np.linspace(0, SPEED_XMAX_KMH, 31)


def main():
    
    # Load the CSV file
    df = pd.read_csv(csv_cloud_motion)
    df_test = pd.read_csv(csv_cloud_motion_test)

    # Filter for mean_response >= 0.4
    df_filtered = df[df["mean_response"] >= RESPONSE_THRESHOLD]
    df_test_filtered = df_test[df_test["mean_response"] >= RESPONSE_THRESHOLD]

    # read classes in df_filtered
    classes = df_filtered["label"].unique()
    classes_test = df_test_filtered["label"].unique()

    # compare if the classes in training and testing are the same
    if not np.array_equal(classes, classes_test):
        print("Warning: Classes in training and testing datasets are not the same.")
        print(f"Training classes: {classes}")
        print(f"Testing classes: {classes_test}")
    # if not, take all the classes
    common_classes = np.union1d(classes, classes_test)


    # Plot distributions for each class
    for class_num in common_classes:

        # filter by label/class if class is present in the filtered dataframe
        if class_num not in df_filtered["label"].values:
            print(f"Class {class_num} not found in training dataset; skipping.")
            continue
        
        class_df = df_filtered[df_filtered["label"] == class_num]
        if class_num not in df_test_filtered["label"].values:
            print(f"Class {class_num} not found in testing dataset; skipping.")
            continue
        class_df_test = df_test_filtered[df_test_filtered["label"] == class_num]

        # Plot mean speeds km/h histograms in same style as the other plots
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.hist(
                class_df["mean_speed_kmh"].dropna(),
                bins=SPEED_BINS,
                label=f"Class {class_num} (training)", 
                density=True, 
                histtype="step",
                linestyle="--",
                color=colors_per_class1_names.get(str(class_num), None),
                linewidth=3,
            )   

        ax.hist(
                class_df_test["mean_speed_kmh"].dropna(),
                bins=SPEED_BINS,
                label=f"Class {class_num} (testing)", 
                density=True, 
                histtype="step",
                linestyle="-",
                color=colors_per_class1_names.get(str(class_num), None),
                linewidth=3,
            )   
        ax.set_xlim(0, SPEED_XMAX_KMH)
        plt.title(f"Distribution of Mean Speeds (km/h) for Class {class_num}")
        plt.xlabel("Mean Speed (km/h)")
        plt.ylabel("Probability density")
        style_axis(ax)
        plt.legend(frameon=False, loc="upper right", fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(OUTPUT_DIR / f"mean_speed_distribution_class_{class_num}.png", transparent=True)
        plt.close()
        
        # Plot mean directions deg to
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.hist(
                class_df["mean_direction_to_deg"],
                bins=30, 
                label=f"Class {class_num} (training)", 
                density=True, 
                histtype="step",
                linestyle="--",
                color=colors_per_class1_names.get(str(class_num), None),
                linewidth=3,
            )   

        ax.hist(
                class_df_test["mean_direction_to_deg"],
                bins=30, 
                label=f"Class {class_num} (testing)", 
                density=True, 
                histtype="step",
                linestyle="-",
                color=colors_per_class1_names.get(str(class_num), None),
                linewidth=3,
            )   
        plt.title(f"Distribution of Mean Directions (deg to) for Class {class_num}")
        plt.xlabel("Mean Direction (deg to)")
        plt.ylabel("Frequency")
        # set axis style
        style_axis(ax)

        # add legend 
        plt.legend(frameon=False, loc="upper right", fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(OUTPUT_DIR / f"mean_direction_to_distribution_class_{class_num}.png", transparent=True)
        plt.close()
        
        # Plot mean directions deg from
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.hist(
                class_df["mean_direction_from_deg"],
                bins=30, 
                label=f"Class {class_num} (training)", 
                density=True, 
                histtype="step",
                linestyle="--",
                color=colors_per_class1_names.get(str(class_num), None),
                linewidth=3,
            )   
        ax.hist(
                class_df_test["mean_direction_from_deg"],
                bins=30, 
                label=f"Class {class_num} (testing)", 
                density=True, 
                histtype="step",
                linestyle="-",
                color=colors_per_class1_names.get(str(class_num), None),
                linewidth=3,
            )   
        plt.title(f"Distribution of Mean Directions (deg from) for Class {class_num}")
        plt.xlabel("Mean Direction (deg from)")
        plt.ylabel("Frequency")
        # set axis style
        style_axis(ax)

        # add legend 
        plt.legend(frameon=False, loc="upper right", fontsize=12)
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(OUTPUT_DIR / f"mean_direction_from_distribution_class_{class_num}.png", transparent=True)
        plt.close()
    
if __name__ == "__main__":
    main()

    
