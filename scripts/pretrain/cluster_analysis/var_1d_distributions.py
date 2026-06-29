""""
code to derive 1d var distributions for each class, for each variable, for the video classified 
using the config process_run_GRL.yaml. 
The code produces a plot of te distributions for each variable, for each class,
and saves it in the figures directory specified in the config file.

author: Claudia Acquistapace
date: 2024-06-29

"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
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

    # search all columns in the file header that end with _mean and loop over them to plot the distributions for each class
    mean_columns = [col for col in video_stats_df.columns if col.endswith("_mean")]

    for col in mean_columns:

        # extract the variable name from the column name
        var_name = col.replace("_mean", "")

        
        for class_num in range(10):

            plt.figure(figsize=(10, 6))
            # filter the dataframe for the current class
            class_df = video_stats_df[video_stats_df["label"] == class_num]
            values = pd.to_numeric(class_df[col], errors="coerce").dropna()

            if var_name in ["precipitation", "euclid_msg_grid"]:
                values = values[values > 0]
                if len(values) > 0:
                    values = values[values <= values.quantile(0.99)]
                bins = 30
            else:
                bins = 50

            # plot the distribution of the current variable for the current class
            ax = plt.gca()
            ax.hist(
                values,
                bins=bins, 
                label=f"Class {class_num}", 
                density=True, 
                histtype="step",
                color=colors_per_class1_names.get(str(class_num), None),
                linewidth=3,
            )

            # remove top and right axis
            style_axis(ax)

            plt.title(f"Distribution of {var_name} for each class")
            plt.xlabel(var_name)
            plt.ylabel("Density")
            ax = set_x_ranges_for_variable(ax, var_name)
            ax.legend(frameon=False, fontsize=11)
            output_filename = f"1d_distr/{var_name}_class{class_num}_distribution.png"
            plt.savefig(os.path.join(OUTPUT_DIR, output_filename), transparent=True, bbox_inches="tight")
            print(f"Distribution plot for {var_name} saved to {os.path.join(OUTPUT_DIR, output_filename)}")
            plt.close()



def set_x_ranges_for_variable(ax, var_name):

    if var_name == "cth":
        ax.set_xlim(500., 14000.)
        ax.set_ylim(0., 0.00025)
    elif var_name == "cot":
        ax.set_xlim(0., 100.)
        ax.set_ylim(0., 0.1)
    #elif var_name == "cma":
        #ax.set_xlim(0., 100.)
        #ax.set_ylim(0., 0.1)
    elif var_name == "precipitation":
        ax.set_xlim(0., 3000.)
        #ax.set_ylim(0., 0.1)
    elif var_name == "euclid_msg_grid":
        ax.set_xlim(0., 800.)
        #ax.set_ylim(0., 0.1)
    #elif var_name == "cth10plus":
        #ax.set_xlim(0., 1.)
        #ax.set_ylim(0., 0.0005)
    #elif var_name == "cot30plus":
        #ax.set_xlim(0, 1.)
        #ax.set_ylim(0., 0.1)
    return ax

if __name__ == "__main__":
    main()
