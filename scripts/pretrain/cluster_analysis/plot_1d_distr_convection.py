""""
with this code, we want to plot histograms of the 1D distribution of the classes identified as convection growing or convection decaying.
We plot the video summary from the training data, using the properties stored therein.
The code does the same as var_1d_distributions.py but grouping classes in the prescribed groups.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

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


convection_growing_classes = [5, 6, 7]
convection_decaying_classes = [3, 9, 0, 8]

night_time_classes = [0, 1, 4, 8]
daytime_classes = [5,6,7]
all_day_classes = [9,2,3]

AXIS_LABEL_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 15
LINEWIDTH = 4


def get_time_of_day_linestyle(class_num):
    if class_num in daytime_classes:
        return "-"
    if class_num in night_time_classes:
        return ":"
    return "-."


def add_distribution_legends(ax):
    class_handles, class_labels = ax.get_legend_handles_labels()
    if class_handles:
        class_legend = ax.legend(
            class_handles,
            class_labels,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
        )
        ax.add_artist(class_legend)

    time_legend_elements = [
        Line2D([0], [0], color="black", lw=LINEWIDTH, linestyle="-", label="Daytime classes"),
        Line2D([0], [0], color="black", lw=LINEWIDTH, linestyle=":", label="Nighttime classes"),
        Line2D([0], [0], color="black", lw=LINEWIDTH, linestyle="-.", label="All day classes"),
    ]
    ax.legend(
        handles=time_legend_elements,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0,
    )


def get_variable_display_label(var_name):
    if var_name == "cma":
        return "cloud cover"
    return var_name


def style_distribution_axis(ax, var_name):
    style_axis(ax)
    ax.set_xlabel(get_variable_display_label(var_name), fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Density", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    return set_x_ranges_for_variable(ax, var_name)



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

        # make one plot for convection growing classes and one for convection decaying classes
        plt.figure(figsize=(12, 7))
        for class_num in range(10):


            # assign linestyle based on whether the class is growing or decaying
            linestyle = get_time_of_day_linestyle(class_num)
        
            if class_num in convection_growing_classes:

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
                    linewidth=LINEWIDTH,
                    linestyle=linestyle,
                )

                # remove top and right axis
                ax = style_distribution_axis(ax, var_name)

        add_distribution_legends(plt.gca())
        output_filename = f"1d_distr/convective_classes/{var_name}_convection_growing_distributions.png"
        plt.savefig(os.path.join(OUTPUT_DIR, output_filename), transparent=True, bbox_inches="tight")
        print(f"Distribution plot for {var_name} saved to {os.path.join(OUTPUT_DIR, output_filename)}")
        plt.close()

        # make one plot for convection growing classes and one for convection decaying classes
        plt.figure(figsize=(12, 7))
        for class_num in range(10):

            # assign linestyle based on whether the class is growing or decaying
            linestyle = get_time_of_day_linestyle(class_num)
            
            if class_num in convection_decaying_classes:

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
                    linewidth=LINEWIDTH,
                    linestyle=linestyle
                )

                # remove top and right axis
                ax = style_distribution_axis(ax, var_name)

        add_distribution_legends(plt.gca())
        output_filename = f"1d_distr/convective_classes/{var_name}_convection_decaying_distributions.png"
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
    #elif var_name == "cloud cover":
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
