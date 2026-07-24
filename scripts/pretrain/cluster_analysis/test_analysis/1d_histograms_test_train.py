"""
This script is used to generate 1D histograms for test and train datasets 
in the context of cluster analysis. For the video summary csv file, It creates a histogram for each variable and for each class
a plot with the test distribution in dashed color, and train distribution in solid colors,
 and saves the plots in the specified output directory.
The code also calculates the KS statistic and Wasserstein distance between the test and train distributions for each variable/class pair, and saves these metrics in a CSV file.


Author: Claudia Acquistapace
Date: 01/07/2026


"""



import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append("/home/claudia/codes/ML_postprocessing")

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except ImportError:
    ks_2samp = None
    wasserstein_distance = None


from utils.configs import load_config
from utils.plotting.plot_class_analysis import plot_hourly_histogram, style_axis, set_x_ranges_for_variable
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
DISTRIBUTION_OUTPUT_DIR = OUTPUT_DIR / "1d_distr"
DISTRIBUTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_distribution_distances(values_train, values_test):
    """Compare train and test distributions for one variable/class pair."""
    if len(values_train) == 0 or len(values_test) == 0:
        return {
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "wasserstein_distance": np.nan,
            "wasserstein_distance_normalized": np.nan,
        }

    train_array = values_train.to_numpy()
    test_array = values_test.to_numpy()

    if ks_2samp is not None:
        ks_result = ks_2samp(train_array, test_array)
        ks_statistic = ks_result.statistic
        ks_pvalue = ks_result.pvalue
    else:
        sorted_values = np.sort(np.unique(np.concatenate([train_array, test_array])))
        train_ecdf = np.searchsorted(np.sort(train_array), sorted_values, side="right") / len(train_array)
        test_ecdf = np.searchsorted(np.sort(test_array), sorted_values, side="right") / len(test_array)
        ks_statistic = np.max(np.abs(train_ecdf - test_ecdf))
        ks_pvalue = np.nan

    if wasserstein_distance is not None:
        wasserstein = wasserstein_distance(train_array, test_array)
    else:
        wasserstein = np.nan

    pooled_std = np.nanstd(np.concatenate([train_array, test_array]))
    wasserstein_normalized = wasserstein / pooled_std if pooled_std > 0 else np.nan

    return {
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "wasserstein_distance": wasserstein,
        "wasserstein_distance_normalized": wasserstein_normalized,
        "pooled_std": pooled_std,
    }


def plot_distribution_distance_metrics(distribution_comparison_df, output_dir):
    """Plot summary figures from the test/train distribution metrics CSV."""
    metrics_to_plot = [
        ("ks_statistic", "KS statistic", "viridis", None),
        ("wasserstein_distance_normalized", "Normalized Wasserstein distance", "viridis", None),
        (
            "test_minus_train_mean_normalized",
            "Normalized test - train mean",
            "coolwarm",
            "symmetric",
        ),
    ]

    for metric_name, metric_label, cmap, color_scale in metrics_to_plot:
        metric_df = distribution_comparison_df.pivot(
            index="variable",
            columns="class",
            values=metric_name,
        )

        fig_width = max(8, 0.8 * len(metric_df.columns) + 3)
        fig_height = max(4, 0.45 * len(metric_df.index) + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        plot_values = metric_df.to_numpy()
        vmin = vmax = None
        if color_scale == "symmetric":
            max_abs_value = np.nanmax(np.abs(plot_values))
            if np.isfinite(max_abs_value) and max_abs_value > 0:
                vmin = -max_abs_value
                vmax = max_abs_value

        image = ax.imshow(plot_values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(metric_label)

        ax.set_xticks(np.arange(len(metric_df.columns)))
        ax.set_xticklabels(metric_df.columns)
        ax.set_yticks(np.arange(len(metric_df.index)))
        ax.set_yticklabels(metric_df.index)
        ax.set_xlabel("Class")
        ax.set_ylabel("Variable")
        ax.set_title(metric_label)
        style_axis(ax)

        output_path = output_dir / f"{metric_name}_heatmap.png"
        fig.savefig(output_path, transparent=True, bbox_inches="tight")
        print(f"{metric_label} heatmap saved to {output_path}")
        plt.close(fig)

    top_metric = "wasserstein_distance_normalized"
    top_rows = (
        distribution_comparison_df.dropna(subset=[top_metric])
        .sort_values(top_metric, ascending=False)
        .head(20)
        .copy()
    )
    top_rows["variable_class"] = (
        top_rows["variable"] + " | class " + top_rows["class"].astype(str)
    )

    fig_height = max(5, 0.35 * len(top_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(
        top_rows["variable_class"],
        top_rows[top_metric],
        color="#4C72B0",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Wasserstein distance")
    ax.set_ylabel("")
    ax.set_title("Largest test/train distribution shifts")
    style_axis(ax)

    output_path = output_dir / "top20_normalized_wasserstein_distance.png"
    fig.savefig(output_path, transparent=True, bbox_inches="tight")
    print(f"Top distribution shift plot saved to {output_path}")
    plt.close(fig)


def main():

    # load the crop statistics CSV file
    videos_training_df = pd.read_csv(CSV_FILES["training"])
    videos_testing_df = pd.read_csv(CSV_FILES["testing"])
    print("Crop statistics CSV files loaded successfully.")


    # drop class with label -100
    videos_training_df["label"] = pd.to_numeric(videos_training_df["label"], errors="coerce")
    videos_training_df = videos_training_df[videos_training_df["label"] != -100]

    videos_testing_df["label"] = pd.to_numeric(videos_testing_df["label"], errors="coerce")
    videos_testing_df = videos_testing_df[videos_testing_df["label"] != -100]

    # search all columns in the file header that end with _mean and loop over them to plot the distributions for each class
    mean_columns_train = [col for col in videos_training_df.columns if col.endswith("_mean")]   
    mean_columns_test = [col for col in videos_testing_df.columns if col.endswith("_mean")]
    distribution_comparison_rows = []

    # loop on the variables stored in the columns of the training dataframe
    for col in mean_columns_train:

        # extract the variable name from the column name
        var_name = col.replace("_mean", "")

        # read corresponding column from the test dataframe
        if col not in mean_columns_test:
            print(f"Column {col} not found in test dataframe. Skipping.")
            continue
        
        # loop on classes and plot the distributions for each class
        for class_num in range(10):

            # create a new figure for each variable and class
            plt.figure(figsize=(10, 6))

            # filter the dataframe for the current class
            class_df_train = videos_training_df[videos_training_df["label"] == class_num]

            values_train = pd.to_numeric(class_df_train[col], errors="coerce").dropna()
            values_test = pd.to_numeric(videos_testing_df[videos_testing_df["label"] == class_num][col], errors="coerce").dropna()
            
            if var_name in ["precipitation", "euclid_msg_grid"]:
                values_train = values_train[values_train > 0]
                values_test = values_test[values_test > 0]
                if len(values_train) > 0:
                    values_train = values_train[values_train <= values_train.quantile(0.99)]
                if len(values_test) > 0:
                    values_test = values_test[values_test <= values_test.quantile(0.99)]
                bins = 30
            else:
                bins = 50

            distance_metrics = calculate_distribution_distances(values_train, values_test)
            distribution_comparison_rows.append(
                {
                    "variable": var_name,
                    "class": class_num,
                    "n_train": len(values_train),
                    "n_test": len(values_test),
                    "train_mean": values_train.mean() if len(values_train) > 0 else np.nan,
                    "test_mean": values_test.mean() if len(values_test) > 0 else np.nan,
                    "test_minus_train_mean": (
                        values_test.mean() - values_train.mean()
                        if len(values_train) > 0 and len(values_test) > 0
                        else np.nan
                    ),
                    "test_minus_train_mean_normalized": (
                        (values_test.mean() - values_train.mean()) / distance_metrics["pooled_std"]
                        if (
                            len(values_train) > 0
                            and len(values_test) > 0
                            and distance_metrics["pooled_std"] > 0
                        )
                        else np.nan
                    ),
                    "train_median": values_train.median() if len(values_train) > 0 else np.nan,
                    "test_median": values_test.median() if len(values_test) > 0 else np.nan,
                    "test_minus_train_median": (
                        values_test.median() - values_train.median()
                        if len(values_train) > 0 and len(values_test) > 0
                        else np.nan
                    ),
                    **distance_metrics,
                }
            )

            # plot the distribution of the current variable for the current class
            ax = plt.gca()

            # plot histogram for train dataset in solid line
            ax.hist(
                values_train,
                bins=bins, 
                label=f"Class {class_num} (training)",
                density=True, 
                histtype="step",
                linestyle="-",
                color=colors_per_class1_names.get(str(class_num), None),
                linewidth=3,
            )

            # plot histogram for test dataset in dashed line
            ax.hist(
                values_test,
                bins=bins, 
                label=f"Class {class_num} (test)", 
                density=True, 
                histtype="step",
                linestyle="--",
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
            output_filename = f"{var_name}_class{class_num}_distribution.png"
            output_path = DISTRIBUTION_OUTPUT_DIR / output_filename
            plt.savefig(output_path, transparent=True, bbox_inches="tight")
            print(f"Distribution plot for {var_name} saved to {output_path}")
            plt.close()

    distribution_comparison_df = pd.DataFrame(distribution_comparison_rows)
    distribution_comparison_df = distribution_comparison_df.sort_values(
        ["ks_statistic", "wasserstein_distance_normalized"],
        ascending=False,
    )
    output_path = DISTRIBUTION_OUTPUT_DIR / "test_train_distribution_distances.csv"
    distribution_comparison_df.to_csv(output_path, index=False)
    print(f"Distribution comparison metrics saved to {output_path}")

    plot_distribution_distance_metrics(distribution_comparison_df, DISTRIBUTION_OUTPUT_DIR)

    print("\nMost different test/train distributions by KS statistic:")
    print(
        distribution_comparison_df.head(20).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

if __name__ == "__main__":
    main()
