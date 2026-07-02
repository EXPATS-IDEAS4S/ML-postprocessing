""""
code to plot the distribution of cosine distances from the centroid of each class for the training and testing 
dataset. 

"""

import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.class_colors import colors_per_class1_names
from utils.plotting.plot_class_analysis import style_axis


# Add the repository root so top-level packages such as `utils` resolve
# regardless of the directory from which this script is launched.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "training": config["output_files"]["features_training"],
    "testing": config["output_files"]["features_testing"],
}
INVALID_LABELS = {-100}


def read_csv_to_dataframe(csv_file):
    """Read a feature CSV and drop debug rows when present."""
    df = pd.read_csv(csv_file)

    if csv_file.endswith("_debug.csv") and "frame" in df.columns:
        df = df[df["frame"] != -710387]
    if df.empty:
        print(f"Warning: {csv_file} is empty after filtering.")

    return df


def prepare_distance_dataframe(df, dataset_name):
    """Keep valid class labels and numeric distances for plotting."""
    df = df[["distance", "label"]].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.dropna(subset=["distance", "label"])
    df["label"] = df["label"].astype(int)

    invalid_count = df["label"].isin(INVALID_LABELS).sum()
    if invalid_count:
        print(f"Dropping {invalid_count} {dataset_name} row(s) with invalid labels: {sorted(INVALID_LABELS)}")
    df = df[~df["label"].isin(INVALID_LABELS)]

    known_labels = {int(label) for label in colors_per_class1_names}
    unknown_labels = sorted(set(df["label"]) - known_labels)
    if unknown_labels:
        print(f"Warning: dropping {dataset_name} row(s) with unknown labels: {unknown_labels}")
        df = df[df["label"].isin(known_labels)]

    return df


def main():

    # read the training and testing CSV files
    df_train = read_csv_to_dataframe(CSV_FILES["training"])
    df_test = read_csv_to_dataframe(CSV_FILES["testing"])
    df_train = prepare_distance_dataframe(df_train, "training")
    df_test = prepare_distance_dataframe(df_test, "testing")

    # select only the columns of the dataframes distance and label and group by label to get the distribution of distances for each class
    df_train_grouped = df_train[["distance", "label"]].groupby("label")
    df_test_grouped = df_test[["distance", "label"]].groupby("label")

    # drop label = -100 from both dataframes
    df_train = df_train[df_train["label"] != -100]
    df_test = df_test[df_test["label"] != -100]

    # find labels that are present in both dataframes and plot the distribution of distances for each class
    labels = sorted(set(df_train["label"]).union(df_test["label"]))
    ncols = 5
    nrows = max(1, math.ceil(len(labels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows), squeeze=False)
    axs = axs.ravel()

    for idx, label in enumerate(labels):
        ax = axs[idx]

        if label in df_train_grouped.groups:
            group = df_train_grouped.get_group(label)
            group["distance"].plot(
                kind="hist",
                bins=30,
                density=True,
                histtype="step",
                ax=ax,
                color=colors_per_class1_names[str(label)],
                linestyle="dashed",
                label="training",
                linewidth=3,
            )


        if label in df_test_grouped.groups:
            group_test = df_test_grouped.get_group(label)
            group_test["distance"].plot(
                kind="hist",
                bins=30,
                density=True,
                histtype="step",
                ax=ax,
                color=colors_per_class1_names[str(label)],
                label="testing",
                linewidth=3,
            )

        ax.set_title(f"Class {label}")
        ax.set_xlabel("Cosine Distance")
        ax.set_ylabel("Density")
        # set axis style
        style_axis(ax)

    for ax in axs[len(labels):]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=3, linestyle="-", label="testing"),
        Line2D([0], [0], color="black", linewidth=3, linestyle="--", label="training"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    
    # save the figure to the output directory
    output_dir = Path(config["output_files"]["figures_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "distance_distributions_per_class.png"
    plt.tight_layout(rect=(0, 0, 0.92, 1))
    plt.savefig(output_file, bbox_inches="tight")



if __name__ == "__main__":
    main()
    
