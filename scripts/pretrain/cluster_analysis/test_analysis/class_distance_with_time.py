"""
code to plot how the distance from the centroid for each class changes over time for the different
temporal intervals identified by the the temporal sequence labels.

- input:
 - feature file with distances for each video of the test dataset 
 - csv with temporal sequence labels for each video of the test dataset

- output:
plot of distance time series for each class, as a function of the temporal sequence labels all views together

"""




import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from scripts.pretrain.cluster_analysis.var_class_temporal_series import read_csv_to_dataframe
from utils.plotting.class_colors import colors_per_class1_names
from utils.plotting.plot_class_analysis import style_axis
from utils.plotting.class_colors import extreme_event_classes

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "testing_video_summary_temporal": config["output_files"]["testing_video_summary_temporal"],
    "feature_file": config["output_files"]["features_testing"],
}

output_dir = config["output_files"]["figures_dir"]

temporal_label_mapping = {
    -4: "> 4 h before event",
    -2: "4h < t < 2h before event",
    -1: "< 2h before event",
    0: "between first and last report",
    1: "< 2h after event",
    2: "2h < t < 4h after event",
    4: "> 4 h after event",
    -100: "far from event",
}


def crop_key(crop_or_path):
    return Path(str(crop_or_path)).stem


def print_crop_match_diagnostics(video_summary_df, feature_df):
    video_crops = set(video_summary_df["crop_key"].dropna())
    feature_crops = set(feature_df["crop_key"].dropna())
    matching_crops = video_crops & feature_crops

    print(f"Video summary unique crops: {len(video_crops)}")
    print(f"Feature file unique crops: {len(feature_crops)}")
    print(f"Matching crops: {len(matching_crops)}")
    print(f"Video crops without feature distance: {len(video_crops - feature_crops)}")
    print(f"Feature crops not in video summary: {len(feature_crops - video_crops)}")

    if video_crops - feature_crops:
        print("Example video crops without feature distance:")
        print(sorted(video_crops - feature_crops)[:5])

    if feature_crops - video_crops:
        print("Example feature crops not in video summary:")
        print(sorted(feature_crops - video_crops)[:5])


def plot_distance_over_temporal_sequence(grouped_df, class_labels, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    for class_num in class_labels:
        class_df = grouped_df[grouped_df["label"] == class_num].sort_values(
            "temporal_sequence_label"
        )
        if class_df.empty:
            continue

        color = colors_per_class1_names.get(str(class_num), "lightgray")
        x_values = class_df["temporal_sequence_label"]
        mean_values = class_df["mean_distance"]

        ax.plot(
            x_values,
            mean_values,
            label=f"Class {class_num}",
            color=color,
            marker="o",
        )

    active_temporal_labels = [
        label
        for label in temporal_label_mapping
        if label != -100 and label in grouped_df["temporal_sequence_label"].values
    ]
    remaining_temporal_labels = [
        label
        for label in sorted(grouped_df["temporal_sequence_label"].unique())
        if label not in active_temporal_labels
    ]
    active_temporal_labels.extend(remaining_temporal_labels)

    ax.set_xlabel("Temporal Sequence Label")
    ax.set_ylabel("Mean Distance from Centroid")
    ax.set_title("Mean Distance from Centroid for Each Class Over Time")
    ax.set_xticks(active_temporal_labels)
    ax.set_xticklabels(
        [
            temporal_label_mapping.get(label, str(label))
            for label in active_temporal_labels
        ],
        rotation=35,
        ha="right",
    )
    style_axis(ax)
    ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    print(f"Distance temporal series plot saved to {output_path}")
    plt.show()
    plt.close(fig)


def main():

    # read the input csv files
    feature_df = read_csv_to_dataframe(CSV_FILES["feature_file"])

    # extract crop name from path by reading the file name at the end of the path
    feature_df["crop_key"] = feature_df["path"].apply(crop_key)

    # extract distance from centroid and crop in a new dataframe
    feature_df = feature_df[["crop_key", "distance"]]

    # read the video summary csv file with temporal sequence labels
    video_summary_df = read_csv_to_dataframe(CSV_FILES["testing_video_summary_temporal"])
    video_summary_df["crop_key"] = video_summary_df["crop"].apply(crop_key)

    print_crop_match_diagnostics(video_summary_df, feature_df)

    # assign distances to the video rows having the corresponding crop name
    video_summary_df = video_summary_df.merge(feature_df, on="crop_key", how="left")

    video_summary_df["label"] = pd.to_numeric(
        video_summary_df["label"], errors="coerce"
    )
    video_summary_df["temporal_sequence_label"] = pd.to_numeric(
        video_summary_df["temporal_sequence_label"], errors="coerce"
    )
    video_summary_df["distance"] = pd.to_numeric(
        video_summary_df["distance"], errors="coerce"
    )
    video_summary_df = video_summary_df.dropna(
        subset=["label", "temporal_sequence_label", "distance"]
    )
    video_summary_df["label"] = video_summary_df["label"].astype(int)
    video_summary_df["temporal_sequence_label"] = video_summary_df[
        "temporal_sequence_label"
    ].astype(int)

    # drop class label -100 and temporal label -100, matching the temporal-series plots
    video_summary_df = video_summary_df[
        (video_summary_df["label"] != -100)
        & (video_summary_df["temporal_sequence_label"] != -100)
    ]
    
    # print first 3 rows of the video summary dataframe
    print(video_summary_df.head(3))

    # group by class and temporal sequence label, and compute mean and std distance
    grouped_df = (
        video_summary_df.groupby(["label", "temporal_sequence_label"])["distance"]
        .agg(mean_distance="mean", std_distance="std", n_crops="count")
        .reset_index()
    )
    print(grouped_df.head())

    # save the plot to the output directory
    output_path = Path(output_dir) / "distance_from_centroid_over_time.png"
    class_labels = sorted(grouped_df["label"].unique())
    plot_distance_over_temporal_sequence(grouped_df, class_labels, output_path)


if __name__ == "__main__":
    main()
