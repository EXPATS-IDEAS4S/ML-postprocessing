"""
Plot temporal class distributions and export crop-group CSV files.

The script reads the test temporal video-summary table defined in
configs/process_run_GRL.yaml:
    output_files.testing_video_summary_temporal

It groups videos by temporal_sequence_label and class label, then saves stacked
class-distribution figures to:
    output_files.figures_dir

Figure outputs:
    class_distributions_over_temporal_sequence.png
    daytime_class_distributions_over_temporal_sequence.png
    nighttime_class_distributions_over_temporal_sequence.png
    group_North_class_distributions_over_temporal_sequence.png
    group_North_daytime_class_distributions_over_temporal_sequence.png
    group_North_nighttime_class_distributions_over_temporal_sequence.png
    group_South_class_distributions_over_temporal_sequence.png
    group_South_daytime_class_distributions_over_temporal_sequence.png
    group_South_nighttime_class_distributions_over_temporal_sequence.png

It also exports raw crop-group CSV subsets to:
    output_files.csv_dir

CSV outputs:
    crops_video_summary_temporal_sequence_north_daytime.csv
    crops_video_summary_temporal_sequence_north_nighttime.csv
    crops_video_summary_temporal_sequence_south_daytime.csv
    crops_video_summary_temporal_sequence_south_nighttime.csv

Daytime is defined as 06:00 <= time_start < 18:00. Nighttime is all other
hours. North views are [7, 8, 9, 4, 5, 6], and South views are [1, 2, 3].

How to run:
    python scripts/pretrain/cluster_analysis/test_analysis/test_temporal_analysis.py

author: Claudia Acquistapace
date: 2027-07-02
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

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "testing_video_summary_temporal": config["output_files"]["testing_video_summary_temporal"],
}
output_dir = config["output_files"]["figures_dir"]
csv_output_dir = Path(config["output_files"]["csv_dir"])
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12
DAYTIME_START_HOUR = 6
DAYTIME_END_HOUR = 18

def main():

    # read csv file with temporal sequence labels
    temporal_df = read_csv_to_dataframe(CSV_FILES["testing_video_summary_temporal"])

    # assign labels to use in the plot for the temporal sequence labels
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

    temporal_df = temporal_df.copy()
    temporal_df["label"] = pd.to_numeric(temporal_df["label"], errors="coerce")
    temporal_df = temporal_df.dropna(subset=["label", "temporal_sequence_label"])
    temporal_df = temporal_df[temporal_df["label"] != -100]
    temporal_df["label"] = temporal_df["label"].astype(int)
    temporal_df["temporal_sequence_label"] = pd.to_numeric(
        temporal_df["temporal_sequence_label"],
        errors="coerce",
    ).astype(int)
    temporal_df["view"] = pd.to_numeric(temporal_df["view"], errors="coerce").astype(int)
    temporal_df["time_start"] = pd.to_datetime(
        temporal_df["time_start"],
        errors="coerce",
    )
    temporal_df = temporal_df.dropna(subset=["time_start"])
    temporal_df["day_night"] = np.where(
        temporal_df["time_start"].dt.hour.between(
            DAYTIME_START_HOUR,
            DAYTIME_END_HOUR - 1,
        ),
        "daytime",
        "nighttime",
    )


    # plot class distributions over temporal sequence
    # ***********************************************************************************************
    # Rows are temporal sequence steps, columns are classes.
    class_distributions = calculate_class_distributions_over_temporal_sequence(
        temporal_df
    )

    temporal_order = [
        label
        for label in [-4, -2, -1, 0, 1, 2, 4, -100]
        if label in class_distributions.index
    ]
    remaining_temporal_labels = [
        label for label in class_distributions.index if label not in temporal_order
    ]
    temporal_order.extend(sorted(remaining_temporal_labels))
    class_distributions = class_distributions.loc[temporal_order]

    class_labels = sorted(class_distributions.columns)
    class_colors = [
        colors_per_class1_names.get(str(class_label), "lightgray")
        for class_label in class_labels
    ]

    # plot class distributions over temporal sequence for all views
    plot_class_distributions_over_temporal_sequence(
        class_distributions,
        class_labels,
        class_colors,
        temporal_label_mapping,
        output_dir, 
    )

    # plot class distributions over temporal sequence for all views, split by daytime/nighttime
    # ***********************************************************************************************
    for day_night_label, day_night_df in temporal_df.groupby("day_night"):
        day_night_distributions = calculate_class_distributions_over_temporal_sequence(
            day_night_df
        )
        day_night_temporal_order = [
            label for label in temporal_order if label in day_night_distributions.index
        ]
        if day_night_temporal_order:
            day_night_distributions = day_night_distributions.loc[
                day_night_temporal_order
            ]

        plot_class_distributions_over_temporal_sequence(
            day_night_distributions,
            class_labels,
            class_colors,
            temporal_label_mapping,
            output_dir,
            output_prefix=f"{day_night_label}_",
        )

    # plot class distributions over the temporal sequence for aggregated views corresponding to different regions of the domain
    # ***********************************************************************************************
    view_groups = {
        "North": [7, 8, 9, 4, 5, 6],
        "South": [1, 2, 3],
    }

    # for each view group, we want to compute the class distributions over the temporal sequence
    for group_name, views in view_groups.items():
        group_df = temporal_df[temporal_df["view"].isin(views)]
        group_class_distributions = calculate_class_distributions_over_temporal_sequence(
            group_df
        )

        # filter out temporal sequence labels that are not present in the group
        group_temporal_order = [
            label for label in temporal_order if label in group_class_distributions.index
        ]
        if group_temporal_order:
            group_class_distributions = group_class_distributions.loc[
                group_temporal_order
            ]

        # plot class distributions over temporal sequence for the view group
        plot_class_distributions_over_temporal_sequence(
            group_class_distributions,
            class_labels,
            class_colors,
            temporal_label_mapping,
            output_dir,
            output_prefix=f"group_{group_name}_",
        )

        for day_night_label, day_night_group_df in group_df.groupby("day_night"):
            save_temporal_subset_csv(
                day_night_group_df,
                csv_output_dir,
                f"{group_name}_{day_night_label}",
            )
            day_night_group_distributions = (
                calculate_class_distributions_over_temporal_sequence(
                    day_night_group_df
                )
            )
            day_night_group_temporal_order = [
                label
                for label in temporal_order
                if label in day_night_group_distributions.index
            ]
            if day_night_group_temporal_order:
                day_night_group_distributions = day_night_group_distributions.loc[
                    day_night_group_temporal_order
                ]

            plot_class_distributions_over_temporal_sequence(
                day_night_group_distributions,
                class_labels,
                class_colors,
                temporal_label_mapping,
                output_dir,
                output_prefix=f"group_{group_name}_{day_night_label}_",
            )

def calculate_class_distributions_over_temporal_sequence(temporal_df):
    if temporal_df.empty:
        return pd.DataFrame()

    class_counts = (
        temporal_df.groupby(["temporal_sequence_label", "label"])
        .size()
        .unstack(fill_value=0)
    )
    return class_counts.div(class_counts.sum(axis=1), axis=0)


def save_temporal_subset_csv(subset_df, csv_output_dir, subset_name):
    csv_output_dir = Path(csv_output_dir)
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        csv_output_dir
        / f"crops_video_summary_temporal_sequence_{subset_name.lower()}.csv"
    )
    subset_df.to_csv(output_file, index=False)
    print(f"Saved temporal subset CSV to {output_file}")



def plot_class_distributions_over_temporal_sequence(
    class_distributions,
    class_labels,
    class_colors,
    temporal_label_mapping,
    output_dir,
    output_prefix="",
):
    # plot class distributions over temporal sequence
    if class_distributions.empty:
        print(f"No data available for {output_prefix or 'all views'}; skipping plot.")
        return None, None

    class_distributions = class_distributions.reindex(columns=class_labels, fill_value=0)

    # plot temporal sequence on x and classes as stacked bars on y
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(class_distributions))
    x_positions = np.arange(len(class_distributions))

    for class_label, class_color in zip(class_labels, class_colors):
        values = class_distributions[class_label].to_numpy()
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            color=class_color,
            edgecolor="white",
            linewidth=0.7,
            label=f"Class {class_label}",
        )
        bottom += values

    # set labels and title
    ax.set_xlabel("Temporal Sequence Label", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Fraction of Videos", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            temporal_label_mapping.get(temporal_label, str(temporal_label))
            for temporal_label in class_distributions.index
        ],
        rotation=35,
        ha="right",
        fontsize=TICK_LABEL_FONTSIZE,
    )
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(0, 1)
    style_axis(ax)

    # set legend
    ax.legend(
        title="Class",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
    )
    plt.tight_layout()
    # save the figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{output_prefix}class_distributions_over_temporal_sequence.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Class distributions over temporal sequence saved to {output_file}")

    return fig, ax








if __name__ == "__main__":
    main()
