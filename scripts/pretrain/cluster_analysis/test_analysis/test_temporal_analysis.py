"""
With this code, we want to read the crops_video_summary_with_temporal_sequence_labels.csv that characterize the 
test dataset in time with respect to the event for each view.
We want to group by temporal sequence label and derive:
    - class distributions over the temporal sequence
    - time series over the temporal sequence for each variable and gradient that characterize the video

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
from utils.plotting.class_colors import extreme_event_classes

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "testing_video_summary_temporal": config["output_files"]["testing_video_summary_temporal"],
    "training_video_summary": config["output_files"]["training_video_summary"],
}
output_dir = config["output_files"]["figures_dir"]
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12

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


    # plot class distributions over temporal sequence
    # ***********************************************************************************************
    # Rows are temporal sequence steps, columns are classes.
    class_counts = (
        temporal_df.groupby(["temporal_sequence_label", "label"])
        .size()
        .unstack(fill_value=0)
    )
    class_distributions = class_counts.div(class_counts.sum(axis=1), axis=0)

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

    # plot class distributions over the temporal sequence for aggregated views corresponding to different regions of the domain
    # ***********************************************************************************************
    view_groups = {
        "North": [7, 8, 9, 4, 5, 6],
        "South": [1, 2, 3],
    }

    # for each view group, we want to compute the class distributions over the temporal sequence
    for group_name, views in view_groups.items():
        group_df = temporal_df[temporal_df["view"].isin(views)]
        group_class_counts = (
            group_df.groupby(["temporal_sequence_label", "label"])
            .size()
            .unstack(fill_value=0)
        )
        group_class_distributions = group_class_counts.div(
            group_class_counts.sum(axis=1), axis=0
        )

        # filter out temporal sequence labels that are not present in the group
        group_temporal_order = [
            label for label in temporal_order if label in group_class_distributions.index
        ]
        group_class_distributions = group_class_distributions.loc[group_temporal_order]

        # plot class distributions over temporal sequence for the view group
        plot_class_distributions_over_temporal_sequence(
            group_class_distributions,
            class_labels,
            class_colors,
            temporal_label_mapping,
            output_dir,
            output_prefix=f"group_{group_name}_",
        )

    # calculate now mean and std of cloud properties and gradients for each temporal sequence label and for each class
    # ***********************************************************************************************

    # drop the temporal sequence label -100, which corresponds to video crops that are far from the event
    temporal_df = temporal_df[temporal_df["temporal_sequence_label"] != -100]

    # drop it also from the labels that need to go on the x axis of the plots
    temporal_label_mapping = {
        label: name
        for label, name in temporal_label_mapping.items() if label != -100
    }
    # group by temporal sequence label and class, and calculate mean and std for each variable and gradient
    grouped = temporal_df.groupby(["temporal_sequence_label", "label"])
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std().reset_index()

    # plot the mean and std for selected variable and gradients over the temporal sequence for each class
    # ***********************************************************************************************
    selected_vars_temporal_series = [
        "cth_mean",
        "cth10plus_mean",
        "cot30plus_mean",
        "cma_mean",
        "cot_mean",
        "precipitation_mean",
        "euclid_msg_grid_mean",
        "cth_gradient",
        "cma_gradient",
        "cot_gradient",
        "precipitation_gradient",
        "euclid_msg_grid_gradient",
        "cot30plus_gradient",
        "cth10plus_gradient"

    ]

    plot_temporal_series_for_variables(
        mean_df,
        std_df,
        selected_vars_temporal_series,
        class_labels,
        colors_per_class1_names,
        temporal_label_mapping,
        output_dir,
    )

    # calculate anomalies of the mean and std values for each variable and gradient over the temporal sequence,
    # with respect to the mean value of the variable/gradient for each class in the training dataset 
    # and plot the anomalies over the temporal sequence for each class
    # ***********************************************************************************************

    # read the training video summary csv file to get the mean values of each variable and gradient for each class
    training_df = read_csv_to_dataframe(CSV_FILES["training_video_summary"])

    # calculate maea values for each variable and gradient for each class in the training dataset
    training_grouped = training_df.groupby("label")
    training_mean_df = training_grouped.mean().reset_index()

    # calculate anomalies of the mean and std values for each variable and gradient over the temporal sequence,
    # with respect to the mean value of the variable/gradient for each class in the training dataset
    anomalies_mean_df = mean_df.copy()
    anomalies_std_df = std_df.copy()
    selected_anomaly_vars_temporal_series = []
    for var in selected_vars_temporal_series:
        anomaly_var = f"{var}_anomaly"
        anomalies_mean_df[var] = anomalies_mean_df.apply(
            lambda row: row[var] - training_mean_df.loc[
                training_mean_df["label"] == row["label"], var
            ].values[0],
            axis=1,
        )
        anomalies_std_df[var] = anomalies_std_df.apply(
            lambda row: row[var] - training_mean_df.loc[
                training_mean_df["label"] == row["label"], var
            ].values[0],
            axis=1,
        )
        # add a name to the variable to indicate that it is an anomaly
        anomalies_mean_df.rename(columns={var: anomaly_var}, inplace=True)
        anomalies_std_df.rename(columns={var: anomaly_var}, inplace=True)
        selected_anomaly_vars_temporal_series.append(anomaly_var)
        
    
    # plot the anomalies over the temporal sequence for each class
    plot_temporal_series_for_variables(
        anomalies_mean_df,
        anomalies_std_df,
        selected_anomaly_vars_temporal_series,
        class_labels,
        colors_per_class1_names,
        temporal_label_mapping,
        output_dir,
    )

    # Repeat the temporal-series and anomaly analysis separately for each view group.
    for group_name, views in view_groups.items():
        group_df = temporal_df[temporal_df["view"].isin(views)]
        if group_df.empty:
            print(f"No temporal data available for {group_name}; skipping temporal series.")
            continue

        group_mean_df, group_std_df = calculate_temporal_mean_std(group_df)
        output_prefix = f"group_{group_name}_"

        plot_temporal_series_for_variables(
            group_mean_df,
            group_std_df,
            selected_vars_temporal_series,
            class_labels,
            colors_per_class1_names,
            temporal_label_mapping,
            output_dir,
            output_prefix=output_prefix,
        )

        group_anomalies_mean_df, group_anomalies_std_df = calculate_anomalies(
            group_mean_df,
            group_std_df,
            selected_vars_temporal_series,
            training_mean_df,
        )
        group_anomaly_vars = [f"{var}_anomaly" for var in selected_vars_temporal_series]
        plot_temporal_series_for_variables(
            group_anomalies_mean_df,
            group_anomalies_std_df,
            group_anomaly_vars,
            class_labels,
            colors_per_class1_names,
            temporal_label_mapping,
            output_dir,
            output_prefix=output_prefix,
        )


def calculate_temporal_mean_std(temporal_df):
    grouped = temporal_df.groupby(["temporal_sequence_label", "label"])
    mean_df = grouped.mean(numeric_only=True).reset_index()
    std_df = grouped.std(numeric_only=True).reset_index()
    return mean_df, std_df


def calculate_anomalies(
    mean_df,
    std_df,
    selected_vars_temporal_series,
    training_mean_df,
):
    anomalies_mean_df = mean_df.copy()
    anomalies_std_df = std_df.copy()

    for var in selected_vars_temporal_series:
        anomaly_var = f"{var}_anomaly"

        anomalies_mean_df[var] = anomalies_mean_df.apply(
            lambda row: row[var] - training_mean_df.loc[
                training_mean_df["label"] == row["label"], var
            ].values[0],
            axis=1,
        )
        anomalies_std_df[var] = anomalies_std_df.apply(
            lambda row: row[var] - training_mean_df.loc[
                training_mean_df["label"] == row["label"], var
            ].values[0],
            axis=1,
        )
        anomalies_mean_df.rename(columns={var: anomaly_var}, inplace=True)
        anomalies_std_df.rename(columns={var: anomaly_var}, inplace=True)

    return anomalies_mean_df, anomalies_std_df



def plot_temporal_series_for_variables(
    mean_df,
    std_df,
    selected_vars_temporal_series,
    class_labels,
    colors_per_class1_names,
    temporal_label_mapping,
    output_dir,
    output_prefix="",
    ):

    # one plot for each variable and gradient, with mean and std for each class
    for var in selected_vars_temporal_series:
        plt.figure(figsize=(10, 6))
        for class_label in class_labels:

            class_mean = mean_df[mean_df["label"] == class_label]
            class_std = std_df[std_df["label"] == class_label]
            
            # plot just the mean values
            plt.plot(
                class_mean["temporal_sequence_label"],
                class_mean[var],
                label=f"Class {class_label}",
                color=colors_per_class1_names.get(str(class_label), "lightgray"),
                marker="o",
            )

            #plt.errorbar(
            #    class_mean["temporal_sequence_label"],
            #    class_mean[var],
            #    yerr=class_std[var],
            #    label=f"Class {class_label}",
            #    color=colors_per_class1_names.get(str(class_label), "lightgray"),
            #    fmt="-o",
            #    capsize=3,
            #)

        plt.xlabel("Temporal Sequence Label", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel(var, fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(
            ticks=list(temporal_label_mapping.keys()),
            labels=[temporal_label_mapping[label] for label in temporal_label_mapping.keys()],
            rotation=35,
            ha="right",
            fontsize=TICK_LABEL_FONTSIZE,
        )
        plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
        style_axis(plt.gca())
        plt.legend(
            title="Class",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_FONTSIZE,
        )
        plt.tight_layout()
        output_file = Path(output_dir) / f"{output_prefix}{var}_temporal_sequence.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Temporal series plot for {var} saved to {output_file}")
        plt.close()
    
    return None



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
