"""
This code calculates, for each class associated to a video, the probability
that the class associated to the consecutive video in time extracted from the
same view of the test dataset is class 0, 1, 2, 3, 4, 5, 6, 7, 8, i.e. the
transition probabilities between classes for consecutive videos in time. The
output is a csv file. it also plots the transition probabilities for each view
 and for all views together, for all times and for selected time intervals.

 input:
    - test dataset csv file with columns: 'crop', 'label', 'time'
    - output csv file name

how to run:
    python transition_probabilities.py 

"""

import pandas as pd
import os
import math
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from pathlib import Path

N_CLASSES = 10
NEXT_VIDEO_OFFSET_MINUTES = 15
PROBABILITY_COLOR_GAMMA = 0.4
TIME_INTERVALS = {
    "All times": None,
    "08:00-21:00": lambda time: (time.dt.hour >= 8) & (time.dt.hour < 21),
    "21:00-08:00": lambda time: (time.dt.hour >= 21) | (time.dt.hour < 8),
}
TIME_INTERVAL_FILENAMES = {
    "All times": "all_times",
    "08:00-21:00": "0800_2100",
    "21:00-08:00": "2100_0800",
}

# read paths from config
sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.plot_class_analysis import plot_hourly_histogram, style_axis
from utils.plotting.class_colors import colors_per_class1_names

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "training": config["output_files"]["training_csv_cth"],
    "testing": config["output_files"]["testing_csv_cth"],
}

# create output directory if it doesn't exist
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])


def append_transition_probability_rows(rows, interval_name, view_name, transitions_df):
    for class_num in range(N_CLASSES):
        class_df = transitions_df[transitions_df["label"] == class_num]
        next_labels = class_df["next_label"].value_counts(normalize=True).to_dict()
        row = {
            "time_interval": interval_name,
            "view": view_name,
            "current_class": class_num,
            "n_transitions": len(class_df),
        }
        row.update({f"next_class_{i}": next_labels.get(i, 0) for i in range(N_CLASSES)})
        rows.append(row)


def main():
    
    # define the input and output paths
    test_csv_path = CSV_FILES["testing"]
    output_csv_path = OUTPUT_DIR

    # read the test dataset csv file
    test_csv_path = os.path.join(test_csv_path)
    test_df = pd.read_csv(test_csv_path, low_memory=False)
    required_columns = {"crop", "label", "time"}
    missing_columns = required_columns.difference(test_df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in {test_csv_path}: {sorted(missing_columns)}. "
            f"Available columns: {list(test_df.columns)}"
        )

    # Extract the numeric part of strings like "view001" from the crop name.
    test_df["view"] = test_df["crop"].str.extract(r"view(\d{3})", expand=False)
    if test_df["view"].isna().any():
        missing_views = test_df.loc[test_df["view"].isna(), "crop"].unique()
        raise ValueError(f"Could not extract view from crop names: {missing_views[:5]}")

    # build one row per video crop, with start/end times from the frame timestamps
    test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce")
    test_df["time"] = pd.to_datetime(test_df["time"])
    video_df = (
        test_df.groupby(["crop", "view"], as_index=False)
        .agg(
            label=("label", "first"),
            start_time=("time", "min"),
            end_time=("time", "max"),
        )
        .sort_values(by=["view", "start_time"])
    )

    # A consecutive video starts 15 minutes after the previous video's end time.
    video_df["expected_next_start_time"] = video_df["end_time"] + pd.Timedelta(
        minutes=NEXT_VIDEO_OFFSET_MINUTES
    )
    next_video_df = video_df[["view", "start_time", "label"]].rename(
        columns={
            "start_time": "expected_next_start_time",
            "label": "next_label",
        }
    )
    transitions_df = video_df.merge(
        next_video_df,
        on=["view", "expected_next_start_time"],
        how="inner",
    )

    # find view-specific transition probabilities for all times and selected time intervals.
    rows = []
    for interval_name, interval_filter in TIME_INTERVALS.items():
        if interval_filter is None:
            interval_df = transitions_df
        else:
            current_in_interval = interval_filter(transitions_df["start_time"])
            next_in_interval = interval_filter(transitions_df["expected_next_start_time"])
            interval_df = transitions_df[current_in_interval & next_in_interval]

        append_transition_probability_rows(
            rows,
            interval_name,
            "all_views",
            interval_df,
        )

        for view, view_df in interval_df.groupby("view"):
            append_transition_probability_rows(rows, interval_name, view, view_df)

    transition_probs_df = pd.DataFrame(rows)

    # save the transition probabilities to a csv file
    output_csv_file = "transition_probabilities_time_intervals.csv"
    output_csv_path = os.path.join(output_csv_path, output_csv_file)
    fig_path = OUTPUT_DIR
    fig_path.mkdir(parents=True, exist_ok=True)
    transition_probs_df.to_csv(output_csv_path, index=False)

    # plot one multipanel figure per time interval, with one panel per view.
    import seaborn as sns
    import matplotlib.pyplot as plt

    next_class_columns = [f"next_class_{i}" for i in range(N_CLASSES)]
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    probability_norm = PowerNorm(gamma=PROBABILITY_COLOR_GAMMA, vmin=0, vmax=1)

    for interval_name in TIME_INTERVALS:
        interval_df = transition_probs_df[
            (transition_probs_df["time_interval"] == interval_name)
            & (transition_probs_df["view"] != "all_views")
        ]
        views = sorted(interval_df["view"].unique())
        if not views:
            print(f"No transitions available for {interval_name}; skipping plot.")
            continue

        n_cols = min(3, len(views))
        n_rows = math.ceil(len(views) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        for i, view in enumerate(views):
            ax = axes[i // n_cols][i % n_cols]
            view_df = interval_df[interval_df["view"] == view]
            matrix = view_df.set_index("current_class").sort_index()[next_class_columns]
            plot_matrix = matrix.mask(matrix == 0)
            annotations = matrix.applymap(lambda value: f"{value:.2f}" if value > 0 else "")

            sns.heatmap(
                plot_matrix,
                annot=annotations,
                fmt="",
                cmap=cmap,
                norm=probability_norm,
                cbar=i == len(views) - 1,
                cbar_ax=cbar_ax if i == len(views) - 1 else None,
                cbar_kws={"label": "Probability"},
                ax=ax,
            )
            ax.set_title(f"View {view}")
            ax.set_xlabel("Next Class")
            ax.set_ylabel("Current Class")

        for i in range(len(views), n_rows * n_cols):
            axes[i // n_cols][i % n_cols].axis("off")

        cbar_ax.tick_params(labelsize=14)
        cbar_ax.set_ylabel("Probability", fontsize=16)

        fig.suptitle(f"Transition Probabilities by View ({interval_name})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 0.9, 0.96])
        fig.savefig(
            os.path.join(
                fig_path,
                f"transition_probabilities_by_view_{TIME_INTERVAL_FILENAMES[interval_name]}.png",
            ),
            dpi=300,
        )
        plt.close(fig)

    for interval_name in TIME_INTERVALS:
        interval_df = transition_probs_df[
            (transition_probs_df["time_interval"] == interval_name)
            & (transition_probs_df["view"] == "all_views")
        ]
        if interval_df.empty:
            print(f"No all-view transitions available for {interval_name}; skipping plot.")
            continue

        matrix = interval_df.set_index("current_class").sort_index()[next_class_columns]
        plot_matrix = matrix.mask(matrix == 0)
        annotations = matrix.applymap(lambda value: f"{value:.2f}" if value > 0 else "")

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            plot_matrix,
            annot=annotations,
            fmt="",
            cmap=cmap,
            norm=probability_norm,
            cbar_kws={"label": "Probability"},
            ax=ax,
        )
        ax.set_title(f"Transition Probabilities - All Views ({interval_name})")
        ax.set_xlabel("Next Class")
        ax.set_ylabel("Current Class")
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                fig_path,
                f"transition_probabilities_all_views_{TIME_INTERVAL_FILENAMES[interval_name]}.png",
            ),
            dpi=300,
        )
        plt.close(fig)
    

if __name__ == "__main__":
    main()
