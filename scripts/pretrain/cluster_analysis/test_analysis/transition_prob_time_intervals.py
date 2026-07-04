"""
With this code, we want to calculate the transition probabilities between classes
for consecutive videos in time. For each class associated with a video, it
calculates the probability that the next video from the same view belongs to
class 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9.

The code groups videos by view and temporal sequence label. It calculates
view-specific transition probabilities, then averages those probabilities over
the views and plots one transition probability matrix for each temporal interval.

Input:
    - test dataset csv file with columns:
      'crop', 'label', 'time_start', 'time_end', 'view', 'temporal_sequence_label'

Output:
    - transition probabilities csv file
    - transition probability matrix plots for each temporal interval
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.class_colors import colors_per_class1_names

N_CLASSES = 10
PROBABILITY_COLOR_GAMMA = 0.4
EXCLUDED_CLASS_LABELS = {-100}
EXCLUDED_TEMPORAL_LABELS = {-100}

TEMPORAL_LABEL_MAPPING = {
    -4: "> 4 h before event",
    -2: "4h < t < 2h before event",
    -1: "< 2h before event",
    0: "between first and last report",
    1: "< 2h after event",
    2: "2h < t < 4h after event",
    4: "> 4 h after event",
    -100: "far from event",
}

TEMPORAL_LABEL_FILENAMES = {
    -4: "more_than_4h_before",
    -2: "4h_to_2h_before",
    -1: "less_than_2h_before",
    0: "during_event",
    1: "less_than_2h_after",
    2: "2h_to_4h_after",
    4: "more_than_4h_after",
    -100: "far_from_event",
}

# Read filename of csv files from config file process_run_GRL.yaml.
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
CSV_FILES = {
    "testing": config["output_files"]["testing_video_summary_temporal"],
}

OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_columns(video_stats_df, csv_path):
    required_columns = {
        "crop",
        "label",
        "time_start",
        "time_end",
        "view",
        "temporal_sequence_label",
    }
    missing_columns = required_columns.difference(video_stats_df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns in {csv_path}: {sorted(missing_columns)}. "
            f"Available columns: {list(video_stats_df.columns)}"
        )


def clean_video_stats(video_stats_df):
    video_stats_df = video_stats_df.copy()
    video_stats_df["label"] = pd.to_numeric(video_stats_df["label"], errors="coerce")
    video_stats_df["view"] = pd.to_numeric(video_stats_df["view"], errors="coerce")
    video_stats_df["temporal_sequence_label"] = pd.to_numeric(
        video_stats_df["temporal_sequence_label"], errors="coerce"
    )
    video_stats_df["time_start"] = pd.to_datetime(
        video_stats_df["time_start"], errors="coerce"
    )
    video_stats_df["time_end"] = pd.to_datetime(
        video_stats_df["time_end"], errors="coerce"
    )
    video_stats_df = video_stats_df.dropna(
        subset=["label", "view", "temporal_sequence_label", "time_start", "time_end"]
    )

    video_stats_df["label"] = video_stats_df["label"].astype(int)
    video_stats_df["view"] = video_stats_df["view"].astype(int)
    video_stats_df["temporal_sequence_label"] = video_stats_df[
        "temporal_sequence_label"
    ].astype(int)

    video_stats_df = video_stats_df[
        ~video_stats_df["label"].isin(EXCLUDED_CLASS_LABELS)
        & ~video_stats_df["temporal_sequence_label"].isin(EXCLUDED_TEMPORAL_LABELS)
    ]
    return video_stats_df


def build_transitions(video_stats_df):
    transitions = []

    for view, view_df in video_stats_df.groupby("view"):
        view_df = view_df.sort_values("time_start").reset_index(drop=True)
        for index in range(len(view_df) - 1):
            current_video = view_df.iloc[index]
            next_video = view_df.iloc[index + 1]

            transitions.append(
                {
                    "view": view,
                    "temporal_sequence_label": current_video[
                        "temporal_sequence_label"
                    ],
                    "current_crop": current_video["crop"],
                    "next_crop": next_video["crop"],
                    "current_time_start": current_video["time_start"],
                    "next_time_start": next_video["time_start"],
                    "time_gap_minutes": (
                        next_video["time_start"] - current_video["time_start"]
                    ).total_seconds()
                    / 60,
                    "current_class": current_video["label"],
                    "next_class": next_video["label"],
                }
            )

    return pd.DataFrame(transitions)


def calculate_view_probabilities(transitions_df):
    rows = []
    next_class_columns = [f"next_class_{class_num}" for class_num in range(N_CLASSES)]

    grouped = transitions_df.groupby(
        ["temporal_sequence_label", "view", "current_class"]
    )
    for (temporal_label, view, current_class), class_df in grouped:
        next_class_probabilities = (
            class_df["next_class"].value_counts(normalize=True).to_dict()
        )
        row = {
            "temporal_sequence_label": temporal_label,
            "view": view,
            "current_class": current_class,
            "n_transitions": len(class_df),
        }
        row.update(
            {
                column: next_class_probabilities.get(class_num, 0)
                for class_num, column in enumerate(next_class_columns)
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_mean_probabilities(view_probabilities_df):
    next_class_columns = [f"next_class_{class_num}" for class_num in range(N_CLASSES)]
    mean_probabilities_df = (
        view_probabilities_df.groupby(["temporal_sequence_label", "current_class"])[
            next_class_columns
        ]
        .mean()
        .reset_index()
    )
    return mean_probabilities_df


def plot_mean_transition_matrices(mean_probabilities_df):
    next_class_columns = [f"next_class_{class_num}" for class_num in range(N_CLASSES)]
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    probability_norm = PowerNorm(gamma=PROBABILITY_COLOR_GAMMA, vmin=0, vmax=1)

    temporal_order = [
        label
        for label in TEMPORAL_LABEL_MAPPING
        if label in mean_probabilities_df["temporal_sequence_label"].values
    ]
    remaining_temporal_labels = [
        label
        for label in sorted(mean_probabilities_df["temporal_sequence_label"].unique())
        if label not in temporal_order
    ]
    temporal_order.extend(remaining_temporal_labels)

    for temporal_label in temporal_order:
        temporal_df = mean_probabilities_df[
            mean_probabilities_df["temporal_sequence_label"] == temporal_label
        ]
        matrix = (
            temporal_df.set_index("current_class")
            .reindex(range(N_CLASSES), fill_value=0)
            .sort_index()[next_class_columns]
        )
        matrix.columns = list(range(N_CLASSES))
        annotations = matrix.applymap(
            lambda value: f"{value:.2f}" if value > 0 else ""
        )

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            matrix.mask(matrix == 0),
            annot=annotations,
            fmt="",
            cmap=cmap,
            norm=probability_norm,
            cbar_kws={"label": "Mean transition probability"},
            ax=ax,
        )
        ax.set_title(
            "Mean Transition Probability Matrix\n"
            f"{TEMPORAL_LABEL_MAPPING.get(temporal_label, temporal_label)}"
        )
        ax.set_xlabel("Next Class")
        ax.set_ylabel("Current Class")
        fig.tight_layout()

        output_name = TEMPORAL_LABEL_FILENAMES.get(
            temporal_label, f"temporal_label_{temporal_label}"
        )
        output_path = OUTPUT_DIR / f"transition_matrix_{output_name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def main():
    video_stats_df = pd.read_csv(CSV_FILES["testing"], low_memory=False)
    validate_input_columns(video_stats_df, CSV_FILES["testing"])
    print("Temporal test video summary CSV loaded successfully.")
    print(f"Total number of samples in the dataset: {len(video_stats_df)}")
    print(
        "Number of samples with temporal label -100: "
        f"{(video_stats_df['temporal_sequence_label'] == -100).sum()}"
    )

    video_stats_df = clean_video_stats(video_stats_df)
    print(f"Samples used after filtering: {len(video_stats_df)}")

    transitions_df = build_transitions(video_stats_df)
    print(f"Transitions found: {len(transitions_df)}")
    if transitions_df.empty:
        raise ValueError("No transitions found after filtering the input dataset.")

    view_probabilities_df = calculate_view_probabilities(transitions_df)
    mean_probabilities_df = calculate_mean_probabilities(view_probabilities_df)

    transitions_df.to_csv(OUTPUT_DIR / "transitions_by_temporal_interval.csv", index=False)
    view_probabilities_df.to_csv(
        OUTPUT_DIR / "transition_probabilities_by_view_temporal_interval.csv",
        index=False,
    )
    mean_probabilities_df.to_csv(
        OUTPUT_DIR / "transition_probabilities_mean_over_views_temporal_interval.csv",
        index=False,
    )

    plot_mean_transition_matrices(mean_probabilities_df)


if __name__ == "__main__":
    main()
