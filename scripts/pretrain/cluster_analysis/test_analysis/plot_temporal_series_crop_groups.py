"""
Plot temporal time series and anomalies for test crop groups.

This script reads the temporal test video summary and the crop-group CSV files
created by test_temporal_analysis.py, then plots class-wise means over temporal
sequence labels for selected variables and gradients.

with the --group argument, you can select which group to plot.
 The available options are:
all_groups
all
north
south
north_day
north_night
south_day
south_night

For groups listed in SELECTED_PREDICTORS_BY_GROUP, only those predictors are
plotted. The script also stores class-wise finite differences between adjacent
temporal sequence bins as warning-threshold CSV files in output_files.csv_dir.
It also saves finite-difference lookup-table heatmaps under:
    output_files.figures_dir/temporal_series_crop_groups/warning_threshold_lookup_tables/

How to run:
    python scripts/pretrain/cluster_analysis/test_analysis/plot_temporal_series_crop_groups.py
    python scripts/pretrain/cluster_analysis/test_analysis/plot_temporal_series_crop_groups.py --group south_day
    python scripts/pretrain/cluster_analysis/test_analysis/plot_temporal_series_crop_groups.py --group south_day --predictors cot_mean_anomaly,cth_gradient_anomaly
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from scripts.pretrain.cluster_analysis.var_class_temporal_series import read_csv_to_dataframe
from utils.plotting.class_colors import colors_per_class1_names
from utils.plotting.plot_class_analysis import style_axis


CONFIG_PATH = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(CONFIG_PATH)
CSV_FILES = {
    "testing_video_summary_temporal": config["output_files"]["testing_video_summary_temporal"],
    "training_video_summary": config["output_files"]["training_video_summary"],
}
OUTPUT_DIR = Path(config["output_files"]["figures_dir"]) / "temporal_series_crop_groups"
CSV_OUTPUT_DIR = Path(config["output_files"]["csv_dir"])

AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 12

TEMPORAL_LABEL_MAPPING = {
    -4: "> 4 h before event",
    -2: "4h < t < 2h before event",
    -1: "< 2h before event",
    0: "between first and last report",
    1: "< 2h after event",
    2: "2h < t < 4h after event",
    4: "> 4 h after event",
}

SELECTED_VARS_TEMPORAL_SERIES = [
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
    "cth10plus_gradient",
]

VIEW_GROUPS = {
    "North": [7, 8, 9, 4, 5, 6],
    "South": [1, 2, 3],
}

SELECTED_CLASSES_BY_GROUP = {
    "south_daytime": [5, 6, 7, 3, 9],
}

SELECTED_PREDICTORS_BY_GROUP = {
    "south_daytime": [
        "cot_mean_anomaly",
        "cot30plus_mean_anomaly",
        "cth_gradient_anomaly",
        "euclid_msg_grid_mean_anomaly",
        "precipitation_mean_anomaly",
    ],
}
def main():
    args = parse_args()
    group_key = normalize_group_name(args.group)

    # Load temporal dataframe and training mean dataframe
    temporal_df = load_temporal_dataframe(CSV_FILES["testing_video_summary_temporal"])
    class_labels = sorted(temporal_df["label"].dropna().astype(int).unique())

    # Load training dataframe and compute mean values for each class
    training_df = read_csv_to_dataframe(CSV_FILES["training_video_summary"])
    training_mean_df = training_df.groupby("label").mean(numeric_only=True).reset_index()

    datasets = {"all": ("", temporal_df)}
    for group_name, views in VIEW_GROUPS.items():
        group_df = temporal_df[temporal_df["view"].isin(views)]
        datasets[group_name.lower()] = (f"group_{group_name}_", group_df)

    for crop_group_name, crop_group_df in load_saved_crop_group_dataframes().items():
        datasets[crop_group_name] = (f"crop_group_{crop_group_name}_", crop_group_df)

    selected_datasets = select_datasets(datasets, args.group)
    for output_prefix, dataset_df in selected_datasets:
        dataset_group_key = get_dataset_group_key(output_prefix, group_key)
        selected_class_labels = get_class_labels_for_group(
            dataset_df,
            class_labels,
            dataset_group_key,
            output_prefix,
        )
        selected_predictors = get_predictors_for_group(
            dataset_group_key,
            args.predictors,
        )
        plot_temporal_series_and_anomalies(
            dataset_df,
            selected_class_labels,
            training_mean_df,
            output_prefix,
            dataset_group_key,
            selected_predictors,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot temporal series and anomalies for selected test crop groups."
    )
    parser.add_argument(
        "--group",
        default="all_groups",
        help=(
            "Group to plot. Use all_groups for every group, all for all views, "
            "north, south, north_day, north_night, south_day, or south_night."
        ),
    )
    parser.add_argument(
        "--predictors",
        default=None,
        help=(
            "Optional comma-separated predictor override, for example "
            "cot_mean_anomaly,cth_gradient_anomaly."
        ),
    )
    return parser.parse_args()


def get_class_labels_for_group(dataset_df, class_labels, group_key, output_prefix):
    selected_classes = SELECTED_CLASSES_BY_GROUP.get(group_key)
    if selected_classes is None:
        return class_labels

    available_classes = set(dataset_df["label"].dropna().astype(int).unique())
    selected_class_labels = [
        class_label
        for class_label in selected_classes
        if class_label in available_classes
    ]
    missing_classes = [
        class_label
        for class_label in selected_classes
        if class_label not in available_classes
    ]

    if missing_classes:
        print(
            f"Classes {missing_classes} are not available for "
            f"{output_prefix or group_key}; skipping them."
        )
    if not selected_class_labels:
        print(
            f"No selected classes found for {output_prefix or group_key}; "
            "falling back to all classes available in this group."
        )
        return sorted(available_classes)

    return selected_class_labels


def get_dataset_group_key(output_prefix, requested_group_key):
    if requested_group_key != "all_groups":
        return requested_group_key

    prefix = output_prefix.rstrip("_")
    if prefix.startswith("crop_group_"):
        return prefix.replace("crop_group_", "", 1).lower()
    if prefix.startswith("group_"):
        return prefix.replace("group_", "", 1).lower()
    if prefix:
        return prefix.lower()
    return "all"


def get_predictors_for_group(group_key, predictor_override):
    if predictor_override:
        return [
            predictor.strip()
            for predictor in predictor_override.split(",")
            if predictor.strip()
        ]

    default_predictors = SELECTED_VARS_TEMPORAL_SERIES + [
        f"{predictor}_anomaly" for predictor in SELECTED_VARS_TEMPORAL_SERIES
    ]
    return SELECTED_PREDICTORS_BY_GROUP.get(group_key, default_predictors)


def select_datasets(datasets, requested_group):
    if requested_group == "all_groups":
        return list(datasets.values())

    group_key = normalize_group_name(requested_group)
    if group_key not in datasets:
        available_groups = ", ".join(["all_groups", *sorted(datasets)])
        raise ValueError(
            f"Unknown group '{requested_group}'. Available groups: {available_groups}"
        )

    return [datasets[group_key]]


def normalize_group_name(group_name):
    group_name = group_name.lower().strip()
    aliases = {
        "north_day": "north_daytime",
        "north_night": "north_nighttime",
        "south_day": "south_daytime",
        "south_night": "south_nighttime",
    }
    return aliases.get(group_name, group_name)


def load_temporal_dataframe(csv_file):
    temporal_df = read_csv_to_dataframe(csv_file)
    return clean_temporal_dataframe(temporal_df)


def clean_temporal_dataframe(temporal_df):
    """
    Clean the temporal dataframe by removing rows with invalid labels and converting
    columns to numeric types.
    """
    
    temporal_df = temporal_df.copy()
    temporal_df["label"] = pd.to_numeric(temporal_df["label"], errors="coerce")
    temporal_df["temporal_sequence_label"] = pd.to_numeric(
        temporal_df["temporal_sequence_label"],
        errors="coerce",
    )
    temporal_df = temporal_df.dropna(subset=["label", "temporal_sequence_label"])
    temporal_df = temporal_df[temporal_df["label"] != -100]
    temporal_df = temporal_df[temporal_df["temporal_sequence_label"] != -100]
    temporal_df["label"] = temporal_df["label"].astype(int)
    temporal_df["temporal_sequence_label"] = temporal_df[
        "temporal_sequence_label"
    ].astype(int)

    if "view" in temporal_df.columns:
        temporal_df["view"] = pd.to_numeric(temporal_df["view"], errors="coerce")
        temporal_df = temporal_df.dropna(subset=["view"])
        temporal_df["view"] = temporal_df["view"].astype(int)

    return temporal_df


def load_saved_crop_group_dataframes():
    crop_group_dfs = {}
    for csv_file in sorted(
        CSV_OUTPUT_DIR.glob("crops_video_summary_temporal_sequence_*.csv")
    ):
        group_name = csv_file.stem.replace("crops_video_summary_temporal_sequence_", "")
        crop_group_dfs[group_name] = clean_temporal_dataframe(pd.read_csv(csv_file))
    return crop_group_dfs


def plot_temporal_series_and_anomalies(
    temporal_df,
    class_labels,
    training_mean_df,
    output_prefix,
    group_key,
    selected_predictors,
):
    if temporal_df.empty:
        print(f"No temporal data available for {output_prefix}; skipping.")
        return

    mean_df, std_df = calculate_temporal_mean_std(temporal_df)
    regular_predictors = [
        predictor
        for predictor in selected_predictors
        if not predictor.endswith("_anomaly")
    ]
    anomaly_base_predictors = [
        remove_suffix(predictor, "_anomaly")
        for predictor in selected_predictors
        if predictor.endswith("_anomaly")
    ]

    plot_temporal_series_for_variables(
        mean_df,
        std_df,
        regular_predictors,
        class_labels,
        colors_per_class1_names,
        TEMPORAL_LABEL_MAPPING,
        OUTPUT_DIR,
        output_prefix=output_prefix,
    )

    anomalies_mean_df, anomalies_std_df = calculate_anomalies(
        mean_df,
        std_df,
        [
            var
            for var in anomaly_base_predictors
            if var in mean_df.columns and var in training_mean_df.columns
        ],
        training_mean_df,
    )
    anomaly_vars = [
        predictor
        for predictor in selected_predictors
        if predictor.endswith("_anomaly") and predictor in anomalies_mean_df.columns
    ]
    plot_temporal_series_for_variables(
        anomalies_mean_df,
        anomalies_std_df,
        anomaly_vars,
        class_labels,
        colors_per_class1_names,
        TEMPORAL_LABEL_MAPPING,
        OUTPUT_DIR,
        output_prefix=output_prefix,
    )
    save_warning_thresholds(
        mean_df,
        anomalies_mean_df,
        regular_predictors,
        anomaly_vars,
        class_labels,
        group_key,
        output_prefix,
    )


def calculate_temporal_mean_std(temporal_df):
    grouped = temporal_df.groupby(["temporal_sequence_label", "label"])
    mean_df = grouped.mean(numeric_only=True).reset_index()
    std_df = grouped.std(numeric_only=True).reset_index()
    return mean_df, std_df


def remove_suffix(value, suffix):
    if value.endswith(suffix):
        return value[: -len(suffix)]
    return value


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


def save_warning_thresholds(
    mean_df,
    anomalies_mean_df,
    regular_predictors,
    anomaly_predictors,
    class_labels,
    group_key,
    output_prefix,
):
    threshold_rows = []
    for predictor in regular_predictors:
        threshold_rows.extend(
            calculate_temporal_difference_threshold_rows(
                mean_df,
                predictor,
                class_labels,
                group_key,
                "temporal_mean",
            )
        )

    for predictor in anomaly_predictors:
        threshold_rows.extend(
            calculate_temporal_difference_threshold_rows(
                anomalies_mean_df,
                predictor,
                class_labels,
                group_key,
                "temporal_anomaly",
            )
        )

    if not threshold_rows:
        print(f"No warning thresholds available for {output_prefix}; skipping CSV.")
        return

    thresholds_df = pd.DataFrame(threshold_rows)
    output_file = (
        CSV_OUTPUT_DIR
        / f"warning_thresholds_temporal_differences_{sanitize_output_prefix(output_prefix, group_key)}.csv"
    )
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds_df.to_csv(output_file, index=False)
    print(f"Warning thresholds saved to {output_file}")
    plot_warning_threshold_lookup_tables(
        thresholds_df,
        OUTPUT_DIR,
        sanitize_output_prefix(output_prefix, group_key),
    )


def calculate_temporal_difference_threshold_rows(
    values_df,
    predictor,
    class_labels,
    group_key,
    predictor_kind,
):
    if predictor not in values_df.columns:
        print(f"{predictor} not available for threshold calculation; skipping.")
        return []

    threshold_rows = []
    for class_label in class_labels:
        class_df = (
            values_df[values_df["label"] == class_label]
            .sort_values("temporal_sequence_label")
            .reset_index(drop=True)
        )
        if len(class_df) < 2:
            continue

        for index in range(len(class_df) - 1):
            start_row = class_df.iloc[index]
            end_row = class_df.iloc[index + 1]
            start_value = start_row[predictor]
            end_value = end_row[predictor]
            if pd.isna(start_value) or pd.isna(end_value):
                continue

            start_temporal_label = int(start_row["temporal_sequence_label"])
            end_temporal_label = int(end_row["temporal_sequence_label"])
            temporal_step = end_temporal_label - start_temporal_label
            finite_difference = end_value - start_value
            gradient_per_temporal_label_step = (
                finite_difference / temporal_step if temporal_step != 0 else pd.NA
            )

            threshold_rows.append(
                {
                    "group": group_key,
                    "class_label": int(class_label),
                    "predictor": predictor,
                    "predictor_kind": predictor_kind,
                    "from_temporal_sequence_label": start_temporal_label,
                    "to_temporal_sequence_label": end_temporal_label,
                    "from_temporal_sequence_name": TEMPORAL_LABEL_MAPPING.get(
                        start_temporal_label,
                        str(start_temporal_label),
                    ),
                    "to_temporal_sequence_name": TEMPORAL_LABEL_MAPPING.get(
                        end_temporal_label,
                        str(end_temporal_label),
                    ),
                    "value_start": start_value,
                    "value_end": end_value,
                    "warning_threshold_finite_difference": finite_difference,
                    "temporal_label_step": temporal_step,
                    "warning_threshold_gradient_per_temporal_label_step": (
                        gradient_per_temporal_label_step
                    ),
                    "warning_threshold_abs_finite_difference": abs(
                        finite_difference
                    ),
                    "direction": get_difference_direction(finite_difference),
                }
            )

    return threshold_rows


def plot_warning_threshold_lookup_tables(thresholds_df, output_dir, output_name):
    output_dir = Path(output_dir) / "warning_threshold_lookup_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds_df = thresholds_df.copy()
    thresholds_df["temporal_interval"] = thresholds_df.apply(
        lambda row: (
            f"{row['from_temporal_sequence_label']} -> "
            f"{row['to_temporal_sequence_label']}"
        ),
        axis=1,
    )

    for predictor, predictor_df in thresholds_df.groupby("predictor"):
        lookup_table = predictor_df.pivot_table(
            index="class_label",
            columns="temporal_interval",
            values="warning_threshold_finite_difference",
            aggfunc="mean",
        ).sort_index()

        if lookup_table.empty:
            continue

        interval_order = get_temporal_interval_order(predictor_df)
        interval_order = [
            interval for interval in interval_order if interval in lookup_table.columns
        ]
        lookup_table = lookup_table.loc[:, interval_order]

        plot_warning_threshold_lookup_table(
            lookup_table,
            predictor,
            output_dir / (
                f"finite_difference_lookup_{output_name}_"
                f"{sanitize_filename(predictor)}.png"
            ),
        )


def get_temporal_interval_order(thresholds_df):
    interval_order_df = (
        thresholds_df[
            [
                "from_temporal_sequence_label",
                "to_temporal_sequence_label",
                "temporal_interval",
            ]
        ]
        .drop_duplicates()
        .sort_values(["from_temporal_sequence_label", "to_temporal_sequence_label"])
    )
    return interval_order_df["temporal_interval"].tolist()


def plot_warning_threshold_lookup_table(lookup_table, predictor, output_file):
    values = lookup_table.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        print(f"No finite threshold values available for {predictor}; skipping plot.")
        return

    max_abs_value = np.nanmax(np.abs(finite_values))
    if max_abs_value == 0:
        max_abs_value = 1.0

    fig_width = max(8, 1.35 * len(lookup_table.columns))
    fig_height = max(3.5, 0.6 * len(lookup_table.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-max_abs_value, vcenter=0, vmax=max_abs_value),
    )

    ax.set_title(f"Finite-difference lookup table: {predictor}", fontsize=14)
    ax.set_xlabel("Temporal interval", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Class", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xticks(np.arange(len(lookup_table.columns)))
    ax.set_xticklabels(
        lookup_table.columns,
        rotation=35,
        ha="right",
        fontsize=TICK_LABEL_FONTSIZE,
    )
    ax.set_yticks(np.arange(len(lookup_table.index)))
    ax.set_yticklabels(lookup_table.index, fontsize=TICK_LABEL_FONTSIZE)

    annotate_lookup_table(ax, values)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Finite difference", fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Finite-difference lookup table saved to {output_file}")


def annotate_lookup_table(ax, values):
    max_abs_value = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    contrast_limit = max_abs_value * 0.55
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if not np.isfinite(value):
                continue
            text_color = "white" if abs(value) >= contrast_limit else "black"
            ax.text(
                column_index,
                row_index,
                format_lookup_value(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )


def format_lookup_value(value):
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def sanitize_filename(value):
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(">", "gt")
        .replace("<", "lt")
    )


def get_difference_direction(finite_difference):
    if finite_difference > 0:
        return "increase"
    if finite_difference < 0:
        return "decrease"
    return "stable"


def sanitize_output_prefix(output_prefix, group_key):
    sanitized = output_prefix.strip("_").lower()
    if sanitized.startswith("crop_group_"):
        sanitized = sanitized.replace("crop_group_", "", 1)
    return sanitized or group_key


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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for var in selected_vars_temporal_series:
        if var not in mean_df.columns:
            print(f"{var} not available for {output_prefix}; skipping.")
            continue

        plt.figure(figsize=(10, 6))
        plotted_any_class = False
        for class_label in class_labels:
            class_mean = mean_df[mean_df["label"] == class_label]
            if class_mean.empty or var not in class_mean.columns:
                continue

            plt.plot(
                class_mean["temporal_sequence_label"],
                class_mean[var],
                label=f"Class {class_label}",
                color=colors_per_class1_names.get(str(class_label), "lightgray"),
                marker="o",
            )
            plotted_any_class = True

        if not plotted_any_class:
            print(
                f"No selected class data available for {var} in "
                f"{output_prefix}; skipping."
            )
            plt.close()
            continue

        plt.xlabel("Temporal Sequence Label", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel(var, fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(
            ticks=list(temporal_label_mapping.keys()),
            labels=[
                temporal_label_mapping[label]
                for label in temporal_label_mapping.keys()
            ],
            rotation=35,
            ha="right",
            fontsize=TICK_LABEL_FONTSIZE,
        )
        plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
        style_axis(plt.gca())
        plt.legend(
            title="Class",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            title_fontsize=LEGEND_FONTSIZE,
        )
        plt.tight_layout()
        output_file = output_dir / f"{output_prefix}{var}_temporal_sequence.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Temporal series plot for {var} saved to {output_file}")
        plt.close()


if __name__ == "__main__":
    main()
