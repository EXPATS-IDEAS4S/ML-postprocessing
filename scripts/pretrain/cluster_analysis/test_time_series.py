"""
code to test the plot of the time series of the variables of some convective classes. 
the goal is to show the graduality of changes (growth, stationarity and decay) between convective classes
 and the temporal variability over the duration of the video of the convective class properties. 
The code will generate plots for visual inspection.
"""




import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.class_colors import colors_per_class1_names
from utils.plotting.plot_class_analysis import style_axis

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILES = {
    "training_csv_cth": config["output_files"]["training_csv_cth"],
    "training_csv_cot": config["output_files"]["training_csv_cot"],
    "training_csv_euclid_msg_grid": config["output_files"]["training_csv_euclid_msg_grid"],
    "training_csv_cth10plus": config["output_files"]["training_csv_cth10plus"],
    "training_csv_cma": config["output_files"]["training_csv_cma"],
    "training_csv_precipitation": config["output_files"]["training_csv_precipitation"],
    "training_distances": config["output_files"]["training_features_distances_centroid"],

}
N_FRAMES = 8
PERCENTILE_COLUMNS = ["25", "50", "75", "95"]
SLOPE_DECIMALS = 2
CLASS_TO_PLOT = {
    "conv_growing": [5, 6, 7],
    "conv_stationary": [0, 8],
    "conv_decaying": [3, 9],
}
PLOT_SPECS = {
    "cth": {
        "csv_key": "training_csv_cth",
        "mode": "percentiles",
        "ylabel_abs": "Cloud-top height (m)",
        "ylabel_delta": r"$\Delta$CTH (m)",
        "slope_unit": "m/frame",
        "output_name": "figuretest_convection_characterization_cth.png",
    },
    "cot": {
        "csv_key": "training_csv_cot",
        "mode": "percentiles",
        "ylabel_abs": "Cloud optical thickness",
        "ylabel_delta": r"$\Delta$COT",
        "slope_unit": "COT/frame",
        "output_name": "figuretest_convection_characterization_cot.png",
    },
    "euclid_msg_grid": {
        "csv_key": "training_csv_euclid_msg_grid",
        "mode": "single_column",
        "value_column": "None",
        "scale": 1,
        "ylabel_abs": "Lightning counts",
        "ylabel_delta": r"$\Delta$Lightning counts",
        "slope_unit": "counts/frame",
        "output_name": "figuretest_convection_characterization_euclid_msg_grid.png",
    },
    "cth10plus": {
        "csv_key": "training_csv_cth10plus",
        "mode": "single_column",
        "value_column": "fraction",
        "scale": 100,
        "ylabel_abs": "CTH10+ area fraction (%)",
        "ylabel_delta": r"$\Delta$CTH10+ area fraction (%)",
        "slope_unit": "%/frame",
        "output_name": "figuretest_convection_characterization_cth10plus.png",
    },
    "cma": {
        "csv_key": "training_csv_cma",
        "mode": "single_column",
        "value_column": "None",
        "scale": 100,
        "ylabel_abs": "Cloud mask area fraction (%)",
        "ylabel_delta": r"$\Delta$Cloud mask area fraction (%)",
        "slope_unit": "%/frame",
        "output_name": "figuretest_convection_characterization_cma.png",
    },
    "precipitation": {
        "csv_key": "training_csv_precipitation",
        "mode": "percentiles",
        "ylabel_abs": "Precipitation (mm)",
        "ylabel_delta": r"$\Delta$Precipitation (mm)",
        "slope_unit": "mm/frame",
        "output_name": "figuretest_convection_characterization_precipitation.png",
    },
}


def main():
    for variable_name, plot_spec in PLOT_SPECS.items():

        # load the variable dataframe from the corresponding CSV file
        variable_df = load_variable_dataframe(CSV_FILES[plot_spec["csv_key"]])

        # read the distance CSV file to get the distance of each video to its class centroid
        video_distances_df = pd.read_csv(CSV_FILES["training_distances"])

        # read column path from video_distances_df and extract the crop name from the path
        video_distances_df["crop"] = video_distances_df["path"].apply(lambda x: os.path.basename(x).split(".")[0])

        # add distance variable from video_distances_df to all the videos in the dataframe
        video_stats_df = variable_df.merge(video_distances_df[["crop", "distance"]], on="crop", how="left")

        # drop class with label -100
        video_stats_df = clean_video_summary_df(video_stats_df)

        # select for each class the 2000 closest samples to the centroid based on the distance column
        video_stats_plot_df = video_stats_df.groupby("label").apply(lambda x: x.nsmallest(2000, "distance")).reset_index(drop=True)

        # prepare the time series for each class and variable
        class_time_series_by_label = prepare_class_time_series(video_stats_plot_df, plot_spec)

        # plot the time series for each class and variable
        plot_variable_time_series(variable_name, plot_spec, class_time_series_by_label)

def clean_video_summary_df(video_stats_df):
    video_stats_df["label"] = pd.to_numeric(video_stats_df["label"], errors="coerce")
    return video_stats_df[video_stats_df["label"] != -100]


def load_variable_dataframe(csv_path):
    return pd.read_csv(csv_path)


def prepare_class_time_series(variable_df, plot_spec):
    variable_df["time"] = pd.to_datetime(variable_df["time"], errors="coerce")
    required_columns = ["crop", "label", "time"]
    central_value_column = "50" if plot_spec["mode"] == "percentiles" else plot_spec["value_column"]
    scale = plot_spec.get("scale", 1)

    if plot_spec["mode"] == "percentiles":
        for column in PERCENTILE_COLUMNS:
            variable_df[column] = pd.to_numeric(variable_df[column], errors="coerce")
        required_columns.extend(PERCENTILE_COLUMNS)
    else:
        value_column = plot_spec["value_column"]
        variable_df[value_column] = pd.to_numeric(variable_df[value_column], errors="coerce")
        required_columns.append(value_column)

    variable_df = variable_df.dropna(subset=required_columns)
    variable_df = variable_df[variable_df["label"] != -100]

    class_time_series_by_label = {}
    for class_label in sorted(variable_df["label"].dropna().astype(int).unique()):
        class_df = variable_df[variable_df["label"] == class_label].copy()
        class_df = class_df.sort_values(["crop", "time"])
        class_df["frame_index"] = class_df.groupby("crop").cumcount()
        class_df = class_df[class_df["frame_index"] < N_FRAMES]

        complete_crops = class_df.groupby("crop")["frame_index"].nunique()
        complete_crops = complete_crops[complete_crops == N_FRAMES].index
        class_df = class_df[class_df["crop"].isin(complete_crops)]

        if class_df.empty:
            continue

        if plot_spec["mode"] == "percentiles":
            class_time_series = class_df.groupby("frame_index")[PERCENTILE_COLUMNS].mean().reset_index()
            central_values_df = class_df[["crop", "frame_index", central_value_column]].copy()
        else:
            value_column = plot_spec["value_column"]
            class_time_series = class_df.groupby("frame_index")[value_column].mean().reset_index()
            class_time_series[value_column] = class_time_series[value_column] * scale
            central_values_df = class_df[["crop", "frame_index", value_column]].copy()
            central_values_df[value_column] = central_values_df[value_column] * scale

        delta_by_crop = central_values_df.pivot(
            index="crop",
            columns="frame_index",
            values=central_value_column,
        )
        delta_by_crop = delta_by_crop.subtract(delta_by_crop[0], axis=0)
        delta_summary = pd.DataFrame({
            "frame_index": delta_by_crop.columns,
            "delta_mean": delta_by_crop.mean(axis=0).to_numpy(),
            "delta_std": delta_by_crop.std(axis=0).to_numpy(),
            "delta_q25": delta_by_crop.quantile(0.25, axis=0).to_numpy(),
            "delta_q75": delta_by_crop.quantile(0.75, axis=0).to_numpy(),
        })
        delta_summary["delta_std"] = delta_summary["delta_std"].fillna(0)

        class_time_series = class_time_series.merge(delta_summary, on="frame_index", how="left")

        class_time_series_by_label[class_label] = class_time_series

    return class_time_series_by_label


def plot_variable_time_series(variable_name, plot_spec, class_time_series_by_label):
    frames = np.arange(N_FRAMES)
    group_names = list(CLASS_TO_PLOT.keys())
    fig, axes = plt.subplots(
        len(group_names),
        1,
        figsize=(8, 3.5 * len(group_names)),
        sharex=True,
        constrained_layout=True,
    )
    if len(group_names) == 1:
        axes = [axes]

    has_any_group = False

    for ax_delta, group_name in zip(axes, group_names):
        plotted_classes = [
            class_label
            for class_label in CLASS_TO_PLOT[group_name]
            if class_label in class_time_series_by_label
        ]

        if not plotted_classes:
            ax_delta.text(
                0.5,
                0.5,
                f"No classes available for {group_name}",
                ha="center",
                va="center",
                transform=ax_delta.transAxes,
            )
            ax_delta.set_title(group_name.replace("_", " ").title(), loc="left")
            style_axis(ax_delta)
            continue

        has_any_group = True
        class_colors = {cls: colors_per_class1_names[str(cls)] for cls in plotted_classes}

        for cls, color in class_colors.items():
            series_df = class_time_series_by_label[cls]
            #print_delta_percentiles(variable_name, group_name, cls, series_df)
            if plot_spec["mode"] == "percentiles":
                median = series_df["50"].values
            else:
                value_column = plot_spec["value_column"]
                median = series_df[value_column].values

            delta = median - median[0]
            ax_delta.plot(
                frames,
                delta,
                color=color,
                marker="o",
                linewidth=2.2,
                label=f"Class {cls}",
            )

            slope = np.polyfit(frames, median, 1)[0]
            ax_delta.text(
                frames[-1] + 0.08,
                delta[-1],
                f"{slope:+.{SLOPE_DECIMALS}f} {plot_spec['slope_unit']}",
                color=color,
                va="center",
                fontsize=9,
            )

        ax_delta.axhline(0, color="0.35", linewidth=1)
        ax_delta.set_ylabel(plot_spec["ylabel_delta"])
        ax_delta.set_title(group_name.replace("_", " ").title(), loc="left")
        ax_delta.legend(frameon=False, ncol=2)
        ax_delta.grid(axis="y", alpha=0.2)
        style_axis(ax_delta)

    if not has_any_group:
        raise ValueError(f"None of the requested classes are available for {variable_name}: {CLASS_TO_PLOT}")

    axes[-1].set_xlabel("Frame index")

    output_file = OUTPUT_DIR / plot_spec["output_name"]
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_delta_percentiles(variable_name, group_name, class_label, series_df):
    percentile_df = series_df[["frame_index", "delta_q25", "delta_q75"]].copy()
    print(f"\n{variable_name} | {group_name} | class {class_label}")
    print(percentile_df.to_string(index=False))


if __name__ == "__main__":
    main()

