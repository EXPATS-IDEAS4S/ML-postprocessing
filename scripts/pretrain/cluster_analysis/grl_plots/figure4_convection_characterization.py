"""
This figure has the goal to demonstrate that we can identify diverse stages of convection 
over the Alps. It works on a selection of videos that are closests to the centroids (distances from centroid smaller than 0.1).
 It will show the graduality of changes (growth, stationarity and decay) between convective 
classes and the temporal variability over the duration of the video of the convective class properties. 
The plot has 3 columns and various rows. Columns are associated with convection growing, convection decaying day and convection decaying night. 
Rows instead contain:
- row 1: scatter plot of CTH mean vs Cloud Cover 
- row 2: scatter plot of Precipitation mean vs EUCLID MSG grid mean
- row 3: temporal evolution of CTH
- row 4: temporal evolution of lightning flash rate
- row 5: temporal evolution of precipitation mean

class_colors = {
    # Decaying daytime
    0: "#2CA25F",
    8: "#006D2C",

    # Decaying nighttime
    3: "#807DBA",
    9: "#54278F",

    # Growing convection
    5: "#FDBE85",
    6: "#F16913",
    7: "#B30000",
}

"""



import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("/home/claudia/codes/ML_postprocessing")

from utils.configs import load_config
from utils.plotting.class_colors import colors_per_class_codes_grl
from utils.plotting.plot_class_analysis import style_axis
# load video summary csv file path from config
REPO_ROOT = Path("/home/claudia/codes/ML_postprocessing")
CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"
config = load_config(str(CONFIG_PATH))

# create output directory if it doesn't exist
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MARKER_SIZE = 300
MARKER_EDGE_COLOR = "grey"
MARKER_EDGE_WIDTH = 1.4
PERCENTILE_ERRORBAR_COLOR = (0.78, 0.78, 0.78, 0.45)
CMA_GRADIENT_LIMIT = 0.2
CTH_MEAN_LIMIT = 8000
CTH_GRADIENT_LIMIT = 800
ZERO_AXIS_LINEWIDTH = 2.2
AXIS_LABEL_FONTSIZE = 23
TITLE_FONTSIZE = 23
TICK_LABEL_FONTSIZE = 23
FONT_SIZE_TEXT = 23 
PANEL_LABEL_FONTSIZE = 23
SECTION_TITLE_OFFSET = 0.045
B_TITLE_GAP_FRACTION = 0.38

LEGEND_FONTSIZE = 18
WIND_DIRECTION_X_LIMITS = (100, 310)
TIME_SERIES_LINEWIDTH = 4
TIME_SERIES_MARKER_SIZE = 10
MAIN_ROW_HSPACE = 0.62
LOWER_TIME_SERIES_ROW_SHIFT = 0.04



# read columns for first scatter plot
scatter_columns = [
    "cth_mean",
    "cth_std", 
    "cma_mean",
    "cma_std",
    "cot_mean",
    "cot_std",
    "precipitation_mean",
    "precipitation_std",
    "euclid_msg_grid_mean",
    "euclid_msg_grid_std",
    "cth10plus_mean",
    "cth10plus_std",
    "cot30plus_mean",
    "cot30plus_std" 
]

# read filename of csv files from config file process_run_GRL.yaml
config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
config = load_config(config_path)
OUTPUT_DIR = Path(config["output_files"]["figures_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILES = {
    "training_video_summary": config["output_files"]["training_video_summary"],
    "training_csv_cth": config["output_files"]["training_csv_cth"],
    "training_csv_cot": config["output_files"]["training_csv_cot"],
    "training_csv_euclid_msg_grid": config["output_files"]["training_csv_euclid_msg_grid"],
    "training_csv_cth10plus": config["output_files"]["training_csv_cth10plus"],
    "training_csv_cma": config["output_files"]["training_csv_cma"],
    "training_csv_precipitation": config["output_files"]["training_csv_precipitation"],
    "training_distances": config["output_files"]["training_features_distances_centroid"],

}
N_CLASSES = 10
N_FRAMES = 8
PERCENTILE_COLUMNS = ["25", "50", "75", "95"]
SLOPE_DECIMALS = 2
CLASS_TO_PLOT = {
    "convection growing": [5, 6, 7],
    "convection decaying day": [3, 9],
    "convection decaying night": [0, 8],
}
PLOT_SPECS = {
    "cth": {
        "csv_key": "training_csv_cth",
        "mode": "percentiles",
        "ylabel_abs": "Cloud-top height (m)",
        "ylabel_delta": "$\\Delta$Cloud-top\nheight (m)",
        "slope_unit": "m/frame",
    },
    "cot": {
        "csv_key": "training_csv_cot",
        "mode": "percentiles",
        "ylabel_abs": "Cloud optical thickness",
        "ylabel_delta": r"$\Delta$COT",
        "slope_unit": "COT/frame",
    },
    "euclid_msg_grid": {
        "csv_key": "training_csv_euclid_msg_grid",
        "mode": "single_column",
        "value_column": "None",
        "scale": 1,
        "ylabel_abs": "Lightning counts",
        "ylabel_delta": r"$\Delta$Lightning counts",
        "slope_unit": "counts/frame",
    },
    "cth10plus": {
        "csv_key": "training_csv_cth10plus",
        "mode": "single_column",
        "value_column": "fraction",
        "scale": 100,
        "ylabel_abs": "CTH10+ area fraction (%)",
        "ylabel_delta": r"$\Delta$CTH10+ area fraction (%)",
        "slope_unit": "%/frame",
    },
    "cma": {
        "csv_key": "training_csv_cma",
        "mode": "single_column",
        "value_column": "None",
        "scale": 100,
        "ylabel_abs": "Cloud cover (%)",
        "ylabel_delta": r"$\Delta$Cloud cover (%)",
        "slope_unit": "%/frame",
    },
    "precipitation": {
        "csv_key": "training_csv_precipitation",
        "mode": "percentiles",
        "ylabel_abs": "Cumulative Precipitation (mm)",
    "ylabel_delta": "$\\Delta$Cumulative\nPrecipitation (mm)",
        "slope_unit": "mm/frame",
    },
}

# select the variables to plot in the figure 4
PLOTS_SPEC_SELECTED = {
    "cth": PLOT_SPECS["cth"],
    "cma": PLOT_SPECS["cma"],
    "precipitation": PLOT_SPECS["precipitation"],
}


def clean_variable_df(df):
    """
    Clean the variable dataframe by removing rows with label -100 and converting the label column to numeric.
    """
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df[df["label"] != -100]

def load_variable_dataframe(csv_path):
    return pd.read_csv(csv_path)


def prepare_class_time_series(variable_df, plot_spec):
    variable_df = variable_df.copy()
    variable_df["time"] = pd.to_datetime(variable_df["time"], errors="coerce")
    required_columns = ["crop", "label", "time"]

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
        else:
            value_column = plot_spec["value_column"]
            scale = plot_spec.get("scale", 1)
            class_time_series = class_df.groupby("frame_index")[value_column].mean().reset_index()
            class_time_series[value_column] = class_time_series[value_column] * scale

        class_time_series_by_label[class_label] = class_time_series

    return class_time_series_by_label


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
    )


def plot_variable_time_series_on_axes(
    axes,
    variable_name,
    plot_spec,
    class_time_series_by_label,
    show_xlabel=False,
    row_index=0,
):
    frames = np.arange(N_FRAMES)

    for column_index, (ax, (group_name, class_labels)) in enumerate(zip(axes, CLASS_TO_PLOT.items())):
        add_panel_label(ax, f"b{row_index * len(CLASS_TO_PLOT) + column_index + 1}")
        plotted_classes = [
            class_label for class_label in class_labels if class_label in class_time_series_by_label
        ]

        if not plotted_classes:
            ax.text(
                0.5,
                0.5,
                f"No classes available for {group_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(group_name.title(), fontsize=TITLE_FONTSIZE)
            style_axis(ax)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            continue

        slope_lines = []

        for class_label in plotted_classes:
            series_df = class_time_series_by_label[class_label]
            if plot_spec["mode"] == "percentiles":
                median = series_df["50"].to_numpy()
            else:
                value_column = plot_spec["value_column"]
                median = series_df[value_column].to_numpy()

            delta = median - median[0]
            ax.plot(
                frames,
                delta,
                color=colors_per_class_codes_grl[str(class_label)],
                marker="o",
                linewidth=TIME_SERIES_LINEWIDTH,
                markersize=TIME_SERIES_MARKER_SIZE,
                label=f"Class {class_label}",
            )

            slope = np.polyfit(frames, median, 1)[0]
            slope_lines.append(
                f"Class {class_label}: {slope:+.{SLOPE_DECIMALS}f} {plot_spec['slope_unit']}"
            )

        ax.axhline(0, color="0.35", linewidth=1)
        if column_index == 0:
            ax.set_ylabel(plot_spec["ylabel_delta"], fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_ylabel("")
        if show_xlabel:
            ax.set_xlabel("Frame index", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.set_xlabel("")
        ax.text(
            0.03,
            0.06,
            "\n".join(slope_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=21,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 3.0},
        )
        ax.grid(axis="y", alpha=0.2)
        style_axis(ax)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)


def add_class_legend(fig):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors_per_class_codes_grl[str(label)],
            markeredgecolor=colors_per_class_codes_grl[str(label)],
            markersize=16,
            label=f"Class {label}",
        )
        for label in range(N_CLASSES)
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.865, 0.5),
        ncol=1,
        frameon=False,
        fontsize=23,
        handletextpad=0.4,
        columnspacing=0.8,
    )


def tighten_lower_time_series_spacing(axes, shift=LOWER_TIME_SERIES_ROW_SHIFT):
    for row_index in range(2, axes.shape[0]):
        row_shift = shift * (row_index - 1)
        for ax in axes[row_index, :]:
            position = ax.get_position()
            ax.set_position([
                position.x0,
                position.y0 + row_shift,
                position.width,
                position.height,
            ])
    
def main():

    # ------------- selecting the 2000 closest samples to the centroid for each class -------------
    # load the crop statistics CSV file
    print("Loading video summary and distances CSV files...")
    video_stats_df = pd.read_csv(CSV_FILES["training_video_summary"])
    video_distances_df = pd.read_csv(CSV_FILES["training_distances"])

    # read column path from video_distances_df and extract the crop name from the path
    video_distances_df["crop"] = video_distances_df["path"].apply(lambda x: os.path.basename(x).split(".")[0])

    print("Merging video summary and distances dataframes...")
    # add distance variable from video_distances_df to all the videos in the dataframe
    video_stats_df = video_stats_df.merge(video_distances_df[["crop", "distance"]], on="crop", how="left")

    print("Cleaning video summary dataframe...")
    # drop class with label -100
    video_stats_df = clean_variable_df(video_stats_df)

    print("Selecting the 2000 closest samples to the centroid for each class...")
    # select for each class the 1000 closest samples to the centroid based on the distance column
    video_stats_plot_df = video_stats_df.groupby("label").apply(lambda x: x.nsmallest(2000, "distance")).reset_index(drop=True)

    # ------------ select the 2000 closest samples to the centroid for each variable file of the frames ------------
    # create a dictionary to store the variable dataframes
    variable_dfs = {}
    for variable_name, plot_spec in PLOT_SPECS.items():

        # load the variable dataframe from the corresponding CSV file
        variable_df = load_variable_dataframe(CSV_FILES[plot_spec["csv_key"]])

        # add distance variable from video_distances_df to all the videos in the dataframe
        variable_df = variable_df.merge(video_distances_df[["crop", "distance"]], on="crop", how="left")

        # drop class with label -100
        variable_df = clean_variable_df(variable_df)    

        # select for each class the 2000 closest samples to the centroid based on the distance column
        variable_df = variable_df.groupby("label").apply(lambda x: x.nsmallest(2000, "distance")).reset_index(drop=True)
        variable_dfs[variable_name] = variable_df
    
    print("Plotting figure 4...")
    #---------------start plotting the figure 4----------------
    n_rows = 1 + len(PLOTS_SPEC_SELECTED)

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(25, 5.4 * n_rows),
        sharey="row",
        gridspec_kw={"height_ratios": [2.6] + [1.0] * (n_rows - 1)},
    )
    fig.subplots_adjust(left=0.08, right=0.84, top=0.92, bottom=0.08, hspace=MAIN_ROW_HSPACE, wspace=0.12)

    # create functions to plot the entire row of scatter plots and time series 
    plot_scatter_row(
        fig,
        axes[0, :],
        video_stats_plot_df,
        class_2_plot=CLASS_TO_PLOT,
    )
    # loop on the variables to be plotter to call the function to plot the time series for each variable
    for i, (variable_name, plot_spec) in enumerate(PLOTS_SPEC_SELECTED.items()):

        # plot cth and cth10plus time series in the same plot with different y-axis
        if variable_name == "cth10plus":
            plot_time_series_row(axes[i+1, :], 
                                variable_dfs["cth"], 
                                class_2_plot=CLASS_TO_PLOT, 
                                variable_name="cth", 
                                plot_spec=PLOTS_SPEC_SELECTED["cth"],
                                show_xlabel=i == len(PLOTS_SPEC_SELECTED) - 1,
                                row_index=i)
            plot_time_series_row(axes[i+1, :], 
                                variable_dfs[variable_name], 
                                class_2_plot=CLASS_TO_PLOT, 
                                variable_name=variable_name, 
                                plot_spec=plot_spec,
                                show_xlabel=i == len(PLOTS_SPEC_SELECTED) - 1,
                                row_index=i)
        else:
            plot_time_series_row(axes[i+1, :], 
                                variable_dfs[variable_name], 
                                class_2_plot=CLASS_TO_PLOT, 
                                variable_name=variable_name, 
                                plot_spec=plot_spec,
                                show_xlabel=i == len(PLOTS_SPEC_SELECTED) - 1,
                                row_index=i)

    tighten_lower_time_series_spacing(axes)
        
    # set the title for each row of the figure
    if n_rows > 1:
        upper_bbox = axes[0, 0].get_position()
        lower_bbox = axes[1, 0].get_position()
        fig.text(
            lower_bbox.x0,
            lower_bbox.y1 + B_TITLE_GAP_FRACTION * (upper_bbox.y0 - lower_bbox.y1),
            "b) Temporal changes in video sequences",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            ha="left",
            va="center",
        )
    # call legend for the classes outside the figure
    add_class_legend(fig)

    # format the figure and save it
    plt.savefig(OUTPUT_DIR / "figure4_convection_characterization.png", dpi=300)
    plt.close(fig)  


def plot_scatter_row(fig, axes, video_stats_plot_df, class_2_plot):
    """
    code inspired by test_2d-scatter to plot the scatter plots of 
    - CTH mean vs Cloud Cover
    - Precipitation mean vs EUCLID MSG grid mean
    For each column, the subplot should be organized in 2 scatter plots, one for each pair of variables. Each column
    corresponds to a different convective class (growing, decaying day, decaying night). Each scatter plot should have
    the median of the entire distribution of the class plotted with a larger marker.

    """
    first_group_name = next(iter(class_2_plot))
    shared_top_ax = None
    shared_bottom_ax = None

    for column_index, (parent_ax, (group_name, class_labels)) in enumerate(zip(axes, class_2_plot.items())):
        parent_spec = parent_ax.get_subplotspec()
        parent_ax.remove()
        subgrid = parent_spec.subgridspec(2, 1, hspace=0.45)
        ax_top = fig.add_subplot(subgrid[0, 0], sharey=shared_top_ax)
        ax_bottom = fig.add_subplot(subgrid[1, 0], sharey=shared_bottom_ax)
        add_panel_label(ax_top, f"a{column_index + 1}")
        add_panel_label(ax_bottom, f"a{len(CLASS_TO_PLOT) + column_index + 1}")
        if shared_top_ax is None:
            shared_top_ax = ax_top
        if shared_bottom_ax is None:
            shared_bottom_ax = ax_bottom

        if group_name == first_group_name:
            bbox = ax_top.get_position()
            fig.text(
                bbox.x0,
                bbox.y1 + SECTION_TITLE_OFFSET,
                "a) Convective life cycle",
                fontsize=TITLE_FONTSIZE,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        for class_label in class_labels:
            class_df = video_stats_plot_df[video_stats_plot_df["label"] == class_label]
            if class_df.empty:
                continue
            positive_lightning_df = class_df[class_df["euclid_msg_grid_mean"] > 0]

            ax_top.scatter(
                class_df["cma_mean"] * 100,
                class_df["cth_mean"],
                color=colors_per_class_codes_grl[str(class_label)],
                label=f"Class {class_label}",
                edgecolor=MARKER_EDGE_COLOR,
                s=MARKER_SIZE,
                alpha=0.5,
            )

            if not positive_lightning_df.empty:
                ax_bottom.scatter(
                    positive_lightning_df["precipitation_mean"],
                    positive_lightning_df["euclid_msg_grid_mean"],
                    color=colors_per_class_codes_grl[str(class_label)],
                    label=f"Class {class_label}",
                    edgecolor=MARKER_EDGE_COLOR,
                    s=MARKER_SIZE,
                    alpha=0.5,
                )

        # loop to plot median of the entire distribution of the class for both scatter plots
        for class_label in class_labels:
            median_row = video_stats_plot_df[video_stats_plot_df["label"] == class_label][
                ["cma_mean", "cth_mean"]
            ].median(numeric_only=True)
            ax_top.scatter(
                median_row["cma_mean"] * 100,
                median_row["cth_mean"],
                color=colors_per_class_codes_grl[str(class_label)],
                edgecolor="black",
                marker="X",
                s=MARKER_SIZE * 10,
                linewidth=2.5,
            )

            positive_bottom_df = video_stats_plot_df[
                (video_stats_plot_df["label"] == class_label)
                & (video_stats_plot_df["euclid_msg_grid_mean"] > 0)
            ]
            if not positive_bottom_df.empty:
                bottom_median_row = positive_bottom_df[
                    ["precipitation_mean", "euclid_msg_grid_mean"]
                ].median(numeric_only=True)
                ax_bottom.scatter(
                    bottom_median_row["precipitation_mean"],
                    bottom_median_row["euclid_msg_grid_mean"],
                    color=colors_per_class_codes_grl[str(class_label)],
                    edgecolor="black",
                    marker="X",
                    s=MARKER_SIZE * 10,
                    linewidth=2.5,
                )
        ax_top.set_title(group_name.replace("_", " ").title(), fontsize=TITLE_FONTSIZE, fontweight="bold", loc="center")
        if column_index == 0:
            ax_top.set_ylabel("Mean cloud \n top height (m)", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax_top.set_ylabel("")
        ax_top.set_xlabel("Cloud Cover (%)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=14)
        ax_top.set_xlim(0., 100.)
        ax_top.grid(axis="y", alpha=0.2)
        style_axis(ax_top)
        ax_top.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

        if column_index == 0:
            ax_bottom.set_ylabel("Lightning counts", fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax_bottom.set_ylabel("")
        ax_bottom.set_xlabel("Mean cumulative Precipitation (mm)", fontsize=AXIS_LABEL_FONTSIZE)
        ax_bottom.set_xlim(0.,6000.)
        ax_bottom.set_yscale("log")
        ax_bottom.grid(axis="y", alpha=0.2)
        style_axis(ax_bottom)
        ax_bottom.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)




def plot_time_series_row(
    axes,
    variable_df,
    class_2_plot,
    variable_name,
    plot_spec,
    show_xlabel=False,
    row_index=0,
):
    """Plot the temporal delta series for one variable across the three class-group axes."""
    class_time_series_by_label = prepare_class_time_series(variable_df, plot_spec)
    plot_variable_time_series_on_axes(
        axes,
        variable_name,
        plot_spec,
        class_time_series_by_label,
        show_xlabel=show_xlabel,
        row_index=row_index,
    )


if __name__ == "__main__":
    main()


