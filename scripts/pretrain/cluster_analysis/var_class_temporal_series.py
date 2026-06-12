from __future__ import annotations

"""
Generate class-wise temporal statistics for crop-based video sequences.

What the script does:
- Reads the training and/or testing CSV files containing per-frame crop statistics
  for one input variable.
- Uses `--mode train`, `--mode test`, or `--mode both` to decide whether to
  process and plot only the training dataset, only the testing dataset, or both
  datasets overlaid.
- Reconstructs the frame index from the sequence identifiers and timestamps.
- Groups samples by class and frame for the selected dataset(s).
- Computes temporal summaries for each class and selected dataset.
- Calculates gradients of the class-mean time series for the selected dataset(s).
- Saves training outputs with the original filenames when training is selected,
  and testing outputs with `_test` in the filename when testing is selected.
- Saves grouped temporal plots with training as solid class-colored lines and
  testing as dashed class-colored lines.
- Normalizes plotted values to 0-1 separately for the training and testing
  datasets, for each class group and plotted variable column.

Supported inputs:
- Continuous variables with percentile columns, such as `cth` and `cot`.
- Single-value variables, such as `cma` and `euclid_msg_grid`.
- Multi-column precipitation input stored in the `precipitation` CSV.

Input:
- Training CSV, required for `--mode train` and `--mode both`, named like:
    `/sat_data/output/grl_2026/csv/crops_stats_var-{variable_name}_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv`
- Testing CSV, required for `--mode test` and `--mode both`, named like:
    `/sat_data/output/grl_2026/csv/crops_stats_var-{variable_name}_stats-50-95-25-75_frames-8_timedim_grl_2026_test_all_7045_imergmin.csv`
- Required columns include sequence identifiers (`crop`, `time`, optionally `lat_mid`, `lon_mid`),
    the class label (`label`), the variable name (`var`), and either percentile columns or a single value column.

Output:
- Training NumPy bundle with gradients, column names, base columns, and class labels:
    `/sat_data/output/grl_2026/npz/mean_gradients_{variable_name}.npz`
- Testing NumPy bundle with gradients, column names, base columns, and class labels:
    `/sat_data/output/grl_2026/npz/mean_gradients_{variable_name}_test.npz`
- Training NumPy bundle with the mean value time series for each class:
    `/sat_data/output/grl_2026/npz/mean_values_time_series_{variable_name}.npz`
- Testing NumPy bundle with the mean value time series for each class:
    `/sat_data/output/grl_2026/npz/mean_values_time_series_{variable_name}_test.npz`
- CSV files with the time series and temporal means for easier inspection:
    `/sat_data/output/grl_2026/npz/mean_values_of_series_{variable_name}.csv`
    `/sat_data/output/grl_2026/npz/mean_values_of_series_{variable_name}_test.csv`

- Grouped temporal plots saved in:
    `/sat_data/output/grl_2026/figs/`

Plot mode:
- `--mode train`: plots only training curves, using solid class-colored lines.
- `--mode test`: plots only testing curves, using dashed class-colored lines.
- `--mode both`: overlays training solid lines and testing dashed lines.
- If `--mode` is not provided, the default is `both`.

For continuous variables:
- The script computes the mean 50th percentile value for each frame and class.
- It then derives the temporal gradient for that 50th percentile series.

For `precipitation`:
- The script computes, for each class and frame:
    - the mean and standard deviation of `50`
    - the mean and standard deviation of `95`
    - the mean and standard deviation of `sum[mm]`
    - the mean and standard deviation of `prec_fraction`
- It also computes sparse-aware diagnostics for each of these four precipitation variables:
    - `<column>_nonzero_fraction`: fraction of samples with value `> 0`
    - `<column>_mean_nonzero`: mean computed only on samples where the value is `> 0`
- It then derives the temporal gradient of the class-mean time series for these four precipitation variables.

For `cma`:
- The script computes the mean cloud-mask value for each frame and class.

For `euclid_msg_grid`:
- For each class and frame, the script computes:
    - `lightning_count`: mean of lightning events over all samples
    - `lightning_count_std`: standard deviation of lightning events over all samples
    - `lightning_nonzero_fraction`: fraction of lightning events samples with count `> 0`
    - `lightning_mean_nonzero`: mean computed only on samples where count `> 0`
- This separates occurrence from intensity:
    - `lightning_nonzero_fraction` tells how often lightning is present
    - `lightning_mean_nonzero` tells how strong it is when it is present

How to run:
- Activate the environment, then run for example:
    `conda activate vissl`
    `python scripts/pretrain/cluster_analysis/var_class_temporal_series.py --var cth`
    `python scripts/pretrain/cluster_analysis/var_class_temporal_series.py --var precipitation`
    `python scripts/pretrain/cluster_analysis/var_class_temporal_series.py --var euclid_msg_grid`
    `python scripts/pretrain/cluster_analysis/var_class_temporal_series.py --var cth --mode train`
    `python scripts/pretrain/cluster_analysis/var_class_temporal_series.py --var cth --mode test`
    `python scripts/pretrain/cluster_analysis/var_class_temporal_series.py --var cth --mode both`

Author: Claudia Acquistapace
Date: 10 sept 2025
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple
import sys
import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pdb


# Add the repository root so top-level packages such as `utils` resolve
# regardless of the directory from which this script is launched.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


VARIABLE_METADATA_PATH = Path(REPO_ROOT) / "configs" / "variables_metadata.yaml"
CSV_DIR = Path("/sat_data/output/grl_2026/csv")
FIG_DIR = Path("/sat_data/output/grl_2026/figs")
NPZ_DIR = Path("/sat_data/output/grl_2026/npz")
CTH_SCALE_TO_KM = 0.001
HISTOGRAM_LINEWIDTH = 3.0
AUTO_VMAX_QUANTILE = 0.995
VALUE_COLUMN_ALIASES = {
    "sum": "sum[mm]",
}
PERCENTILE_COLUMNS = ("50",)
PRECIPITATION_VALUE_COLUMNS = ("50", "95", "sum[mm]", "prec_fraction")
CMA_VALUE_COLUMNS = ("categorical", "None")
SCALAR_VARIABLE_COLUMNS = {
    "cma": "categorical",
    "euclid_msg_grid": "lightning_count",
}
SPARSE_AWARE_DIAGNOSTIC_COLUMNS = {
    "euclid_msg_grid": "lightning_nonzero_fraction",
}
CONDITIONAL_MEAN_DIAGNOSTIC_COLUMNS = {
    "euclid_msg_grid": "lightning_mean_nonzero",
}
TEST_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">")

from utils.plotting.class_colors import colors_per_class1_names, class_groups
from utils.configs import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot class-wise 1D variable histograms.")
    parser.add_argument("--var", default="cth", help="Variable name as defined in variables_metadata.yaml")
    parser.add_argument("--percentile", default="50", help="Percentile column to plot for continuous variables")
    parser.add_argument(
        "--mode",
        default="both",
        choices=("train", "test", "both"),
        help="Dataset to plot: training only, testing only, or both overlaid.",
    )
    return parser.parse_args()


def load_variable_metadata(variable_name: str) -> dict:
    """"
    Load variable metadata from the YAML configuration file and return the dictionary for the specified variable.
    Args:
        variable_name (str): Name of the variable to load metadata for.
    Returns:
        dict: Metadata dictionary for the specified variable.
    Raises:
        ValueError: If the variable is not found in the configuration file.
    """
    config = load_config(str(VARIABLE_METADATA_PATH))
    variables = config.get("variables", {})

    if variable_name not in variables:
        available_variables = ", ".join(sorted(variables))
        raise ValueError(
            f"Variable '{variable_name}' not found in {VARIABLE_METADATA_PATH}. "
            f"Available variables: {available_variables}"
        )

    return variables[variable_name]


def resolve_stats_csv_files(variable_name: str, mode: str) -> Tuple[Optional[Path], Optional[Path]]:
    train_csv_file = (
        CSV_DIR
        / f"crops_stats_var-{variable_name}_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv"
    )
    test_csv_file = (
        CSV_DIR
        / f"crops_stats_var-{variable_name}_stats-50-95-25-75_frames-8_timedim_grl_2026_test_all_7045_imergmin.csv"
    )

    selected_files = []
    if mode in ("train", "both"):
        selected_files.append(train_csv_file)
    if mode in ("test", "both"):
        selected_files.append(test_csv_file)

    missing_files = [str(csv_file) for csv_file in selected_files if not csv_file.exists()]
    if missing_files:
        raise FileNotFoundError("Missing required crop statistics CSV(s): " + ", ".join(missing_files))

    return (
        train_csv_file if mode in ("train", "both") else None,
        test_csv_file if mode in ("test", "both") else None,
    )


def main(variable_name: str = "cth", mode: str = "both"):
    mode = mode.lower()
    if mode not in ("train", "test", "both"):
        raise ValueError("mode must be one of: train, test, both")

    # reading metadata for the variable to plot and get value scale and axis label
    metadata = load_variable_metadata(variable_name)

    # read the selected train and/or test csv files for the selected variable
    train_csv_file, test_csv_file = resolve_stats_csv_files(variable_name, mode)
    df = read_csv_to_dataframe(str(train_csv_file)) if train_csv_file is not None else None
    df_test = read_csv_to_dataframe(str(test_csv_file)) if test_csv_file is not None else None

    # normalize variable-specific column names for downstream processing
    if df is not None:
        df = normalize_variable_columns(df, variable_name)
        print("Training column titles:", df.columns.tolist())
    if df_test is not None:
        df_test = normalize_variable_columns(df_test, variable_name)
        print("Test column titles:", df_test.columns.tolist())
    print("Reading CSV files...")

    # define variable string for plot titles and axis labels using metadata, or variable name if not defined in metadata
    var_name = variable_name
    var_string = metadata.get('long_name', metadata.get('label', var_name.upper()))
    reference_df = df if df is not None else df_test
    value_columns = get_value_columns(reference_df, var_name) # get the value columns to process for the variable (percentiles or single value column)
    if df is not None and df_test is not None and get_value_columns(df_test, var_name) != value_columns:
        raise ValueError(
            f"Train and test CSVs resolved different value columns: "
            f"{value_columns} vs {get_value_columns(df_test, var_name)}"
        )
    gradient_columns = get_gradient_columns(var_name, value_columns) # get the columns for which to compute gradients (value columns and optionally std and sparse-aware diagnostics)

    df_all_mean = {}
    df_all_mean_test = {}
    class_labels = []
    test_class_labels = []
    all_grad_class = None
    all_grad_class_test = None

    if df is not None:
        df_all_mean, class_labels, _, all_grad_class = build_class_temporal_statistics(
            df,
            var_name,
            value_columns,
            gradient_columns,
            dataset_name="training",
        )
    if df_test is not None:
        df_all_mean_test, test_class_labels, _, all_grad_class_test = build_class_temporal_statistics(
            df_test,
            var_name,
            value_columns,
            gradient_columns,
            dataset_name="test",
        )

    print(f"Saving mean gradients for variable {var_name}...")
    if df is not None:
        save_mean_time_series_files(df_all_mean, class_labels, value_columns, var_name)
    if df_test is not None:
        save_mean_time_series_files(df_all_mean_test, test_class_labels, value_columns, var_name, suffix="_test")
    
    # save the mean gradients for each class and each percentile in a numpy file
    if all_grad_class is not None:
        np.savez(
            os.path.join(NPZ_DIR, f'mean_gradients_{var_name}.npz'),
            gradients=all_grad_class, # save the gradients of all columns, including std and sparse-aware diagnostics when present, for all classes
            columns=np.array(gradient_columns, dtype=object), # save column names as an array of strings in the npz file
            base_columns=np.array(value_columns, dtype=object), # save the base value columns as an array of strings in the npz file
            class_labels=np.array(class_labels), # save the class labels as an array in the npz file
        )
    if all_grad_class_test is not None:
        np.savez(
            os.path.join(NPZ_DIR, f'mean_gradients_{var_name}_test.npz'),
            gradients=all_grad_class_test,
            columns=np.array(gradient_columns, dtype=object),
            base_columns=np.array(value_columns, dtype=object),
            class_labels=np.array(test_class_labels),
        )

    df_all_mean_for_plot = df_all_mean if mode in ("train", "both") else {}
    df_all_mean_test_for_plot = df_all_mean_test if mode in ("test", "both") else None

    # plot 50th percentile for selected group of classes normalized between min and max
    if uses_extended_column_stats(var_name):
        for column_name in get_plot_columns(var_name, value_columns):
            plot_temporal_series_perc_by_group(
                df_all_mean_for_plot,
                var_name,
                column_name,
                output_dir=str(FIG_DIR),
                df_all_mean_test=df_all_mean_test_for_plot,
            )
    else:
        for percentile in ('50', '75', '95'):
            if percentile in value_columns:
                plot_temporal_series_perc_by_group(
                    df_all_mean_for_plot,
                    var_name,
                    percentile,
                    output_dir=str(FIG_DIR),
                    df_all_mean_test=df_all_mean_test_for_plot,
                )


def build_class_temporal_statistics(
    df: pd.DataFrame,
    var_name: str,
    value_columns: list[str],
    gradient_columns: list[str],
    dataset_name: str,
    compute_gradients: bool = True,
) -> Tuple[dict, list, np.ndarray, np.ndarray]:
    """Build class-wise mean temporal series, optionally with gradient summaries."""
    print(f"Variables in the {dataset_name} dataframe:", df['var'].unique())
    print(f"Labels in the {dataset_name} dataframe:", df['label'].unique())

    # drop all rows with label -100
    df = df[df['label'] != -100]
    print(f"Labels in the {dataset_name} dataframe after dropping -100:", df['label'].unique())

    # assign frame numbers within each video sequence using the timestamp order
    df = assign_frame_numbers(df)

    # loop on classes and plot temporal series of mean percentiles for each class and concatenate resulting dataframes
    df_all_mean = {}
    class_labels = sorted(df['label'].unique()) # sort class labels in ascending order

    # define matrix for gradients
    # initialize mean gradients array (average of the gradient of the time series)
    if is_scalar_variable(var_name):
        mean_grad_class = np.zeros((len(class_labels), 1)) # gradient of cloud cover, one single value per class
    else:
        mean_grad_class = np.zeros((len(class_labels), len(value_columns))) # gradient of percentiles, one value per class and per percentile
    all_grad_class = np.full((len(class_labels), len(gradient_columns)), np.nan) # gradient of all columns, including std and sparse-aware diagnostics when present

    # loop on unique labels of the classes
    for label_index, label_sel in enumerate(class_labels):

        # select class
        df_class = df[df['label'] == label_sel]
        print(f"Number of {dataset_name} samples in class {label_sel}: {len(df_class)}")

        print(f"Processing variable: {var_name}")

        # select only var columns equal to var_name
        df_class_var = df_class[df_class['var'] == var_name]
        print(f"Number of {dataset_name} samples in class {label_sel} for variable {var_name}: {len(df_class_var)}")

        # if var_name is cot or cth group and mean the percentiles, otherwise group and mean categorical variable
        if uses_extended_column_stats(var_name):
            if df_class_var.empty:
                print(f"Skipping {dataset_name} class {label_sel}: no rows found for variable {var_name}")
                continue

            # build aggregation mapping for mean, std and sparse-aware diagnostics
            aggregation = build_extended_stats_aggregation(var_name, value_columns)
            
            # group by frame and compute mean, std and sparse-aware diagnostics for the variable value columns
            df_grouped = df_class_var.groupby('frame').agg(**aggregation).reset_index()
            # fill NaN values in std columns with 0, since NaNs in std indicate that there is only one sample for that frame and class, and thus the gradient should be 0 since there is no variability between samples
            for value_column in value_columns:
                std_column = get_std_column(value_column)
                if std_column in df_grouped:
                    df_grouped[std_column] = df_grouped[std_column].fillna(0.0)

            # print the value columns, the corresponding std columns and sparse-aware diagnostic columns if they exist in the grouped dataframe
            for value_column in value_columns:
                print(df_grouped[value_column])
                sparse_diagnostic_column = get_sparse_diagnostic_column(var_name, value_column) # get the sparse-aware diagnostic column name for the variable and value column, if it exists
                # if the sparse-aware diagnostic column is defined, print it as well
                if sparse_diagnostic_column is not None:
                    print(df_grouped[sparse_diagnostic_column])
                conditional_mean_column = get_conditional_mean_column(var_name, value_column) # get the conditional mean diagnostic column name for the variable and value column, if it exists
                if conditional_mean_column is not None:
                    print(df_grouped[conditional_mean_column])

            if compute_gradients:
                # compute gradients for all columns and store in the all_grad_class array
                all_grad = compute_gradients_for_columns(df_grouped, gradient_columns)
                all_grad_class[label_index, :] = all_grad

                # calculate mean gradient of the time series for each value column
                if is_scalar_variable(var_name):
                    mean_grad_class[label_index] = np.nanmedian(np.gradient(df_grouped[value_columns[0]]))
                else:
                    mean_grad = np.zeros(len(value_columns))
                    # calculate the mean gradient for each value column (e.g., each percentile) and store in the mean_grad_class array
                    for ind, value_column in enumerate(value_columns):
                        mean_grad[ind] = np.nanmedian(np.gradient(df_grouped[value_column]))
                    mean_grad_class[label_index, :] = mean_grad
                    print(f"Mean gradients for class {label_sel}: {mean_grad}")
                    print("Mean gradient array:", mean_grad_class[label_index, :])
                    print(" all gradients array:", mean_grad_class)

            # plot temporal series of mean categorical for the class
            #plot_temporal_series_perc_by_class(df_grouped, 'categorical', var_name, 'Mean cloud cover', label_sel, output_dir)

        else:

            if df_class_var.empty:
                print(f"Skipping {dataset_name} class {label_sel}: no rows found for variable {var_name}")
                continue

            # group by frame and compute mean of the percentile columns for the variable
            df_grouped = (
                df_class_var.groupby('frame')[value_columns]
                .mean()
                .reset_index()
            )

            if compute_gradients:
                # compute gradients for all columns and store in the all_grad_class array
                all_grad = compute_gradients_for_columns(df_grouped, gradient_columns)
                all_grad_class[label_index, :] = all_grad

                # calculate mean gradient of the time series for each percentile
                mean_grad = np.zeros(len(value_columns))
                for ind, perc in enumerate(value_columns):
                    mean_grad[ind] = np.nanmedian(np.gradient(df_grouped[perc]))
                mean_grad_class[label_index, :] = mean_grad
                print(f"Mean gradients for class {label_sel}: {mean_grad}")
                for perc in value_columns:
                    print(f"gradient {perc}", np.gradient(df_grouped[perc]))
                print("mean gradient", mean_grad)
                print("Mean gradient array:", mean_grad_class[label_index, :])
                print(" all gradients array:", mean_grad_class)
                #pdb.set_trace()

        # store the grouped dataframe for the class in the df_all_mean dictionary for later plotting of temporal series by group of classes
        df_all_mean[label_sel] = df_grouped

    class_labels = sorted(df_all_mean)
    return df_all_mean, class_labels, mean_grad_class, all_grad_class


def save_mean_time_series_files(
    df_all_mean: dict,
    class_labels: list,
    value_columns: list[str],
    var_name: str,
    suffix: str = "",
) -> None:
    """Save class-wise mean time series to npz and csv files."""
    np.savez(
        os.path.join(NPZ_DIR, f'mean_values_time_series_{var_name}{suffix}.npz'),
        mean_values=df_all_mean,
        columns=np.array(value_columns, dtype=object),
        class_labels=np.array(class_labels),
    )

    df_mean_values = pd.DataFrame({
        'label': class_labels,
    })
    if df_all_mean and class_labels:
        grouped_stat_columns = [
            column for column in df_all_mean[class_labels[0]].columns if column != 'frame'
        ]
        for value_column in grouped_stat_columns:
            # Store the full frame series plus its temporal mean for each class.
            df_mean_values[value_column] = [df_all_mean[label][value_column].values for label in class_labels]
            df_mean_values[f'{value_column}_temporal_mean'] = [
                df_all_mean[label][value_column].mean() for label in class_labels
            ]

    df_mean_values.to_csv(os.path.join(NPZ_DIR, f'mean_values_of_series_{var_name}{suffix}.csv'), index=False)


def get_test_marker(class_id: int) -> str:
    return TEST_MARKERS[int(class_id) % len(TEST_MARKERS)]


def get_group_value_range(data_by_label, group_labels, column_name: str) -> Tuple[float, float]:
    """Return min/max over one dataset for one class group and column."""
    values = []
    for label in group_labels:
        df_class = data_by_label.get(label)
        if df_class is None or column_name not in df_class:
            continue
        values.extend(df_class[column_name].dropna().to_numpy(dtype=float))

    if not values:
        return 0.0, 1.0

    values = np.asarray(values, dtype=float)
    return float(np.nanmin(values)), float(np.nanmax(values))


def normalize_to_unit_interval(values, value_min: float, value_max: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    value_range = value_max - value_min
    if value_range == 0:
        return np.zeros_like(values, dtype=float)
    return (values - value_min) / value_range


def normalized_axis_label(var_name: str, column_name: str) -> str:
    if uses_extended_column_stats(var_name):
        return f'Normalized {get_column_axis_label(var_name, column_name)}'
    return f'Normalized {var_name} {column_name}th Percentile'


def plot_temporal_series_perc_by_class(df_grouped, arg, var_name, var_string, label, output_dir):
    """
    function to plot percentiles time series or cloud cover for each class
    Args:
        df_grouped (pd.DataFrame): Dataframe with mean values for the class.
        arg (str): Column name to plot ('25', '50', '75', '95' or 'categorical').
        var_name (str): Name of the variable to plot (cot, cth, cma).
        var_string (str): String to use in the title of the plot.
        label (int): Class label.
        output_dir (str): Directory to save the plot.
    """
    # plot temporal series of categorical mean variable
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped['frame'], df_grouped[arg], label=var_string, marker='o')
    plt.title(f'Temporal Series of Mean \n {var_name} for Class {label}')
    plt.xlabel('Frame')
    plt.ylabel(f'{var_name} Mean Categorical')
    plt.legend()
    plt.grid()
    plt.xticks(df_grouped['frame'])
    plt.tight_layout()
    plt_path = os.path.join(output_dir, f'temporal_series_{var_name}_class_{label}.png')
    plt.savefig(plt_path)
    print(f"Plot saved to {plt_path}")  
    plt.close()
    return()
        
def plot_temporal_series_perc_by_group(df_all_mean, var_name, perc_sel, output_dir, df_all_mean_test=None):
    """
    function to plot percentiles time series for user defined group of classes (ascending, descending, mixed) 
    Args:
        df_all_mean (list of pd.DataFrame): List of dataframes with mean values for each class.
        var_name (str): Name of the variable to plot.
        perc_sel (int): Percentile to plot (e.g., 50, 75, 25, 95).
        output_dir (str): Directory to save the plot.
    Output:
        Saves the plot to the specified output directory.
    """
    # convert perc_sel to string
    perc_sel_str = str(perc_sel)

    # read groups convective, broken and dissipative from utils/plotting/class_colors.py
    for group_name, group_labels in class_groups.items():

        print(group_name, group_labels)
        train_value_min, train_value_max = get_group_value_range(
            df_all_mean,
            group_labels,
            perc_sel_str,
        )
        test_value_min, test_value_max = get_group_value_range(
            df_all_mean_test or {},
            group_labels,
            perc_sel_str,
        )

        # plot temporal evolution of the selected percentile for the group of classes
        plt.figure(figsize=(10, 6))
        plotted_frames = None
        for i in group_labels:
            df_class = df_all_mean.get(i)
            df_class_test = (df_all_mean_test or {}).get(i)

            if df_class is not None and perc_sel_str not in df_class:
                df_class = None
            if df_class_test is not None and perc_sel_str not in df_class_test:
                df_class_test = None
            if df_class is None and df_class_test is None:
                continue

            plotted_frames = df_class['frame'] if df_class is not None else df_class_test['frame']
            # plot class using the color defined in utils/plotting/class_colors.py
            class_color = colors_per_class1_names.get(str(i))
            if class_color is not None:
                if df_class is not None:
                    plot_values = normalize_to_unit_interval(df_class[perc_sel_str], train_value_min, train_value_max)
                    plt.plot(df_class['frame'], 
                             plot_values, 
                             label=f'Class {i}', 
                             linewidth=3,
                             color=class_color)

                if df_class_test is not None:
                    test_plot_values = normalize_to_unit_interval(df_class_test[perc_sel_str], test_value_min, test_value_max)
                    plt.plot(
                        df_class_test['frame'],
                        test_plot_values,
                        linestyle='--',
                        linewidth=3,
                        label=f'Class {i} test',
                        color=class_color,
                    )

                if df_class is not None and var_name != 'cma' and is_primary_value_column(var_name, perc_sel_str):
                    std_column = get_std_column(perc_sel_str)
                    if std_column in df_class:
                        lower_bound = df_class[perc_sel_str] - df_class[std_column]
                        upper_bound = df_class[perc_sel_str] + df_class[std_column]
                        lower_bound = normalize_to_unit_interval(lower_bound, train_value_min, train_value_max)
                        upper_bound = normalize_to_unit_interval(upper_bound, train_value_min, train_value_max)
                        lower_bound = np.clip(lower_bound, 0.0, 1.0)
                        upper_bound = np.clip(upper_bound, 0.0, 1.0)
                        plt.fill_between(df_class['frame'], lower_bound, upper_bound, color=class_color, alpha=0.15)
            else:
                if df_class is not None:
                    plot_values = normalize_to_unit_interval(df_class[perc_sel_str], train_value_min, train_value_max)
                    plt.plot(df_class['frame'], 
                             plot_values, 
                             label=f'Class {i}', 
                             linewidth=3)
                if df_class_test is not None:
                    test_plot_values = normalize_to_unit_interval(df_class_test[perc_sel_str], test_value_min, test_value_max)
                    plt.plot(
                        df_class_test['frame'],
                        test_plot_values,
                        linestyle='--',
                        linewidth=3,
                        label=f'Class {i} test',
                    )
        if uses_extended_column_stats(var_name):
            plt.title(get_column_plot_title(var_name, perc_sel_str, group_name), fontsize=20)
        else:
            plt.title(
                f'Temporal Series of {var_name} {perc_sel_str}th Percentile\n'
                f'for {group_name.capitalize()} Classes',
                fontsize=20,
            )
        plt.xlabel('Frame', fontsize=20)
        # remove upper and right border
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel(normalized_axis_label(var_name, perc_sel_str), fontsize=20)
        plt.ylim(0, 1)
        plt.legend(
            fontsize=14,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
        )
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        if plotted_frames is None:
            plt.close()
            continue
        plt.xticks(plotted_frames, fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt_path = os.path.join(output_dir, 
                            f'temporal_series_{perc_sel_str}_{var_name}_{group_name}_classes.png')
        plt.savefig(plt_path, transparent=True, bbox_inches='tight')
        print(f"Plot saved to {plt_path}")
        plt.close()


    return

    # plot percentile for selected group of classes normalized between min and max

    plt.figure(figsize=(10, 6))
    for i in class_indices:
        df_class = df_all_mean[i]
        plt.plot(df_class['frame'], 
                 (df_class[perc_sel_str] - df_class[perc_sel_str].min()) / (df_class[perc_sel_str].max() - df_class[perc_sel_str].min()), 
                 label=f'Class {i}', 
                 marker='o', 
                 color=plt.cm.tab10(i),
                 linewidth=3)
    plt.title(f'Temporal Series of {var_name} {perc_sel_str}th Percentile for {group.capitalize()} Classes')
    plt.xlabel('Frame')
    plt.ylabel(f'{var_name} {perc_sel_str}th Percentile')
    plt.legend()
    plt.grid()
    plt.xticks(df_class['frame'])
    plt.tight_layout()
    plt_path = os.path.join(output_dir, f'temporal_series_{perc_sel_str}_{var_name}_{group}_classes.png')
    plt.savefig(plt_path)
    print(f"Plot saved to {plt_path}")

    plt.close()
    return

def plot_temporal_series_min_max_perc_all_classes(df_all_mean, var_name, perc_sel, output_dir):
    """
    Plot temporal series of the selected percentile for all classes between min and max in different subplots.
    the percentile (50, 75, 25, 95) is passed as argument
    
    Args:
        df_all_mean (list of pd.DataFrame): List of dataframes with mean values for each class.
        var_name (str): Name of the variable to plot.
        perc_sel (int): Percentile to plot (e.g., 50, 75, 25, 95).
        output_dir (str): Directory to save the plot.

    Output:
        Saves the plot to the specified output directory.
    """

    if var_name == 'cma':
        print(f"Plotting {var_name} mean for all classes between min and max...")
        # plot cloud mask for all classes between min and max each in a different subplot
        plt.figure(figsize=(10, 20))

        for i, df_class in enumerate(df_all_mean):
            ax = plt.subplot(len(df_all_mean), 1, i + 1)
            # choose a different color for each class
            plt.plot(df_class['frame'], 
                     df_class['categorical'],
                       label=f'Class {i}', marker='o', 
                        color=plt.cm.tab10(i), 
                        linewidth=3)
            plt.legend()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.suptitle(f'Temporal Series of {var_name} for All Classes', fontsize=20)
        plt.xlabel('Frame')
        plt.ylabel('Cloud cover')
        plt.xticks(df_class['frame'])
        plt.tight_layout()
        plt_path = os.path.join(output_dir, f'temporal_series_min_max_{var_name}_all_classes.png')
        plt.savefig(plt_path)
        print(f"Plot saved to {plt_path}")
        plt.close()
    else:
        print(f"Plotting {var_name} {perc_sel}th percentile for all classes between min and max...")    
        # convert perc_sel to string
        perc_sel_str = str(perc_sel)

        # plot percentile for all classes between min and max each in a different subplot
        plt.figure(figsize=(10, 20))
        for i, df_class in enumerate(df_all_mean):
            ax = plt.subplot(len(df_all_mean), 1, i + 1)
            # choose a different color for each class
            plt.plot(df_class['frame'], 
                    df_class[perc_sel_str], 
                    label=f'Class {i}', 
                    marker='o', 
                    color=plt.cm.tab10(i),
                    linewidth=3)
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.suptitle(f'Temporal Series of {var_name} {perc_sel_str}th Percentile (Min-Max) for All Classes', fontsize=20)
        plt.xlabel('Frame')
        plt.ylabel(f'{var_name} {perc_sel_str}th Percentile (Min-Max)')
        plt.xticks(df_class['frame'])

        plt.tight_layout()
        plt_path = os.path.join(output_dir, f'temporal_series_min_max_{perc_sel_str}_{var_name}_all_classes.png')
        plt.savefig(plt_path)

        print(f"Plot saved to {plt_path}")

        plt.close()
    return

def read_csv_to_dataframe(csv_file):
    """Reads the CSV file into a pandas DataFrame.
    If the file ends with '_debug.csv', it filters out rows where 'frame' equals -710387.
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        pd.DataFrame: The DataFrame containing the CSV data, with debug rows removed if applicable.
    """

    df = pd.read_csv(csv_file)

    # if file ends with _debug.csv, remove all lines with frame = -710387
    if csv_file.endswith('_debug.csv'):
        df = df[df['frame'] != -710387]
    if df.empty:
        print(f"Warning: {csv_file} is empty after filtering.")

    return df


def get_percentile_columns(df: pd.DataFrame) -> list[str]:
    """Return percentile columns present in the CSV in numeric order.
    This function checks for the presence of expected percentile columns (e.g., '25', '50', '75', '95') in 
    the DataFrame and returns a list of those that are found, sorted in numeric order.
     If no percentile columns are found, it raises a ValueError.
    Args:
        df (pd.DataFrame): The input DataFrame read from the CSV file.
    Returns:
        list[str]: A list of percentile column names that are present in the DataFrame, sorted in numeric order.
    Raises:
        ValueError: If no percentile columns are found in the DataFrame.
    """
    available_columns = [column for column in PERCENTILE_COLUMNS if column in df.columns]
    if not available_columns:
        raise ValueError("No percentile columns found in the CSV file.")
    return available_columns


def get_cma_value_column(df: pd.DataFrame) -> str:
    """Return the column containing the cloud-mask mean for cma files.
    This function checks for the presence of expected cloud-mask value columns (e.g., 'categorical', 'None')
    in the DataFrame and returns the first one that is found. If no cloud-mask value column is found, it raises a ValueError.
    Args:
        df (pd.DataFrame): The input DataFrame read from the CSV file.
    Returns:
        str: The name of the cloud-mask value column.
    Raises:
        ValueError: If no cloud-mask value column is found in the DataFrame.
    """
    for column in CMA_VALUE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError("No cma value column found in the CSV file.")


def get_value_columns(df: pd.DataFrame, variable_name: str) -> list[str]:
    """Return the data columns used to compute temporal statistics for the variable.
    For example, for `cma` files, return the scalar value column. For `precipitation`, return the four value columns.
    For other variables, return the percentile columns.
    Args:
    df (pd.DataFrame): The input DataFrame read from the CSV file.
    variable_name (str): The name of the variable being processed.
    Returns:
    list[str]: The list of column names to use for computing temporal statistics.
    """

    if is_scalar_variable(variable_name):
        return [get_scalar_value_column(df, variable_name)]
    if variable_name == 'precipitation':
        return [column for column in PRECIPITATION_VALUE_COLUMNS if column in df.columns]
    return get_percentile_columns(df)


def normalize_variable_columns(df: pd.DataFrame, variable_name: str) -> pd.DataFrame:
    """Normalize CSV-specific column names for downstream processing.
    For example, for `cma` files, rename the `None` column to `categorical` for consistency with the variable metadata.
    Args:
    df (pd.DataFrame): The input DataFrame read from the CSV file.
    variable_name (str): The name of the variable being processed.
    Returns:
    pd.DataFrame: The DataFrame with normalized column names.
    """
    if not is_scalar_variable(variable_name) or 'None' not in df.columns:
        return df
    target_column = SCALAR_VARIABLE_COLUMNS[variable_name]
    if target_column in df.columns:
        return df
    return df.rename(columns={'None': target_column})


def is_scalar_variable(variable_name: str) -> bool:
    """Return True when the variable uses a single scalar value column instead of percentiles."""
    return variable_name in SCALAR_VARIABLE_COLUMNS


def uses_extended_column_stats(variable_name: str) -> bool:
    """Return True when the variable uses mean/std and optional sparse diagnostics for each data column.
    This is true for `precipitation` and `euclid_msg_grid`, which have value columns with many zeros 
    and benefit from additional diagnostics to separate occurrence from intensity.
    Are also considered scalar variables, which have a single value column and a std column, but no sparse-aware diagnostics.
    """ 
    return is_scalar_variable(variable_name) or variable_name == 'precipitation'


def get_scalar_value_column(df: pd.DataFrame, variable_name: str) -> str:
    """Return the scalar-value column for variables such as cma and euclid_msg_grid.
    This function checks for the presence of the expected scalar value column for the variable 
    (e.g., 'categorical' for 'cma', 'lightning_count' for 'euclid_msg_grid') 
    in the DataFrame and returns it if found. If the expected column is not found,
     it checks for any of the known aliases for scalar value columns (e.g., 'None')
      and returns it if found. If no scalar value column is found, it raises a ValueError."""
    expected_column = SCALAR_VARIABLE_COLUMNS[variable_name]
    if expected_column in df.columns:
        return expected_column
    for column in CMA_VALUE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No scalar value column found in the CSV file for {variable_name}.")


def get_scalar_value_column_name(variable_name: str) -> str:
    """Return the normalized scalar-value column name for the variable.
    This function returns the standardized column name for the scalar value column of the variable,
    which is defined in the SCALAR_VARIABLE_COLUMNS mapping. 
    This is used for consistent naming of the scalar value column across different CSV files, 
    even if they use different column names (e.g., 'None' vs 'categorical' for cma). 
    If the variable is not defined as a scalar variable, it raises a ValueError.
    Args:
    variable_name (str): The name of the variable being processed.
    Returns:
    str: The standardized column name for the scalar value column of the variable.
    Raises:
    ValueError: If the variable is not defined as a scalar variable in the SCALAR_VARIABLE_COLUMNS mapping.
    """
    return SCALAR_VARIABLE_COLUMNS[variable_name]


def get_std_column(column_name: str) -> str:
    """Return the standard-deviation column name for a base data column."""
    return f'{column_name}_std'


def get_sparse_diagnostic_column(variable_name: str, column_name: str | None = None) -> str | None:
    """Return the sparse-aware diagnostic column for variables that need one.
    For `precipitation`, this is defined for each value column as `prec_nonzero_fraction`.
    For `euclid_msg_grid`, this is defined as `lightning_nonzero_fraction`.
    For other variables, this is defined in the SPARSE_AWARE_DIAGNOSTIC_COLUMNS mapping.
    Args:
    variable_name (str): The name of the variable being processed.
    column_name (str | None): The name of the base data column, 
    if the sparse-aware diagnostic column is defined per data column (e.g., for precipitation). 
    If the sparse-aware diagnostic column is not defined per data column (e.g., for euclid_msg_grid), this can be
     left as None.
    Returns:
    str | None: The name of the sparse-aware diagnostic column for the variable and base data   
        column, or None if no sparse-aware diagnostic column is defined for the variable.
    """

    if variable_name == 'precipitation' and column_name is not None:
        return f'prec_nonzero_fraction'
    return SPARSE_AWARE_DIAGNOSTIC_COLUMNS.get(variable_name)


def get_conditional_mean_column(variable_name: str, column_name: str | None = None) -> str | None:
    """Return the conditional mean diagnostic column for sparse variables.
    For `precipitation`, this is defined for each value column as `prec_mean_nonzero`, 
    which is the mean over non-zero samples and thus reflects the intensity of precipitation when it occurs,
     without being confounded by the large number of zero-precipitation samples.
     For `euclid_msg_grid`, this is defined as `lightning_mean_nonzero`, 
     which is the mean over samples with lightning counts > 0,
      and thus reflects the intensity of lightning when it occurs without being confounded by the large number of samples with zero lightning counts.
    For other variables, this is defined in the CONDITIONAL_MEAN_DIAGNOSTIC_COLUMNS mapping.
    Args:
        variable_name (str): The name of the variable being processed.
        column_name (str | None): The name of the base data column, 
        if the conditional mean diagnostic column is defined per data column (e.g., for precipitation). 
        If the conditional mean diagnostic column is not defined per data column (e.g., for euclid_msg_grid), this can be
         left as None.
    Returns:
    str | None: The name of the conditional mean diagnostic column for the variable and base data column, or None if no conditional mean diagnostic column is defined for the variable.
    """
    if variable_name == 'precipitation' and column_name is not None:
        return f'{column_name}_mean_nonzero'
    return CONDITIONAL_MEAN_DIAGNOSTIC_COLUMNS.get(variable_name)


def build_extended_stats_aggregation(variable_name: str, value_columns: list[str]) -> dict:
    """Build the aggregation mapping for variables with std and sparse-aware diagnostics.
    For each base value column, this includes the mean, the std, and optionally the sparse-aware diagnostics.
    Args:
        variable_name (str): The name of the variable being processed.
        value_columns (list[str]): The list of base data columns for the variable.
    Returns:
        dict: The aggregation mapping to use in the pandas groupby aggregation.
    """ 
    aggregation = {}
    # For each value column, compute the mean, the std, and optionally the sparse-aware diagnostics
    for value_column in value_columns:
        aggregation[value_column] = (value_column, 'mean') # compute the mean of the base value column
        aggregation[get_std_column(value_column)] = (value_column, 'std') # compute the std of the base value column
        sparse_diagnostic_column = get_sparse_diagnostic_column(variable_name, value_column) # compute the sparse-aware diagnostic column, if applicable for the variable and value column
        # if the sparse-aware diagnostic column is defined, compute it as the fraction of samples with value > 0
        if sparse_diagnostic_column is not None:
            aggregation[sparse_diagnostic_column] = (value_column, lambda values: (values > 0).mean())
        conditional_mean_column = get_conditional_mean_column(variable_name, value_column) # compute the conditional mean diagnostic column, if applicable for the variable and value column
        # if the conditional mean diagnostic column is defined, compute it as the mean over samples with value > 0, or 0 if there are no samples with value > 0 to avoid NaNs
        if conditional_mean_column is not None:
            aggregation[conditional_mean_column] = (
                value_column,
                lambda values: values[values > 0].mean() if (values > 0).any() else 0.0,
            )
    return aggregation


def get_plot_columns(variable_name: str, value_columns: list[str]) -> list[str]:
    """Return the ordered list of data and diagnostic series to plot.
    For variables with extended stats, this includes the base value columns, the std columns, and the sparse-aware diagnostic columns.
    For other variables, this includes only the base value columns.
    Args:
        variable_name (str): The name of the variable being processed.
        value_columns (list[str]): The list of base data columns for the variable.
    Returns:
        list[str]: The ordered list of columns to plot.
    """
    plot_columns = list(value_columns)
    if not uses_extended_column_stats(variable_name):
        return plot_columns

    for value_column in value_columns:
        sparse_diagnostic_column = get_sparse_diagnostic_column(variable_name, value_column)
        if sparse_diagnostic_column is not None:
            plot_columns.append(sparse_diagnostic_column)
        conditional_mean_column = get_conditional_mean_column(variable_name, value_column)
        if conditional_mean_column is not None:
            plot_columns.append(conditional_mean_column)
    return plot_columns


def get_gradient_columns(variable_name: str, value_columns: list[str]) -> list[str]:
    """Return the ordered list of columns whose gradients should be saved.
    For variables with extended stats, this includes the base value columns, the std columns, and the sparse-aware diagnostic columns.
    For other variables, this includes only the base value columns.
    Args:
    variable_name (str): The name of the variable being processed.
    value_columns (list[str]): The list of base data columns for the variable.
    Returns:
    list[str]: The ordered list of columns for which to compute and save gradients. 
    """
    gradient_columns = list(value_columns)
    if not uses_extended_column_stats(variable_name):
        return gradient_columns

    for value_column in value_columns:
        gradient_columns.append(get_std_column(value_column))
        sparse_diagnostic_column = get_sparse_diagnostic_column(variable_name, value_column)
        if sparse_diagnostic_column is not None:
            gradient_columns.append(sparse_diagnostic_column)
        conditional_mean_column = get_conditional_mean_column(variable_name, value_column)
        if conditional_mean_column is not None:
            gradient_columns.append(conditional_mean_column)
    return gradient_columns


def compute_gradients_for_columns(df_grouped: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Compute the median gradient for each requested grouped time-series column."""
    gradients = np.full(len(columns), np.nan)
    for index, column_name in enumerate(columns):
        if column_name not in df_grouped:
            continue
        gradients[index] = np.nanmedian(np.gradient(df_grouped[column_name]))
    return gradients


def is_primary_value_column(variable_name: str, column_name: str) -> bool:
    """Return True when the plotted column is a base data column with a std band.
    This is true for the scalar value column of scalar variables, and for the value columns of precipitation.
    For these columns, we plot a std band around the mean line in the temporal series plots, 
    since they reflect the main variable of interest and the std provides useful information 
    about the variability of the samples that are averaged to compute the mean.
    For other columns such as sparse-aware diagnostics, we do not plot a std band since
     they are derived metrics that do not directly reflect the variability of the underlying 
     samples in the same way as the base value columns.   
    """
    if is_scalar_variable(variable_name):
        return column_name == get_scalar_value_column_name(variable_name)
    if variable_name == 'precipitation':
        return column_name in PRECIPITATION_VALUE_COLUMNS
    return False


def get_scalar_series_label(variable_name: str) -> str:
    """Return a human-readable label for scalar time series plots.
    For example, for `cma`, return 'cloud mask', and for `euclid_msg_grid`, return 'lightning counts'.
    For other variables, return the variable name itself.
    """
    if variable_name == 'cma':
        return 'cloud mask'
    if variable_name == 'euclid_msg_grid':
        return 'lightning counts'
    return variable_name


def get_column_plot_title(variable_name: str, column_name: str, group_name: str) -> str:
    """Return the plot title for supported variables and diagnostics.
     For scalar variables, this depends on whether the column is the base value column, 
     the sparse-aware diagnostic column, or the conditional mean diagnostic column.
     For precipitation, this depends on the column suffix indicating whether it is a base value column,
     the non-zero fraction diagnostic, or the conditional mean diagnostic.
     For other variables, this is a generic title with the variable name, column name, and
     group name.
     Args:
        variable_name (str): The name of the variable being processed.
        column_name (str): The name of the data or diagnostic column being plotted.
        group_name (str): The name of the group of classes being plotted (e.g.,
         'convective', 'broken', 'dissipative').
    Returns:
        str: The title to use for the plot of the temporal series of the column for the group of classes.
    """
    if is_scalar_variable(variable_name):
        if column_name == get_sparse_diagnostic_column(variable_name, column_name):
            return (
                f'Temporal Series of Non-Zero {get_scalar_series_label(variable_name)} Fraction\n'
                f'for {group_name.capitalize()} Classes'
            )
        if column_name == get_conditional_mean_column(variable_name, column_name):
            return (
                f'Temporal Series of Conditional Mean {get_scalar_series_label(variable_name)}\n'
                f'for {group_name.capitalize()} Classes'
            )
        return (
            f'Temporal Series of Mean {get_scalar_series_label(variable_name)}\n'
            f'for {group_name.capitalize()} Classes'
        )

    if variable_name == 'precipitation':
        if column_name.endswith('_nonzero_fraction'):
            base_column = strip_suffix(column_name, '_nonzero_fraction')
            return (
                f'Temporal Series of Non-Zero Fraction for {format_column_label(variable_name, base_column)}\n'
                f'in {group_name.capitalize()} Classes'
            )
        if column_name.endswith('_mean_nonzero'):
            base_column = strip_suffix(column_name, '_mean_nonzero')
            return (
                f'Temporal Series of Conditional Mean for {format_column_label(variable_name, base_column)}\n'
                f'in {group_name.capitalize()} Classes'
            )
        return (
            f'Temporal Series of Mean {format_column_label(variable_name, column_name)}\n'
            f'for {group_name.capitalize()} Classes'
        )

    return (
        f'Temporal Series of {variable_name} {column_name}\n'
        f'for {group_name.capitalize()} Classes'
    )


def get_column_axis_label(variable_name: str, column_name: str) -> str:
    """Return the y-axis label for supported variables and diagnostics."""
    if is_scalar_variable(variable_name):
        if column_name == get_sparse_diagnostic_column(variable_name, column_name):
            return 'Fraction of non-zero samples'
        if column_name == get_conditional_mean_column(variable_name, column_name):
            return 'Mean over non-zero samples'
        if variable_name == 'cma':
            return 'Mean cloud mask'
        if variable_name == 'euclid_msg_grid':
            return 'Mean lightning counts'
        return f'Mean {variable_name}'

    if variable_name == 'precipitation':
        if column_name.endswith('_nonzero_fraction'):
            return 'Fraction of non-zero samples'
        if column_name.endswith('_mean_nonzero'):
            return f'Conditional mean {format_column_label(variable_name, strip_suffix(column_name, "_mean_nonzero"))}'
        return f'Mean {format_column_label(variable_name, column_name)}'

    return f'Mean {column_name}'


def strip_suffix(value: str, suffix: str) -> str:
    """Remove a suffix in a way that is compatible with Python 3.8."""
    if value.endswith(suffix):
        return value[:-len(suffix)]
    return value


def format_column_label(variable_name: str, column_name: str) -> str:
    """Return a readable label for a data column."""
    if column_name in PERCENTILE_COLUMNS:
        return f'{column_name}th percentile'
    if column_name in {'25', '75', '95', '99'}:
        return f'{column_name}th percentile'
    if variable_name == 'precipitation' and column_name == 'prec_fraction':
        return 'precipitation fraction'
    return VALUE_COLUMN_ALIASES.get(column_name, column_name)


def assign_frame_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 0-based frame indices within each crop/lat/lon video sequence."""
    required_columns = {'crop', 'time'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_string = ", ".join(sorted(missing_columns))
        raise ValueError(f"Cannot assign frame numbers. Missing columns: {missing_string}")

    group_columns = [column for column in ['crop', 'lat_mid', 'lon_mid'] if column in df.columns]

    frame_df = df.copy()
    frame_df['time'] = pd.to_datetime(frame_df['time'], errors='raise')
    sorted_index = frame_df.sort_values(group_columns + ['time']).index

    frame_df.loc[sorted_index, 'frame'] = (
        frame_df.loc[sorted_index]
        .groupby(group_columns, dropna=False)
        .cumcount()
        .to_numpy()
    )
    frame_df['frame'] = frame_df['frame'].astype(int)

    group_sizes = frame_df.groupby(group_columns, dropna=False)['frame'].max() + 1
    if not group_sizes.eq(8).all():
        invalid_sequences = group_sizes[~group_sizes.eq(8)]
        print(
            "Warning: found sequences with a number of frames different from 8:",
            invalid_sequences.value_counts().sort_index().to_dict(),
        )

    return frame_df


if __name__ == "__main__":
    args = parse_args()
    main(variable_name=args.var, mode=args.mode)
