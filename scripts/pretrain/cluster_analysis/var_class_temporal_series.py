from __future__ import annotations

"""
Generate class-wise temporal statistics for crop-based video sequences.

What the script does:
- Reads a CSV file containing per-frame crop statistics for one input variable.
- Reconstructs the frame index from the sequence identifiers and timestamps.
- Groups samples by class and frame.
- Computes temporal summaries for each class.
- Calculates gradients of the class-mean time series.
- Saves gradient arrays and grouped temporal plots.

Supported inputs:
- Continuous variables with percentile columns, such as `cth` and `cot`.
- Single-value variables, such as `cma` and `euclid_msg_grid`.
- Multi-column precipitation input stored in the `precipitation` CSV.

Input:
- A CSV file named like:
    `/sat_data/output/grl_2026/csv/crops_stats_var-{variable_name}_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv`
- Required columns include sequence identifiers (`crop`, `time`, optionally `lat_mid`, `lon_mid`),
    the class label (`label`), the variable name (`var`), and either percentile columns or a single value column.

Output:
- A single NumPy bundle with gradients, column names, base columns, and class labels:
    `/sat_data/output/grl_2026/figs/mean_gradients_{variable_name}.npz`
- Grouped temporal plots saved in:
    `/sat_data/output/grl_2026/figs/`

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
    - `lightning_count`: mean over all samples
    - `lightning_count_std`: standard deviation over all samples
    - `lightning_nonzero_fraction`: fraction of samples with count `> 0`
    - `lightning_mean_nonzero`: mean computed only on samples where count `> 0`
- This separates occurrence from intensity:
    - `lightning_nonzero_fraction` tells how often lightning is present
    - `lightning_mean_nonzero` tells how strong it is when it is present

How to run:
- Activate the environment, then run for example:
    `conda activate vissl`
    `python var_class_temporal_series.py --var cth`

Author: Claudia Acquistapace
Date: 10 sept 2025
"""
import argparse
import argparse
from pathlib import Path
from typing import Tuple

import sys
import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import pdb


# Add the repository root so top-level packages such as `utils` resolve
# regardless of the directory from which this script is launched.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


VARIABLE_METADATA_PATH = Path(REPO_ROOT) / "configs" / "variables_metadata.yaml"
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

from utils.plotting.class_colors import colors_per_class1_names, class_groups
from utils.configs import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot class-wise 1D variable histograms.")
    parser.add_argument("--var", default="cth", help="Variable name as defined in variables_metadata.yaml")
    parser.add_argument("--percentile", default="50", help="Percentile column to plot for continuous variables")
    return parser.parse_args()


def load_variable_metadata(variable_name: str) -> dict:
    config = load_config(str(VARIABLE_METADATA_PATH))
    variables = config.get("variables", {})

    if variable_name not in variables:
        available_variables = ", ".join(sorted(variables))
        raise ValueError(
            f"Variable '{variable_name}' not found in {VARIABLE_METADATA_PATH}. "
            f"Available variables: {available_variables}"
        )

    return variables[variable_name]

def main(variable_name: str = "cth"):

    # reading metadata for the variable to plot and get value scale and axis label
    metadata = load_variable_metadata(variable_name)

    # read csv file
    output_dir = '/sat_data/output/grl_2026/figs/'

    # read variable to plot and then read the correpsonding csv file
    csv_file = f'/sat_data/output/grl_2026/csv/crops_stats_var-{variable_name}_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv'

    df = read_csv_to_dataframe(csv_file)
    df = normalize_variable_columns(df, variable_name)
    print("Column titles:", df.columns.tolist())
    print("Reading CSV file...")

    # list all variables in the dataframe
    print("Variables in the dataframe:", df['var'].unique())
    print("Labels in the dataframe:", df['label'].unique())

    # drop all rows with label -100
    df = df[df['label'] != -100]
    print("Labels in the dataframe after dropping -100:", df['label'].unique())
    
    # assign frame numbers within each video sequence using the timestamp order
    df = assign_frame_numbers(df)

    # loop on classes and plot temporal series of mean percentiles for each class and concatenate resulting dataframes
    df_all_mean = {}

    var_name = variable_name
    var_string = metadata.get('long_name', metadata.get('label', var_name.upper()))
    class_labels = sorted(df['label'].unique())
    value_columns = get_value_columns(df, var_name)
    gradient_columns = get_gradient_columns(var_name, value_columns)

    # define matrix for gradients
    # initialize mean gradients array (average of the gradient of the time series)
    if is_scalar_variable(var_name):
        mean_grad_class = np.zeros((len(class_labels), 1)) # gradient of cloud cover, one single value per class
    else:
        mean_grad_class = np.zeros((len(class_labels), len(value_columns)))
    all_grad_class = np.full((len(class_labels), len(gradient_columns)), np.nan)

    # loop on unique labels of the classes
    for label_index, label_sel in enumerate(class_labels):

        # select class
        df_class = df[df['label'] == label_sel]
        print(f"Number of samples in class {label_sel}: {len(df_class)}")

        print(f"Processing variable: {var_name}")

        # select only var columns equal to var_name
        df_class_var = df_class[df_class['var'] == var_name]
        print(f"Number of samples in class {label_sel} for variable {var_name}: {len(df_class_var)}")

        # if var_name is cot or cth group and mean the percentiles, otherwise group and mean categorical variable
        if uses_extended_column_stats(var_name):
            if df_class_var.empty:
                print(f"Skipping class {label_sel}: no rows found for variable {var_name}")
                continue

            aggregation = build_extended_stats_aggregation(var_name, value_columns)

            df_grouped = df_class_var.groupby('frame').agg(**aggregation).reset_index()
            for value_column in value_columns:
                std_column = get_std_column(value_column)
                if std_column in df_grouped:
                    df_grouped[std_column] = df_grouped[std_column].fillna(0.0)

            for value_column in value_columns:
                print(df_grouped[value_column])
                sparse_diagnostic_column = get_sparse_diagnostic_column(var_name, value_column)
                if sparse_diagnostic_column is not None:
                    print(df_grouped[sparse_diagnostic_column])
                conditional_mean_column = get_conditional_mean_column(var_name, value_column)
                if conditional_mean_column is not None:
                    print(df_grouped[conditional_mean_column])

            all_grad = compute_gradients_for_columns(df_grouped, gradient_columns)
            all_grad_class[label_index, :] = all_grad

            if is_scalar_variable(var_name):
                mean_grad_class[label_index] = np.nanmedian(np.gradient(df_grouped[value_columns[0]]))
            else:
                mean_grad = np.zeros(len(value_columns))
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
                print(f"Skipping class {label_sel}: no rows found for variable {var_name}")
                continue

            df_grouped = (
                df_class_var.groupby('frame')[value_columns]
                .mean()
                .reset_index()
            )

            all_grad = compute_gradients_for_columns(df_grouped, gradient_columns)
            all_grad_class[label_index, :] = all_grad

            # plot temporal series of mean categorical for the class
            #plot_temporal_series_perc_by_class(df_grouped, '25', var_name, var_string, label_sel, output_dir)
            #plot_temporal_series_perc_by_class(df_grouped, '50', var_name, var_string, label_sel, output_dir)
            #plot_temporal_series_perc_by_class(df_grouped, '75', var_name, var_string, label_sel, output_dir)
            #plot_temporal_series_perc_by_class(df_grouped, '95', var_name, var_string, label_sel, output_dir)

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

        df_all_mean[label_sel] = df_grouped

    # store mean gradients for each class and each percentile in python file using numpy save
    # and also as a .py file with the array as a list

    print(f"Saving mean gradients for variable {var_name}...")

    np.savez(
        os.path.join(output_dir, f'mean_gradients_{var_name}.npz'),
        gradients=all_grad_class,
        columns=np.array(gradient_columns, dtype=object),
        base_columns=np.array(value_columns, dtype=object),
        class_labels=np.array(class_labels),
    )

    # plot 50th percentile for selected group of classes normalized between min and max
    if uses_extended_column_stats(var_name):
        for column_name in get_plot_columns(var_name, value_columns):
            plot_temporal_series_perc_by_group(df_all_mean, var_name, column_name, output_dir=output_dir)
    else:
        for percentile in ('50', '75', '95'):
            if percentile in value_columns:
                plot_temporal_series_perc_by_group(df_all_mean, var_name, percentile, output_dir=output_dir)



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
    plt.title(f'Temporal Series of Mean {var_name} for Class {label}')
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
        
def plot_temporal_series_perc_by_group(df_all_mean, var_name, perc_sel, output_dir):
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

        # plot temporal evolution of the selected percentile for the group of classes
        plt.figure(figsize=(10, 6))
        plotted_frames = None
        for i in group_labels:
            if i not in df_all_mean:
                continue
            df_class = df_all_mean[i]
            if perc_sel_str not in df_class:
                continue
            plotted_frames = df_class['frame']
            # plot class using the color defined in utils/plotting/class_colors.py
            class_color = colors_per_class1_names.get(str(i))
            if class_color is not None:
                # plot values normalized between min and max
                if uses_extended_column_stats(var_name):
                    plot_values = df_class[perc_sel_str]
                else:
                    value_range = df_class[perc_sel_str].max() - df_class[perc_sel_str].min()
                    if value_range == 0:
                        plot_values = np.zeros(len(df_class[perc_sel_str]))
                    else:
                        plot_values = (df_class[perc_sel_str] - df_class[perc_sel_str].min()) / value_range
                plt.plot(df_class['frame'], 
                         plot_values, 
                         label=f'Class {i}', 
                         linewidth=3,
                         color=class_color)
                if is_primary_value_column(var_name, perc_sel_str):
                    std_column = get_std_column(perc_sel_str)
                    if std_column in df_class:
                        lower_bound = df_class[perc_sel_str] - df_class[std_column]
                        upper_bound = df_class[perc_sel_str] + df_class[std_column]
                        plt.fill_between(df_class['frame'], lower_bound, upper_bound, color=class_color, alpha=0.15)
            else:
                plt.plot(df_class['frame'], 
                         df_class[perc_sel_str], 
                         label=f'Class {i}', 
                         linewidth=3)
        if uses_extended_column_stats(var_name):
            plt.title(get_column_plot_title(var_name, perc_sel_str, group_name), fontsize=20)
        else:
            plt.title(f'Temporal Series of {var_name} {perc_sel_str}th Percentile for {group_name.capitalize()} Classes', fontsize=20)
        plt.xlabel('Frame', fontsize=20)
        # remove upper and right border
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if uses_extended_column_stats(var_name):
            plt.ylabel(get_column_axis_label(var_name, perc_sel_str), fontsize=20)
        else:
            plt.ylabel(f'{var_name} {perc_sel_str}th Percentile', fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        if plotted_frames is None:
            plt.close()
            continue
        plt.xticks(plotted_frames, fontsize=16)
        plt.tight_layout()
        plt_path = os.path.join(output_dir, 
                            f'temporal_series_{perc_sel_str}_{var_name}_{group_name}_classes.png')
        plt.savefig(plt_path, transparent=True)
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
    """Reads the CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_file)

    # if file ends with _debug.csv, remove all lines with frame = -710387
    if csv_file.endswith('_debug.csv'):
        df = df[df['frame'] != -710387]
    if df.empty:
        print(f"Warning: {csv_file} is empty after filtering.")

    return df


def get_percentile_columns(df: pd.DataFrame) -> list[str]:
    """Return percentile columns present in the CSV in numeric order."""
    available_columns = [column for column in PERCENTILE_COLUMNS if column in df.columns]
    if not available_columns:
        raise ValueError("No percentile columns found in the CSV file.")
    return available_columns


def get_cma_value_column(df: pd.DataFrame) -> str:
    """Return the column containing the cloud-mask mean for cma files."""
    for column in CMA_VALUE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError("No cma value column found in the CSV file.")


def get_value_columns(df: pd.DataFrame, variable_name: str) -> list[str]:
    """Return the data columns used to compute temporal statistics for the variable."""
    if is_scalar_variable(variable_name):
        return [get_scalar_value_column(df, variable_name)]
    if variable_name == 'precipitation':
        return [column for column in PRECIPITATION_VALUE_COLUMNS if column in df.columns]
    return get_percentile_columns(df)


def normalize_variable_columns(df: pd.DataFrame, variable_name: str) -> pd.DataFrame:
    """Normalize CSV-specific column names for downstream processing."""
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
    """Return True when the variable uses mean/std and optional sparse diagnostics for each data column."""
    return is_scalar_variable(variable_name) or variable_name == 'precipitation'


def get_scalar_value_column(df: pd.DataFrame, variable_name: str) -> str:
    """Return the scalar-value column for variables such as cma and euclid_msg_grid."""
    expected_column = SCALAR_VARIABLE_COLUMNS[variable_name]
    if expected_column in df.columns:
        return expected_column
    for column in CMA_VALUE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No scalar value column found in the CSV file for {variable_name}.")


def get_scalar_value_column_name(variable_name: str) -> str:
    """Return the normalized scalar-value column name for the variable."""
    return SCALAR_VARIABLE_COLUMNS[variable_name]


def get_std_column(column_name: str) -> str:
    """Return the standard-deviation column name for a base data column."""
    return f'{column_name}_std'


def get_sparse_diagnostic_column(variable_name: str, column_name: str | None = None) -> str | None:
    """Return the sparse-aware diagnostic column for variables that need one."""
    if variable_name == 'precipitation' and column_name is not None:
        return f'{column_name}_nonzero_fraction'
    return SPARSE_AWARE_DIAGNOSTIC_COLUMNS.get(variable_name)


def get_conditional_mean_column(variable_name: str, column_name: str | None = None) -> str | None:
    """Return the conditional mean diagnostic column for sparse variables."""
    if variable_name == 'precipitation' and column_name is not None:
        return f'{column_name}_mean_nonzero'
    return CONDITIONAL_MEAN_DIAGNOSTIC_COLUMNS.get(variable_name)


def build_extended_stats_aggregation(variable_name: str, value_columns: list[str]) -> dict:
    """Build the aggregation mapping for variables with std and sparse-aware diagnostics."""
    aggregation = {}
    for value_column in value_columns:
        aggregation[value_column] = (value_column, 'mean')
        aggregation[get_std_column(value_column)] = (value_column, 'std')
        sparse_diagnostic_column = get_sparse_diagnostic_column(variable_name, value_column)
        if sparse_diagnostic_column is not None:
            aggregation[sparse_diagnostic_column] = (value_column, lambda values: (values > 0).mean())
        conditional_mean_column = get_conditional_mean_column(variable_name, value_column)
        if conditional_mean_column is not None:
            aggregation[conditional_mean_column] = (
                value_column,
                lambda values: values[values > 0].mean() if (values > 0).any() else 0.0,
            )
    return aggregation


def get_plot_columns(variable_name: str, value_columns: list[str]) -> list[str]:
    """Return the ordered list of data and diagnostic series to plot."""
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
    """Return the ordered list of columns whose gradients should be saved."""
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
    """Return True when the plotted column is a base data column with a std band."""
    if is_scalar_variable(variable_name):
        return column_name == get_scalar_value_column_name(variable_name)
    if variable_name == 'precipitation':
        return column_name in PRECIPITATION_VALUE_COLUMNS
    return False


def get_scalar_series_label(variable_name: str) -> str:
    """Return a human-readable label for scalar time series plots."""
    if variable_name == 'cma':
        return 'cloud mask'
    if variable_name == 'euclid_msg_grid':
        return 'lightning counts'
    return variable_name


def get_column_plot_title(variable_name: str, column_name: str, group_name: str) -> str:
    """Return the plot title for supported variables and diagnostics."""
    if is_scalar_variable(variable_name):
        if column_name == get_sparse_diagnostic_column(variable_name, column_name):
            return f'Temporal Series of Non-Zero {get_scalar_series_label(variable_name)} Fraction for {group_name.capitalize()} Classes'
        if column_name == get_conditional_mean_column(variable_name, column_name):
            return f'Temporal Series of Conditional Mean {get_scalar_series_label(variable_name)} for {group_name.capitalize()} Classes'
        return f'Temporal Series of Mean {get_scalar_series_label(variable_name)} for {group_name.capitalize()} Classes'

    if variable_name == 'precipitation':
        if column_name.endswith('_nonzero_fraction'):
            base_column = strip_suffix(column_name, '_nonzero_fraction')
            return f'Temporal Series of Non-Zero Fraction for {format_column_label(variable_name, base_column)} in {group_name.capitalize()} Classes'
        if column_name.endswith('_mean_nonzero'):
            base_column = strip_suffix(column_name, '_mean_nonzero')
            return f'Temporal Series of Conditional Mean for {format_column_label(variable_name, base_column)} in {group_name.capitalize()} Classes'
        return f'Temporal Series of Mean {format_column_label(variable_name, column_name)} for {group_name.capitalize()} Classes'

    return f'Temporal Series of {variable_name} {column_name} for {group_name.capitalize()} Classes'


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
    main(variable_name=args.var)

