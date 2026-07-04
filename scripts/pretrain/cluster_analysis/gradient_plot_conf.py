
"""Plot class-wise mean-gradient scatter diagnostics.

Reads:
- Mean-gradient NPZ files from /sat_data/output/grl_2026/npz/:
  mean_gradients_cot.npz, mean_gradients_cth.npz, mean_gradients_cma.npz,
  mean_gradients_precipitation.npz, and mean_gradients_euclid_msg_grid.npz.
- Older all-percentile NPY gradient files from /sat_data/output/grl_2026/figs/
  for grouped CTH/COT/cloud cover scatter plots.

Outputs:
- Scatter-plot PNG files saved to /sat_data/output/grl_2026/figs/.
- Current precipitation-lightning outputs include sum[mm] vs lightning_count
  and prec_fraction vs lightning_mean_nonzero.

What it does:
- Loads mean gradient values per cloud class.
- Selects specific gradient columns either by position or by NPZ column name.
- Creates scatter plots comparing cloud, precipitation, and lightning gradient
  diagnostics across classes, with shared axis limits where useful.

Output files:
- gradient_scatter_cma_cth_perc50.png: CTH vs cloud cover gradients for the 50th percentile.
- gradient_scatter_cot_cth_perc50.png: CTH vs COT gradients for the 50th percentile.
- {class_name}_gradient_scatter_cma_cth.png: CTH vs cloud cover gradients across all percentiles for grouped classes.
- gradient_scatter_prec_sum_lightning_count.png: Precipitation sum vs lightning count gradients.
- gradient_scatter_prec_fraction_lightning_mean_nonzero.png: Precipitation fraction vs lightning mean non-zero gradients.


How to call it
- activate vissl env with conda activate vissl
- Run from the repository root with:
  python scripts/pretrain/cluster_analysis/gradient_plot_conf.py
- Select which dataset split to plot with --mode:
  python scripts/pretrain/cluster_analysis/gradient_plot_conf.py --mode train
  python scripts/pretrain/cluster_analysis/gradient_plot_conf.py --mode test
  python scripts/pretrain/cluster_analysis/gradient_plot_conf.py --mode both
- The default is --mode both, which overlays training and testing points.
- Ensure the required NPZ/NPY files are in place and that the output directory is writable.
Notes:
- The script uses colors defined in utils.plotting.class_colors for class-wise coloring.
- Fit lines are added for class groups defined in class_groups, with colors from FIT_COLORS_BY_GROUP.
- Axis limits are computed to be shared across classes and percentiles for better visual comparison.
- The script is modular, with helper functions for loading data, preparing fit lines, and plotting, making it easier to extend for additional variables or diagnostics in the future.
- The script assumes that the mean gradient NPZ files contain arrays of shape (num_classes, num_percentiles) or (num_classes,) and that the columns metadata is correctly stored for NPZ files that include it.
- The script also assumes that the class IDs in the mean gradient arrays correspond to those defined in colors_per_class1_names and class_groups for consistent coloring and grouping in the plots.

Author: Claudia Acquistapace
date: 2026-06-08


"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import xarray as xr
import pandas as pd
import pdb
import sys
from array import array
import argparse
from pathlib import Path
from typing import Tuple

import sys
import os
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

from utils.plotting.class_colors import colors_per_class1_names, class_groups

VARIABLE_METADATA_PATH = Path(REPO_ROOT) / "configs" / "variables_metadata.yaml"
CTH_SCALE_TO_KM = 0.001
HISTOGRAM_LINEWIDTH = 3.0
AUTO_VMAX_QUANTILE = 0.995
VALUE_COLUMN_ALIASES = {
    "sum": "sum[mm]",
}
PERCENTILE_COLUMNS = ("25", "50", "75", "95", "99")
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
FIT_COLORS_BY_GROUP = {
    "Convection": "red",
    "Overcast": "blue",
    "Broken Clouds": "green",
}
TRAIN_MARKER = "o"
TEST_MARKER = "^"
LEGEND_GREY = "0.4"
SCATTER_SIZE = 200

from utils.plotting.class_colors import colors_per_class1_names, class_groups
from utils.configs import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot class-wise gradient scatter diagnostics.")
    parser.add_argument("--var", default="cth", help="Variable name as defined in variables_metadata.yaml")
    parser.add_argument("--percentile", default="50", help="Percentile column to plot for continuous variables")
    parser.add_argument(
        "--mode",
        choices=("train", "test", "both"),
        default="both",
        help="Dataset split to plot: train, test, or both.",
    )
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


def main(mode: str = "both"):
    if mode not in {"train", "test", "both"}:
        raise ValueError("mode must be one of: train, test, both")
    plot_train = mode in {"train", "both"}
    plot_test = mode in {"test", "both"}


    # read gradient npz files and save figures
    input_dir = '/sat_data/output/grl_2026/npz/'
    input_dir_npy = '/sat_data/output/grl_2026/figs/'
    output_dir = '/sat_data/output/grl_2026/figs/'

    # read as arrays the content of the .py files with mean gradients for each class and each percentile
    file_path_cot = os.path.join(input_dir, f'mean_gradients_cot.npz')
    file_path_cth = os.path.join(input_dir, f'mean_gradients_cth.npz')
    file_path_cma = os.path.join(input_dir, f'mean_gradients_cma.npz')
    file_path_prec = os.path.join(input_dir, f'mean_gradients_precipitation.npz')
    file_path_lightning = os.path.join(input_dir, f'mean_gradients_euclid_msg_grid.npz')
    file_path_cot_test = os.path.join(input_dir, f'mean_gradients_cot_test.npz')
    file_path_cth_test = os.path.join(input_dir, f'mean_gradients_cth_test.npz')
    file_path_cma_test = os.path.join(input_dir, f'mean_gradients_cma_test.npz')
    file_path_prec_test = os.path.join(input_dir, f'mean_gradients_precipitation_test.npz')
    file_path_lightning_test = os.path.join(input_dir, f'mean_gradients_euclid_msg_grid_test.npz')

    # read the 50th percentile gradients for cth and cma to compute shared axis limits \
    # across classes and percentiles for the scatter plots
    mean_grad_class_cot = read_gradient_files(file_path_cot)
    mean_grad_class_cth = read_gradient_files(file_path_cth)
    mean_grad_class_cma = read_gradient_files(file_path_cma)
    mean_grad_class_prec, prec_columns, _ = read_gradient_npz(file_path_prec)
    mean_grad_class_lightning, lightning_columns, _ = read_gradient_npz(file_path_lightning)
    mean_grad_class_cot_test, _, class_labels_test_cot = read_gradient_npz(file_path_cot_test)
    mean_grad_class_cth_test, _, class_labels_test_cth = read_gradient_npz(file_path_cth_test)
    mean_grad_class_cma_test, _, class_labels_test_cma = read_gradient_npz(file_path_cma_test)
    mean_grad_class_prec_test, prec_columns_test, class_labels_test_prec = read_gradient_npz(file_path_prec_test)
    mean_grad_class_lightning_test, lightning_columns_test, class_labels_test_lightning = read_gradient_npz(file_path_lightning_test)
    perc = '50'

    # extract the 50th percentile gradients for cth, cot, cma, and precipitation for all classes
    mean_grad_cot_50 = get_gradient_column(mean_grad_class_cot, 0)
    mean_grad_cth_50 = get_gradient_column(mean_grad_class_cth, 0)
    mean_grad_cma = get_gradient_column(mean_grad_class_cma, 0)
    mean_grad_prec_sum = get_gradient_column_by_name(mean_grad_class_prec, prec_columns, "sum[mm]")
    mean_grad_prec_fraction = get_gradient_column_by_name(mean_grad_class_prec, prec_columns, "prec_fraction")
    mean_grad_lightning_count = get_gradient_column_by_name(mean_grad_class_lightning, lightning_columns, "lightning_count")
    mean_grad_lightning_mean_nonzero = get_gradient_column_by_name(
        mean_grad_class_lightning,
        lightning_columns,
        "lightning_mean_nonzero",
    )
    mean_grad_cot_50_test = get_gradient_column(mean_grad_class_cot_test, 0)
    mean_grad_cth_50_test = get_gradient_column(mean_grad_class_cth_test, 0)
    mean_grad_cma_test = get_gradient_column(mean_grad_class_cma_test, 0)
    mean_grad_prec_sum_test = get_gradient_column_by_name(mean_grad_class_prec_test, prec_columns_test, "sum[mm]")
    mean_grad_prec_fraction_test = get_gradient_column_by_name(mean_grad_class_prec_test, prec_columns_test, "prec_fraction")
    mean_grad_lightning_count_test = get_gradient_column_by_name(mean_grad_class_lightning_test, lightning_columns_test, "lightning_count")
    mean_grad_lightning_mean_nonzero_test = get_gradient_column_by_name(
        mean_grad_class_lightning_test,
        lightning_columns_test,
        "lightning_mean_nonzero",
    )

    shared_limits = compute_shared_limits(
        combine_mode_values(mean_grad_cth_50, mean_grad_cth_50_test, mode),
        combine_mode_values(mean_grad_cma, mean_grad_cma_test, mode),
    )

    # plot gradients of cma and 50th perc cth for each class using colors from colors_per_class1_names
    plt.figure(figsize=(8, 6))
    if plot_train:
        for j in range(len(mean_grad_cth_50)):
            plt.scatter(mean_grad_cma[j],
                         mean_grad_cth_50[j],
                         color=get_class_color(j),
                         marker=TRAIN_MARKER,
                           s=SCATTER_SIZE)
    if plot_test:
        scatter_test_points(plt.gca(), mean_grad_cma_test, mean_grad_cth_50_test, class_labels_test_cma)

    fit_handles, fit_labels = ([], [])
    test_fit_handles, test_fit_labels = ([], [])
    if plot_train:
        fit_handles, fit_labels = add_group_fit_lines(plt.gca(), mean_grad_cma, mean_grad_cth_50)
    if plot_test:
        test_fit_handles, test_fit_labels = add_testing_group_fit_lines(
            plt.gca(),
            mean_grad_cma_test,
            mean_grad_cth_50_test,
            class_labels_test_cma,
        )

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.xlabel('Mean Gradient cloud cover', fontsize=14)
    plt.ylabel('Mean Gradient cloud top height', fontsize=14)
    plt.title(f'Mean Gradients of Cloud Cover vs CTH for {perc}th Percentile', fontsize=16)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    apply_shared_limits(plt.gca(), shared_limits)
    add_color_and_symbol_legends(
        plt.gca(),
        range(len(mean_grad_cth_50)),
        fit_handles + test_fit_handles,
        fit_labels + test_fit_labels,
        mode=mode,
    )
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gradient_scatter_cma_cth_perc{perc}.png'), dpi=300)
    plt.close()

    # plot gradients of 50th perc COT and 50th perc cth for each class using colors from colors_per_class1_names
    cot_cth_limits = compute_shared_limits(
        combine_mode_values(mean_grad_cth_50, mean_grad_cth_50_test, mode),
        combine_mode_values(mean_grad_cot_50, mean_grad_cot_50_test, mode),
    )

    plt.figure(figsize=(8, 6))
    if plot_train:
        for j in range(len(mean_grad_cot_50)):
            plt.scatter(mean_grad_cot_50[j],
                         mean_grad_cth_50[j],
                         color=get_class_color(j),
                         marker=TRAIN_MARKER,
                           s=SCATTER_SIZE)
    if plot_test:
        scatter_test_points(plt.gca(), mean_grad_cot_50_test, mean_grad_cth_50_test, class_labels_test_cot)
    fit_handles, fit_labels = ([], [])
    test_fit_handles, test_fit_labels = ([], [])
    if plot_train:
        fit_handles, fit_labels = add_group_fit_lines(plt.gca(), mean_grad_cot_50, mean_grad_cth_50)
    if plot_test:
        test_fit_handles, test_fit_labels = add_testing_group_fit_lines(
            plt.gca(),
            mean_grad_cot_50_test,
            mean_grad_cth_50_test,
            class_labels_test_cot,
        )
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)

    plt.xlabel('Mean Gradient cloud optical thickness', fontsize=14)
    plt.ylabel('Mean Gradient cloud top height', fontsize=14)
    #plt.title(f'Mean Gradients of COT vs CTH for {perc}th Percentile', fontsize=16)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    apply_shared_limits(plt.gca(), cot_cth_limits)
    add_color_and_symbol_legends(
        plt.gca(),
        range(len(mean_grad_cot_50)),
        fit_handles + test_fit_handles,
        fit_labels + test_fit_labels,
        mode=mode,
    )
    # remove top and right spines and make remaining thicker
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gradient_scatter_cot_cth_perc{perc}.png'), dpi=300)
    plt.close()


    # read now old .npy files containing all percentiles for cth and cma and plot them in
    # the same scatter plot for the classes in class_groups using the function defined below
    all_percentiles_file_path_cot = os.path.join(input_dir_npy, f'mean_gradients_cot.npy')
    all_percentiles_file_path_cth = os.path.join(input_dir_npy, f'mean_gradients_cth.npy')
    all_percentiles_file_path_cma = os.path.join(input_dir_npy, f'mean_gradients_cma.npy')
    mean_grad_class_cot = read_gradient_files(all_percentiles_file_path_cot)
    mean_grad_class_cth = read_gradient_files(all_percentiles_file_path_cth)
    mean_grad_class_cma = read_gradient_files(all_percentiles_file_path_cma)    


    # read class name and class_ids from class_groups
    for class_name, class_ids in class_groups.items():
        print(f"Plotting gradients for class group: {class_name} with class ids: {class_ids}")
        plot_scatter_gradcth_gradcma_all_perc_grouped_class(
            mean_grad_class_cth,
            mean_grad_class_cma,
            mean_grad_class_cot,
            mean_grad_class_cth_test,
            mean_grad_class_cma_test,
            mean_grad_class_cot_test,
            class_labels_test_cth,
            class_ids,
            class_name,
            output_dir,
            shared_limits,
            mode,
        )


    plot_class_scatter(
        mean_grad_prec_sum,
        mean_grad_lightning_count,
        x_values_test=mean_grad_prec_sum_test,
        y_values_test=mean_grad_lightning_count_test,
        test_class_labels=class_labels_test_prec,
        xlabel='Mean Gradient precipitation sum [mm]',
        ylabel='Mean Gradient lightning count',
        title='Mean Gradients of Precipitation Sum vs Lightning Count',
        output_path=os.path.join(output_dir, 'gradient_scatter_prec_sum_lightning_count.png'),
        mode=mode,
    )

    plot_class_scatter(
        mean_grad_prec_fraction,
        mean_grad_lightning_mean_nonzero,
        x_values_test=mean_grad_prec_fraction_test,
        y_values_test=mean_grad_lightning_mean_nonzero_test,
        test_class_labels=class_labels_test_prec,
        xlabel='Mean Gradient precipitation fraction',
        ylabel='Mean Gradient lightning mean non-zero',
        title='Mean Gradients of Precipitation Fraction vs Lightning Mean Non-Zero',
        output_path=os.path.join(output_dir, 'gradient_scatter_prec_fraction_lightning_mean_nonzero.png'),
        mode=mode,
    )



def plot_scatter_gradcth_gradcma_all_perc_grouped_class(
    mean_grad_class_cth,
    mean_grad_class_cma,
    mean_grad_class_cot,
    mean_grad_class_cth_test,
    mean_grad_class_cma_test,
    mean_grad_class_cot_test,
    test_class_labels,
    class_ids,
    class_name,
    output_dir='/sat_data/output/grl_2026/figs/',
    shared_limits=None,
    mode="both",
):
    """
    plotting function to create a scatter plot of mean gradients of cth vs cma for selected classes across all percentiles
    Inputs:
    - mean_grad_class_cth: numpy array of shape (num_classes, num_percentiles
    - mean_grad_class_cma: numpy array of shape (num_classes, num_percentiles)
    - class_ids: list of class ids to plot
    - class_name: name of the class group for title and filename
    Outputs:
    - saves a scatter plot figure in output_dir

    """
    # plot for class 2, 3, 4 the gradients of cth across percentiles in the same scatter plot
    plt.figure(figsize=(8, 6))
    # plot for convective class from class_groups

    mean_grad_cma = get_gradient_column(mean_grad_class_cma, 0)
    mean_grad_cma_test = get_gradient_column(mean_grad_class_cma_test, 0)
    mean_grad_class_cth = np.asarray(mean_grad_class_cth)
    mean_grad_class_cth_test = np.asarray(mean_grad_class_cth_test)
    if mean_grad_class_cth.ndim == 1:
        mean_grad_class_cth = mean_grad_class_cth[:, np.newaxis]
    if mean_grad_class_cth_test.ndim == 1:
        mean_grad_class_cth_test = mean_grad_class_cth_test[:, np.newaxis]
    if mean_grad_class_cma.ndim == 1:
        mean_grad_class_cma = mean_grad_class_cma[:, np.newaxis]
    if mean_grad_class_cot.ndim == 1:
        mean_grad_class_cot = mean_grad_class_cot[:, np.newaxis]
    if mean_grad_class_cot_test.ndim == 1:
        mean_grad_class_cot_test = mean_grad_class_cot_test[:, np.newaxis]

    percentiles = ['50'] if mean_grad_class_cth.shape[1] == 1 else ['25', '50', '75', '95'][:mean_grad_class_cth.shape[1]]
    if mode in {"train", "both"}:
        for class_id in class_ids:
            for i, perc in enumerate(percentiles):
                plt.scatter(mean_grad_cma[class_id],
                             mean_grad_class_cth[class_id, i],
                             color=get_class_color(class_id),
                             marker=TRAIN_MARKER,
                             # set black edge color
                                edgecolor='black',
                            s=300)
    if mode in {"test", "both"}:
        scatter_test_points_for_group(
            plt.gca(),
            mean_grad_cma_test,
            mean_grad_class_cth_test,
            test_class_labels,
            class_ids,
        )
    fit_handles, fit_labels = ([], [])
    test_fit_handles, test_fit_labels = ([], [])
    if mode in {"train", "both"}:
        fit_handles, fit_labels = add_selected_group_fit_lines(
            plt.gca(),
            mean_grad_cma,
            mean_grad_class_cth,
            [class_name],
        )
    if mode in {"test", "both"}:
        test_fit_handles, test_fit_labels = add_testing_group_fit_lines(
            plt.gca(),
            mean_grad_cma_test,
            mean_grad_class_cth_test,
            test_class_labels,
            [class_name],
        )
    add_color_and_symbol_legends(
        plt.gca(),
        class_ids,
        fit_handles + test_fit_handles,
        fit_labels + test_fit_labels,
        fontsize=12,
        mode=mode,
    )
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.xlabel('Mean Gradient cloud cover', fontsize=18)
    plt.ylabel('Mean Gradient CTH', fontsize=18)
    if shared_limits is not None:
        apply_shared_limits(plt.gca(), shared_limits)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # remove top and right spines and make remaining thicker
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)

    #plt.title(f'Mean Gradients of cloud cover vs CTH for Selected Classes', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{class_name}_gradient_scatter_cma_cth.png'), dpi=300, transparent=True)
    plt.close()


    # do the same plot for cma vs cot gradients for the same classes and percentiles
    plt.figure(figsize=(8, 6))
    if mode in {"train", "both"}:
        for class_id in class_ids:
            for i, perc in enumerate(percentiles):
                plt.scatter(mean_grad_cma[class_id],
                             mean_grad_class_cot[class_id, i],
                             color=get_class_color(class_id),
                             marker=TRAIN_MARKER,
                             edgecolor='black',
                            s=300)
    if mode in {"test", "both"}:
        scatter_test_points_for_group(
            plt.gca(),
            mean_grad_cma_test,
            mean_grad_class_cot_test,
            test_class_labels,
            class_ids,
        )
    fit_handles, fit_labels = ([], [])
    test_fit_handles, test_fit_labels = ([], [])
    if mode in {"train", "both"}:
        fit_handles, fit_labels = add_selected_group_fit_lines(
            plt.gca(),
            mean_grad_cma,
            mean_grad_class_cot,
            [class_name],
        )
    if mode in {"test", "both"}:
        test_fit_handles, test_fit_labels = add_testing_group_fit_lines(
            plt.gca(),
            mean_grad_cma_test,
            mean_grad_class_cot_test,
            test_class_labels,
            [class_name],
        )
    add_color_and_symbol_legends(
        plt.gca(),
        class_ids,
        fit_handles + test_fit_handles,
        fit_labels + test_fit_labels,
        fontsize=12,
        mode=mode,
    )
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.xlabel('Mean Gradient cloud cover', fontsize=18)
    plt.ylabel('Mean Gradient cloud optical thickness', fontsize=18)
    if shared_limits is not None:
        apply_shared_limits(plt.gca(), shared_limits)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # remove top and right spines and make remaining thicker
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)        
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    #plt.title(f'Mean Gradients of cloud cover vs cloud optical thickness for Selected Classes', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{class_name}_gradient_scatter_cma_cot.png'), dpi=300, transparent=True)
    plt.close()

    return


def add_group_fit_lines(ax, x_values, y_values):
    """Add linear and quadratic fit lines for each configured class group."""
    handles = []
    labels = []
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    for group_name, class_ids in class_groups.items():
        color = FIT_COLORS_BY_GROUP.get(group_name, 'black')
        group_handles, group_labels = add_fit_lines(
            ax,
            x_values[class_ids],
            y_values[class_ids],
            group_name,
            color,
        )
        handles.extend(group_handles)
        labels.extend(group_labels)
    return handles, labels


def add_selected_group_fit_lines(ax, x_values, y_values, group_names):
    """Add training fit lines for selected class groups."""
    handles = []
    labels = []
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    for group_name in group_names:
        class_ids = class_groups.get(group_name, [])
        if not class_ids:
            continue
        color = FIT_COLORS_BY_GROUP.get(group_name, 'black')
        group_handles, group_labels = add_fit_lines(
            ax,
            x_values[class_ids],
            y_values[class_ids],
            group_name,
            color,
        )
        handles.extend(group_handles)
        labels.extend(group_labels)
    return handles, labels


def get_class_color(class_id):
    return colors_per_class1_names.get(str(int(class_id)), f"C{int(class_id) % 10}")


def combine_mode_values(train_values, test_values, mode):
    values = []
    if mode in {"train", "both"}:
        values.append(np.asarray(train_values).reshape(-1))
    if mode in {"test", "both"}:
        values.append(np.asarray(test_values).reshape(-1))
    return np.concatenate(values)


def scatter_test_points(ax, x_values, y_values, class_labels):
    for index, class_id in enumerate(class_labels):
        if index >= len(x_values) or index >= len(y_values):
            continue
        ax.scatter(
            x_values[index],
            y_values[index],
            color=get_class_color(class_id),
            marker=TEST_MARKER,
            s=SCATTER_SIZE,
        )


def scatter_test_points_for_group(ax, x_values, y_values, class_labels, class_ids):
    class_ids = set(class_ids)
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    for index, class_id in enumerate(class_labels):
        class_id = int(class_id)
        if class_id not in class_ids or index >= len(x_values):
            continue
        y_row = y_values[index]
        if np.ndim(y_row) == 0:
            y_row = [y_row]
        for y_value in y_row:
            ax.scatter(
                x_values[index],
                y_value,
                color=get_class_color(class_id),
                marker=TEST_MARKER,
                s=300,
                edgecolor='black',
            )


def add_testing_group_fit_lines(ax, x_values, y_values, class_labels, group_names=None):
    """Add dotted linear fits through testing points for selected class groups."""
    handles = []
    labels = []
    group_names = list(group_names or class_groups.keys())

    for group_name in group_names:
        group_handles, group_labels = add_testing_group_fit_line(
            ax,
            x_values,
            y_values,
            class_labels,
            group_name,
        )
        handles.extend(group_handles)
        labels.extend(group_labels)

    return handles, labels


def add_testing_group_fit_line(ax, x_values, y_values, class_labels, group_name):
    class_ids = set(class_groups.get(group_name, []))
    selected_x = []
    selected_y = []

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    for index, class_id in enumerate(class_labels):
        class_id = int(class_id)
        if class_id not in class_ids or index >= len(x_values) or index >= len(y_values):
            continue

        y_row = y_values[index]
        if np.ndim(y_row) == 0:
            y_row = [y_row]

        for y_value in y_row:
            if np.isfinite(x_values[index]) and np.isfinite(y_value):
                selected_x.append(x_values[index])
                selected_y.append(y_value)

    if len(selected_x) < 2 or len(np.unique(selected_x)) < 2:
        return [], []

    selected_x = np.asarray(selected_x)
    selected_y = np.asarray(selected_y)
    x_fit = np.linspace(np.min(selected_x), np.max(selected_x), 100)
    linear_fit = np.poly1d(np.polyfit(selected_x, selected_y, deg=1))
    color = FIT_COLORS_BY_GROUP.get(group_name, 'black')
    handle, = ax.plot(
        x_fit,
        linear_fit(x_fit),
        color=color,
        linestyle=':',
        linewidth=2.5,
        label=f'Testing {group_name} fit',
    )
    return [handle], [f'Testing {group_name} fit']


def add_color_and_symbol_legends(ax, class_labels, extra_handles=None, extra_labels=None, fontsize=10, mode="both"):
    class_handles = [
        Line2D(
            [0],
            [0],
            marker=TRAIN_MARKER,
            linestyle='None',
            markerfacecolor=get_class_color(class_id),
            markeredgecolor=get_class_color(class_id),
            markersize=8,
            label=f'Class {class_id}',
        )
        for class_id in class_labels
    ]
    class_legend = ax.legend(
        handles=class_handles,
        title='Classes',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=fontsize,
        frameon=False,
        labelspacing=1.0,
        handletextpad=0.8,
        borderaxespad=0.8,
    )
    ax.add_artist(class_legend)

    dataset_handles = []
    dataset_labels = []
    if mode in {"train", "both"}:
        dataset_handles.append(Line2D(
            [0],
            [0],
            marker=TRAIN_MARKER,
            linestyle='None',
            markerfacecolor=LEGEND_GREY,
            markeredgecolor=LEGEND_GREY,
            markersize=8,
            label='Training',
        ))
        dataset_labels.append('Training')
    if mode in {"test", "both"}:
        dataset_handles.append(Line2D(
            [0],
            [0],
            marker=TEST_MARKER,
            linestyle='None',
            markerfacecolor=LEGEND_GREY,
            markeredgecolor=LEGEND_GREY,
            markersize=8,
            label='Testing',
        ))
        dataset_labels.append('Testing')
    handles = dataset_handles + list(extra_handles or [])
    labels = dataset_labels + list(extra_labels or [])
    ax.legend(
        handles,
        labels,
        title='Symbols',
        bbox_to_anchor=(1.05, 0.45),
        loc='upper left',
        fontsize=fontsize,
        frameon=False,
        labelspacing=1.0,
        handletextpad=0.8,
        borderaxespad=0.8,
    )


def add_fit_lines(ax, x_values, y_values, label_prefix, color):
    """Add linear and quadratic fits for the provided x/y values."""
    x_values, y_values = prepare_fit_values(x_values, y_values)
    if len(x_values) == 0:
        return [], []

    x_unique = np.unique(x_values)
    if len(x_unique) < 2:
        return [], []

    x_fit = np.linspace(np.min(x_values), np.max(x_values), 100)
    handles = []
    labels = []

    linear_fit = np.poly1d(np.polyfit(x_values, y_values, deg=1))
    linear_handle, = ax.plot(
        x_fit,
        linear_fit(x_fit),
        color=color,
        linestyle='--',
        linewidth=1.5,
        label=f'{label_prefix} linear fit',
    )
    handles.append(linear_handle)
    labels.append(f'{label_prefix} linear fit')

    if len(x_unique) >= 3:
        quadratic_fit = np.poly1d(np.polyfit(x_values, y_values, deg=2))
        quadratic_handle, = ax.plot(
            x_fit,
            quadratic_fit(x_fit),
            color=color,
            linestyle=':',
            linewidth=0.5,
            label=f'{label_prefix} quadratic fit',
        )
        handles.append(quadratic_handle)
        labels.append(f'{label_prefix} quadratic fit')

    return handles, labels


def prepare_fit_values(x_values, y_values):
    """Return finite 1D x/y arrays, repeating x across percentile columns when needed."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if y_values.ndim == 2 and x_values.ndim == 1:
        x_values = np.repeat(x_values[:, np.newaxis], y_values.shape[1], axis=1)

    x_values = x_values.reshape(-1)
    y_values = y_values.reshape(-1)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[finite_mask], y_values[finite_mask]


def plot_class_scatter(
    x_values,
    y_values,
    xlabel,
    ylabel,
    title,
    output_path,
    x_values_test=None,
    y_values_test=None,
    test_class_labels=None,
    mode="both",
):
    """Plot one point per class for two named mean-gradient columns."""
    limit_x_values = combine_mode_values(x_values, x_values_test, mode)
    limit_y_values = combine_mode_values(y_values, y_values_test, mode)
    shared_limits = compute_shared_limits(limit_y_values, limit_x_values)

    plt.figure(figsize=(8, 6))
    if mode in {"train", "both"}:
        for j in range(len(x_values)):
            plt.scatter(
                x_values[j],
                y_values[j],
                color=get_class_color(j),
                marker=TRAIN_MARKER,
                s=SCATTER_SIZE,
            )
    if mode in {"test", "both"} and x_values_test is not None and y_values_test is not None and test_class_labels is not None:
        scatter_test_points(plt.gca(), x_values_test, y_values_test, test_class_labels)
    fit_handles, fit_labels = ([], [])
    test_fit_handles = []
    test_fit_labels = []
    if mode in {"train", "both"}:
        fit_handles, fit_labels = add_group_fit_lines(plt.gca(), x_values, y_values)
    if mode in {"test", "both"} and x_values_test is not None and y_values_test is not None and test_class_labels is not None:
        test_fit_handles, test_fit_labels = add_testing_group_fit_lines(
            plt.gca(),
            x_values_test,
            y_values_test,
            test_class_labels,
        )
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    apply_shared_limits(plt.gca(), shared_limits)
    add_color_and_symbol_legends(
        plt.gca(),
        range(len(x_values)),
        fit_handles + test_fit_handles,
        fit_labels + test_fit_labels,
        mode=mode,
    )
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def set_axis_limits_with_padding(ax, x_values, y_values, padding_ratio=0.08):
    """Set data-driven axis limits with a small padding so all classes remain visible."""
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    x_span = x_max - x_min
    y_span = y_max - y_min

    x_padding = max(x_span * padding_ratio, 0.001)
    y_padding = max(y_span * padding_ratio, 1.0)

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)


def compute_shared_limits(mean_grad_class_cth, mean_grad_class_cma):
    """Compute one shared set of axis limits across all classes and percentiles."""
    x_values = mean_grad_class_cma.reshape(-1).tolist()
    y_values = mean_grad_class_cth.reshape(-1).tolist()

    fig, ax = plt.subplots()
    set_axis_limits_with_padding(ax, x_values, y_values)
    limits = {
        'xlim': ax.get_xlim(),
        'ylim': ax.get_ylim(),
    }
    plt.close(fig)
    return limits


def apply_shared_limits(ax, shared_limits):
    """Apply precomputed shared axis limits to a scatter plot."""
    ax.set_xlim(*shared_limits['xlim'])
    ax.set_ylim(*shared_limits['ylim'])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

def read_gradient_files(file_path):

    loaded_file = np.load(file_path, allow_pickle=True)
    if isinstance(loaded_file, np.lib.npyio.NpzFile):
        arr = loaded_file['gradients']
        loaded_file.close()
    else:
        arr = loaded_file

    return arr

def read_gradient_npz(file_path):
    """Read gradients and column metadata from a mean-gradient NPZ file."""
    with np.load(file_path, allow_pickle=True) as loaded_file:
        gradients = loaded_file['gradients']
        columns = loaded_file['columns'].tolist()
        class_labels = loaded_file['class_labels']
    return gradients, columns, class_labels

def get_gradient_column(gradients, column_index=0):
    """Return one gradient column as a 1D array."""
    gradients = np.asarray(gradients)
    if gradients.ndim == 1:
        return gradients
    if gradients.ndim == 2:
        return gradients[:, column_index]
    raise ValueError(f"Expected a 1D or 2D gradient array, got shape {gradients.shape}")


def get_gradient_column_by_name(gradients, columns, column_name):
    """Return one gradient column from an NPZ gradients array using its stored column name."""
    if column_name not in columns:
        available_columns = ", ".join(columns)
        raise ValueError(f"Column '{column_name}' not found. Available columns: {available_columns}")
    return get_gradient_column(gradients, columns.index(column_name))


if __name__ == "__main__":
    args = parse_args()
    main(mode=args.mode)
