
import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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


def main():


    # read csv file
    output_dir = '/sat_data/output/grl_2026/figs/'

    # read as arrays the content of the .py files with mean gradients for each class and each percentile
    file_path_cot = os.path.join(output_dir, f'mean_gradients_cot.npy')
    file_path_cth = os.path.join(output_dir, f'mean_gradients_cth.npy')
    file_path_cma = os.path.join(output_dir, f'mean_gradients_cma.npy')


    mean_grad_class_cot = read_gradient_files(file_path_cot)
    mean_grad_class_cth = read_gradient_files(file_path_cth)
    mean_grad_class_cma = read_gradient_files(file_path_cma)
    shared_limits = compute_shared_limits(mean_grad_class_cth, mean_grad_class_cma)

    perc = '50'
    # plot gradients of cma and 50th perc cth for each class using colors from colors_per_class1_names
    plt.figure(figsize=(8, 6))
    colors = list(colors_per_class1_names.values())
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    for j in range(mean_grad_class_cot.shape[0]):
        plt.scatter(mean_grad_class_cma[j],
                     mean_grad_class_cth[j, 1],
                     color=colors[j % len(colors)],
                     marker=markers[j % len(markers)],
                     label=f'Class {j}',
                       s=200)   
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.xlabel('Mean Gradient CMA', fontsize=14)
    plt.ylabel('Mean Gradient CTH', fontsize=14)
    plt.title(f'Mean Gradients of CMA vs CTH for {perc}th Percentile', fontsize=16)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    apply_shared_limits(plt.gca(), shared_limits)
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gradient_scatter_cma_cth_perc{perc}.png'), dpi=300)
    plt.close()

    # read class name and class_ids from class_groups
    for class_name, class_ids in class_groups.items():
        print(f"Plotting gradients for class group: {class_name} with class ids: {class_ids}")
        plot_scatter_gradcth_gradcma_all_perc_grouped_class(
            mean_grad_class_cth,
            mean_grad_class_cma,
            class_ids,
            class_name,
            output_dir,
            shared_limits,
        )



def plot_scatter_gradcth_gradcma_all_perc_grouped_class(mean_grad_class_cth, mean_grad_class_cma, class_ids, class_name, output_dir='/sat_data/output/grl_2026/figs/', shared_limits=None):
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

    colors = list(colors_per_class1_names.values())

    # use markers to distinguish percentiles

    markers = ['o', 's', 'X', 'D']
    percentiles = ['25', '50', '75', '95']
    for class_id in class_ids:
        for i, perc in enumerate(percentiles):
            plt.scatter(mean_grad_class_cma[class_id],
                         mean_grad_class_cth[class_id, i],
                         color=colors[class_id % len(colors)],
                         marker=markers[i % len(markers)],
                         # set black edge color
                            edgecolor='black',
                         label=f'Class {class_id} - {perc}th',
                        s=300)
    # construct a legend with three labels for the three classes without the percentiles
    handles_class = []
    labels_class = []
    for class_id in class_ids:
        # assign colors of the classes from colors_per_class1_names
        handles_class.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Class {class_id}',
                          markerfacecolor=colors[class_id % len(colors)], markersize=10))
        labels_class.append(f'Class {class_id}')

    handles_perc = []
    labels_perc = []
    for i, perc in enumerate(percentiles):
        # use the markers defined above
        handles_perc.append(plt.Line2D([0], [0], marker=markers[i], color='w', label=f'{perc}th',
                          markerfacecolor='gray', markeredgecolor='black', markersize=10))
        labels_perc.append(f'{perc}th')
    
    # combine labels to create a single legend
    handles_combined = handles_class + handles_perc
    labels_combined = labels_class + labels_perc

    # plot legend for classes
    plt.legend(handles_combined, 
               labels_combined, 
                 bbox_to_anchor=(1.05, 1),
                   loc='upper left', fontsize=16,
                   frameon=False)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.xlabel('Mean Gradient cloud cover', fontsize=18)
    plt.ylabel('Mean Gradient CTH', fontsize=18)
    if shared_limits is not None:
        apply_shared_limits(plt.gca(), shared_limits)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.title(f'Mean Gradients of cloud cover vs CTH for Selected Classes', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{class_name}_gradient_scatter_cma_cth.png'), dpi=300, transparent=True)
    plt.close()
    return


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

    arr = np.load(file_path)

    return arr

if __name__ == "__main__":
    main()