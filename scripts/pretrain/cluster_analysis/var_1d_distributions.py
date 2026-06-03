"""
Plot class-wise histograms for 1D crop statistics.

This script reads one of the per-variable CSV files produced under
/sat_data/output/grl_2026/csv/ and generates density histograms for:
1. predefined class groups
2. each individual class

The plotted variable is selected through `main(variable_name=..., percentile=...)`
or through the CLI arguments `--var` and `--percentile`.

Variable display metadata are loaded from:
/home/claudia/codes/ML_postprocessing/configs/variables_metadata.yaml

The metadata file provides, for the selected variable:
- `long_name` for the x-axis label
- `unit` for the x-axis label
- `vmin` and `vmax` for the x-axis limits and histogram bins
- `categorical` to decide whether integer-like categorical bins should be used

Current input data pattern:
/sat_data/output/grl_2026/csv/crops_stats_var-{variable_name}_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv

Current output directory:
/sat_data/output/grl_2026/figs/

Generated outputs:
- grouped histograms:
    distribution_{variable_name}_{percentile}th_perc_{class_name}_classes.png
- per-class histograms:
    distribution_{variable_name}_{percentile}th_perc_class_{class_id}.png

Implementation note:
- `cth` values in the CSV are currently stored in meters, while the metadata file
    defines `cth` in km. This script converts `cth` values to km before plotting so
    the plotted values match the metadata-derived axis limits and labels.

Author: Claudia Acquistapace
Date: 13 sept 2025, modified 3rd June 2026

example call:
python var_1d_distributions.py --var cth --percentile 50

"""

import argparse
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add the repository root so top-level packages such as `utils` resolve
# regardless of the directory from which this script is launched.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utils.configs import load_config
from utils.plotting.class_colors import colors_per_class1_names, class_groups
from scripts.pretrain.cluster_analysis.var_class_temporal_series import read_csv_to_dataframe


VARIABLE_METADATA_PATH = Path(REPO_ROOT) / "configs" / "variables_metadata.yaml"
CTH_SCALE_TO_KM = 0.001
HISTOGRAM_LINEWIDTH = 3.0


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


def get_value_scale(variable_name: str, metadata: dict) -> float:
    # The cth CSV currently stores values in meters while metadata is defined in km.
    if variable_name == "cth" and metadata.get("unit") == "km":
        return CTH_SCALE_TO_KM
    return 1.0


def format_axis_label(metadata: dict) -> str:
    long_name = metadata.get("long_name") or "variable"
    unit = metadata.get("unit")
    if unit:
        return f"{long_name} [{unit}]"
    return long_name


def build_histogram_bins(metadata: dict):
    vmin = metadata.get("vmin")
    vmax = metadata.get("vmax")

    if metadata.get("categorical") and vmin is not None and vmax is not None:
        return np.arange(vmin, vmax + 2) - 0.5

    if vmin is not None and vmax is not None:
        return np.linspace(vmin, vmax, 51)

    return 50



def main(variable_name: str = "cth", percentile: str = "50"):
    metadata = load_variable_metadata(variable_name)
    value_scale = get_value_scale(variable_name, metadata)
    x_label = format_axis_label(metadata)
    histogram_bins = build_histogram_bins(metadata)
    x_min = metadata.get("vmin")
    x_max = metadata.get("vmax")

    # read csv file
    output_dir = '/sat_data/output/grl_2026/figs/'

    # read variable to plot and then read the correpsonding csv file
    csv_file = f'/sat_data/output/grl_2026/csv/crops_stats_var-{variable_name}_stats-50-95-25-75_frames-8_timedim_grl_2026_all_240216_imergmin.csv'

    df = read_csv_to_dataframe(csv_file)
    print("Column titles:", df.columns.tolist())
    # select data for the variable of interest
    df_var = df[df['var'] == variable_name]

    # loop on class groups and plot distributions for each class in the group in the same plot
    for class_name, class_ids in class_groups.items():
        print(f"Processing class group: {class_name} with classes {class_ids}")
        plt.figure(figsize=(10, 8))
        # plot distributions of 50th percentile of cth for each class in the group
        for class_id in class_ids:

            # read values of 50th percentile of cth for the class
            df_class = df_var[(df_var['label'] == class_id)]
            values = df_class[percentile].dropna().to_numpy(dtype=float) * value_scale

            if values.size == 0:
                print(f"Skipping class {class_id} in group {class_name}: no values for percentile {percentile}")
                continue

            # plot distribution of 50th percentile of cth for the class
            plt.hist(values,
             bins=histogram_bins,
             density=True,
             histtype='step',
             linewidth=HISTOGRAM_LINEWIDTH,
             label=f'Class {class_id}',
             color=colors_per_class1_names[str(class_id)])
            # add legend outside the plot
            plt.legend(frameon=False, fontsize=18)
            if x_min is not None and x_max is not None:
                plt.xlim(x_min, x_max)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel(f'{x_label} - {percentile}th perc', fontsize=20)
            plt.ylabel('Density', fontsize=20)
            plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
            # remove top and right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            # enlarge fonts of all texts
            plt.rcParams.update({'font.size': 20})
            # make axis thicker
            plt.gca().spines['left'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)
            # make ticks thicker
            plt.gca().tick_params(width=1.5, length=7)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                  f'distribution_{variable_name}_{percentile}th_perc_{class_name}_classes.png'), 
                                  dpi=300, 
                                  transparent=True)
        plt.close()

    # add loop on individual classes and plot distributions for each class separately
    for class_id in range(15):
        print(f"Processing class: {class_id}")
        df_class = df_var[(df_var['label'] == class_id)]
        values = df_class[percentile].dropna().to_numpy(dtype=float) * value_scale

        if values.size == 0:
            print(f"Skipping class {class_id}: no values for percentile {percentile}")
            continue

        plt.figure(figsize=(10, 8))
        # plot distributions of 50th percentile of cth for the class
        plt.hist(values,
            bins=histogram_bins,
            density=True,
            histtype='step',
            linewidth=HISTOGRAM_LINEWIDTH,
            color=colors_per_class1_names[str(class_id)])
        if x_min is not None and x_max is not None:
            plt.xlim(x_min, x_max)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(f'{x_label} - {percentile}th perc', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        # remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # enlarge fonts of all texts
        plt.rcParams.update({'font.size': 20})
        # make axis thicker
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        # make ticks thicker
        plt.gca().tick_params(width=1.5, length=7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                  f'distribution_{variable_name}_{percentile}th_perc_class_{class_id}.png'), 
                                  dpi=300, 
                                  transparent=True)
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(variable_name=args.var, percentile=args.percentile)