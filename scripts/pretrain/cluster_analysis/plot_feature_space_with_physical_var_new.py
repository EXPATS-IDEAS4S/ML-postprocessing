"""
Description:
------------
This script visualizes the distribution of selected physical variables 
(e.g., cloud fraction, ice cloud fraction, precipitation) over a 2D embedding 
obtained from a dimensionality reduction method such as t-SNE.

It reads a merged CSV containing T-SNE components, labels, and variable values,
and produces class-wise contour plots for each variable. Interpolation and KDE
filtering are applied to generate smooth visualizations.

Features:
- Modular functions for grid interpolation and plotting.
- Optional subsampling based on distance.
- Colorbars with consistent min/max ranges across classes.
- Saves output plots per class to a specified directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
import cmcrameri.cm as cmc

# Add custom functions path
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))
from utils.buckets.aux_functions_from_buckets import get_variable_info

# -----------------------------
# Configuration
# -----------------------------
CONFIG = {
    "reduction_method": "tsne",
    "run_name": "dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA",
    "random_state": "3",
    "sampling_type": "all",  # Options: 'random', 'closest', 'farthest', 'all'
    "variables_to_plot": ["cma-None", "cph-None"],
    "output_base_dir": "/data1/fig/",
    "n_subsample": None,  # None for no subsampling
    "grid_res": 100,      # Grid resolution for interpolation
    "colormap": cmc.imola
}

# -----------------------------
# Helper Functions
# -----------------------------
def load_merged_csv(config):
    """Load the merged CSV containing T-SNE components, labels, and variables."""
    path = os.path.join(
        config["output_base_dir"],
        config["run_name"],
        config["sampling_type"],
        f'merged_tsne_variables_{config["run_name"]}_{config["sampling_type"]}_{config["random_state"]}.csv'
    )
    df = pd.read_csv(path)
    print(f"Loaded merged dataframe with {len(df)} samples and labels: {df['label'].unique()}")
    return df

def create_output_dir(variable_name, config):
    """Create directory for saving plots for a specific variable."""
    if config["n_subsample"]:
        output_dir = os.path.join(config["output_base_dir"], config["run_name"], config["sampling_type"], f"{variable_name}_{config['n_subsample']}")
    else:
        output_dir = os.path.join(config["output_base_dir"], config["run_name"], config["sampling_type"], variable_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def interpolate_variable(df_class, variable, grid_res):
    """Interpolate variable values onto a uniform grid for contour plotting."""
    grid_x, grid_y = np.mgrid[
        df_class['Component_1'].min():df_class['Component_1'].max():grid_res*1j,
        df_class['Component_2'].min():df_class['Component_2'].max():grid_res*1j
    ]
    grid_z = griddata(
        (df_class['Component_1'], df_class['Component_2']),
        df_class[variable],
        (grid_x, grid_y),
        method='linear',
        fill_value=np.nan
    )
    return grid_x, grid_y, grid_z

def plot_class_distribution(df_class, variable, info_var, vmin, vmax, output_file, cmap, grid_res):
    """Plot a single class's variable distribution as a contour plot."""
    grid_x, grid_y, grid_z = interpolate_variable(df_class, variable, grid_res)
    if np.isnan(grid_z).all():
        print(f"Skipping class {df_class['label'].iloc[0]} due to all NaN after interpolation.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    if variable.startswith('cot'):
        contour = ax.contourf(grid_x, grid_y, grid_z, alpha=0.6, cmap=cmap, levels=np.linspace(vmin, 20, 100), extend='max')
    elif variable.startswith('precipitation'):
        contour = ax.contourf(grid_x, grid_y, grid_z, alpha=0.6, cmap=cmap, levels=np.linspace(vmin, 10, 100), extend='max')
    else:
        contour = ax.contourf(grid_x, grid_y, grid_z, alpha=0.6, cmap=cmap, levels=np.linspace(vmin, vmax, 100))

    ax.set_title(f"Class {int(df_class['label'].iloc[0])}", fontsize=14)
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)

    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05)
    if info_var['unit'] == 'None':
        cbar.set_label(f"{info_var['long_name']}", fontsize=12)
    else:
        cbar.set_label(f"{info_var['long_name']} ({info_var['unit']})", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot for class {df_class['label'].iloc[0]}: {output_file}")

# -----------------------------
# Main Execution
# -----------------------------
def main(config):
    df_merged = load_merged_csv(config)

    for variable in config["variables_to_plot"]:
        stat = variable.split('-')[1]
        var_name = variable.split('-')[0]
        info_var = get_variable_info(var_name)
        vmin, vmax = df_merged[variable].min(), df_merged[variable].max()
        output_dir = create_output_dir(variable, config)

        for class_label in df_merged['label'].unique():
            df_class = df_merged[df_merged['label'] == class_label]

            if config["n_subsample"]:
                df_class = df_class.nlargest(config["n_subsample"], 'distance')

            df_class = df_class.dropna(subset=['Component_1', 'Component_2', variable])
            if df_class.empty:
                print(f"Skipping class {class_label} due to missing data.")
                continue

            output_file = os.path.join(output_dir, f"{config['run_name']}_class_{int(class_label)}_{variable}_plot.png")
            plot_class_distribution(df_class, variable, info_var, vmin, vmax, output_file, config["colormap"], config["grid_res"])

if __name__ == "__main__":
    main(CONFIG)
