"""
Code to analyze gradients across frames for different classes and variables
Author: Claudia Acquistapace
Date: 10 sept 2025

"""


import os
import matplotlib.pyplot as plt
import numpy as np
import sys


# Add the repository root so top-level packages such as `utils` resolve
# regardless of the directory from which this script is launched.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


sys.path.append(os.path.abspath("/Users/claudia/Documents/ML-postprocessing"))
from utils.plotting.class_colors import colors_per_class1_names, class_groups



# directory where the .npy files with mean gradients for each class and each percentile are stored
output_dir = '/sat_data/output/grl_2026/figs/'


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
    mean_grad_precip_50_nonzero = read_named_gradient_series(
        'precipitation',
        '50_nonzero_fraction',
        output_dir,
    )
    mean_grad_lightning_nonzero = read_named_gradient_series(
        'euclid_msg_grid',
        'lightning_nonzero_fraction',
        output_dir,
    )

    print("Mean gradients of CTH for each class and each percentile:")
    print(mean_grad_class_cth)
    print("Mean gradients of COT for each class and each percentile:")
    print(mean_grad_class_cot)
    print("Mean gradients of CMA for each class and each percentile:")
    print(mean_grad_class_cma)


    # plot scatter plot of cot vs cth mean gradients for each class and each percentile
    percentiles = ['25', '50', '75', '95']
    plt.figure(figsize=(10, 8))

    # one subplot for each percentile
    colors = list(colors_per_class1_names.values())
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    for i, perc in enumerate(percentiles):
        plt.subplot(2, 2, i + 1)
        for j in range(mean_grad_class_cot.shape[0]):
            plt.scatter(mean_grad_class_cot[j, i],
                         mean_grad_class_cth[j, i],
                         color=colors[j % len(colors)],
                         marker=markers[j % len(markers)],
                         label=f'Class {j}',
                           s=100)
        plt.title(f'Percentile {perc}th', fontsize=16)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.xlabel('Mean Gradient of COT', fontsize=16)
        plt.ylabel('Mean Gradient of CTH', fontsize=16)
        # remove upper and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # enlarge fonts of all texts
        plt.rcParams.update({'font.size': 16})
    plt.suptitle('Mean Gradients of COT vs CTH for all Classes', fontsize=20)
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(os.path.join(output_dir, 'mean_gradients_cot_vs_cth.png'), transparent=True)
    plt.close()

    # plot scatter plot of 50th and 75th percentiles for group of classes (Convection, Overcast, Broken Clouds)
    plt.figure(figsize=(10, 8))
    for j in range(mean_grad_class_cot.shape[0]):
        # plot colors for class groups from class_colors.py
        plt.scatter(mean_grad_class_cot[j, 1],
                    mean_grad_class_cth[j, 1],
                    color=colors[j % len(colors)],
                    marker=markers[j % len(markers)],
                    label=f'Class {j}',
                    s=100)
    plt.title(f'Percentile {perc}th', fontsize=16)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Mean Gradient of COT', fontsize=16)
    plt.ylabel('Mean Gradient of CTH', fontsize=16)

    plt.figure(figsize=(10, 8))
    colors = list(colors_per_class1_names.values())
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    for class_index in range(len(mean_grad_precip_50_nonzero)):
        plt.scatter(
            mean_grad_precip_50_nonzero[class_index],
            mean_grad_lightning_nonzero[class_index],
            color=colors[class_index % len(colors)],
            marker=markers[class_index % len(markers)],
            label=f'Class {class_index}',
            s=100,
        )
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('Mean Gradient of Precipitation 50th Non-Zero Fraction', fontsize=16)
    plt.ylabel('Mean Gradient of Lightning Non-Zero Fraction', fontsize=16)
    plt.title('Mean Gradients of Precipitation vs Lightning Non-Zero Fraction', fontsize=20)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(
        os.path.join(output_dir, 'mean_gradients_precipitation50_nonzero_vs_lightning_nonzero.png'),
        transparent=True,
    )
    plt.close()

def read_gradient_files(file_path):

    arr = np.load(file_path)

    return arr


def read_named_gradient_series(variable_name, column_name, output_dir):
    """Read one named gradient series from the saved full-gradient bundles."""
    gradient_path = os.path.join(output_dir, f'mean_gradients_all_{variable_name}.npy')
    columns_path = os.path.join(output_dir, f'mean_gradients_all_{variable_name}_columns.npy')

    gradients = np.load(gradient_path, allow_pickle=True)
    columns = np.load(columns_path, allow_pickle=True).tolist()

    if column_name not in columns:
        available_columns = ', '.join(columns)
        raise ValueError(f"Column '{column_name}' not found for {variable_name}. Available columns: {available_columns}")

    return gradients[:, columns.index(column_name)]


if __name__ == "__main__":
    main()


