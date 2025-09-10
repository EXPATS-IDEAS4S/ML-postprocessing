"""
Code to analyze gradients across frames for different classes and variables
Author: Claudia Acquistapace
Date: 10 sept 2025

"""


import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import pdb
from array import array

# csv file with 2D+1 output
csv_file = '/Users/claudia/Documents/data_ml_spacetime/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000.csv'
output_dir = '/Users/claudia/Documents/data_ml_spacetime/figs/'

def main():

    # read as arrays the content of the .py files with mean gradients for each class and each percentile
    file_path_cot = os.path.join(output_dir, f'mean_gradients_cot.npy')
    file_path_cth = os.path.join(output_dir, f'mean_gradients_cth.npy')
    file_path_cma = os.path.join(output_dir, f'mean_gradients_cma.npy')


    mean_grad_class_cot = read_gradient_files(file_path_cot)
    mean_grad_class_cth = read_gradient_files(file_path_cth)
    mean_grad_class_cma = read_gradient_files(file_path_cma)
    print("Mean gradients of CTH for each class and each percentile:")
    print(mean_grad_class_cth)
    print("Mean gradients of COT for each class and each percentile:")
    print(mean_grad_class_cot)
    print("Mean gradients of CMA for each class and each percentile:")
    print(mean_grad_class_cma)


    # plot scatter plot of cot vs cth mean gradients for each class and each percentile
    percentiles = ['25', '50', '75', '99']
    plt.figure(figsize=(10, 8))

    # one subplot for each percentile
    # colors based on class
    colors = ['darkgray', 'darkslategrey', 'peru', 'orangered', 'lightcoral',
              'gold', 'yellowgreen', 'limegreen', 'deepskyblue', 'navy']
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
        plt.title(f'Percentile {perc}')
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.xlabel('Mean Gradient of COT')
        plt.ylabel('Mean Gradient of CTH')
    plt.suptitle('Mean Gradients of COT vs CTH for all Classes')
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig(os.path.join(output_dir, 'mean_gradients_cot_vs_cth.png'))
    plt.close()

def read_gradient_files(file_path):

    arr = np.load(file_path)

    return arr


if __name__ == "__main__":
    main()


