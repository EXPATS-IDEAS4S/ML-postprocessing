
import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import pdb
import sys
from array import array
sys.path.append(os.path.abspath("/Users/claudia/Documents/ML-postprocessing"))
from utils.plotting.class_colors import colors_per_class1_names, class_groups



# csv file with 2D+1 output
csv_file = '/Users/claudia/Documents/data_ml_spacetime/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000_debug.csv'
output_dir = '/Users/claudia/Documents/data_ml_spacetime/figs/'

def main():

    # read as arrays the content of the .py files with mean gradients for each class and each percentile
    file_path_cot = os.path.join(output_dir, f'mean_gradients_cot.npy')
    file_path_cth = os.path.join(output_dir, f'mean_gradients_cth.npy')
    file_path_cma = os.path.join(output_dir, f'mean_gradients_cma.npy')


    mean_grad_class_cot = read_gradient_files(file_path_cot)
    mean_grad_class_cth = read_gradient_files(file_path_cth)
    mean_grad_class_cma = read_gradient_files(file_path_cma)

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
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gradient_scatter_cma_cth_perc{perc}.png'), dpi=300)
    plt.close()

    class_name = 'Convective'
    class_ids = [2, 3, 4]
    plot_scatter_gradcth_gradcma_all_perc_grouped_class(mean_grad_class_cth, mean_grad_class_cma, class_ids, class_name)
    class_name = 'Overcast'
    class_ids = [5, 6, 7]
    plot_scatter_gradcth_gradcma_all_perc_grouped_class(mean_grad_class_cth, mean_grad_class_cma, class_ids, class_name)
    class_name = 'Broken'
    class_ids = [0, 1, 8]
    plot_scatter_gradcth_gradcma_all_perc_grouped_class(mean_grad_class_cth, mean_grad_class_cma, class_ids, class_name)

def plot_scatter_gradcth_gradcma_all_perc_grouped_class(mean_grad_class_cth, mean_grad_class_cma, class_ids, class_name):
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
    percentiles = ['25', '50', '75', '99']
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
    # set less xticks on x axis to avoid that they overlap
    plt.xticks(np.arange(-0.005, 0.005, 0.0025), fontsize=18)
    plt.yticks(fontsize=18)
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # fix x and y lims
    plt.xlim(-0.005, 0.005)
    plt.ylim(-30, 35)
    #plt.title(f'Mean Gradients of cloud cover vs CTH for Selected Classes', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{class_name}_gradient_scatter_cma_cth.png'), dpi=300, transparent=True)
    plt.close()
    return

def read_gradient_files(file_path):

    arr = np.load(file_path)

    return arr

if __name__ == "__main__":
    main()