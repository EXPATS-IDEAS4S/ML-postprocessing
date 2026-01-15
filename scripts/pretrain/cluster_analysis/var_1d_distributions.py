"""
plot distributions of 1d variables for each class
Author: Claudia Acquistapace
Date: 13 sept 2025

"""

import sys
import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import pdb
from array import array 

sys.path.append(os.path.abspath("/Users/claudia/Documents/ML-postprocessing"))
from utils.plotting.class_colors import colors_per_class1_names, class_groups
from scripts.pretrain.cluster_analysis.var_class_temporal_series import read_csv_to_dataframe

def main():

    # read csv file
    csv_file = '/Users/claudia/Documents/data_ml_spacetime/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000_debug.csv' 
    output_dir = '/Users/claudia/Documents/data_ml_spacetime/figs/'
    df = read_csv_to_dataframe(csv_file)
    print("Column titles:", df.columns.tolist())


    # read all variables of interest in the file
    var = ['cth']
    var_string =['Cloud Top Height [m]']

    # select data for the variable of interest
    df_var = df[df['var'] == var[0]]

    # loop on class groups and plot distributions for each class in the group in the same plot
    for class_name, class_ids in class_groups.items():
        print(f"Processing class group: {class_name} with classes {class_ids}")
        plt.figure(figsize=(10, 8))
        # plot distributions of 50th percentile of cth for each class in the group
        for class_id in class_ids:

            # read values of 50th percentile of cth for the class
            df_class = df_var[(df_var['label'] == class_id)]
            values = df_class['50'].values

            # plot distribution of 50th percentile of cth for the class
            plt.hist(values,  
             bins=50, 
             density=True, 
             alpha=0.5,
            label=f'Class {class_id}',
            color=colors_per_class1_names[str(class_id)])  
            # add legend outside the plot
            plt.legend(frameon=False, fontsize=18)
            plt.xlim(0., 12500.)
            plt.ylim(0., 0.0005)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel(f'{var_string[0]}- 50th perc', fontsize=20)
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
                                  f'distribution_{var[0]}_50th_perc_{class_name}_classes.png'), 
                                  dpi=300, 
                                  transparent=True)

if __name__ == "__main__":
    main()  