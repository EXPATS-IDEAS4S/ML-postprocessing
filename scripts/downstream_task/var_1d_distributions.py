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


def main():

    # read csv file
    csv_file = '/data1/fig/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/epoch_50/all/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-5_timedim_coords-datetime_supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1_all_12683.csv' 
    #output_dir = '/data1/fig/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/epoch_50/all/'
    output_dir = '/data1/fig/dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1/downstream_task/'
    df = pd.read_csv(csv_file)
    print("Column titles:", df.columns.tolist())
    #print(df)

    #prediction_file = "/data1/fig/supervised_ir108-cm_75x75_5frames_12k_nc_r2dplus1/epoch_50/all/val_filepaths_true_predicted_labels_features.csv"
    prediction_file = "/data1/runs/dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1/hail_linear_classif/features/epoch_50/val_filepaths_true_predicted_labels_features.csv"
    df_pred = pd.read_csv(prediction_file)
    #from file_path extract the crop
    df_pred['crop'] = df_pred['file_path'].apply(lambda x: os.path.basename(x))
    #change column name of  'true_label (0=hail; 1=no_hail)', 'predicted_label (0=hail; 1=no_hail)' to 'label' and 'predicted_label'
    df_pred = df_pred.rename(columns={'true_label (0=hail; 1=no_hail)': 'true_label', 'predicted_label (0=hail; 1=no_hail)': 'predicted_label'})
    print("Column titles:", df_pred.columns.tolist())
    #print(df_pred)

    #merge the two dataframes on the crop column (the first one has several row corresponding to the same crop, so each value of the second df should be repeated )
    df_merged = df.merge(df_pred[['crop', 'hail_class', 'true_label', 'predicted_label']], on='crop', how='left')
    print("Merged DataFrame:")
    #print(df_merged)
    print(df_merged.columns.tolist())
    
    # read all variables of interest in the file
    # var = ['cma']
    # var_string = ['Cloud Cover']

    var = ['cth']
    var_string =['Cloud Top Height [m]']




    # ----------------------
    # Split categories
    # ----------------------
    df_true0_pred0 = df_merged[(df_merged["true_label"] == 0) & (df_merged["predicted_label"] == 0)]  # TP
    df_true1_pred1 = df_merged[(df_merged["true_label"] == 1) & (df_merged["predicted_label"] == 1)]  # TN
    df_true0_pred1 = df_merged[(df_merged["true_label"] == 0) & (df_merged["predicted_label"] == 1)]  # FN
    df_true1_pred0 = df_merged[(df_merged["true_label"] == 1) & (df_merged["predicted_label"] == 0)]  # FP

    # Group into "correct" and "errors"
    groups = {
        "correct_predictions": (
            [df_true0_pred0, df_true1_pred1],
            ["True Positive (hail)", "True Negative (no_hail)"],
            ["blue", "green"],
        ),
        "errors": (
            [df_true0_pred1, df_true1_pred0],
            ["False Negative (missed hail)", "False Positive (false alarm)"],
            ["orange", "red"],
        ),
    }

    # ----------------------
    # Plot each group
    # ----------------------
    for group_name, (dfs, labels, colors) in groups.items():
        plt.figure(figsize=(10, 8))

        for df_cat, cat_name, color in zip(dfs, labels, colors):
            df_var = df_cat[df_cat['var'] == var[0]]
            #values = df_var['None'].values 
            values = df_var['50'].values
            
            plt.hist(
                values,
                bins=30,
                density=True,
                alpha=0.7,               # transparent bars
                label=cat_name,
                color=color,
                histtype="step",
                linewidth=4,
            )

        # Formatting
        #plt.legend(frameon=False, fontsize=18)
        

        #plt.xlim(0., 1)
        #plt.ylim(0., 18)

        plt.xlim(0., 12500.)
        plt.ylim(0., 0.0005)

        #define custom x ticks
        # = np.arange(0, 13000, 2000) 
        #plt.xticks(x_ticks)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.xlabel(f'{var_string[0]}', fontsize=20, color='white')
        plt.xlabel(f'{var_string[0]}- 50th perc', fontsize=20, color='white')
        plt.ylabel('Density', fontsize=20, color='white')
        plt.grid(color='white', linestyle='--', linewidth=0.5)

        # White axes, ticks, legend
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.tick_params(colors='white', width=1.5, length=7)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # # Legend outside on the right
        # legend = plt.legend(
        #     frameon=False,
        #     fontsize=18,
        #     loc="center left",          # anchor relative to the axes
        #     bbox_to_anchor=(1.02, 0.5)  # (x, y): shift outside the plot
        # )
        # for text in legend.get_texts():
        #     text.set_color('white')

        

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'distribution_{var[0]}_cm_{group_name}.png'),
            dpi=300,
            transparent=True,
        )
        plt.close()


if __name__ == "__main__":
    main()  