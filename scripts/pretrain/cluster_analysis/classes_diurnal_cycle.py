"""
Code to calculate the occurrence of different classes across the day. 
We select the time to associate to each video by calculating the mean time of the 8 frames. 
We then group by time in the hours of the day (0-23) and calculate the occurrence of each class
Author: Claudia Acquistapace
Date: 10 sept 2025

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

from scripts.pretrain.cluster_analysis.var_class_temporal_series import read_csv_to_dataframe

from utils.plotting.class_colors import colors_per_class1_names, class_groups


# csv file with 2D+1 output
csv_file = '/Users/claudia/Documents/data_ml_spacetime/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000_debug.csv'
output_dir = '/Users/claudia/Documents/data_ml_spacetime/figs/'

def main():

    # read csv file
    df = read_csv_to_dataframe(csv_file)
    print("Column titles:", df.columns.tolist())

    # select one variable to be sure to select 8 frames
    df8 = df[df['var'] == 'cth']
    

    # find all values of crop_index in df
    crop_indices = df8['crop_index'].unique()
    print("Number of unique crop indices (videos):", len(crop_indices))

    # loop on crop indeces and calculate mean time for each video
    times, labels = derive_time_class(df8, crop_indices)
    print("Times shape:", times.shape)
    print("Labels shape:", labels.shape)

    print("Number of videos with 8 frames:", len(times))
    print("First 5 times:", times[:5])
    print("First 5 labels:", labels[:5])


    # create a new dataframe with times and labels
    df_times = pd.DataFrame({'time': times, 'label': labels})
    # ensure 'time' column is in datetime format before extracting hour
    df_times['time'] = pd.to_datetime(df_times['time'])
    df_times['hour'] = df_times['time'].dt.hour
    print(df_times.head())

    # group by hour and label and count occurrences
    df_grouped = df_times.groupby(['hour', 'label']).size().unstack(fill_value=0)
    print(df_grouped.head())    
    # normalize by total occurrences per hour
    df_grouped = df_grouped.div(df_grouped.sum(axis=1), axis=0)
    print(df_grouped.head())

    # plot occurrence of each class across the day
    plt.figure(figsize=(10, 6))
    for label in df_grouped.columns:
        plt.plot(df_grouped.index, 
                 df_grouped[label],
                 linewidth=3,
                label=f'Class {label}')
        
    plt.xlabel('Hour of the day', fontsize=16)
    plt.ylabel('Occurrence', fontsize=16)
    plt.title('Occurrence of each class across the day', fontsize=16)
    plt.xticks(range(0, 24, 2), fontsize=14)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_occurrence_diurnal_cycle.png'))
    plt.close()

    # plot of diurnal cycles for class groups

    # read class groups from utils/plotting/class_colors.py
    for group_name, group_labels in class_groups.items():
        print(group_name, group_labels)


        # plot diurnal cycle for each element of the group
        plt.figure(figsize=(10, 6))
        for label in group_labels:
            # plot class using the color defined in utils/plotting/class_colors.py
            if str(label) in colors_per_class1_names:
                plt.plot(df_grouped.index, 
                         df_grouped[label], 
                         label=f'Class {label}', 
                         linewidth=4,
                         color=colors_per_class1_names[str(label)])
            else:
                plt.plot(df_grouped.index, 
                         df_grouped[label], 
                         marker='o', 
                         markersize=10, 
                         label=f'Class {label}')

        plt.xlabel('Hour of the day', fontsize=20)
        plt.ylabel('Occurrence', fontsize=20)
        plt.title('Occurrence of class groups across the day', fontsize=20)
        plt.xticks(range(0, 24), fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        # enlarge fonts of all texts
        plt.rcParams.update({'font.size': 20})
        # enlarge xticks and yticks fonts
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        # remove upper and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 f'{group_name}_class_groups_occurrence_diurnal_cycle.png'), 
                                 transparent=True)


    # plot single diurnal cycle for each class
    for label in df_grouped.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df_grouped.index, 
                 df_grouped[label], 
                 marker='o', 
                 markersize=10, 
                 color='C'+str(label))
        plt.xlabel('Hour of the day [hh]')
        plt.ylabel('Normalized Occurrence')
        plt.title(f'Occurrence of Class {label} across the day')
        plt.xticks(range(0, 24))
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        # remove upper and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # make axis thicker
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        # make ticks thicker
        plt.gca().tick_params(width=1.5, length=7, labelsize=12)
        
        plt.savefig(os.path.join(output_dir, f'class_{label}_diurnal_cycle.png'), transparent=True)
        plt.close()

    



def derive_time_class(df, crop_indices):
    """
    Derive mean time for each video and calculate occurrence of each class across the day
    
    Parameters
    ----------
    df : pandas dataframe
        
        Dataframe with columns: 'crop_index', 'datetime', 'label'
    crop_indices : array-like
        Array of unique crop indices (videos)
    
    Returns
    -------
    array of video mean times and classification labels"""
    mean_times = []
    labels = [] 

    for crop_index in crop_indices:
        df_crop = df[df['crop_index'] == crop_index]

        if len(df_crop) != 8:
            continue  # skip if not 8 frames
        # convert datetime to pandas datetime
        df_crop['datetime'] = pd.to_datetime(df_crop['time'])
        # calculate mean time
        mean_time = df_crop['datetime'].mean()
        mean_times.append(mean_time)
        # get label (assuming all frames have the same label)
        label = df_crop['label'].iloc[0]
        labels.append(label)

    return np.array(mean_times), np.array(labels)

if __name__ == "__main__":
    main()

