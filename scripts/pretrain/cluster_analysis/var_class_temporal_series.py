"""
code to generate the temporal series of the specific input variables 
for each class
"""
import sys
import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd


# csv file with 2D+1 output
csv_file = '/Users/claudia/Documents/data_ml_spacetime/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000.csv'
output_dir = '/Users/claudia/Documents/data_ml_spacetime/figs/'

def main():

    print("Reading CSV file...")
    df = read_csv_to_dataframe(csv_file)
    #print(df.head())
    print("Column titles:", df.columns.tolist())

    # list all variables in the dataframe
    print("Variables in the dataframe:", df['var'].unique())
    print("Labels in the dataframe:", df['label'].unique())

    # loop on classes and plot temporal series of mean percentiles for each class and concatenate resulting dataframes
    df_all_mean = []
    df_all_std = []


    # loop on unique labels of the classes 
    for label_sel in df['label'].unique():

        # select class
        df_class = df[df['label'] == label_sel]
        print(f"Number of samples in class {label_sel}: {len(df_class)}")

        # select variable of interest
        var_name = 'cma'  # change to your variable of interest
        if var_name == 'cma':
            mean_grad_class = np.zeros((len(df['label'].unique()), 1))
        else:
            mean_grad_class = np.zeros((len(df['label'].unique()), 4))  # to store mean gradients for each class and each percentile

        # select only var columns equal to var_name
        df_class_var = df_class[df_class['var'] == var_name]
        print(f"Number of samples in class {label_sel} for variable {var_name}: {len(df_class_var)}")

        # if var_name is cot or cth group and mean the percentiles, otherwise group and mean categorical variable
        if var_name == 'cma':
            # group by frame and calculate mean values of cma categories
            df_grouped = df_class_var.groupby('frame').agg({
                'categorical': 'mean',
            }).reset_index()
        else:
            # group by frame and calculate mean values of cth percentiles
            df_grouped = df_class_var.groupby('frame').agg({
                '25': 'mean',
                '50': 'mean',
                '75': 'mean',
                '99': 'mean'
            }).reset_index()

        # group by frame and calculate std values of cth percentiles
        df_std = df_class_var.groupby('frame').agg({
            '25': 'std',
            '50': 'std',
            '75': 'std',
            '99': 'std'
        }).reset_index()
        #print("Grouped DataFrame:")
        #print(df_grouped)

        if var_name == 'cma':

            # plot temporal series of categorical mean variable
            plt.figure(figsize=(10, 6))
            plt.plot(df_grouped['frame'], df_grouped['categorical'], label='Mean cloud cover', marker='o')
            plt.title(f'Temporal Series of Mean {var_name} for Class {label_sel}')
            plt.xlabel('Frame')
            plt.ylabel(f'{var_name} Mean Categorical')
            plt.legend()
            plt.grid()
            plt.xticks(df_grouped['frame'])
            plt.tight_layout()
            plt_path = os.path.join(output_dir, f'temporal_series_{var_name}_class_{label_sel}.png')
            plt.savefig(plt_path)
            print(f"Plot saved to {plt_path}")  
            plt.close()

            # calculate mean gradient of the time series for each percentile
            mean_grad_class[label_sel] = np.nanmedian(np.gradient(df_grouped['categorical']))

        else:

            # plot temporal series of mean percentiles for the class
            plt.figure(figsize=(10, 6))
            plt.plot(df_grouped['frame'], df_grouped['25'], label='25th Percentile', marker='o')
            plt.plot(df_grouped['frame'], df_grouped['50'], label='50th Percentile', marker='o')
            plt.plot(df_grouped['frame'], df_grouped['75'], label='75th Percentile', marker='o')
            plt.plot(df_grouped['frame'], df_grouped['99'], label='99th Percentile', marker='o')
            plt.title(f'Temporal Series of {var_name} Percentiles for Class {label_sel}')
            plt.xlabel('Frame')
            plt.ylabel(f'{var_name} Percentiles')
            plt.legend()
            plt.grid()
            plt.xticks(df_grouped['frame'])
            plt.tight_layout()
            plt_path = os.path.join(output_dir, f'temporal_series_{var_name}_class_{label_sel}.png')
            plt.savefig(plt_path)
            print(f"Plot saved to {plt_path}")  

            # calculate mean gradient of the time series for each percentile
            mean_grad = np.zeros(4)
            for ind, perc in enumerate(['25', '50', '75', '99']):
                mean_grad[ind] = np.nanmedian(np.gradient(df_grouped[perc]))
            mean_grad_class[label_sel, :] = mean_grad

        df_all_mean.append(df_grouped)
        df_all_std.append(df_std)
    # store mean gradients for each class and each percentile in python file using numpy save
    # and also as a .py file with the array as a list
    np.save(os.path.join(output_dir, f'mean_gradients_{var_name}.npy'), mean_grad_class)





    # plot 50th percentile for all classes non normalized
    #plt.figure(figsize=(10, 6))
    #for i, df_class in enumerate(df_all):
    #    plt.plot(df_class['frame'], df_class['50'], label=f'Class {i}', marker='o')
    #plt.title(f'Temporal Series of {var_name} 50th Percentile for All Classes')
    #plt.xlabel('Frame')
    #plt.ylabel(f'{var_name} 50th Percentile')
    #plt.legend()
    #plt.grid()
    #plt.xticks(df_class['frame'])
    #plt.tight_layout()
    #plt_path = os.path.join(output_dir, f'temporal_series_{var_name}_all_classes.png')
    #plt.savefig(plt_path)
    #print(f"Plot saved to {plt_path}")

    # plot 50th percentile for all classes between min and max each in a different subplot
    plot_temporal_series_min_max_perc_all_classes(df_all_mean, df_all_std, var_name, perc_sel=50, output_dir=output_dir)
    plot_temporal_series_min_max_perc_all_classes(df_all_mean, df_all_std, var_name, perc_sel=75, output_dir=output_dir)
    plot_temporal_series_min_max_perc_all_classes(df_all_mean, df_all_std, var_name, perc_sel=25, output_dir=output_dir)
    plot_temporal_series_min_max_perc_all_classes(df_all_mean, df_all_std, var_name, perc_sel=99, output_dir=output_dir)

    # plot 50th percentile for selected group of classes normalized between min and max
    if var_name == 'cth':
        plot_temporal_series_perc_by_group(df_all_mean, var_name,  50, 'ascending', output_dir=output_dir)
        plot_temporal_series_perc_by_group(df_all_mean, var_name,  50, 'descending', output_dir=output_dir)
        plot_temporal_series_perc_by_group(df_all_mean, var_name,  50, 'mixed', output_dir=output_dir)
 



        
def plot_temporal_series_perc_by_group(df_all_mean, var_name, perc_sel, group, output_dir):
    """
    function to plot percentiles time series for user defined group of classes (ascending, descending, mixed) 
    Args:
        df_all_mean (list of pd.DataFrame): List of dataframes with mean values for each class.
        var_name (str): Name of the variable to plot.
        perc_sel (int): Percentile to plot (e.g., 50, 75, 25, 99).
        group (str): Group of classes to plot ('ascending', 'descending', 'mixed').
        output_dir (str): Directory to save the plot.
    Output:
        Saves the plot to the specified output directory.
    """
    print(var_name)
    if var_name == 'cth':
        groups = {
            'ascending': [0, 1, 5, 6],
            'descending': [3, 4, 8],
            'mixed': [2, 7]
        }
    elif var_name == 'cot':
        groups = {
            'ascending': [4, 3],
            'descending': [1],
            'mixed': [0, 2, 5, 6, 7, 8]
        }

    print(groups)

    class_indices = groups[group]

    # convert perc_sel to string
    perc_sel_str = str(perc_sel)

    # plot percentile for selected group of classes normalized between min and max

    plt.figure(figsize=(10, 6))
    for i in class_indices:
        df_class = df_all_mean[i]
        plt.plot(df_class['frame'], 
                 (df_class[perc_sel_str] - df_class[perc_sel_str].min()) / (df_class[perc_sel_str].max() - df_class[perc_sel_str].min()), 
                 label=f'Class {i}', 
                 marker='o', 
                 color=plt.cm.tab10(i),
                 linewidth=3)
    plt.title(f'Temporal Series of {var_name} {perc_sel_str}th Percentile for {group.capitalize()} Classes')
    plt.xlabel('Frame')
    plt.ylabel(f'{var_name} {perc_sel_str}th Percentile')
    plt.legend()
    plt.grid()
    plt.xticks(df_class['frame'])
    plt.tight_layout()
    plt_path = os.path.join(output_dir, f'temporal_series_{perc_sel_str}_{var_name}_{group}_classes.png')
    plt.savefig(plt_path)
    print(f"Plot saved to {plt_path}")

    plt.close()
    return

def plot_temporal_series_min_max_perc_all_classes(df_all_mean, df_all_std, var_name, perc_sel, output_dir):
    """
    Plot temporal series of the selected percentile for all classes between min and max in different subplots.
    the percentile (50, 75, 25, 99) is passed as argument
    
    Args:
        df_all_mean (list of pd.DataFrame): List of dataframes with mean values for each class.
        df_all_std (list of pd.DataFrame): List of dataframes with std values for each class.
        var_name (str): Name of the variable to plot.
        perc_sel (int): Percentile to plot (e.g., 50, 75, 25, 99).
        output_dir (str): Directory to save the plot.

    Output:
        Saves the plot to the specified output directory.
    """

    if var_name == 'cma':
        print(f"Plotting {var_name} mean for all classes between min and max...")
        # plot cloud mask for all classes between min and max each in a different subplot
        plt.figure(figsize=(10, 20))

        for i, df_class in enumerate(df_all_mean):
            ax = plt.subplot(len(df_all_mean), 1, i + 1)
            # choose a different color for each class
            plt.plot(df_class['frame'], 
                     df_class['categorical'],
                       label=f'Class {i}', marker='o', 
                        color=plt.cm.tab10(i), 
                        linewidth=3)
            plt.legend()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.suptitle(f'Temporal Series of {var_name} for All Classes', fontsize=20)
        plt.xlabel('Frame')
        plt.ylabel('Cloud cover')
        plt.xticks(df_class['frame'])
        plt.tight_layout()
        plt_path = os.path.join(output_dir, f'temporal_series_min_max_{var_name}_all_classes.png')
        plt.savefig(plt_path)
        print(f"Plot saved to {plt_path}")
        plt.close()
    else:
        print(f"Plotting {var_name} {perc_sel}th percentile for all classes between min and max...")    
        # convert perc_sel to string
        perc_sel_str = str(perc_sel)

        # plot percentile for all classes between min and max each in a different subplot
        plt.figure(figsize=(10, 20))
        for i, df_class in enumerate(df_all_mean):
            ax = plt.subplot(len(df_all_mean), 1, i + 1)
            # choose a different color for each class
            plt.plot(df_class['frame'], 
                    df_class[perc_sel_str], 
                    label=f'Class {i}', 
                    marker='o', 
                    color=plt.cm.tab10(i),
                    linewidth=3)

            # plot std as shaded area
            #plt.fill_between(df_class['frame'], 
            #                 df_class[perc_sel_str] + df_all_std[i][perc_sel_str], 
            #                 df_class[perc_sel_str] - df_all_std[i][perc_sel_str], 
            #                 color=plt.cm.tab10(i), 
            #                 alpha=0.2, 
            #                 label='Std Dev')
            #plt.legend()
            #plt.ylim(df_class[perc_sel_str].min(), df_class[perc_sel_str].max())
            #plt.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.suptitle(f'Temporal Series of {var_name} {perc_sel_str}th Percentile (Min-Max) for All Classes', fontsize=20)
        plt.xlabel('Frame')
        plt.ylabel(f'{var_name} {perc_sel_str}th Percentile (Min-Max)')
        plt.xticks(df_class['frame'])

        plt.tight_layout()
        plt_path = os.path.join(output_dir, f'temporal_series_min_max_{perc_sel_str}_{var_name}_all_classes.png')
        plt.savefig(plt_path)

        print(f"Plot saved to {plt_path}")

        plt.close()
    return

def read_csv_to_dataframe(csv_file):
    """Reads the CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_file)
    return df


if __name__ == "__main__":
    main()



