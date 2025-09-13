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
import pdb
sys.path.append(os.path.abspath("/Users/claudia/Documents/ML-postprocessing"))
from utils.plotting.class_colors import colors_per_class1_names, class_groups


# csv file with 2D+1 output
csv_file = '/Users/claudia/Documents/data_ml_spacetime/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000_debug.csv'
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

    # select variable of interest
    var_name = 'cth'  # change to your variable of interest
    var_string = 'Cloud Top Height'  # string to use in the title of the plot

    # define matrix for gradients
    # initialize mean gradients array (average of the gradient of the time series)
    if var_name == 'cma':
        mean_grad_class = np.zeros((len(df['label'].unique()), 1)) # gradient of cloud cover, one single value per class
    else:
        mean_grad_class = np.zeros((len(df['label'].unique()), 4))  # to store mean gradients for each class and each percentile

    # loop on unique labels of the classes
    for label_sel in df['label'].unique():

        # select class
        df_class = df[df['label'] == label_sel]
        print(f"Number of samples in class {label_sel}: {len(df_class)}")

        print(f"Processing variable: {var_name}")

        # select only var columns equal to var_name
        df_class_var = df_class[df_class['var'] == var_name]
        print(f"Number of samples in class {label_sel} for variable {var_name}: {len(df_class_var)}")

        # if var_name is cot or cth group and mean the percentiles, otherwise group and mean categorical variable
        if var_name == 'cma':

            # group by frame and calculate mean values of cma categories and skip nans
            df_grouped = df_class_var.groupby('frame').agg({
                'categorical': 'mean',
            }).reset_index()

            print(df_grouped['categorical'])
            mean_grad_class[label_sel] = np.nanmedian(np.gradient(df_grouped['categorical']))

            # plot temporal series of mean categorical for the class
            #plot_temporal_series_perc_by_class(df_grouped, 'categorical', var_name, 'Mean cloud cover', label_sel, output_dir)

        else:

            # group by frame and calculate mean values of cth percentiles
            df_grouped = df_class_var.groupby('frame').agg({
                '25': 'mean',
                '50': 'mean',
                '75': 'mean',
                '99': 'mean'
            }).reset_index()

            # plot temporal series of mean categorical for the class
            #plot_temporal_series_perc_by_class(df_grouped, '25', var_name, var_string, label_sel, output_dir)
            #plot_temporal_series_perc_by_class(df_grouped, '50', var_name, var_string, label_sel, output_dir)
            #plot_temporal_series_perc_by_class(df_grouped, '75', var_name, var_string, label_sel, output_dir)
            #plot_temporal_series_perc_by_class(df_grouped, '99', var_name, var_string, label_sel, output_dir)

            # calculate mean gradient of the time series for each percentile
            mean_grad = np.zeros(4)
            for ind, perc in enumerate(['25', '50', '75', '99']):
                mean_grad[ind] = np.nanmedian(np.gradient(df_grouped[perc]))
            mean_grad_class[label_sel, :] = mean_grad
            print(f"Mean gradients for class {label_sel}: {mean_grad}")
            print("gradient", np.gradient(df_grouped['50']))
            print("gradient", np.gradient(df_grouped['75']))
            print("gradient", np.gradient(df_grouped['99']))
            print("gradient", np.gradient(df_grouped['25']))
            print("mean gradient", mean_grad)
            print("Mean gradient array:", mean_grad_class[label_sel, :])
            print(" all gradients array:", mean_grad_class)
            #pdb.set_trace()

        df_all_mean.append(df_grouped)

    # store mean gradients for each class and each percentile in python file using numpy save
    # and also as a .py file with the array as a list

    print(f"Saving mean gradients for variable {var_name}...")
    print(mean_grad_class)

    np.save(os.path.join(output_dir, f'mean_gradients_{var_name}.npy'), mean_grad_class)

    # plot 50th percentile for all classes between min and max each in a different subplot
    #plot_temporal_series_min_max_perc_all_classes(df_all_mean, var_name, perc_sel=50, output_dir=output_dir)
    #plot_temporal_series_min_max_perc_all_classes(df_all_mean, var_name, perc_sel=75, output_dir=output_dir)
    #plot_temporal_series_min_max_perc_all_classes(df_all_mean, var_name, perc_sel=25, output_dir=output_dir)
    #plot_temporal_series_min_max_perc_all_classes(df_all_mean, var_name, perc_sel=99, output_dir=output_dir)

    # plot 50th percentile for selected group of classes normalized between min and max
    plot_temporal_series_perc_by_group(df_all_mean, var_name,  50, output_dir=output_dir)
    plot_temporal_series_perc_by_group(df_all_mean, var_name,  75, output_dir=output_dir)
    plot_temporal_series_perc_by_group(df_all_mean, var_name,  99, output_dir=output_dir)



def plot_temporal_series_perc_by_class(df_grouped, arg, var_name, var_string, label, output_dir):
    """
    function to plot percentiles time series or cloud cover for each class
    Args:
        df_grouped (pd.DataFrame): Dataframe with mean values for the class.
        arg (str): Column name to plot ('25', '50', '75', '99' or 'categorical').
        var_name (str): Name of the variable to plot (cot, cth, cma).
        var_string (str): String to use in the title of the plot.
        label (int): Class label.
        output_dir (str): Directory to save the plot.
    """
    # plot temporal series of categorical mean variable
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped['frame'], df_grouped[arg], label=var_string, marker='o')
    plt.title(f'Temporal Series of Mean {var_name} for Class {label}')
    plt.xlabel('Frame')
    plt.ylabel(f'{var_name} Mean Categorical')
    plt.legend()
    plt.grid()
    plt.xticks(df_grouped['frame'])
    plt.tight_layout()
    plt_path = os.path.join(output_dir, f'temporal_series_{var_name}_class_{label}.png')
    plt.savefig(plt_path)
    print(f"Plot saved to {plt_path}")  
    plt.close()
    return()
        
def plot_temporal_series_perc_by_group(df_all_mean, var_name, perc_sel, output_dir):
    """
    function to plot percentiles time series for user defined group of classes (ascending, descending, mixed) 
    Args:
        df_all_mean (list of pd.DataFrame): List of dataframes with mean values for each class.
        var_name (str): Name of the variable to plot.
        perc_sel (int): Percentile to plot (e.g., 50, 75, 25, 99).
        output_dir (str): Directory to save the plot.
    Output:
        Saves the plot to the specified output directory.
    """
    # convert perc_sel to string
    perc_sel_str = str(perc_sel)

    # read groups convective, broken and dissipative from utils/plotting/class_colors.py
    for group_name, group_labels in class_groups.items():

        print(group_name, group_labels)

        # plot temporal evolution of the selected percentile for the group of classes
        plt.figure(figsize=(10, 6))
        for i in group_labels:
            df_class = df_all_mean[i]
            # plot class using the color defined in utils/plotting/class_colors.py
            if str(i) in colors_per_class1_names:
                # plot values normalized between min and max
                plt.plot(df_class['frame'], 
                         (df_class[perc_sel_str] - df_class[perc_sel_str].min()) / (df_class[perc_sel_str].max() - df_class[perc_sel_str].min()), 
                         label=f'Class {i}', 
                         linewidth=3,
                         color=colors_per_class1_names[str(i)])
            else:
                plt.plot(df_class['frame'], 
                         df_class[perc_sel_str], 
                         label=f'Class {i}', 
                         linewidth=3)
        plt.title(f'Temporal Series of {var_name} {perc_sel_str}th Percentile for {group_name.capitalize()} Classes', fontsize=20)
        plt.xlabel('Frame', fontsize=20)
        # remove upper and right border
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel(f'{var_name} {perc_sel_str}th Percentile', fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.xticks(df_class['frame'], fontsize=16)
        plt.tight_layout()
        plt_path = os.path.join(output_dir, 
                            f'temporal_series_{perc_sel_str}_{var_name}_{group_name}_classes.png')
        plt.savefig(plt_path, transparent=True)
        print(f"Plot saved to {plt_path}")


    return

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

def plot_temporal_series_min_max_perc_all_classes(df_all_mean, var_name, perc_sel, output_dir):
    """
    Plot temporal series of the selected percentile for all classes between min and max in different subplots.
    the percentile (50, 75, 25, 99) is passed as argument
    
    Args:
        df_all_mean (list of pd.DataFrame): List of dataframes with mean values for each class.
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

    # if file ends with _debug.csv, remove all lines with frame = -710387
    if csv_file.endswith('_debug.csv'):
        df = df[df['frame'] != -710387]
    if df.empty:
        print(f"Warning: {csv_file} is empty after filtering.")

    return df


if __name__ == "__main__":
    main()



