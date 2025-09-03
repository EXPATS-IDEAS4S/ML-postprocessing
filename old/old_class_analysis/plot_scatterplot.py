##########################################################
## plot distr for 2 relevant varables in a scatterplot  ##
##########################################################

import pandas as pd
#import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from utils.processing.aux_functions import pick_variable

run_names = ['10th-90th', '10th-90th_CMA']

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

stat = None #'50'

# Read data
n_subsample = 100  # Number of samples per cluster

data_type =  'continuous' #'space-time'  #'continuous' 'topography' 'era5-land'

var_1 = 'cot'
var_2 = 'ctp'

# Select the correct varable information based on the data type  
vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)
#print(vars, vars_long_name, vars_units, vars_logscale, vars_dir)

# Combine the lists into a dictionary for easier lookup
variable_info = {
    var: {
        'long_name': long_name,
        'units': unit,
        'logscale': logscale,
        'dir': direction
    }
    for var, long_name, unit, logscale, direction in zip(vars, vars_long_name, vars_units, vars_logscale, vars_dir)
}
print(variable_info)


for run_name in run_names:

    # Path to fig folder for outputs
    output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/'

    # Load CSV file with the crops path and labels into a pandas DataFrame
    df_crops = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv')

    # Check if var_1 and var_2 exist in the variable_info
    if var_1 not in variable_info or var_2 not in variable_info:
        raise ValueError(f"Variables {var_1} or {var_2} not found in variable_info.")

    # Extract relevant information for plotting
    var_1_column = var_1
    var_2_column = var_2
    var_1_long_name = variable_info[var_1]['long_name']
    var_2_long_name = variable_info[var_2]['long_name']
    var_1_units = variable_info[var_1]['units'] if variable_info[var_1]['units'] else ''
    var_2_units = variable_info[var_2]['units'] if variable_info[var_2]['units'] else ''
    var_1_dir = variable_info[var_1]['dir']
    var_2_dir = variable_info[var_2]['dir']


    # Get the label column (assuming it's named 'label' in the CSV)
    label_column = 'label'

    # Ensure the label column exists in the dataframe
    if label_column not in df_crops.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe columns.")

    # Create scatterplot with seaborn, coloring points by the labels
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df_crops[var_1_column], 
        y=df_crops[var_2_column], 
        hue=df_crops[label_column], 
        palette='viridis',  # Choose a color palette
        s=30,               # Adjust size of points
        #edgecolor='k',       # Add edge color for better visibility
        alpha=0.7            # Transparency
    )

    # Add plot labels and title
    plt.xlabel(f'{var_1_long_name} [{var_1_units}]')
    plt.ylabel(f'{var_2_long_name} [{var_2_units}]')
    plt.title(f'{var_1_long_name} vs {var_2_long_name} - {run_name} - {n_subsample} - {sampling_type}')

    # Reverse y axis if direction is 'decr'
    if var_2_dir == 'decr':
        ax.invert_yaxis()
    if var_1_dir == 'decr':
        ax.invert_xaxis()

    # Add a legend
    plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the figure
    output_fig_path = f'{output_path}{var_1}_vs_{var_2}_scatterplot_{run_name}_{sampling_type}_{n_subsample}_{stat}.png'
    fig.savefig(output_fig_path, bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()
    exit()

        
        