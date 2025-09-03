import pandas as pd
from utils.processing.aux_functions import pick_variable, plot_single_vars

run_names = ['dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA']  #['10th-90th', '10th-90th_CMA']

# Define sampling type
sampling_types = ['closest','farthest']  # Options: 'random', 'closest', 'farthest', 'all'

# Pick the statistics to compute for each crop, the percentile values
#'1%', '5%', '25%', '50%', '75%', '95%', '99%' ,None if all points are needed
#use e.g '25-75' if interquartile range want to be calculated
stat = 50  #['1','50','99','25-75'] 

data_type = 'continuous' #'continuous' 'topography' 'era5-land' 'cateogrical' 'space-time'

# Read data
n_subsample = 100  # Number of samples per cluster

for run_name in run_names:

    # Path to fig folder for outputs
    output_path = f'/home/Daniele/fig/{run_name}/'

    # Read the data for each sampling type and concatenate them for comparison
    combined_data = []

    for sampling_type in sampling_types:
        # Read the data for the specific sampling type
        df_stat = pd.read_csv(f'{output_path}{sampling_type}/{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv')
        # Add a column to label the sampling type
        df_stat['Sampling_Type'] = sampling_type

        # Append to combined_data list
        combined_data.append(df_stat)

    # Concatenate the data from both sampling types into one DataFrame
    df_combined = pd.concat(combined_data, ignore_index=True)
    
    # Print combined data to check
    print(df_combined)

    # # df_medians and df_iqrs are two separate DataFrames for different statistics

    # # Merging the dataframes to add the new statistic type
    # df_stat1['stat_type'] = stat_1
    # df_stat2['stat_type'] = stat_2

    # # Concatenate the two dataframes
    # df_combined = pd.concat([df_stat1, df_stat2], ignore_index=True)
    # print(df_combined)

    vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)

    # Plotting box plots for both statistics
    for var, long_name, unit, direction, scale in zip(vars, vars_long_name, vars_units, vars_dir, vars_logscale):
        plot_single_vars(df_combined, n_subsample, var, long_name, unit, direction, scale, output_path, run_name, sampling_type, f'{sampling_types[0]} vs {sampling_types[1]}',legend='Sampling Type', hue='Sampling_Type')
        

