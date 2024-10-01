import pandas as pd
from aux_functions import pick_variable, plot_single_vars

run_name = '10th-90th_CMA'

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

# Pick the statistics to compute for each crop, the percentile values
#'1%', '5%', '25%', '50%', '75%', '95%', '99%' ,None if all points are needed
#use e.g '25-75' if interquartile range want to be calculated
stat_1 = '50'
stat_2 = '25-75'

data_type = 'era5-land' #'continuous' 'topography' 'era5-land'

# Path to fig folder for outputs
output_path = f'/home/daniele/Documenti/Data/Fig/{run_name}/{sampling_type}/'

# Read data
n_subsample = 1000  # Number of samples per cluster

# Open csv file with the stats
df_stat1 = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat_1}.csv')
df_stat2 = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat_2}.csv')

print(df_stat1)
print(df_stat2)

# df_medians and df_iqrs are two separate DataFrames for different statistics

# Merging the dataframes to add the new statistic type
df_stat1['stat_type'] = stat_1
df_stat2['stat_type'] = stat_2

# Concatenate the two dataframes
df_combined = pd.concat([df_stat1, df_stat2], ignore_index=True)
print(df_combined)

vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)

# Plotting box plots for both statistics
for var, long_name, unit, direction, scale in zip(vars, vars_long_name, vars_units, vars_dir, vars_logscale):
    plot_single_vars(df_combined, n_subsample, var, long_name, unit, direction, scale, output_path, run_name, sampling_type, f'{stat_1}-{stat_2}',legend='Percentile', hue='stat_type')
    

