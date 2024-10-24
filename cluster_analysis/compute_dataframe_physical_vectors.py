import pandas as pd
from aux_functions import pick_variable

run_name = '10th-90th_CMA' #['10th-90th', '10th-90th_CMA']
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'
stat = '50' #None  # '50'
n_subsample = 33792  # Number of samples per cluster

# Define the data types to retrieve variables from
data_types = ['space-time', 'continuous', 'categorical', 'topography', 'era5-land']

# Initialize an empty list to hold all variables
correlation_vars = []

# Loop over each data type to retrieve variables and append to the list
for data_type in data_types:
    vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)
    correlation_vars.extend(vars)  # Append each variable list to the main correlation_vars list

print(correlation_vars)

merged_df = None

#for run_name in run_names:
    
# Path to fig folder for outputs
output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/'

# Loop over each data type to retrieve variables and append to the list
for data_type in data_types:

    # Load each CSV into a DataFrame
    if data_type=='space-time':
        df_crops = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_None.csv')
    elif data_type=='categorical':
        df_crops = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_True.csv')
    else:
        df_crops = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv')
    
    # For the first iteration, set the initial DataFrame
    if merged_df is None:
        merged_df = df_crops#.drop(columns=['label'])
    else:
        # Merge DataFrames by 'label', avoiding duplicates
        merged_df = pd.concat([merged_df, df_crops.drop(columns=['label'])], axis=1)

print(merged_df)

#save dataframe
merged_df.to_csv(f'{output_path}physical_feature_vectors_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)