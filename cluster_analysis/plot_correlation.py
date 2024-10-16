import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from aux_functions import pick_variable

run_name = '10th-90th' #['10th-90th', '10th-90th_CMA']
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'
stat = '50' #None  # '50'
n_subsample = 1000  # Number of samples per cluster

# # Define the data types to retrieve variables from
# data_types = ['space-time', 'continuous', 'topography', 'era5-land', 'categorical']

# # Initialize an empty list to hold all variables
# correlation_vars = []

# # Loop over each data type to retrieve variables and append to the list
# for data_type in data_types:
#     vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)
#     correlation_vars.extend(vars)  # Append each variable list to the main correlation_vars list

# print(correlation_vars)

# merged_df = None

#for run_name in run_names:
    
# Path to fig folder for outputs
output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/'

# # Loop over each data type to retrieve variables and append to the list
# for data_type in data_types:

#     # Load each CSV into a DataFrame
#     if data_type=='space-time':
#         df_crops = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_None.csv')
#     elif data_type=='categorical':
#         df_crops = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_True.csv')
#     else:
#         df_crops = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv')
    
#     # For the first iteration, set the initial DataFrame
#     if merged_df is None:
#         merged_df = df_crops.drop(columns=['label'])
#     else:
#         # Merge DataFrames by 'label', avoiding duplicates
#         merged_df = pd.concat([merged_df, df_crops.drop(columns=['label'])], axis=1)

# print(merged_df)

# Load the saved CSV into a new DataFrame
df_loaded = pd.read_csv(f'{output_path}physical_feature_vectors_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv')

# Create a DataFrame with only the selected variables for correlation
df_corr = df_loaded.drop(columns=['label']) #df_crops[correlation_vars]

correlation_vars = df_corr.columns.tolist()
print(correlation_vars)

# Calculate the correlation matrix
corr_matrix = df_corr.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', xticklabels=correlation_vars, yticklabels=correlation_vars)#, annot=True, fmt=".2f",)

# Set plot labels and title
plt.title(f'Correlation Heatmap - {run_name} - {n_subsample} - {sampling_type}')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Save the heatmap figure
output_fig_path = f'{output_path}correlation_heatmap_{run_name}_{sampling_type}_{n_subsample}_{stat}.png'
plt.savefig(output_fig_path, bbox_inches='tight', dpi=300)
print(f'fig saved in: {output_path}correlation_heatmap_{run_name}_{sampling_type}_{n_subsample}_{stat}.png')

# Show the heatmap
#plt.show()


#Masking for correlation values above 0.5 or below -0.5
corr_masked = corr_matrix.where((corr_matrix > 0.5) | (corr_matrix < -0.5), other=np.nan)

# Plot the masked heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_masked, cmap='coolwarm', xticklabels=correlation_vars, yticklabels=correlation_vars, annot=True, fmt=".2f", mask=np.isnan(corr_masked))

# Set plot labels and title for the masked heatmap
plt.title(f'Masked Correlation Heatmap - {run_name} - {n_subsample} - {sampling_type}')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Save the masked heatmap figure
output_fig_masked_path = f'{output_path}correlation_heatmap_filtered_{run_name}_{sampling_type}_{n_subsample}_{stat}.png'
plt.savefig(output_fig_masked_path, bbox_inches='tight', dpi=300)
print(f'Filtered correlation heatmap saved in: {output_fig_masked_path}')

# Show the filtered heatmap
#plt.show()