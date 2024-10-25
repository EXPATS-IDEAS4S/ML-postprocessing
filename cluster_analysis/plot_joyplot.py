import pandas as pd
import xarray as xr
import numpy as np
from glob import glob

from aux_functions import pick_variable, plot_joyplot


run_names = ['10th-90th_CMA', '10th-90th']

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

# Pick the statistics to compute for each crop, the percentile values
stats = [None] #[1,50,99,'25-75']   #'1%', '5%', '25%', '50%', '75%', '95%', '99%' ,None if all points are needed
#use e.g '25-75' if interquartile range want to be calculated

# Paths to CMSAF cloud properties crops
cloud_properties_path = '/data1/crops/cmsaf_2013-2014_expats/nc_clouds/'
cloud_properties_crop_list = sorted(glob(cloud_properties_path + '*.nc'))
n_samples = len(cloud_properties_crop_list)

# Read data
if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:
    n_subsample = 100


# Path to topography data (DEM and land-sea mask)
era5_path = f'/data1/other_data/ERA5-Land_t2m_snowc_u10_v10_sp_tp/'

data_types = ['continuous'] #['era5-land','continuous','topography'] # 'era5-land'

# Select the correct varable information based on the data type  
vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)

# Save in case of crop stats are calculated
#if stat: 
df_continuous.to_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
print('Continous Stats for each crop are saved to CSV files.')

# Compute stats for continuous variables
continuous_stats = df_continuous.groupby('label').agg(['mean', 'std'])
continuous_stats.columns = ['_'.join(col).strip() for col in continuous_stats.columns.values]
continuous_stats.reset_index(inplace=True)

Save continuous stats to a CSV file
continuous_stats.to_csv(f'{output_path}{data_type}_clusters_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
print('Overall Continous Stats for each cluster are saved to CSV files.')

#extract class label
label_names = np.unique(df_continuous['label'].values)
label_names = label_names[label_names != -100]

# Plotting continuous variables box plots
for var, long_name, unit, direction, scale in zip(vars, vars_long_name, vars_units, vars_dir, vars_logscale):
    #plot_single_vars(df_continuous, n_subsample, var, long_name, unit, direction, scale, output_path, run_name, sampling_type, stat)
    for class_label in label_names:
        plot_joyplot(df_continuous, class_label, var, long_name, unit, n_subsample, output_path, run_name, sampling_type)

                

