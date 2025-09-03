import pandas as pd
import xarray as xr
import numpy as np
from glob import glob

from utils.processing.aux_functions import pick_variable, plot_joyplot


run_names = ['10th-90th_CMA', '10th-90th']

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

# Pick the statistics to compute for each crop, the percentile values
#'1%', '5%', '25%', '50%', '75%', '95%', '99%' ,None if all points are needed
#use e.g '25-75' if interquartile range want to be calculated
stat = None #[1,50,99,'25-75']   

# Paths to CMSAF cloud properties crops
# To use n samples if set sampling_type = 'all'
cloud_properties_path = '/data1/crops/cmsaf_2013-2014_expats/nc_clouds/'
cloud_properties_crop_list = sorted(glob(cloud_properties_path + '*.nc'))
n_samples = len(cloud_properties_crop_list)

if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:   
    n_subsample = 100

data_types = ['continuous'] #['era5-land','continuous','topography'] # 'era5-land'

vars_to_plot = ['ctp'] #['cph', 'cma', 'ctp', 'cot']

for data_type in data_types:
    for run_name in run_names:
        print(data_type, run_name)

        # Path to fig folder for outputs
        output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/'

        # Select the correct varable information based on the data type  
        vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)

        # Save in case of crop stats are calculated
        #if stat: 
        df = pd.read_csv(f'{output_path}{data_type}_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv')
        #print(df)
        class_df = df[df['label'] == 0]
        #print(class_df)

        #extract class label
        label_names = np.unique(df['label'].values)
        label_names = label_names[label_names != -100]

        # Plotting continuous variables box plots
        for var, long_name, unit, direction, scale in zip(vars, vars_long_name, vars_units, vars_dir, vars_logscale):
            if var in vars_to_plot:
                for class_label in label_names:
                    print(var, class_label)
                    plot_joyplot(df, class_label, var, long_name, unit, n_subsample, output_path, run_name, sampling_type)
                    exit()

                    

