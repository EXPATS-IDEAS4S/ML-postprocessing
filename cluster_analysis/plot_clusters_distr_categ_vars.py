############################################################
## Compute stats and plot distr for Categorical variables ##
############################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np

from aux_functions import concatenate_values, extend_labels

run_names = ['10th-90th','10th-90th_CMA']

# Define sampling type
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'

# Pick the statistics to compute for each crop, the percentile values
stat = True  #True or False, if True 
#CMA: it computes the fraction of clouds in images, 
#CPH: it computes the percentage of water clouds over the number of cloudy points

# Read data
if sampling_type == 'all':
    n_subsample = 33792  # Number of samples per cluster
else:
    n_subsample = 100  # Number of samples per cluster

for run_name in run_names:

    # Path to fig folder for outputs
    output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/' 

    # Load CSV file with the crops path and labels into a pandas DataFrame
    df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsample}_{sampling_type}.csv')

    print(df_labels)

    categorical_vars = ['cph', 'cma']
    cat_vars_long_name = ['cloud phase', 'cloud mask']
    #cat_var_units = [('clear','liquid','ice'),('clear','cloudy')] #0:clear, 1:liquid, 2:ice

    # Initialize lists to hold data for continuous and categorical variables
    categorical_data = {var: [] for var in categorical_vars}
    labels = []

    # Read the .nc files and extract data
    for i, row in df_labels.iterrows():
        ds = xr.open_dataset(row['path'])

        for var in categorical_vars:
            values = ds[var].values.flatten()
            if stat:
                if var == 'cma':
                    # Compute the fraction of cloud pixels (value == 1) over total pixels
                    total_pixels = len(values)
                    cloud_pixels = np.sum(values == 1)
                    fraction_cloudy = cloud_pixels / total_pixels if total_pixels > 0 else 0
                    values = fraction_cloudy
                    #categorical_data[var].append(fraction_cloudy)

                elif var == 'cph':
                    # Compute the fraction of liquid clouds (value == 1) over cloudy pixels (value 1 or 2)
                    cloudy_pixels = np.sum((values == 1) | (values == 2))  # pixels with value 1 or 2
                    liquid_pixels = np.sum(values == 1)  # liquid clouds (value 1)
                    
                    if cloudy_pixels > 0:
                        fraction_liquid = liquid_pixels / cloudy_pixels
                    else:
                        fraction_liquid = 0  # If no cloudy pixels, set the fraction to 0
                    values = fraction_liquid
                    #categorical_data[var].append(fraction_liquid)
                else:
                    print('wrong variable names!')

            # Ensure values is a list or array before extending
            categorical_data = concatenate_values(values, var, categorical_data)
        
        # Extend labels based on the number of valid entries for this dataset
        labels = extend_labels(values, labels, row)

    # Create DataFrames for continuous and categorical variables
    df_categorical = pd.DataFrame(categorical_data)
    df_categorical['label'] = labels

    print(df_categorical)

    if stat:
        # Save Dataframe in csv only for stats
        df_categorical.to_csv(f'{output_path}categorical_crops_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
        print('Categorical Stats for each crop are saved to CSV files.')

        # Compute stats for continuous variables
        df_cat_stats = df_categorical.groupby('label').agg(['mean', 'std'])
        df_cat_stats.columns = ['_'.join(col).strip() for col in df_cat_stats.columns.values]
        df_cat_stats.reset_index(inplace=True)
    else:
        # Compute stats for categorical variables excluding the label column
        categorical_percentages = df_categorical.groupby('label').apply(
            lambda x: x.drop(columns='label').apply(lambda col: col.value_counts(normalize=True))
        )
        print(categorical_percentages)

        df_cat_stats = pd.DataFrame(categorical_percentages)

    # Save categorical stats to a CSV file
    df_cat_stats.to_csv(f'{output_path}categorical_clusters_stats_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv', index=False)
    print('Categorical Stats for each cluster are saved to CSV files.')


    #Compute relative frequency of each label
    # Useful only in case subsampling is not used
    label_counts = df_categorical['label'].value_counts(normalize=True) * 100

    print(label_counts)

    df_rel_freq = pd.DataFrame(label_counts)

    # Save categorical stats to a CSV file
    df_rel_freq.to_csv(f'{output_path}label_rel_freq.csv', index=False)

    # Define a consistent color palette for each variable name
    cph_palette = {'clear': 'blue', 'liquid': 'green', 'ice': 'red'}
    cma_palette = {'clear': 'blue', 'cloudy': 'gray'}

    # Mapping dictionaries
    cph_mapping = {0: 'clear', 1: 'liquid', 2: 'ice'}
    cma_mapping = {0: 'clear', 1: 'cloudy'}

    # Adding new columns
    df_categorical['cph_name'] = df_categorical['cph'].map(cph_mapping)
    df_categorical['cma_name'] = df_categorical['cma'].map(cma_mapping)

    print(df_categorical)

    # Plotting categorical with consistent colors
    for var, long_name in zip(categorical_vars, cat_vars_long_name):
        fig, axes = plt.subplots(1, 1, figsize=(8, 4), sharey=True)

        #Define var name
        if stat and var=='cma':
            var_name = 'cloud fraction'
        elif stat and var=='cph':
            var_name = 'liquid fraction'
        else:
            var_name = var

        # Set the color palette based on the variable being plotted
        if var == 'cph':
            palette = cph_palette
            hue_order = list(cph_palette.keys())  # Consistent order of categories
        elif var == 'cma':
            palette = cma_palette
            hue_order = list(cma_palette.keys())  # Consistent order of categories

        if stat:
            sns.boxplot(data=df_categorical, x='label', y=var, ax=axes, showfliers=False)
            axes.set_ylabel(f'{var_name}', fontsize=14)
        else:
            sns.countplot(x='label', hue=var + '_name', data=df_categorical, ax=axes,
                    palette=palette, hue_order=hue_order)
            axes.set_ylabel('Counts',fontsize=14)
        
            # Move the legend outside on the right and center it vertically
            axes.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., title="cloud state", fontsize=12, title_fontsize=14)
        axes.set_title(f'{long_name} ({var}) - {n_subsample} samples {sampling_type} to centroids', fontsize=14, fontweight='bold')
        axes.set_xlabel('Cloud Class Label', fontsize=14)
        # Increase tick label font size
        axes.tick_params(axis='both', which='major', labelsize=12)  
        
        # Save figure
        fig.savefig(f'{output_path}barplot_{var}_{run_name}_{n_subsample}_{sampling_type}_{stat}.png', bbox_inches='tight')
        print(f'Figure saved: {output_path}{var}_bar_{n_subsample}.png')

        # Close Fig
        plt.close()


