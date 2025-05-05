##########################################################
## plot distr for 2 relevant varables in a scatterplot  ##
##########################################################
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np
import cmcrameri.cm as cmc

from aux_functions_from_buckets import extract_variable_values, compute_categorical_values
from get_data_from_buckets import Initialize_s3_client
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL


# Initialize S3 client
BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'
BUCKET_CROP_MSG = 'expats-msg-training'

# === USER CONFIGURATION ===
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
sampling_type = 'closest'  # 'all' or 'closest'
n_subsamples = 1000
heatmap = True  # Set True for heatmaps, False for scatterplots

s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

# List of variables to plot
varslist = ['cot', 'cth', 'cma', 'cph']
categ_vars = ['cma', 'cph']  # Categorical variables

# Optional: variable info for nicer labels
variable_info = {
    'cot': {'long_name': 'Cloud Optical Thickness', 'units': None, 'dir': 'incr', 'log': True, 'limit': (0.01, 155), 'stat': '95th_percentile', 'threshold': None},
    'cth': {'long_name': 'Cloud Top Height', 'units': 'm', 'dir': 'incr', 'log': False, 'limit':(0, 17000),  'stat': '95th_percentile', 'threshold': None},
    'cma': {'long_name': 'Cloud Cover', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1.1), 'stat': 'cloud_ratio', 'threshold': 0.},
    'cph': {'long_name': 'Ice ratio', 'units': None, 'dir': 'incr', 'log': False, 'limit': (0, 1.1), 'stat': 'ice_ratio', 'threshold': 0.},
}

additional_vars = {
     '6.2-10.8': {'long_name': 'WV 6.2 - IR 10.8 µm', 'units': 'K', 'dir': 'decr', 'log': False, 'stat': 'OT_ratio', 'threshold': 0.},
     'prec_ratio': {'long_name': 'Precipitation Ratio', 'units': None, 'dir': 'incr', 'log': False, 'stat': 'wet_ratio', 'threshold': 0.1}
}

    
# Read crop data
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_subsamples}_{sampling_type}.csv')

# Remove the rows with label invalid (-100)
df = df_labels[df_labels['label'] != -100]
print(df)

# Take a sample (for testing)
#df = df.sample(n=100)

labels = sorted(df['label'].unique())

# === PLOTTING LOOP ===
for var_1, var_2 in combinations(varslist, 2):
    print(f"Processing pair: {var_1} vs {var_2}")

    for label in labels:
        df_label = df[df['label'] == label]
        #df_label = df_label.sample(n=10)
        #print(df_label)
        
        # Initialize data list for the scatterplot
        data_list_1 = []
        data_list_2 = []

        #exctract values for each row of the dataframe
        for index, row in df_label.iterrows():
            values_1 = extract_variable_values(row, var_1, s3, BUCKET_CROP_MSG, BUCKET_IMERG_NAME, BUCKET_CMSAF_NAME)
            values_2 = extract_variable_values(row, var_2, s3, BUCKET_CROP_MSG, BUCKET_IMERG_NAME, BUCKET_CMSAF_NAME)
            if var_1 in categ_vars:
                values_1 = compute_categorical_values(values_1, var_1)
            else:
                #compute 95th percentile
                values_1 = np.nanpercentile(values_1, 95)
            #print(f"Values for {var_1}: {values_1}")
            #append values to the list
            data_list_1.append(values_1)
            if var_2 in categ_vars:
                values_2 = compute_categorical_values(values_2, var_2)
            else:
                #compute 95th percentile
                values_2 = np.nanpercentile(values_2, 90)
            #print(f"Values for {var_2}: {values_2}")
            #append values to the list
            data_list_2.append(values_2)
        
        print(f"Data list 1: {data_list_1}")
        print(f"Data list 2: {data_list_2}")
        
        x = data_list_1
        y = data_list_2

        fig, ax = plt.subplots(figsize=(6, 5))

        if heatmap:
            hb = ax.hexbin(x, y, gridsize=40, cmap=cmc.buda, mincnt=1)
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('Counts')
        else:
            sns.scatterplot(x=x, y=y, ax=ax, s=20, alpha=0.5)

        # Set axis labels using metadata
        def format_label(var):
            info = variable_info.get(var, {})
            name = info.get('long_name', var)
            units = info.get('units', '')
            return f"{name} [{units}]" if units else name
        
        # Set axis limits using metadata
        ax.set_xlim(variable_info.get(var_1, {}).get('limit', (0, 1)))
        ax.set_ylim(variable_info.get(var_2, {}).get('limit', (0, 1)))

        #Apply log scale if specified
        if variable_info.get(var_1, {}).get('log', False):
            ax.set_xscale('log')
        if variable_info.get(var_2, {}).get('log', False):
            ax.set_yscale('log')

        ax.set_xlabel(format_label(var_1), fontsize=12)
        ax.set_ylabel(format_label(var_2), fontsize=12)
        ax.set_title(f'{format_label(var_1)} vs {format_label(var_2)} | Label {label}', fontsize=12, fontweight='bold')

        # Reverse axes if direction is 'decr'
        if variable_info.get(var_1, {}).get('dir') == 'decr':
            ax.invert_xaxis()
        if variable_info.get(var_2, {}).get('dir') == 'decr':
            ax.invert_yaxis()

        # Save plot
        out_dir = f'{output_path}scatterplots/label_{label}/'
        os.makedirs(out_dir, exist_ok=True)
        if heatmap:
            fname = f"{out_dir}{var_1}_vs_{var_2}_label{label}_{run_name}_{sampling_type}_heatmap.png"
        else:
            fname = f"{out_dir}{var_1}_vs_{var_2}_label{label}_{run_name}_{sampling_type}.png"
        fig.savefig(os.path.join(output_path, fname), dpi=300, bbox_inches='tight')
        plt.close()

print("All plots generated.")

#nohup 1991256 and 1991579