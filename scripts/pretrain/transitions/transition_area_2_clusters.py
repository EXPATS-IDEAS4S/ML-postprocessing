

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
no sense, or doing it on a the distances of the 2d space, so I need to reduce the space with the centroids of the 2 clusters,
or just plot the gomitoli plo, showing the transition between the 2 clusters 
'''
# Example input: Replace with your actual file or dataset
data_path = '/home/Daniele/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/all/physical_feature_vectors_dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_all_67425_50.csv'

sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'

run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
output_path = f'/home/Daniele/fig/{run_name}/{sampling_type}/'

# Convert to DataFrame
df = pd.read_csv(data_path)
print(df)

var='ctp'


# Define the two labels to filter
label1 = 1
label2 = 2

# Filter the dataframe for rows with the two specified labels
filtered_df = df[df['label'].isin([label1, label2])]

# Split into two subsets based on the labels
df_label1 = filtered_df[filtered_df['label'] == label1].reset_index(drop=True)
df_label2 = filtered_df[filtered_df['label'] == label2].reset_index(drop=True)

# Ensure the two subsets have the same number of rows
min_len = min(len(df_label1), len(df_label2))
df_label1 = df_label1.iloc[:min_len]
df_label2 = df_label2.iloc[:min_len]

# Compute the metric d1/(d1 + d2) using the 'distance' column
d1 = df_label1['distance']
d2 = df_label2['distance']
metric = d1 / (d1 + d2)

# Create a new dataframe with the metric and a selected variable
result_df = pd.DataFrame({
    'metric': metric,
    'value': df_label1[var]  # Replace 'ctp' with the column to analyze
})

# Sort by the computed metric
result_df = result_df.sort_values(by='metric')

# Plot the ordered values
plt.figure(figsize=(10, 6))
plt.plot(result_df['metric'], result_df['value'], marker='o', color='blue')#, label=var)
plt.xlabel("Normalized Metric (d1 / (d1 + d2))")
plt.ylabel(var)
plt.title("Ordered Values by Normalized Metric")
#plt.legend()
plt.grid()
#plt.show()

plt.savefig(f'{output_path}transition_analisys_clusters_{label1}{label2}_{var}_{sampling_type}.png', bbox_inches='tight')