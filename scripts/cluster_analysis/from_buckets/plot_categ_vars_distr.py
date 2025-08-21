import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


# Configuration
reduction_method = 'tsne'
run_name = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'
random_state = '3'
sampling_type = 'all'
var_names = ['cma-None', 'cph-None']  # Variables to extract
long_names = ['cloud cover', 'ice ratio']  # Long names for variables
# Path to figure folder for outputs
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
output_dir = os.path.join(output_path, "categ_var_distr")
os.makedirs(output_dir, exist_ok=True)

# Open dataframe with t-SNE and variables merged together
df_merged = pd.read_csv(f'{output_path}merged_tsne_variables_{run_name}_{sampling_type}_{random_state}.csv')

# Get unique labels and their colors
label_colors = df_merged[['label', 'color']].drop_duplicates().set_index('label')['color'].to_dict()

# Plot distribution for each variable and label
sns.set_context("talk")  # Set larger default sizes
fontsize_title = 18
fontsize_labels = 14
fontsize_ticks = 12

for var, long_name in zip(var_names, long_names):
    for label in df_merged['label'].unique():
        plt.figure(figsize=(7, 5))

        # Extract values for the current label and variable
        values = df_merged[df_merged['label'] == label][var].dropna()

        # Create histogram plot
        sns.histplot(values, bins=50, element="bars",
                fill=True,
                edgecolor=None,
                color='blue',
                stat="probability")

        # Titles and labels
        plt.title(f"Class {label}", fontsize=fontsize_title)
        plt.xlabel(long_name, fontsize=fontsize_labels)
        plt.ylabel("Frequency", fontsize=fontsize_labels)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.grid(True, linestyle="--", alpha=0.5)

        # Save figure
        output_file = os.path.join(output_dir, f"{var}_distribution_label_{label}.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved: {output_file}")
