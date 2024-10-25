import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from glob import glob

def compare_label_distributions(run_name_1, run_name_2, csv_path_1, csv_path_2, output_path):
    # Load the data for both runs
    df_run1 = pd.read_csv(csv_path_1)
    df_run2 = pd.read_csv(csv_path_2)
    
    # Create a mapping of paths to labels in the second run
    path_to_label_run2 = dict(zip(df_run2['path'], df_run2['label']))
    
    # Prepare a dictionary to store the cross-distribution counts
    cross_distribution = {label: [] for label in df_run1['label'].unique()}

    # For each label in the first run, collect the mapped labels in the second run
    for label in df_run1['label'].unique():
        # Filter rows with the current label
        df_label = df_run1[df_run1['label'] == label]
        
        # Map each path in the current label of run 1 to the corresponding label in run 2
        mapped_labels = [path_to_label_run2.get(path, None) for path in df_label['path']]
        
        # Count occurrences of each mapped label and store them in cross_distribution
        cross_distribution[label] = Counter(mapped_labels)

    # Plot the distributions in a 3x3 grid (assuming max of 9 unique labels in run 1)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Label Distribution Comparison: {run_name_1} to {run_name_2}", fontsize=16)
    
    for idx, (label, label_counts) in enumerate(cross_distribution.items()):
        row, col = divmod(idx, 3)  # Get subplot row and column indices

        # Create a histogram for the current label's distribution in run 2
        sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), ax=axes[row, col])
        axes[row, col].set_title(f"Run 1 Label {label}")
        axes[row, col].set_xlabel("Run 2 Labels")
        axes[row, col].set_ylabel("Count")

    # Hide unused subplots
    for idx in range(len(cross_distribution), 9):
        fig.delaxes(axes.flat[idx])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the main title
    fig.savefig(f'{output_path}label_distribution_comparison_{run_name_1}_to_{run_name_2}.png')


run_1 = '10th-90th'
run_2 = '10th-90th_CMA'

sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'

# Paths to CMSAF cloud properties crops
cloud_properties_path = '/data1/crops/cmsaf_2013-2014_expats/nc_clouds/'
cloud_properties_crop_list = sorted(glob(cloud_properties_path + '*.nc'))

n_samples = len(cloud_properties_crop_list)

output_path = F'/home/Daniele/fig/cma_analysis/{run_1}/{sampling_type}/'

csv_run_1 = f'/home/Daniele/fig/cma_analysis/{run_1}/{sampling_type}/crop_list_{run_1}_{n_samples}_{sampling_type}.csv'
csv_run_2 = f'/home/Daniele/fig/cma_analysis/{run_2}/{sampling_type}/crop_list_{run_2}_{n_samples}_{sampling_type}.csv'

# Example usage
compare_label_distributions(
    run_name_1=run_1,
    run_name_2=run_2,
    csv_path_1=csv_run_1,
    csv_path_2=csv_run_2,
    output_path=output_path
)
