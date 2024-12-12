import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
from glob import glob
import re
import numpy as np

# Paths to CMSAF cloud properties crops
cloud_properties_path = '/data1/crops/cmsaf_2013-2014-2015-2016_expats/nc_clouds/'
cloud_properties_crop_list = sorted(glob(cloud_properties_path + '*.nc'))
n_samples = len(cloud_properties_crop_list)

sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'

run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
output_path = f'/home/Daniele/fig/{run_name}/{sampling_type}/'
#Load the path and labels of the nc crops
csv_file = f'{output_path}crop_list_{run_name}_{n_samples}_{sampling_type}.csv'

time_step = 15  # Time step in minutes
n_steps = 10  # Number of time steps to look ahead this should correspond to 2.5 hours
classes = list(range(9))  # Class labels (0 to 8 for a 3x3 plot)
time_delta = dt.timedelta(minutes=time_step * n_steps)  # Calculate total time offset

# Load data
df = pd.read_csv(csv_file)

def extract_timestamp(file_path):
    # Extract the filename
    filename = file_path.split('/')[-1]
    
    # Use regex to capture date and time patterns
    match = re.search(r'(\d{8})_(\d{2}:\d{2})', filename)
    
    if match:
        date_str, time_str = match.groups()
        
        # Combine and parse date and time
        timestamp = dt.datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H:%M")
        #print(timestamp)
        return timestamp
    else:
        return None

df['timestamp'] = df['path'].apply(extract_timestamp)

# Plotting setup
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

# Generate distribution for each class
for i, class_label in enumerate(classes):
    # Get rows with the current class label
    df_class = df[df['label'] == class_label]

    # Initialize a dictionary to store count of labels at time delta
    label_distribution = {label: 0 for label in classes}

    # Iterate over each row in df_class
    for _, row in df_class.iterrows():
        # Find crops n steps after the current crop
        future_time = row['timestamp'] + time_delta
        future_crops = df[(df['timestamp'] == future_time)]

        # Count label occurrences in future crops
        for _, future_row in future_crops.iterrows():
            future_label = future_row['label']
            label_distribution[future_label] += 1

    # Convert distribution to a DataFrame for easier plotting
    dist_df = pd.DataFrame.from_dict(label_distribution, orient='index', columns=['count'])
    dist_df.reset_index(inplace=True)
    dist_df.columns = ['label', 'count']

    # Plot label distribution for the current class
    sns.barplot(data=dist_df, x='label', y='count', ax=axes[i])
    axes[i].set_title(f'Class {class_label}')
    axes[i].set_xlabel('Future Label')
    axes[i].set_ylabel('Count')

# Adjust layout
plt.tight_layout()
plt.suptitle('Distribution of Labels at Future Time Steps by Class', y=1.02, fontsize=16)
fig.savefig(f'{output_path}transition_probability_{sampling_type}_{n_steps}.png', bbox_inches='tight')


# Plot the transition probability matrix

# Initialize a matrix to store normalized transition probabilities
transition_matrix = np.zeros((len(classes), len(classes)))

# Compute the transition matrix
for i, class_label in enumerate(classes):
    # Get rows with the current class label
    df_class = df[df['label'] == class_label]

    # Initialize a dictionary to store count of labels at time delta
    label_distribution = {label: 0 for label in classes}

    # Iterate over each row in df_class
    for _, row in df_class.iterrows():
        # Find crops n steps after the current crop
        future_time = row['timestamp'] + time_delta
        future_crops = df[(df['timestamp'] == future_time)]

        # Count label occurrences in future crops
        for _, future_row in future_crops.iterrows():
            future_label = future_row['label']
            label_distribution[future_label] += 1

    # Normalize using the total count possible (n_crops_in_class * n_steps)
    total_transitions_possible = len(df_class) * n_steps
    if total_transitions_possible > 0:
        for j, dest_label in enumerate(classes):
            transition_matrix[i, j] = label_distribution[dest_label] / total_transitions_possible

# Plot the transition matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    transition_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="Blues", 
    xticklabels=classes, 
    yticklabels=classes
)
plt.title("Transition Probability Matrix")
plt.xlabel("Destination Class")
plt.ylabel("Originating Class")
plt.savefig(f'{output_path}transition_matrix_{sampling_type}_{n_steps}_normalized.png', bbox_inches='tight')
#plt.show()

# Threshold for highlighting transitions (1 / number of classes)
threshold = 1 / len(classes)

# Mask transition probabilities below the threshold
highlighted_matrix = np.where(transition_matrix > threshold, transition_matrix, 0)

# Plot the highlighted transition matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    highlighted_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="Reds", 
    xticklabels=classes, 
    yticklabels=classes, 
    mask=(highlighted_matrix == 0)  # Mask cells with zero values
)
plt.title("Transition Probabilities Above Random Threshold")
plt.xlabel("Destination Class")
plt.ylabel("Originating Class")
plt.savefig(f'{output_path}transition_matrix_highlighted_{sampling_type}_{n_steps}.png', bbox_inches='tight')
#plt.show()