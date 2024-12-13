import numpy as np
import pandas as pd
from glob import glob
import torch
import matplotlib.pyplot as plt

from embedding_plotting_func import plot_embedding_dots_iterative_test_msg_icon, scale_to_01_range, name_to_rgb, extract_hour, plot_embedding_dots, plot_embedding_filled, plot_embedding_crops, plot_embedding_dots_iterative_case_study

# Name of the run
scale = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon'

output_path = f'/home/Daniele/fig/{scale}/'

case_study_msg = True
case_study_icon = True

image_path = f'/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA_test_msg_icon/1/' #cot_2013_128_germany, ir108_2013_128_germany
crop_path_list = sorted(glob(image_path+'*.tif'))
#print(len(crop_path_list))

if case_study_msg:
    #get only the case study images to plot
    #image_path_msg = '/data1/other_data/Case_Studies/Marche_Flood_22/MSG/image_to_plot/'
    #case_study_msg_images = sorted(glob(image_path_msg+'*.tif'))
    #get the path to the crops used to test the model
    case_study_msg_crops = sorted(glob(image_path+'IR_108_128x128_20220915_*_200K-300K_greyscale_CMA.tif'))
    #print(len(case_study_msg_crops))
    #print(case_study_msg_images)

if case_study_icon:
    # get path to images to plot
    #image_path_icon = '/data1/other_data/Case_Studies/Marche_Flood_22/ICON/image_to_plot/'
    #case_study_icon_images = sorted(glob(image_path_icon+'*.tif'))
    #get the path to the crops used to test the model
    case_study_icon_crops = sorted(glob(image_path+'cropped_icon_*_200K-300K_greyscale.tif'))
    #print(case_study_icon_images)
    #print(len(case_study_icon_crops))

# Load the feature indices, targets, and embeddings
filename1 = 'rank0_chunk0_train_heads_inds.npy' 
filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

path_feature = f'/data1/runs/{scale}/features/'

checkpoints_path = f'/data1/runs/{scale}/checkpoints/'  

assignments = torch.load(checkpoints_path+'assignments_800ep.pt',map_location='cpu')
sample_list = np.load(checkpoints_path+'samples_k7_800ep.npy')

data1 = np.load(path_feature+filename1)

print(data1.shape, data1)
print(assignments) #TODO it has 3 rows, DC used the first (why?)


# Define the color names for each class
colors_per_class1_names = {
    '0': 'darkgray', 
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green'
}

colors_per_class1_rgb = {k: name_to_rgb(v) for k, v in colors_per_class1_names.items()}
print(colors_per_class1_rgb)





# Create a DataFrame for labels
df_labels = pd.DataFrame({'y': '', 'index': data1})
df_labels.set_index('index', inplace=True)
print(df_labels)

labels = assignments[0, :].cpu().numpy()
selected_labels = labels[data1]


#for index in range(len(data1)):
#    df_labels.loc[index] = assignments[0, :].cpu()[index]

df_labels['y'] = selected_labels
print(df_labels)

print(type(data1))
print(data1)
# Using a list comprehension to get elements at indices in data1
selected_elements = [crop_path_list[i] for i in data1]

df_labels['location'] = selected_elements


if case_study_msg:
    # Add the 'case_study' column based on whether 'location' is in 'case_study_crops'
    df_labels['case_study_msg'] = df_labels['location'].isin(case_study_msg_crops)

if case_study_icon:
    df_labels['case_study_icon'] = df_labels['location'].isin(case_study_icon_crops)


print(df_labels)

# Filter out invalid labels (-100)
df_labels = df_labels[df_labels['y'] != -100]

unique_labels = sorted(df_labels['y'].unique())


# Plot class transition and class  distribution for the case study


# Filter rows where case_study_msg or case_study_icon is True
msg_data = df_labels[df_labels['case_study_msg']]
icon_data = df_labels[df_labels['case_study_icon']]

# Extract hours from the 'location' column using regex for msg and icon
msg_hours = msg_data['location'].str.extract(r'IR_108_\d+x\d+_\d{8}_(\d{2}:\d{2})')[0]
icon_hours = icon_data['location'].str.extract(r'cropped_icon_\d{4}-\d{2}-\d{2}T(\d{2}:\d{2})')[0]

# Convert extracted hours to a uniform numeric format (e.g., hours as float)
msg_hours = msg_hours.str.split(':').apply(lambda x: int(x[0])  if isinstance(x, list) else np.nan)
icon_hours = icon_hours.str.split(':').apply(lambda x: int(x[0]) if isinstance(x, list) else np.nan)

# Extract classes for msg and icon
msg_classes = msg_data['y']
icon_classes = icon_data['y']

# Create a histogram for class frequencies
msg_class_counts = msg_classes.value_counts().sort_index()
icon_class_counts = icon_classes.value_counts().sort_index()

# Set up the plot
fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 1]}, figsize=(12, 6))

# Initialize counts for all unique labels, defaulting to 0 if a label is missing
msg_class_counts = msg_classes.value_counts().reindex(unique_labels, fill_value=0)
icon_class_counts = icon_classes.value_counts().reindex(unique_labels, fill_value=0)

# Time transition plot (left)
ax[0].plot(msg_hours, msg_classes, label='MSG', color='red', marker='o', linestyle='-')
ax[0].plot(icon_hours, icon_classes, label='ICON', color='blue', marker='o', linestyle='--')
ax[0].set_xlabel('Time (Hour)', fontsize=14)
ax[0].set_ylabel('Class Labels', fontsize=14)
ax[0].set_title('Class Transitions Over Time', fontsize=16)
ax[0].grid(True)
ax[0].tick_params(axis='both', which='major', labelsize=12)

# Histogram (right, aligned with y-axis of the time transition plot)
bar_width = 0.4
indices = np.arange(len(unique_labels))  # Indices for all unique labels
ax[1].barh(
    indices - bar_width / 2, 
    msg_class_counts.values, 
    height=bar_width, 
    color='red', 
    alpha=0.7, 
    label='MSG'
)
ax[1].barh(
    indices + bar_width / 2, 
    icon_class_counts.values, 
    height=bar_width, 
    color='blue', 
    alpha=0.7, 
    label='ICON'
)

# Set y-ticks for all unique labels
ax[1].set_yticks(indices)
ax[1].set_yticklabels(unique_labels)
ax[1].set_xlabel('counts', fontsize=14)
ax[1].set_title('Class Occurence', fontsize=16)
# Adjust the limits of the y-axis to reduce blank space
y_min, y_max = -0.5, len(unique_labels) - 0.5  # Tighten the range
ax[0].set_ylim(y_min, y_max)  # Apply same limits to left plot
ax[1].set_ylim(y_min, y_max)


# Tick spacing adjustment
ax[1].tick_params(axis='y', labelsize=12, length=5)
ax[1].tick_params(axis='both', which='major', labelsize=12)

# Add a single legend for both plots to the right of the histogram
fig.legend(loc='center right', fontsize=14, labels=['MSG', 'ICON'], bbox_to_anchor=(1.02, 0.5))

# Adjust layout
fig.savefig(output_path+'class_transition_case_study.png', bbox_inches='tight')
