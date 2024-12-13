
import numpy as np
import pandas as pd
from glob import glob
import torch

from embedding_plotting_func import plot_embedding_crops_new, plot_embedding_dots_iterative_test_msg_icon, scale_to_01_range, name_to_rgb, extract_hour, plot_embedding_dots, plot_embedding_filled, plot_embedding_crops, plot_embedding_dots_iterative_case_study

scale = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon'
random_state = '3' #all visualization were made with random state 3
reduction_method = 'tsne' # Options: 'tsne', 'isomap',

output_path = f'/home/Daniele/fig/{scale}/'
filename = f'{reduction_method}_pca_cosine_{scale}_{random_state}.npy'

# Open csv file with already labels and dim red features
df_labels = pd.read_csv(f'{output_path}reduced_features_labels_{scale}.csv')
print(df_labels)

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


# add column with path to images for case studied
# Add a new column based on condition

path_msg_images = '/data1/other_data/Case_Studies/Marche_Flood_22/MSG/image_to_plot/'
# add the path to the images for the case study
#df_labels['msg_image_path'] = df_labels['case_study_msg'].map(lambda x: path_msg_images if x else '')
df_labels['msg_image_path'] = df_labels.apply(
lambda row: path_msg_images if row['vector_type'] == 'msg' and row['case_study'] else '',
axis=1
)


path_icon_images = '/data1/other_data/Case_Studies/Marche_Flood_22/ICON/image_to_plot/'
# add the path to the images for the case study
#df_labels['icon_image_path'] = df_labels['case_study_icon'].map(lambda x: path_icon_images if x else '')
df_labels['icon_image_path'] = df_labels.apply(
lambda row: path_icon_images if row['vector_type'] == 'icon' and row['case_study'] else '',
axis=1
) 

# Apply the extraction function to get the hours
#df_labels['hour'] = df_labels['location'].apply(extract_hour)

# Filter out invalid labels (-100)
df_subset1 = df_labels[df_labels['y'] != -100]

# Map labels to colors
df_subset1['color'] = df_subset1['y'].map(lambda x: colors_per_class1_names[str(int(x))])

# Sample 20,000 points for plotting
df_subset2 = df_subset1.sample(n=20000)

print(df_subset2)

# Plot embedding with dots

#plot_embedding_dots(df_subset2, colors_per_class1_names, output_path, filename)
#plot_embedding_filled(df_subset2, colors_per_class1_names, output_path, filename, df_subset)
#plot_embedding_dots_iterative_case_study(df_subset1, colors_per_class1_names, output_path+'trajectory_iter/', filename, df_subset1)
plot_embedding_dots_iterative_test_msg_icon(df_subset1, colors_per_class1_names, output_path+'trajectory_iter/', filename, df_subset1, legend=True)
#plot_embedding_crops_new(df_subset2, output_path, filename)



