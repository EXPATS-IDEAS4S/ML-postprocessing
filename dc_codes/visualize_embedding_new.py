
import numpy as np
import pandas as pd
from glob import glob
import torch

from embedding_plotting_func import plot_average_crop_shapes, plot_embedding_crops_table, plot_embedding_crops_new, plot_embedding_dots_iterative_test_msg_icon, scale_to_01_range, name_to_rgb, extract_hour, plot_embedding_dots, plot_embedding_filled, plot_embedding_crops, plot_embedding_dots_iterative_case_study

run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3' #all visualization were made with random state 3
sampling_type = 'all' 
reduction_method = 'tsne' # Options: 'tsne', 'isomap',

output_path = f'/data1/fig/{run_name}/{sampling_type}/'
filename = f'{reduction_method}_pca_cosine_{run_name}_{random_state}.npy'

# List of the image crops
image_crops_path = f'/data1/crops/{run_name}/1/'
list_image_crops = sorted(glob(image_crops_path+'*.tif'))
n_samples = len( list_image_crops)
print('n samples: ', n_samples)

# Read data
if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:
    n_subsample = 1000

# Open csv file with already labels and dim red features
df_labels = pd.read_csv(f'{output_path}merged_tsne_variables_{run_name}_{sampling_type}_{random_state}.csv')
#list all columns name
df_labels = df_labels.loc[:, ~df_labels.columns.str.contains('^color')]
print(df_labels.columns)


#Remove column called olor


# # Define the color names for each class from dataframe

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

# path_msg_images = '/data1/other_data/Case_Studies/Marche_Flood_22/MSG/image_to_plot/'
# # add the path to the images for the case study
# #df_labels['msg_image_path'] = df_labels['case_study_msg'].map(lambda x: path_msg_images if x else '')
# df_labels['msg_image_path'] = df_labels.apply(
# lambda row: path_msg_images if row['vector_type'] == 'msg' and row['case_study'] else '',
# axis=1
# )


# path_icon_images = '/data1/other_data/Case_Studies/Marche_Flood_22/ICON/image_to_plot/'
# # add the path to the images for the case study
# #df_labels['icon_image_path'] = df_labels['case_study_icon'].map(lambda x: path_icon_images if x else '')
# df_labels['icon_image_path'] = df_labels.apply(
# lambda row: path_icon_images if row['vector_type'] == 'icon' and row['case_study'] else '',
# axis=1
# ) 

# Apply the extraction function to get the hours
#df_labels['hour'] = df_labels['location'].apply(extract_hour)

# Filter out invalid labels (-100)
df_subset1 = df_labels[df_labels['label'] != -100]

# Map labels to colors
df_subset1['color'] = df_subset1['label'].map(lambda x: colors_per_class1_names[str(int(x))])

# Sample 20,000 points for plotting
df_subset2 = df_subset1.sample(n=200)

print(df_subset2)

# Plot embedding with dots

#plot_embedding_dots(df_subset2, colors_per_class1_names, output_path, filename)
#plot_embedding_filled(df_subset2, colors_per_class1_names, output_path, filename, df_subset)
#plot_embedding_dots_iterative_case_study(df_subset1, colors_per_class1_names, output_path+'trajectory_iter/', filename, df_subset1)
#plot_embedding_dots_iterative_test_msg_icon(df_subset1, colors_per_class1_names, output_path+'trajectory_iter/', filename, df_subset1, legend=True)
#plot_embedding_crops_new(df_subset2, output_path, filename)
#plot_embedding_crops_table(df_subset1, output_path, filename, n=10 ,selection='random')
plot_average_crop_shapes(df_subset1, output_path+'avarage_maps/', filename, n=50, selection="closest", alpha=0.05)


