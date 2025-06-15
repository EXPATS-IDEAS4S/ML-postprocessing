import numpy as np
import pandas as pd
from glob import glob
import os
import sys

sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))
from feature_space_plot_functions import name_to_rgb, scale_to_01_range

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

reduction_method = 'tsne' #'tsne
run_name = 'dcv2_ir108_100x100_k9_expats_35k_nc'
random_state = '3' #all visualization were made with random state 3
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all''
file_extension = 'nc'  # Image file extension
epoch = 500  # Epoch number for the run
FROM_CROP_STATS = False  # If True, use crop stats to get labels and paths, otherwise jsut crop list 

# Get number of samples
if sampling_type == 'all':
    image_path = f'/data1/crops/{run_name}/{file_extension}/1/' 
    crop_path_list = sorted(glob(image_path+'*.'+file_extension))
    n_samples = len(crop_path_list)
else:
    n_samples = 1000  # Number of samples per cluster

# Path to fig folder for outputs
output_path = f'/data1/fig/{run_name}/epoch_{epoch}/{sampling_type}/'

# Open the T-SNE file
#tsne_path = f'/data1/fig/{run_name}/epoch_{epoch}/{sampling_type}/'
filename = f'{reduction_method}_pca_cosine_perp-50_{run_name}_{random_state}_epoch_{epoch}.npy' 
tsne = np.load(output_path+filename)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# Get index of the crops
# filename1 = 'rank0_chunk0_train_heads_inds.npy' 
# path_feature = f'/data1/runs/{run_name}/features/'
# data1 = np.load(path_feature+filename1)
#print(data1)
data1 = np.arange(n_samples)

# Create a DataFrame for T-SNE coordinates
df_tsne = pd.DataFrame({'Component_1': tx, 'Component_2': ty, 'crop_index': data1})
#print(df_tsne)

#Load the path and labels of the nc crops
if FROM_CROP_STATS:
    # Load crop stats file
    df_labels = pd.read_csv(f'{output_path}crops_stats_{run_name}_{sampling_type}_{n_samples}.csv')
else:
    # Load crop list file
    df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{sampling_type}_{n_samples}.csv')


# Get the list of index from  the dataframe in crescent order
data_index = sorted(df_labels.crop_index.values)

#Filter df_tsen usinf the data_index
df_tsne = df_tsne[df_tsne.crop_index.isin(data_index)]
#print(df_tsne)

# Order df_labels by crop_index
df_labels = df_labels.set_index('crop_index').loc[data_index].reset_index()
#print(df_labels)

# merge df_tsne and df_labels by crop_index and remove one crop index column
df_tsne = df_tsne.set_index('crop_index').join(df_labels.set_index('crop_index')).reset_index()
#print(df_tsne) 

#Find the index with labels = -100
#print(df_tsne[df_tsne['label'] == -100])

# Remore rows with label = -100
df_tsne = df_tsne[df_tsne['label'] != -100]
#print(df_tsne)

# Map labels to colors
df_tsne['color'] = df_tsne['label'].map(lambda x: colors_per_class1_names[str(int(x))])
print(df_tsne)

#Save the merged dataframe
if FROM_CROP_STATS:
    df_tsne.to_csv(f'{output_path}merged_tsne_crop_stats_{run_name}_{sampling_type}_{random_state}_epoch_{epoch}.csv', index=False)
else:
    df_tsne.to_csv(f'{output_path}merged_tsne_crop_list_{run_name}_{sampling_type}_{random_state}_epoch_{epoch}.csv', index=False)

