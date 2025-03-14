import numpy as np
import pandas as pd
from glob import glob
import os
import sys

sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))
from feature_space_plot_functions import name_to_rgb, scale_to_01_range, colors_per_class1_names

reduction_method = 'tsne' #'tsne
run_name = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'
random_state = '3' #all visualization were made with random state 3
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all''

# Get number of samples
if sampling_type == 'all':
    image_path = f'/data1/crops/{run_name}/1/' 
    crop_path_list = sorted(glob(image_path+'*.tif'))
    n_samples = len(crop_path_list)
else:
    n_subsample = 100  # Number of samples per cluster

# Path to fig folder for outputs
output_path = f'/data1/fig/{run_name}/{sampling_type}/'

# Open the T-SNE file
tsne_path = f'/data1/fig/{run_name}/'
filename = f'{reduction_method}_pca_cosine_{run_name}_{random_state}.npy' 
tsne = np.load(tsne_path+filename)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# Get index of the crops
filename1 = 'rank0_chunk0_train_heads_inds.npy' 
path_feature = f'/data1/runs/{run_name}/features/'
data1 = np.load(path_feature+filename1)
print(data1)

# Create a DataFrame for T-SNE coordinates
df_tsne = pd.DataFrame({'Component_1': tx, 'Component_2': ty, 'crop_index': data1})
print(df_tsne)

#Load the path and labels of the nc crops
#TODO change this with the dataframe containig the variable for the characterization
df_labels = pd.read_csv(f'{output_path}crop_list_{run_name}_{n_samples}_{sampling_type}.csv')
print(df_labels)

# Get the list of index from  the dataframe in crescent order
data_index = sorted(df_labels.crop_index.values)

#Filter df_tsen usinf the data_index
df_tsne = df_tsne[df_tsne.crop_index.isin(data_index)]
print(df_tsne)

# Order df_labels by crop_index
df_labels = df_labels.set_index('crop_index').loc[data_index].reset_index()
print(df_labels)

# merge df_tsne and df_labels by crop_index and remove one crop index column
df_tsne = df_tsne.set_index('crop_index').join(df_labels.set_index('crop_index')).reset_index()
print(df_tsne) 

#Find the index with labels = -100
#print(df_tsne[df_tsne['label'] == -100])

# Remore rows with label = -100
df_tsne = df_tsne[df_tsne['label'] != -100]
print(df_tsne)

# Map labels to colors
df_tsne['color'] = df_tsne['label'].map(lambda x: colors_per_class1_names[str(int(x))])
print(df_tsne)

# Create a column called 'cth-50'and fill it with random values from 1000 to 10000
#TODO remove this line when the variabels are computed
df_tsne['cth-50'] = np.random.randint(1000, 10000, df_tsne.shape[0])
print(df_tsne)

#Save the merged dataframe
df_tsne.to_csv(f'{output_path}merged_tsne_variables_{run_name}_{sampling_type}_{random_state}.csv', index=False)

