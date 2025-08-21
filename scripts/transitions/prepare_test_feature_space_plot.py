
import numpy as np
import pandas as pd
from glob import glob
import torch

from embedding_plotting_func import plot_embedding_crops_new, plot_embedding_dots_iterative_test_msg_icon, scale_to_01_range, name_to_rgb, extract_hour, plot_embedding_dots, plot_embedding_filled, plot_embedding_crops, plot_embedding_dots_iterative_case_study

training_run =  'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
test_run = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon'

random_state = '3' #all visualization were made with random state 3

output_path = f'/home/Daniele/fig/{test_run}/'

reduction_method = 'tsne' # Options: 'tsne', 'isomap',
n_random_samples = None #30000

case_study_msg = False
case_study_icon = False
centroids = False




#### Load feature vectors for the training crops ###


# load list of training crops
image_train_path = f'/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA/1/'
crop_train_path_list = sorted(glob(image_train_path+'*.tif'))

# Load the feature indices, targets, and embeddings
filename1 = 'rank0_chunk0_train_heads_inds.npy' 
filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

path_feature = f'/data1/runs/{training_run}/features/'

checkpoints_path = f'/data1/runs/{training_run}/checkpoints/'  

assignments = torch.load(checkpoints_path+'assignments_800ep.pt',map_location='cpu')
sample_list = np.load(checkpoints_path+'samples_k7_800ep.npy')

distances = torch.load(checkpoints_path+'distance_800ep.pt',map_location='cpu')

data1 = np.load(path_feature+filename1)

# Create a DataFrame for labels
df_labels = pd.DataFrame({'y': '', 'index': data1})
df_labels.set_index('index', inplace=True)

labels = assignments[0, :].cpu().numpy()
selected_labels = labels[data1]

# add the class labels to the dataframe
df_labels['y'] = selected_labels

# Add the 'distance' column
dist = distances[0, :].cpu().numpy()
df_labels['distance'] = dist[data1]

# add the dataset column
#df_labels['dataset'] = 'train'

# add the sample type column
df_labels['vector_type'] = 'msg'

# Using a list comprehension to get elements at indices in data1
selected_elements = [crop_train_path_list[i] for i in data1]

df_labels['location'] = selected_elements

df_labels['case_study'] = False

print(df_labels)


#TODO add centroids from the training

#centroids = torch.load(checkpoints_path+'centroids0.pt',map_location='cpu')
#centroids_np = centroids.numpy()  # Convert to numpy array if necessary




### Load feature vectors for the test crops ###

image_test_path = f'/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA_test_msg_icon/1/' #cot_2013_128_germany, ir108_2013_128_germany
#print(len(crop_path_list))
crop_test_path_list = sorted(glob(image_test_path+'*.tif'))
print(len(crop_test_path_list))


case_study_msg_crops = sorted(glob(image_test_path+'IR_108_128x128_20220915_*_200K-300K_greyscale_CMA.tif'))
case_study_icon_crops = sorted(glob(image_test_path+'cropped_icon_*_200K-300K_greyscale.tif'))
print(len(case_study_msg_crops))
print(len(case_study_icon_crops))

# Load the feature indices, targets, and embeddings
filename1 = 'rank0_chunk0_train_heads_inds.npy' 
filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

path_feature = f'/data1/runs/{test_run}/features/'

checkpoints_path = f'/data1/runs/{test_run}/checkpoints/'  

assignments = torch.load(checkpoints_path+'assignments_800ep.pt',map_location='cpu')
sample_list = np.load(checkpoints_path+'samples_k7_800ep.npy')
distances = torch.load(checkpoints_path+'distance_800ep.pt',map_location='cpu')

data1 = np.load(path_feature+filename1)

# Create a DataFrame for labels
df_labels_test = pd.DataFrame({'y': '', 'index': data1})
df_labels_test.set_index('index', inplace=True)

labels = assignments[0, :].cpu().numpy()
selected_labels = labels[data1]

# add the class labels to the dataframe
df_labels_test['y'] = selected_labels

# Add the 'distance' column
dist = distances[0, :].cpu().numpy()
df_labels_test['distance'] = dist[data1]

# add the dataset column
#df_labels_test['dataset'] = 'test'

# Using a list comprehension to get elements at indices in data1
#selected_elements_icon = [crop_test_path_list[i] for i in data1 ] 
df_labels_test['location'] = crop_test_path_list


# Add the 'case_study' column based on whether 'location' is in 'case_study_crops'
df_labels_test['case_study_msg'] = df_labels_test['location'].isin(case_study_msg_crops)
df_labels_test['case_study_icon'] = df_labels_test['location'].isin(case_study_icon_crops)
print(df_labels_test['case_study_msg'].sum())
print(df_labels_test['case_study_icon'].sum())

# Add the 'vector_type' column based on whether 'location' is in 'case_study_msg_crops'
df_labels_test['vector_type'] = np.where(df_labels_test['case_study_icon'], 'icon', 'msg')

# Add the 'case_study' column based on the OR operation of 'case_study_msg' and 'case_study_icon'
df_labels_test['case_study'] = df_labels_test['case_study_msg'] | df_labels_test['case_study_icon']

print(df_labels_test['case_study'].sum())

# Drop the 'case_study_msg' and 'case_study_icon' columns
df_labels_test.drop(columns=['case_study_msg', 'case_study_icon'], inplace=True)

# Drop rows where 'case_study' is False
df_labels_test = df_labels_test[df_labels_test['case_study']]

#print(df_labels_test)


# Reset the index of df_labels_test
df_labels_test.reset_index(drop=True, inplace=True)

# Concatenate df_labels and df_labels_test
df_labels = pd.concat([df_labels, df_labels_test], ignore_index=True)

print(df_labels)


### Open dimensionality reduced feature space ###

dim_red_features = np.load(f'{output_path}/tsne_pca_cosine_{test_run}_{random_state}.npy')
print(dim_red_features.shape)
print(dim_red_features)

df_labels['Component_1'] = dim_red_features[:, 0]
df_labels['Component_2'] = dim_red_features[:, 1]

print(df_labels)

#save the dataframe to a csv file
df_labels.to_csv(f'{output_path}reduced_features_labels_{test_run}.csv', index=False)




