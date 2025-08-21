
import numpy as np
import pandas as pd
from glob import glob
import torch
import gc
import openTSNE

from embedding_plotting_func import plot_embedding_crops_new, plot_embedding_dots_iterative_test_msg_icon, scale_to_01_range, name_to_rgb, extract_hour, plot_embedding_dots, plot_embedding_filled, plot_embedding_crops, plot_embedding_dots_iterative_case_study

training_run =  'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
test_run = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon'

random_state = 3 #all visualization were made with random state 3

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

n_crops = len(crop_train_path_list)
n_dim = 128

filename3 = 'rank0_chunk0_train_heads_features.npy'
filename1 = 'rank0_chunk0_train_heads_inds.npy' 


common_path = f'/data1/runs/{training_run}/features/'  

data1 = np.load(common_path+filename1)
data3 = np.load(common_path+filename3)

data_train = np.reshape(data3,(n_crops,n_dim))

print(data_train.shape)
print(data_train)

# Create column names like dim_1, dim_2, ..., dim_n
column_names = [f'dim_{i+1}' for i in range(n_dim)]

# Create a DataFrame from the NumPy array
df_features_train = pd.DataFrame(data_train, columns=column_names, index=data1)

df_features_train['vector_type'] = 'msg'
df_features_train['case_study'] = False

print(df_features_train)



### Load feature for test crops ###

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

data1 = np.load(path_feature+filename1)

data3 = np.load(path_feature+filename3)

data_test = np.reshape(data3,(n_crops,n_dim))

# Create column names like dim_1, dim_2, ..., dim_n
column_names = [f'dim_{i+1}' for i in range(n_dim)]

# Create a DataFrame from the NumPy array
df_features_test = pd.DataFrame(data_test, columns=column_names, index=data1)

print(df_features_test)


# add the dataset column
#df_labels_test['dataset'] = 'test'

# Using a list comprehension to get elements at indices in data1
#selected_elements_icon = [crop_test_path_list[i] for i in data1 ] 
df_features_test['location'] = crop_test_path_list


# Add the 'case_study' column based on whether 'location' is in 'case_study_crops'
df_features_test['case_study_msg'] = df_features_test['location'].isin(case_study_msg_crops)
df_features_test['case_study_icon'] = df_features_test['location'].isin(case_study_icon_crops)
print(df_features_test['case_study_msg'].sum())
print(df_features_test['case_study_icon'].sum())

# Add the 'vector_type' column based on whether 'location' is in 'case_study_msg_crops'
df_features_test['vector_type'] = np.where(df_features_test['case_study_icon'], 'icon', 'msg')

# Add the 'case_study' column based on the OR operation of 'case_study_msg' and 'case_study_icon'
df_features_test['case_study'] = df_features_test['case_study_msg'] | df_features_test['case_study_icon']

print(df_features_test['case_study'].sum())


# Drop rows where 'case_study' is False
df_features_test = df_features_test[df_features_test['case_study']]

# Drop the 'case_study_msg' and 'case_study_icon' columns
df_features_test.drop(columns=['case_study_msg', 'case_study_icon', 'location'], inplace=True)


print(df_features_test)


# Reset the index of df_labels_test
df_features_test.reset_index(drop=True, inplace=True)

# Concatenate df_labels and df_labels_test
df_features = pd.concat([df_features_train, df_features_test], ignore_index=True)

print(df_features)

# Save the DataFrame to a CSV file
df_features.to_csv(f'{output_path}features_{test_run}.csv', index=False)




