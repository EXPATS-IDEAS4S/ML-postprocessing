"""
This script performs dimensionality reduction on high-dimensional feature data using the Isomap method, enabling
visualization in a lower-dimensional space. The script loads feature data, applies optional random sampling for 
subsetting, and uses Isomap to reduce the data to two components. The transformed data is then saved as a .npy file.

Modules:
    - numpy: For data manipulation and saving the reduced embeddings.
    - sklearn.manifold.Isomap: For applying Isomap dimensionality reduction on the feature data.
    - random: For generating random samples if data subsampling is enabled.

Parameters:
    - run_names: List of dataset identifiers to specify the source data for processing.
    - reduction_method: Method of dimensionality reduction ('isomap' here; other methods can be added if needed).
    - random_samples: Boolean to indicate if a random subset of the data should be used.
    - n_random_samples: Number of samples to select if random sampling is enabled.
    - n_features: Dimension of the original feature space.
    - n_epochs: Number of epochs for training (included for reference but not directly used in this script).
    - n_components: Number of dimensions for the output embeddings.

Workflow:
    1. Load feature and index data from specified file paths.
    2. Optionally, apply random sampling to reduce the dataset size.
    3. Reshape data if needed, preparing it for dimensionality reduction.
    4. Apply Isomap with cosine metric to create a 2D representation of the feature space.
    5. Save the transformed data to an output directory, with filenames indicating the reduction method and dataset.

Usage:
    - Adjust `run_names`, `n_random_samples`, and `n_components` as per data specifications.
    - Modify file paths in `common_path` and `output_path` to match the data and output structure.
    - If additional reduction methods are added, update `reduction_method` accordingly.

Note:
    - Garbage collection (`gc.collect()`) is included to manage memory usage, especially for large datasets.
    - Ensure compatibility with other reduction methods if extending this script.
"""


import numpy as np
from sklearn.manifold import Isomap
import random

run_names = ['dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon']

reduction_method = 'isomap' # Options: 'tsne', 'isomap', 
#if using tsne select the random state

#Set to the number of n subsample if you want to use a random sample of the data
random_samples = True
n_random_samples = 30000

#Set other parameters
n_features = 128 #Dimension of the feature space
n_epochs = 800 #Number of epochs for the training
n_components = 2 #Number of components to visualize for the embedding

for run_name in run_names:
    #Path to the features 
    common_path = f'/data1/runs/{run_name}/features/'  ##
    #Path to save the output
    output_path = f'/home/Daniele/fig/{run_name}/'
    
    #filenames
    filename1 = 'rank0_chunk0_train_heads_inds.npy' 
    #filename2 = 'rank0_chunk0_train_heads_targets.npy'
    filename3 = 'rank0_chunk0_train_heads_features.npy'

    #Open data
    data1 = np.load(common_path+filename1)
    #data2 = np.load(common_path+filename2)
    data3 = np.load(common_path+filename3)

    # get the number of crops
    n_crops = data1.shape[0]  #33792

    # Randomly select some rows
    if random_samples:
        random_indices = random.sample(range(n_crops), n_random_samples)
        n_crops = n_random_samples

        # Extract the corresponding values from data3
        data1 = data1[random_indices]
        data3 = data3[random_indices, :]
    #print(data3)
    # Reshape data if necessary
    data = np.reshape(data3,(n_crops,n_features))  #DC value 664332, it should be the train num samples
    print(f'Feature shape: {np.shape(data3)}')
    print(f'Index shape: {data1.shape}')
    
    # Clear memory
    import gc    
    gc.collect()


    print('Processing isomap...')
    #isomap from sklearn:
    #https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap

    #default parameters
    #class sklearn.manifold.Isomap(*, n_neighbors=5, radius=None, n_components=2, 
    #eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', 
    #n_jobs=None, metric='minkowski', p=2, metric_params=None)[source]

    #tol: tolerance in the eigenvalue solver to attain convergence.
    #While a lower value might result in a more accurate solution, it might also lengthen the computation time.

    #max_iter: The maximum number of times the eigenvalue solver can run. 
    #It continues if None is selected, unless convergence or additional stopping conditions are satisfied

    embedding = Isomap(n_components=n_components, metric='cosine', n_jobs=-1)#, n_neighbors=10)#eigen_solver='dense')
    X_transformed = embedding.fit_transform(data)

    print(f'{reduction_method} calculated')
    print(X_transformed.shape)
    print(X_transformed)

    np.save(f'{output_path}{reduction_method}_cosine_{run_name}_{n_crops}.npy', X_transformed)
    np.save(f'{output_path}{reduction_method}_cosine_{run_name}_{n_crops}_indeces.npy', data1)
    
    print(f'{reduction_method} calculated and saved')
    # nohup 1321124

