
import numpy as np
from sklearn.manifold import Isomap
import pandas as pd
import random
import torch
import openTSNE

run_names = ['dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA']

reduction_method = 'isomap' # Options: 'tsne', 'isomap', 
#if using tsne select the random state
random_state = 3 # [0,3,16,23,57]

#Set to the number of n subsample if you want to use a random sample of the data
random_samples = True
n_random_samples = 10000

#Set True if you want to iclude centroids in the analysis
centroids = False

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

    if centroids:
        #Open centroids (TODO understand why centroids0) 
        path_centroids = f'/data1/runs/{run_name}/checkpoints/centroids0.pt'
        centroids = torch.load(path_centroids,map_location='cpu')
        centroids_np = centroids.numpy()  # Convert to numpy array if necessary
        print(f'Centroids shape: {np.shape(centroids_np)}')
        del centroids

        n_classes = centroids_np.shape[0]  #9
        centroids_np = np.reshape(centroids_np,(n_classes,n_features))  #DC value 664332, it should be the train num samples
        #print(centroids_np) 

        # Concatenate crops and centroids
        # Add a 'Type' column to indicate if the row is a 'crop' or 'centroid'
        crops_features = np.concatenate([data3, centroids_np], axis=0)
        print(f'Features with type shape: {np.shape(crops_features)}')
        type_labels = np.array(['crop'] * len(data3) + ['centroid'] * len(centroids_np))

        # Combine indices and centroids with a placeholder index for centroids (e.g., -1 for simplicity)
        #indices = np.concatenate([data1, -1 * np.ones(centroids_np.shape[0])])
        indices = np.concatenate([data1,range(n_classes) ]) # use class label as index for centroids

        #Define output filename
        output_filename = f'{reduction_method}_{run_name}_{n_components}d_cosine_{n_epochs}ep_{n_crops}samples_with_centroids.csv'
    else:
        crops_features = data3
        type_labels = np.array(['crop'] * len(data3))
        indices = data1
        output_filename = f'{reduction_method}_{run_name}_{n_components}d_cosine_{n_epochs}ep_{n_crops}samples.csv'

    #check how the data look without reduction (plot projection of the data)

    # # Step 1: Calculate variance along each dimension and select the top 3 dimensions
    # variances = np.var(crops_features, axis=0)
    # top_three_dims = np.argsort(variances)[:3]  # Indices of top 3 dimensions with highest variance

    # # Step 2: Generate all pairs of these 3 dimensions
    # from itertools import combinations
    # dimension_pairs = list(combinations(top_three_dims, 2))

    # # Step 3: Loop over each pair, create and save the plot
    # for dim_x, dim_y in dimension_pairs:
    #     # Extract the two selected dimensions for plotting
    #     data_x = crops_features[:, dim_x]
    #     data_y = crops_features[:, dim_y]

    #     # Separate data based on `type_labels`
    #     crop_points_x = data_x[type_labels == 'crop']
    #     crop_points_y = data_y[type_labels == 'crop']
    #     centroid_points_x = data_x[type_labels == 'centroid']
    #     centroid_points_y = data_y[type_labels == 'centroid']

    #     # Create plot
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(8, 6))

    #     # Plot 'crop' points as small blue dots
    #     plt.scatter(crop_points_x, crop_points_y, s=10, color='blue', alpha=0.5, label='Crop', marker='o')

    #     # Plot 'centroid' points as larger orange stars
    #     plt.scatter(centroid_points_x, centroid_points_y, s=100, color='orange', alpha=0.7, label='Centroid', marker='*')

    #     # Customize plot appearance
    #     plt.xlabel(f'Dimension {dim_x + 1} (High Variance)')
    #     plt.ylabel(f'Dimension {dim_y + 1} (High Variance)')
    #     plt.title(f"2D Projection Along Dimensions {dim_x + 1} and {dim_y + 1}")
    #     plt.legend()

    #     # Save the plot with a filename including the dimension numbers
    #     plot_filename = f"{output_path}embedding_2d_projection_dim_{dim_x + 1}_vs_dim_{dim_y + 1}.png"
    #     plt.savefig(plot_filename, bbox_inches='tight')
    #     print(f"Saved plot as {plot_filename}")

    #     # Close the plot to save memory
    #     plt.close()
    
    #apply the reduction method

    import gc    
    gc.collect()


    if reduction_method == 'isomap':
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
        X_transformed = embedding.fit_transform(crops_features)
    elif reduction_method == 'tsne':
        print('Processing tsne...')
        #tsne parameters here:
        #https://opentsne.readthedocs.io/en/stable/api/sklearn.html#openTSNE.sklearn.TSNE
        
        X_transformed = openTSNE.TSNE(perplexity=30,initialization="pca",metric="cosine",n_jobs=-1,random_state=random_state).fit(data)
        # from sklearn.manifold import TSNE
        # X_transformed = TSNE(n_components=2, random_state=random_state, perplexity=30, metric='cosine', init='pca').fit_transform(crops_features)
    else:
        print('reduction method not recognized')

    print(f'{reduction_method} calculated')
    print(X_transformed.shape)
    print(X_transformed)
    np.save(f'{output_path}{reduction_method}_pca_cosine_{run_name}_{random_state}.npy', X_transformed)

    # Create a DataFrame with the Isomap-transformed data and the random indices
    df = pd.DataFrame(X_transformed, columns=[f'Component_{i+1}' for i in range(n_components)])
    df['Type'] = type_labels
    df['Selected_Index'] = indices
    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv(f'{output_path}{output_filename}', index=False)
    print(f'{reduction_method} calculated and saved')
    # nohup

