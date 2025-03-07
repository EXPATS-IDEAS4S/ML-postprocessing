"""
This script performs t-SNE (t-distributed Stochastic Neighbor Embedding) dimensionality reduction on high-dimensional
feature data to visualize it in a lower-dimensional space. The script loads feature data, reshapes it if necessary,
and applies t-SNE with specified parameters such as perplexity, initialization method, and metric. The resulting
embeddings are saved as .npy files for future use or visualization.

Modules:
    - numpy: For data manipulation and saving the embeddings.
    - openTSNE: For applying t-SNE on the feature data.
    - gc: To manage garbage collection and free up memory during data processing.

Parameters:
    - scales: List of dataset scale identifiers to load and process.
    - random_states: List of random seed values to ensure reproducibility of t-SNE results.
    - n_crops: Number of samples in the feature dataset (used to reshape the data).

Workflow:
    1. Load feature data from specified file paths.
    2. Reshape the data into the desired format (e.g., (n_crops, 128) for a 128-dimensional feature space).
    3. Apply t-SNE with parameters for initialization, metric, and random state.
    4. Save the generated 2D embeddings to an output directory with a filename that includes the scale and random state.

Usage:
    - Adjust `scales`, `random_states`, and `n_crops` to match the specific datasets and configurations.
    - Modify t-SNE parameters (e.g., perplexity, metric) as needed based on data characteristics.
    - Ensure that file paths in `common_path` and `output_path` are correctly set.
"""


import numpy as np
import os, glob
import pandas as pd
#from sklearn.manifold import TSNE
#from openTSNE import TSNE
#import torch

import openTSNE

scales = ['dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA']
random_states = [3] #[0,3,16,23,57] #old was 3, already computed

n_crops =  70094 #67473

#filename1 = 'rank0_chunk0_train_heads_inds.npy' 
#filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

import gc
gc.collect()


for scale in scales:
    for random_state in random_states:

        output_path = f'/data1/fig/{scale}/'
        os.makedirs(output_path, exist_ok=True)

        # # Open datafrae with the features
        # df_features = pd.read_csv(f'{output_path}features_{scale}.csv')
        # #extract the feature vectors from the DataFrame      
        # data3 = df_features.iloc[:, :-2].values

        common_path = f'/data1/runs/{scale}/features/'          

        #data1 = np.load(common_path+filename1)
        #data2 = np.load(common_path+filename2)
        data3 = np.load(common_path+filename3)
        #data = data3
        data = np.reshape(data3,(n_crops,128))  #DC value 664332, it should be the train num samples

        # path_centroids = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{scale}/checkpoints/centroids0.pt'
        # centroids = torch.load(path_centroids,map_location='cpu')
        # centroids_np = centroids.numpy()  # Convert to numpy array if necessary
        # print(f'Centroids shape: {np.shape(centroids_np)}')

        # n_classes = centroids_np.shape[0]  #9
        # centroids_np = np.reshape(centroids_np,(n_classes,128))  #DC value 664332, it should be the train num samples
        # #print(centroids_np) 

        # # Concatenate crops and centroids
        # # Add a 'Type' column to indicate if the row is a 'crop' or 'centroid'
        # data = np.concatenate([data, centroids_np], axis=0)


        #tsne parameters here:
        #https://opentsne.readthedocs.io/en/stable/api/sklearn.html#openTSNE.sklearn.TSNE
        #%time
        embedding_pca_cosine = openTSNE.TSNE(
            perplexity=30,
            initialization="pca",
            metric="cosine",
            n_jobs=-1,
            random_state=random_state,
        ).fit(data)

        #tsne = TSNE(n_components=2, random_state=5).fit_transform(data)

        np.save(f'{output_path}tsne_pca_cosine_{scale}_{random_state}.npy', embedding_pca_cosine)  ##

        print(f'{output_path}tsne_pca_cosine_{scale}_{random_state}.npy calculated and saved')

"""
embedding_annealing = openTSNE.TSNE(
    perplexity=500, metric="cosine", initialization="pca", n_jobs=-1, random_state=3
).fit(data)

embedding_annealing.affinities.set_perplexities([50])

embedding_annealing = embedding_annealing.optimize(250)

np.save(output_path+'/tsnegermany_pca_cosine_500annealing50_800ep.npy', embedding_annealing) ##


affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
    data,
    perplexities=[50, 500],
    metric="cosine",
    n_jobs=-1,
    random_state=3,
)

init = openTSNE.initialization.pca(data, random_state=42)

embedding_multiscale = openTSNE.TSNE(n_jobs=-1).fit(
    affinities=affinities_multiscale_mixture,
    initialization=init,
)

np.save(output_path+'tsnegermany_pca_cosine_500multiscale50_800ep.npy', embedding_multiscale)  ##
"""