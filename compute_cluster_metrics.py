import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
import pandas as pd
import glob

# Metrics definition here:
# https://scikit-learn.org/stable/modules/clustering.html

scales = ['10th-90th','10th-90th_CMA']
path_out = '/home/Daniele/fig/'

# Initialize a list to store mean and std of results for all scales
results = []

for scale in scales:
    print(f"Processing scale: {scale}")

    # Define the path to the t-SNE files
    tsne_path = f'/home/Daniele/fig/dcv_ir108_128x128_k9_30k_grey_{scale}/'

    # Use glob to find all t-SNE files matching the pattern
    tsne_filenames = glob.glob(tsne_path + f'tsne_pca_cosine_{scale}_*.npy')

    # Lists to collect metrics for all t-SNE versions
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []

    # Loop over each t-SNE file (different random states)
    for tsne_filename in tsne_filenames:
        print(f"Processing t-SNE file: {tsne_filename}")

        # Load the 2D feature vectors
        X = np.load(tsne_filename)

        # Open the clustering assignments
        checkpoints_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{scale}/checkpoints/'  
        assignments = torch.load(checkpoints_path+'assignments_800ep.pt', map_location='cpu')
        cluster_labels = assignments[0].cpu()

        # Compute the metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)

        # Store the metrics in the lists
        silhouette_scores.append(silhouette_avg)
        calinski_harabasz_scores.append(calinski_harabasz)
        davies_bouldin_scores.append(davies_bouldin)

    # Compute the mean and standard deviation for each metric
    mean_silhouette = np.mean(silhouette_scores)
    std_silhouette = np.std(silhouette_scores)

    mean_calinski_harabasz = np.mean(calinski_harabasz_scores)
    std_calinski_harabasz = np.std(calinski_harabasz_scores)

    mean_davies_bouldin = np.mean(davies_bouldin_scores)
    std_davies_bouldin = np.std(davies_bouldin_scores)

    # Store the mean and std in the results list
    results.append({
        "Scale": scale,
        "Mean Silhouette Score": mean_silhouette,
        "Std Silhouette Score": std_silhouette,
        "Mean Calinski-Harabasz Score": mean_calinski_harabasz,
        "Std Calinski-Harabasz Score": std_calinski_harabasz,
        "Mean Davies-Bouldin Score": mean_davies_bouldin,
        "Std Davies-Bouldin Score": std_davies_bouldin
    })

# Convert the results list into a pandas DataFrame
results_df = pd.DataFrame(results)

# Define the path for saving the results CSV file
output_csv_path = f'{path_out}clustering_metrics_summary_{scales}.csv'

# Save the DataFrame to a CSV file
results_df.to_csv(output_csv_path, index=False)

print(f"Metrics saved to {output_csv_path}")
