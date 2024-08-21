import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
import pandas as pd

scales = ['min-max','1th-99th','5th-95th']
path_out = '/home/Daniele/fig/'

# Initialize a list to store results for all scales
results = []

for scale in scales:
    print(scale)

    #open the feature vectores in 2d after dimensionality reduction

    tsne_path = f'/home/Daniele/fig/dcv_ir108_128x128_k9_30k_grey_{scale}/'

    tsne_filename = 'tsnegermany_pca_cosine_500ep.npy'

    X = np.load(tsne_path+tsne_filename)
    #print(X.shape)

    # Open the clustering assignments

    checkpoints_path = f'/home/Daniele/codes/vissl/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{scale}/checkpoints/'  

    assignments = torch.load(checkpoints_path+'assignments_800ep.pt',map_location='cpu')

    cluster_labels = assignments[0].cpu()
    #print(cluster_labels.shape)


    # Compute the metrics score

    silhouette_avg = silhouette_score(X, cluster_labels)

    print(f"The average silhouette score is: {silhouette_avg:.4f}")

    calinski_harabasz = calinski_harabasz_score(X, cluster_labels)

    print(f"The calinski harabasz score is: {calinski_harabasz:.4f}")

    davies_bouldin = davies_bouldin_score(X, cluster_labels)

    print(f"The davies bouldin score is: {davies_bouldin:.4f}")

    # Store the results in the list
    results.append({
        "Scale": scale,
        "Silhouette Score": silhouette_avg,
        "Calinski-Harabasz Score": calinski_harabasz,
        "Davies-Bouldin Score": davies_bouldin
    })

# Convert the results list into a pandas DataFrame
results_df = pd.DataFrame(results)

# Define the path for saving the results CSV file
output_csv_path = path_out + 'clustering_metrics_colorscale.csv'

# Save the DataFrame to a CSV file
results_df.to_csv(output_csv_path, index=False)

print(f"Metrics saved to {output_csv_path}")