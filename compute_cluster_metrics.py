import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import torch
import pandas as pd
import glob
import os

# Metrics definition here:
# https://scikit-learn.org/stable/modules/clustering.html

# Configuration
scales = ['dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA']
epochs = [100, 200, 300, 400, 500]

# Initialize a list to store results
results = []

for epoch in epochs:
    for scale in scales:
        print(f"==> Processing scale: {scale}, epoch: {epoch}")

        # Define output directory
        output_dir = f"/data1/fig/{scale}/clustering_metrics_output/"
        os.makedirs(output_dir, exist_ok=True)

        # Define path to t-SNE files
        tsne_path = f'/data1/fig/{scale}/epoch_{epoch}/'
        tsne_filenames = glob.glob(os.path.join(tsne_path, f'tsne_pca_cosine_{scale}_*.npy'))

        # Define path to clustering assignments TODO 
        checkpoint_path = f'/data1/runs/{scale}/checkpoints/assignments_800ep.pt'

        if not tsne_filenames:
            print(f"⚠️ No t-SNE files found for epoch {epoch}. Skipping.")
            continue

        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Assignment file not found for epoch {epoch}: {checkpoint_path}")
            continue

        # Load assignments
        assignments = torch.load(checkpoint_path, map_location='cpu')
        cluster_labels = assignments[0].cpu().numpy()
        print(cluster_labels)

        # Initialize metric lists
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []

        for tsne_file in tsne_filenames:
            print(f"  → Processing t-SNE file: {tsne_file}")
            X = np.load(tsne_file)

            silhouette_scores.append(silhouette_score(X, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
            davies_scores.append(davies_bouldin_score(X, cluster_labels))

        # Store stats for this epoch
        results.append({
            "Scale": scale,
            "Epoch": epoch,
            "Silhouette Mean": np.mean(silhouette_scores),
            "Silhouette Std": np.std(silhouette_scores),
            "Calinski-Harabasz Mean": np.mean(calinski_scores),
            "Calinski-Harabasz Std": np.std(calinski_scores),
            "Davies-Bouldin Mean": np.mean(davies_scores),
            "Davies-Bouldin Std": np.std(davies_scores)
        })

# Save to CSV
df = pd.DataFrame(results)
output_file = os.path.join(output_dir, f"clustering_metrics_summary.csv")
df.to_csv(output_file, index=False)

print(f"✅ Metrics saved to {output_file}")
