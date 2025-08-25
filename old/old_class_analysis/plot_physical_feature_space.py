import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler




run_name = '10th-90th' #['10th-90th', '10th-90th_CMA']
sampling_type = 'closest'  # Options: 'random', 'closest', 'farthest', 'all'
stat = '50' #None  # '50'
n_subsample = 1000  # Number of samples per cluster

# Path to fig folder for outputs
output_path = f'/home/Daniele/fig/cma_analysis/{run_name}/{sampling_type}/'

# Load the saved CSV into a new DataFrame
merged_df = pd.read_csv(f'{output_path}physical_feature_vectors_{run_name}_{sampling_type}_{n_subsample}_{stat}.csv')

# remove rows with NaN values
merged_df = merged_df.dropna()

#get the labels in a separate array
labels = merged_df['label'].tolist()
print(len(labels))
labels_array = np.array(labels)
print(np.unique(labels_array))

# Drop non-numerical columns (if any), keeping only the features for PCA and t-SNE
X = merged_df.drop(columns=['label'])  # Assuming 'label' is the only non-numerical column
print(X)

# Standardize the features (PCA and t-SNE work better on standardized data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- PCA --------------------

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca.shape)

# Create a DataFrame for PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['label'] = labels

# Plot PCA result
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='label', palette='Set2', data=df_pca, alpha=0.7)
plt.title('PCA - 2D Feature Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Labels')
#plt.show()
# Save the figure
output_fig_path = f'{output_path}PCA_physical_vectors_{run_name}_{sampling_type}_{n_subsample}_{stat}.png'
plt.savefig(output_fig_path, bbox_inches='tight', dpi=300)
print(f'fig saved in: {output_fig_path}')

# -------------------- t-SNE --------------------

# Apply t-SNE to reduce to 2 dimensions
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create a DataFrame for t-SNE results
df_tsne = pd.DataFrame(data=X_tsne, columns=['t-SNE1', 't-SNE2'])
df_tsne['label'] = labels

# Plot t-SNE result
plt.figure(figsize=(8, 6))
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='label', palette='Set2', data=df_tsne, alpha=0.7)
plt.title('t-SNE - 2D Feature Space')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Labels')
plt.show()
# Save the figure
output_fig_path = f'{output_path}tSNE_physical_vectors_{run_name}_{sampling_type}_{n_subsample}_{stat}.png'
plt.savefig(output_fig_path, bbox_inches='tight', dpi=300)
print(f'fig saved in: {output_fig_path}')
