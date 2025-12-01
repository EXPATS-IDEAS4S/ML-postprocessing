import pandas as pd
from sklearn.manifold import TSNE
import openTSNE
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


# === CONFIG ===
csv_dir = "/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/test/teamx/"   # change to your csv path
input_csv = 'features_train_dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA.csv'
output_suffix = "_with_tsne"  # will save as your_file_with_tsne.csv
tsne_random_state = 42        # for reproducibility
n_jobs = -1                   # m cores

# === LOAD CSV ===
print(f"Reading {input_csv}...")

df = pd.read_csv(csv_dir + input_csv)
print(df)


# === SELECT FEATURE COLUMNS (dim_1 ... dim_128) ===
feature_cols = [col for col in df.columns if col.startswith("dim_")]
X = df[feature_cols].values
print(f"Feature matrix shape: {X.shape}")
print(np.isnan(X).any())  # should be False

n_crops = 67425
feature_file = os.path.join(
        f"/data1/runs/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/features/",
        "rank0_chunk0_train_heads_features.npy")
data = np.load(feature_file)
X = np.reshape(data, (n_crops, -1))

X_scaled = StandardScaler().fit_transform(X)

# === RUN T-SNE ===
print("Running t-SNE (this might take a while for large datasets)...")
tsne = TSNE(
    n_components=2,
    random_state=tsne_random_state,
    perplexity=50,
    n_jobs=n_jobs,
    #initialization= "pca",
    init= "pca",
    metric= "cosine"          # speeds up convergence
    #learning_rate="auto" # recommended in sklearn >= 1.2
)
X_embedded = tsne.fit_transform(X_scaled)

# === ADD TO DATAFRAME ===
df["comp_1"] = X_embedded[:, 0]
df["comp_2"] = X_embedded[:, 1]

# === SAVE TO NEW CSV ===
base, ext = input_csv.split(".")
output_csv = f"{csv_dir}{base}{output_suffix}.{ext}"
print(f"Saving to {output_csv}...")
df.to_csv(output_csv, index=False)

print(f"✅ Done! Saved with t-SNE results to: {output_csv}")
