import os
import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE
import openTSNE

# ================================================
# CONFIG
# ================================================
run_name = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
output_path = f"/data1/fig/{run_name}/epoch_800/test_traj"
output_test_csv = os.path.join(
    output_path, f"features_train_test_{run_name}_2nd_labels.csv"
)
centroids_file = os.path.join(
    f"/data1/runs/{run_name}/checkpoints", "centroids0.pt"
)

vector_types = ["TRAIN", "PRECIP", "HAIL", "MIXED"]
tsne_out_csv = os.path.join(output_path, "tsne_all_vectors_with_centroids.csv")



# ================================================
# LOAD DATA
# ================================================
df = pd.read_csv(output_test_csv, low_memory=False)
print(df)
#change vector_type colum id is TEST to the value contained in the column storm_type
df.loc[df["vector_type"] == "TEST", "vector_type"] = df.loc[df["vector_type"] == "TEST", "storm_type"].values
print(df)
#use indentifies from the path basename
df["filename"] = df["path"].apply(lambda x: os.path.basename(x))
print(df["filename"])
#print(df["vector_type"].value_counts())

feature_cols = [c for c in df.columns if c.startswith("dim_")]
print(f"Using {len(feature_cols)} feature dimensions")

# ================================================
# COLLECT FEATURES PER VECTOR TYPE
# ================================================
X_list = []
labels = []
classes = []
filenames = []

for vtype in vector_types:
    df_v = df[df["vector_type"] == vtype]
    X_v = df_v[feature_cols].values.astype(np.float32)

    X_list.append(X_v)
    labels.extend([vtype] * len(X_v))
    classes.extend(df_v["label"].values.tolist())
    filenames.extend(df_v["filename"].values.tolist())

    print(f"{vtype}: {X_v.shape[0]} samples")

# ================================================
# LOAD & APPEND CENTROIDS
# ================================================
centroids = torch.load(centroids_file, map_location="cpu").numpy().astype(np.float32)
print(f"Centroids shape: {centroids.shape}")

X_list.append(centroids)
labels.extend(["CENTROID"] * centroids.shape[0])
classes.extend(range(centroids.shape[0]))
filenames.extend(["CENTROID_"+str(i) for i in range(centroids.shape[0])])

# ================================================
# STACK EVERYTHING
# ================================================
X_all = np.vstack(X_list)
labels = np.array(labels)
classes = np.array(classes)
filenames = np.array(filenames)

print(f"Total samples for t-SNE: {X_all.shape}")

# ================================================
# t-SNE REDUCTION
# ================================================
# tsne = TSNE(
#     n_components=2,
#     perplexity=50,
#     learning_rate="auto",
#     init="pca",
#     random_state=42,
#     n_iter=1000,
# )

tsne = openTSNE.TSNE(random_state=42, 
                      n_jobs=-1,
                     perplexity=50,
                    initialization="pca",
                    metric="cosine")

X_tsne = tsne.fit(X_all)

# ================================================
# SAVE TO CSV
# ================================================
df_tsne = pd.DataFrame({
    "tsne_dim_1": X_tsne[:, 0],
    "tsne_dim_2": X_tsne[:, 1],
    "vector_type": labels,
    "label": classes,
    "filename": filenames,
})

df_tsne.to_csv(tsne_out_csv, index=False)
print(f"t-SNE saved to: {tsne_out_csv}")

#214820