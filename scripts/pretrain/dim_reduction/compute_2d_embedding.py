"""
Dimensionality Reduction Pipeline (Isomap & t-SNE)

This script performs dimensionality reduction on high-dimensional feature data 
to visualize it in a lower-dimensional space. It supports both Isomap and t-SNE 
(using scikit-learn and openTSNE backends). Results are saved as .npy files 
for further visualization or analysis.

Modules:
    - numpy: For data manipulation and saving the embeddings.
    - openTSNE: For applying t-SNE with advanced options.
    - sklearn.manifold: For Isomap and t-SNE (basic).
    - os: For path management and directory creation.
    - gc: To manage garbage collection and free memory during processing.

Parameters:
    - scales: List of dataset scale identifiers to process.
    - epochs: List of training epochs to process features from.
    - random_states: List of random seed values to ensure reproducibility.
    - n_crops: Number of samples in the feature dataset (used for reshaping).
    - perplexities: List of perplexity values for t-SNE runs.
    - methods: List of dimensionality reduction methods to apply 
               (options: "isomap", "tsne_sklearn", "tsne_opentsne").

Workflow:
    1. Load feature data from specified directories.
    2. Reshape features to (n_crops, 128).
    3. Apply chosen dimensionality reduction method(s):
        - Isomap (sklearn.manifold.Isomap)
        - t-SNE (sklearn.manifold.TSNE)
        - t-SNE (openTSNE.TSNE with PCA initialization)
    4. Save embeddings to an output directory with filenames that include 
       the scale, epoch, perplexity, method, and random seed.

Usage:
    - Adjust `scales`, `epochs`, `random_states`, `n_crops`, and `perplexities` 
      to match the dataset and experiment configuration.
    - Add or remove methods in `methods` to control which algorithms run.
    - Ensure file paths in `common_path` and `output_path` are correctly set.
    - Run the script to compute and save embeddings for all combinations.

Example output filename:
    tsne_opentsne_pca_cosine_perp-50_scale-randomstate-3_epoch-500.npy
"""


import os
import gc
import numpy as np
from sklearn.manifold import Isomap, TSNE as SklearnTSNE
import openTSNE

# ---------------- CONFIG ---------------- #
CONFIG = {
    "scales": ["dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA"],
    "epochs": [500, 800],
    "random_states": [0, 3, 16, 23, 57],
    "n_crops": 70094,
    "sampling_type": "all",  # Options: 'random', 'closest', 'farthest', 'all'
    "methods": ["isomap", "tsne_sklearn", "tsne_opentsne"],
    # Parameters for Isomap
    "isomap": {
        "n_neighbors": 10,
        "n_components": 2
    },
    # Parameters for TSNE (sklearn)
    "tsne_sklearn": {
        "n_components": 2,
        "perplexity": 30,
        "learning_rate": 200,
        "n_iter": 1000
    },
    # Parameters for TSNE (openTSNE)
    "tsne_opentsne": {
        "perplexity": 50,
        "initialization": "pca",
        "metric": "cosine"
    }
}

# ---------------- UTILS ---------------- #
def load_features(scale: str, epoch: int, n_crops: int) -> np.ndarray:
    """Load and reshape feature data for a given scale and epoch."""
    feature_file = os.path.join(
        f"/data1/runs/{scale}/features/epoch_{epoch}/",
        "rank0_chunk0_train_heads_features.npy"
    )
    data = np.load(feature_file)
    return np.reshape(data, (n_crops, -1))


def save_embedding(embedding, scale: str, epoch: int, method: str, random_state: int, extra_tag: str = ""):
    """Save embedding results as .npy file."""
    output_path = f"/data1/fig/{scale}/epoch_{epoch}/{CONFIG['sampling_type']}/"
    os.makedirs(output_path, exist_ok=True)
    filename = f"{method}{extra_tag}_{scale}_{random_state}_epoch_{epoch}.npy"
    np.save(os.path.join(output_path, filename), embedding)
    print(f"✅ Saved: {filename}")


# ---------------- DIMENSIONALITY REDUCTIONS ---------------- #
def run_isomap(data: np.ndarray, random_state: int):
    params = CONFIG["isomap"]
    model = Isomap(**params)
    return model.fit_transform(data)


def run_tsne_sklearn(data: np.ndarray, random_state: int):
    params = CONFIG["tsne_sklearn"]
    model = SklearnTSNE(random_state=random_state, **params)
    return model.fit_transform(data)


def run_tsne_opentsne(data: np.ndarray, random_state: int):
    params = CONFIG["tsne_opentsne"]
    model = openTSNE.TSNE(random_state=random_state, n_jobs=-1, **params)
    return model.fit(data)


# ---------------- MAIN ---------------- #
def main():
    gc.collect()

    for epoch in CONFIG["epochs"]:
        for scale in CONFIG["scales"]:
            data = load_features(scale, epoch, CONFIG["n_crops"])
            print(f"📂 Loaded {scale} @ epoch {epoch}, shape={data.shape}")

            for random_state in CONFIG["random_states"]:
                for method in CONFIG["methods"]:
                    print(f"➡️ Running {method} (rs={random_state})")
                    if method == "isomap":
                        embedding = run_isomap(data, random_state)
                    elif method == "tsne_sklearn":
                        embedding = run_tsne_sklearn(data, random_state)
                    elif method == "tsne_opentsne":
                        embedding = run_tsne_opentsne(data, random_state)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    save_embedding(embedding, scale, epoch, method, random_state)


if __name__ == "__main__":
    main()
