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
    - perplexities: List of perplexity values for t-SNE runs.
    - methods: List of dimensionality reduction methods to apply 
               (options: "isomap", "tsne_sklearn", "tsne_opentsne").

Workflow:
    1. Load feature data from specified directories.
    2. Load features and flatten trailing dimensions when needed.
    3. Apply chosen dimensionality reduction method(s):
        - Isomap (sklearn.manifold.Isomap)
        - t-SNE (sklearn.manifold.TSNE)
        - t-SNE (openTSNE.TSNE with PCA initialization)
    4. Save embeddings to an output directory with filenames that include 
       the scale, epoch, perplexity, method, and random seed.

Usage:
        - Adjust `scales`, `epochs`, `random_states`, and `perplexities` 
      to match the dataset and experiment configuration.
    - Add or remove methods in `methods` to control which algorithms run.
    - Ensure file paths in `common_path` and `output_path` are correctly set.
    - Run the script to compute and save embeddings for all combinations.

Example output filename:
    tsne_opentsne_pca_cosine_perp-50_scale-randomstate-3_epoch-500.npy

Call of the script
conda run -n vissl python scripts/pretrain/embedding_visualization/compute_2d_embedding.py
"""


import os
import gc
import sys
import time
from pathlib import Path
from typing import Optional, Union
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.configs import load_config

# ---------------- CONFIG ---------------- #
CONFIG = {}
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "process_run_GRL.yaml"


def get_reduction_config(config_path: Optional[Union[str, os.PathLike]] = None) -> dict:
    """Load reduction settings from YAML."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    loaded = load_config(str(config_path))
    reduction_config = loaded.get("reduction", {})

    if not reduction_config:
        raise ValueError(f"Missing 'reduction' section in config: {config_path}")

    return reduction_config

# ---------------- UTILS ---------------- #
def get_feature_label() -> str:
    """Build a stable label for logs and output filenames from the feature path."""
    feature_path = Path(CONFIG["feature_folder"]).resolve()

    if len(feature_path.parts) >= 3:
        return feature_path.parent.parent.name

    return feature_path.parent.name or "features"


def load_features(epoch: int) -> np.ndarray:
    """Load feature data for a given epoch."""
    feature_file = os.path.join( CONFIG["feature_folder"]
        ,
        "rank0_chunk0_train_heads_features.npy"
    )
    data = np.load(feature_file)

    if data.ndim == 1:
        raise ValueError(f"Expected at least 2 dimensions in feature file, got shape {data.shape}")

    if data.ndim == 2:
        return data

    return data.reshape(data.shape[0], -1)


def save_embedding(embedding, scale: str, epoch: int, method: str, random_state: int, extra_tag: str = ""):
    """Save embedding results as .npy file."""
    
    os.makedirs(CONFIG["output_path"], exist_ok=True)
    filename = f"{method}{extra_tag}_perpl-{CONFIG['tsne_opentsne']['perplexity']}_{scale}_{random_state}_epoch_{epoch}{CONFIG['filename_suffix']}.npy"
    #np.save(os.path.join(CONFIG["output_path"], f"epoch_{epoch}", CONFIG['sampling_type'], filename), embedding)
    np.save(os.path.join(CONFIG["output_path"], filename), embedding)
    print(f"✅ Saved: {filename}")


def log_with_timestamp(message: str):
    """Print a timestamped progress message."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


# ---------------- DIMENSIONALITY REDUCTIONS ---------------- #
def run_isomap(data: np.ndarray, random_state: int):
    from sklearn.manifold import Isomap

    params = CONFIG["isomap"]
    model = Isomap(**params)
    return model.fit_transform(data)


def run_tsne_sklearn(data: np.ndarray, random_state: int):
    from sklearn.manifold import TSNE as SklearnTSNE

    params = CONFIG["tsne_sklearn"]
    model = SklearnTSNE(random_state=random_state, **params)
    return model.fit_transform(data)


def run_tsne_opentsne(data: np.ndarray, random_state: int):
    import openTSNE

    params = dict(CONFIG["tsne_opentsne"])
    params.setdefault("verbose", True)
    params.setdefault("callbacks_every_iters", 50)

    log_with_timestamp(
        "Starting openTSNE "
        f"with n_samples={data.shape[0]}, n_features={data.shape[1]}, "
        f"perplexity={params.get('perplexity')}, initialization={params.get('initialization')}, "
        f"metric={params.get('metric')}, callbacks_every_iters={params.get('callbacks_every_iters')}"
    )

    model = openTSNE.TSNE(random_state=random_state, n_jobs=-1, **params)
    start_time = time.perf_counter()
    embedding = model.fit(data)
    elapsed = time.perf_counter() - start_time

    log_with_timestamp(f"Completed openTSNE in {elapsed:.2f} seconds")
    return embedding


# ---------------- MAIN ---------------- #
def main(config_path: Optional[Union[str, os.PathLike]] = None):
    global CONFIG
    CONFIG = get_reduction_config(config_path)
    gc.collect()

    feature_label = get_feature_label()

    for epoch in CONFIG["epochs"]:
        data = load_features(epoch)
        print(f"📂 Loaded {feature_label} @ epoch {epoch}, shape={data.shape}")

        for random_state in CONFIG["random_states"]:
            for method in CONFIG["methods"]:
                log_with_timestamp(f"Running {method} (rs={random_state})")
                method_start_time = time.perf_counter()
                if method == "isomap":
                    embedding = run_isomap(data, random_state)
                elif method == "tsne_sklearn":
                    embedding = run_tsne_sklearn(data, random_state)
                elif method == "tsne_opentsne":
                    embedding = run_tsne_opentsne(data, random_state)
                else:
                    raise ValueError(f"Unknown method: {method}")

                log_with_timestamp(
                    f"Finished {method} (rs={random_state}) in {time.perf_counter() - method_start_time:.2f} seconds"
                )
                save_embedding(embedding, feature_label, epoch, method, random_state)


if __name__ == "__main__":
    main()
#648862