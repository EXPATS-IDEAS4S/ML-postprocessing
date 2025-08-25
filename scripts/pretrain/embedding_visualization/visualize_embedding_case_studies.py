"""
Feature Embedding Visualization with Test Cases

This script visualizes high-dimensional feature embeddings (t-SNE / Isomap) in 2D space. 
It supports both scatter-based visualizations (color-coded by class) and optional overlay 
of image crops. Test cases (MSG/ICON) can be highlighted for case study analysis.

Workflow:
    1. Load features, labels, and assignments.
    2. Reduce dimensionality (precomputed embeddings).
    3. Prepare DataFrame with labels, coordinates, and image paths.
    4. Plot scatter embeddings and optionally crops with class colors.
"""

import numpy as np
import pandas as pd
from glob import glob
import torch

from scripts.pretrain.dim_reduction.plot_embedding_utils import (
    plot_embedding_crops_new,
    plot_embedding_dots_iterative_test_msg_icon,
    scale_to_01_range,
    name_to_rgb,
    extract_hour,
    plot_embedding_dots,
    plot_embedding_filled,
    plot_embedding_crops,
    plot_embedding_dots_iterative_case_study
)

# ---------------- CONFIG ---------------- #
CONFIG = {
    "scale": "dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon",
    "random_state": 3,
    "reduction_method": "tsne",  # options: 'tsne', 'isomap'
    "n_random_samples": None,    # e.g. 30000 for subsampling
    "case_study_msg": False,
    "case_study_icon": False,
    "output_dir": "/home/Daniele/fig/",
    "image_path": "/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA_test_msg_icon/1/",
    "features_path": "/data1/runs/{scale}/features/",
    "checkpoints_path": "/data1/runs/{scale}/checkpoints/",
}

COLORS_PER_CLASS = {
    '0': 'darkgray', 
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green'
}


# ---------------- FUNCTIONS ---------------- #

def load_embeddings(cfg):
    """Load embeddings, feature indices, and assignments."""
    scale = cfg["scale"]
    method = cfg["reduction_method"]
    random_state = cfg["random_state"]
    n_random = cfg["n_random_samples"]

    tsne_path = f'{cfg["output_dir"]}{scale}/'

    if method == "tsne":
        emb_file = f'{method}_pca_cosine_{scale}_{random_state}.npy'
    else:  # isomap
        emb_file = f'{method}_cosine_{scale}_{n_random}.npy'
        indeces_file = f'{method}_cosine_{scale}_{n_random}_indeces.npy'
    embeddings = np.load(tsne_path + emb_file)

    feature_inds = np.load(cfg["features_path"].format(scale=scale) + "rank0_chunk0_train_heads_inds.npy")
    assignments = torch.load(cfg["checkpoints_path"].format(scale=scale) + "assignments_800ep.pt", map_location="cpu")

    return embeddings, feature_inds, assignments


def prepare_dataframe(embeddings, feature_inds, assignments, cfg):
    """Build dataframe with coordinates, labels, image paths, and optional case study flags."""
    # Normalize coords
    tx, ty = embeddings[:, 0], embeddings[:, 1]
    tx, ty = scale_to_01_range(tx), scale_to_01_range(ty)

    # Build base DataFrame
    df = pd.DataFrame({
        "Component_1": tx,
        "Component_2": ty,
        "index": feature_inds
    }).set_index("index")

    labels = assignments[0, :].cpu().numpy()
    df["y"] = labels[feature_inds]

    # Map images
    crop_paths = sorted(glob(cfg["image_path"] + "*.tif"))
    df["location"] = [crop_paths[i] for i in feature_inds]

    # Add hour info
    df["hour"] = df["location"].apply(extract_hour)

    # Filter invalid labels
    df = df[df["y"] != -100]

    # Color mapping
    df["color"] = df["y"].map(lambda x: COLORS_PER_CLASS[str(int(x))])

    return df


def plot_embeddings(df, cfg):
    """Generate scatter and optional crop plots."""
    scale = cfg["scale"]
    output_path = f'{cfg["output_dir"]}{scale}/'
    filename = f'{cfg["reduction_method"]}_{scale}'

    # Scatter plot
    df_sample = df.sample(n=20000)
    plot_embedding_dots(df_sample, COLORS_PER_CLASS, output_path, filename)
    plot_embedding_filled(df_sample, COLORS_PER_CLASS, output_path, filename)

    # Optional crops
    plot_size = [1000, 1000]
    max_img_size = 80
    offset = [max_img_size // 2] * 2
    img_area = [plot_size[0] - max_img_size, plot_size[1] - max_img_size]
    tsne_canvas = 255 * np.ones((plot_size[0], plot_size[1], 3), np.uint8)

    for y_val in df["y"].unique():
        subset = df.query("y == @y_val").sample(n=50)
        indices = subset.index.tolist()
        selected_imgs = [subset.loc[i, "location"] for i in indices]

        plot_embedding_crops(
            indices, selected_imgs, subset, tsne_canvas,
            output_path, filename, img_area, offset, max_img_size,
            subset["Component_1"].min(), subset["Component_1"].max(),
            subset["Component_2"].min(), subset["Component_2"].max(),
            COLORS_PER_CLASS
        )


# ---------------- MAIN ---------------- #

def main():
    embeddings, feature_inds, assignments = load_embeddings(CONFIG)
    df = prepare_dataframe(embeddings, feature_inds, assignments, CONFIG)
    plot_embeddings(df, CONFIG)


if __name__ == "__main__":
    main()
