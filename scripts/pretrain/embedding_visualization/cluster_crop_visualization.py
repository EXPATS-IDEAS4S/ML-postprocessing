"""
Image Crop Visualization and Processing Module

This script provides functions to process and visualize image crops grouped by cluster/class labels.
It supports morphological closing to fill black patches, generating sample grids for visual inspection,
and saving processed images in structured directories.

Main Features:
- Apply morphological closing to remove black patches in images.
- Plot sample grids per class with selection options (closest, furthest, random, all).
- Save processed crops per class as PNGs in organized folders.
- Configurable selection mode, number of samples, and morphological processing.

Configuration parameters (set at the top):
- reduction_method: Dimensionality reduction method used ('tsne', 'umap', etc.).
- run_name: Identifier for the run/experiment.
- random_state: Random seed for reproducible sampling.
- sampling_type: Sampling method used for class selection ('closest', 'furthest', 'random', 'all').
- output_path: Base directory for outputs.
- csv_path: Path to merged CSV containing image paths, labels, and distances.
"""

import os
import cv2
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
from tqdm import tqdm

# ========================
# CONFIGURATION PARAMETERS
# ========================
reduction_method = 'tsne'
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3'
sampling_type = 'all'
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
csv_path = f'{output_path}merged_{reduction_method}_variables_{run_name}_{sampling_type}_{random_state}.csv'

# Load dataset
df = pd.read_csv(csv_path)
print("Unique labels:", df['label'].unique())


# ========================
# HELPER FUNCTIONS
# ========================
def fill_closed_black_patches(img, structure=np.ones((3, 3))):
    """
    Apply morphological closing to black patches (pixels with value 0)
    and replace them with the mean of surrounding pixels.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.
    structure : np.ndarray
        Structuring element used for morphological closing.

    Returns
    -------
    np.ndarray
        Processed image with black patches filled.
    """
    img = img.copy()
    mask = img == 0
    closed_mask = binary_closing(mask, structure=structure)
    filled_mask = closed_mask & ~mask

    for y, x in zip(*np.where(filled_mask)):
        neighborhood = img[max(0, y-1):y+2, max(0, x-1):x+2]
        valid_neighbors = neighborhood[neighborhood > 0]
        if valid_neighbors.size > 0:
            img[y, x] = np.mean(valid_neighbors)
    return img


def plot_sample_grid_per_class(
    df,
    n_cols=5,
    selection_mode='closest',
    output_dir=None,
    title="Sample Images per Class",
    close_black_patches=False
):
    """
    Plot a grid of sample images per class.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ['path', 'label', 'distance'].
    n_cols : int
        Number of columns per row.
    selection_mode : str
        One of {'closest', 'furthest', 'random', 'all'}.
    output_dir : str or None
        Directory to save the figure. If None, the figure is displayed.
    title : str
        Figure title.
    close_black_patches : bool
        Whether to apply morphological closing to black patches.
    """
    unique_labels = sorted(df['label'].unique())
    n_rows = len(unique_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, label in enumerate(unique_labels):
        class_df = df[df['label'] == label].copy()

        if selection_mode == 'closest':
            class_df = class_df.sort_values(by='distance', ascending=False)
        elif selection_mode == 'furthest':
            class_df = class_df.sort_values(by='distance', ascending=True)
        elif selection_mode == 'random':
            class_df = class_df.sample(frac=1, random_state=42)

        selected_paths = class_df['path'].head(n_cols).to_list()

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis('off')
            if col_idx >= len(selected_paths):
                continue

            crop_path = selected_paths[col_idx]
            if not os.path.exists(crop_path):
                continue
            try:
                img = imageio.imread(crop_path)
                if close_black_patches:
                    img = fill_closed_black_patches(img)
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                if row_idx == 0:
                    ax.set_title(f"#{col_idx + 1}", fontsize=16)
                if col_idx == 0:
                    ax.set_ylabel(f"Label {label}", fontsize=16)
            except Exception as e:
                print(f"Failed to load image: {crop_path} | Error: {e}")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"sample_grid_{selection_mode}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Saved: {output_file}")
    else:
        plt.show()
    plt.close()


def process_and_save_crops(df, output_base_dir, apply_closing=True, n_samples=None, selection_mode='all'):
    """
    Processes and saves image crops as PNGs per class.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ['path', 'label', 'distance'].
    output_base_dir : str
        Directory to save processed crops.
    apply_closing : bool
        Whether to apply morphological closing.
    n_samples : int or None
        Number of images per class to save. If None, saves all.
    selection_mode : str
        One of {'closest', 'furthest', 'random', 'all'}.
    """
    assert selection_mode in ['closest', 'furthest', 'random', 'all'], "Invalid selection_mode"
    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) if apply_closing else None

    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label].copy()

        if selection_mode == 'closest':
            class_df = class_df.sort_values(by='distance', ascending=False)
        elif selection_mode == 'furthest':
            class_df = class_df.sort_values(by='distance', ascending=True)
        elif selection_mode == 'random':
            class_df = class_df.sample(frac=1, random_state=42)

        if n_samples is not None:
            class_df = class_df.head(n_samples).reset_index(drop=True)
        else:
            class_df = class_df.reset_index(drop=True)

        label_dir = os.path.join(output_base_dir, selection_mode, str(label))
        os.makedirs(label_dir, exist_ok=True)

        for idx, row in tqdm(class_df.iterrows(), total=len(class_df), desc=f"Label {label}"):
            try:
                img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: could not load image at {row['path']}")
                    continue
                if apply_closing:
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, struct_element)
                out_path = os.path.join(label_dir, f"{idx + 1}.png")
                cv2.imwrite(out_path, img)
            except Exception as e:
                print(f"Error processing {row['path']}: {e}")


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    output_png_base = os.path.join(output_path, "crops_png")
    os.makedirs(output_png_base, exist_ok=True)

    # Example: Save processed crops
    # process_and_save_crops(df, output_png_base, apply_closing=True, n_samples=50, selection_mode='closest')

    # Example: Plot sample grid
    plot_sample_grid_per_class(
        df=df,
        n_cols=10,
        selection_mode='furthest',
        output_dir=os.path.join(output_path, "sample_grids"),
        close_black_patches=True
    )
