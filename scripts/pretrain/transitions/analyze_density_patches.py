"""
Analyze disconnected patches in t-SNE density contours for each pathway.

This script:
1. Loads pathway data and t-SNE embeddings
2. Computes density contours using KDE for each pathway
3. Identifies disconnected patches using connected component labeling
4. Maps data points to their corresponding patches
5. Computes cloud property statistics (CMA, CTH, COT) for each patch
6. Exports results to CSV and visualizations to PNG
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from scipy import ndimage
import cmcrameri.cm as cmc
from PIL import Image

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.usetex": False,
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")
from utils.plotting.class_colors import CLOUD_CLASS_INFO

# Configuration
RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUT_DIR = f"{BASE_DIR}/pathway_analysis"
PATCHES_OUT_DIR = os.path.join(OUT_DIR, "density_patches_analysis")
os.makedirs(PATCHES_OUT_DIR, exist_ok=True)

SELECTED_CLASSES = [1, 2, 4]
DENSITY_THRESHOLD_PERCENTILE = 50  # Use 50th percentile as boundary


def load_data():
    """Load pathway and t-SNE data."""
    path = f"{OUT_DIR}/df_pathways_merged_no_dominance.csv"
    df = pd.read_csv(path, low_memory=False)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    csv_tnse_file = "tsne_all_vectors_with_centroids.csv"
    df_tsne = pd.read_csv(os.path.join(BASE_DIR, csv_tnse_file))
    df_tsne_centroids = df_tsne[df_tsne["vector_type"] == "CENTROID"]
    df_tsne_train = df_tsne[df_tsne["vector_type"] == "TRAIN"]
    df_tsne_train = df_tsne_train[df_tsne_train["label"] != -100]
    
    return df, df_tsne_train, df_tsne_centroids


def load_crop_percentiles(base_dir):
    """Load external crop-level CTH/COT percentiles used by pathway panel d."""
    pct_path = os.path.join(base_dir, "merged_crops_stats_cot_percentiles.csv")
    if not os.path.exists(pct_path):
        print(f"Crop percentiles file not found: {pct_path}")
        return None, None

    try:
        df_crop_pct = pd.read_csv(pct_path, low_memory=False)
        df_crop_cth_pct = (
            df_crop_pct[df_crop_pct["var"] == "cth"]
            .rename(columns={"25": "cth25_pct", "50": "cth50_pct", "75": "cth75_pct"})
            [["crop", "cth25_pct", "cth50_pct", "cth75_pct"]]
        )
        df_crop_cot_pct = (
            df_crop_pct[df_crop_pct["var"] == "cot_percentiles"]
            .rename(columns={"25": "cot25_pct", "50": "cot50_pct", "75": "cot75_pct"})
            [["crop", "cot25_pct", "cot50_pct", "cot75_pct"]]
        )

        # Normalize key and enforce one row per crop to avoid accidental duplicate joins.
        df_crop_cth_pct["crop"] = df_crop_cth_pct["crop"].astype(str).str.strip()
        df_crop_cot_pct["crop"] = df_crop_cot_pct["crop"].astype(str).str.strip()
        df_crop_cth_pct = df_crop_cth_pct.groupby("crop", as_index=False).median(numeric_only=True)
        df_crop_cot_pct = df_crop_cot_pct.groupby("crop", as_index=False).median(numeric_only=True)

        print(f"Loaded crop percentiles from: {pct_path}")
        return df_crop_cth_pct, df_crop_cot_pct
    except Exception as e:
        print(f"Failed to load crop percentiles: {e}")
        return None, None


def add_cth_cot50_columns(df_pw, df_crop_cth_pct, df_crop_cot_pct):
    """
    Add cth50/cot50 columns using the same priority as panel d:
    1) crop-level external percentiles (cth50_pct/cot50_pct)
    2) fallback to pathway-level columns when available.
    """
    df_pw_stats = df_pw.copy()

    # Keep original pathway columns for fallback after merge.
    src_cth50 = df_pw_stats["cth50"] if "cth50" in df_pw_stats.columns else None
    src_cot_thick = df_pw_stats["cot_thick"] if "cot_thick" in df_pw_stats.columns else None

    if (
        df_crop_cth_pct is not None
        and df_crop_cot_pct is not None
        and "crop" in df_pw_stats.columns
    ):
        df_pw_stats["crop"] = df_pw_stats["crop"].astype(str).str.strip()
        df_pw_stats = (
            df_pw_stats.merge(df_crop_cth_pct, on="crop", how="left")
            .merge(df_crop_cot_pct, on="crop", how="left")
        )

        # Expose unified names expected in patch statistics output.
        df_pw_stats["cth50"] = df_pw_stats["cth50_pct"]
        df_pw_stats["cot50"] = df_pw_stats["cot50_pct"]

        # If crop-level percentiles are missing for some crops, fallback to pathway columns.
        if src_cth50 is not None:
            df_pw_stats["cth50"] = df_pw_stats["cth50"].fillna(src_cth50)
        if src_cot_thick is not None:
            df_pw_stats["cot50"] = df_pw_stats["cot50"].fillna(src_cot_thick)
    else:
        if "cth50" not in df_pw_stats.columns:
            df_pw_stats["cth50"] = np.nan
        if "cot50" not in df_pw_stats.columns:
            # Some datasets use cot_thick as the available COT proxy.
            if "cot_thick" in df_pw_stats.columns:
                df_pw_stats["cot50"] = df_pw_stats["cot_thick"]
            else:
                df_pw_stats["cot50"] = np.nan

    return df_pw_stats


def identify_patches(test_vectors, x_min, x_max, y_min, y_max, grid_resolution=100):
    """
    Identify disconnected patches in KDE density above threshold.
    
    Returns:
        patch_labels: 2D array with patch labels (0 = background)
        xx, yy: Grid coordinates
        kde: KDE object
        density_threshold: Density value at the threshold percentile
    """
    if len(test_vectors) < 4:
        return None, None, None, None, None
    
    # Compute KDE
    kde = gaussian_kde(test_vectors.T)
    
    # Create grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(positions).reshape(xx.shape)
    
    # Compute threshold
    density_values = kde(test_vectors.T)
    density_threshold = np.percentile(density_values, DENSITY_THRESHOLD_PERCENTILE)
    
    # Create binary mask of high-density regions
    binary_mask = zz >= density_threshold
    
    # Label connected components
    labeled_array, num_features = ndimage.label(binary_mask)
    
    return labeled_array, (xx, yy), kde, density_threshold, zz


def assign_points_to_patches(test_vectors, xx, yy, labeled_array):
    """
    Assign each data point to its nearest grid cell's patch label.
    
    Returns:
        patch_ids: Array of patch IDs for each point (0 if no patch)
        grid_to_patch: Mapping from grid indices to patch labels
    """
    # Create interpolators to map points to grid patches
    x_coords = test_vectors[:, 0]
    y_coords = test_vectors[:, 1]
    
    x_grid = xx[0, :]
    y_grid = yy[:, 0]
    
    # Normalize coordinates to grid indices
    x_indices = np.searchsorted(x_grid, x_coords)
    y_indices = np.searchsorted(y_grid, y_coords)
    
    # Clamp to valid grid range
    x_indices = np.clip(x_indices, 0, len(x_grid) - 1)
    y_indices = np.clip(y_indices, 0, len(y_grid) - 1)
    
    # Get patch labels for each point
    patch_ids = labeled_array[y_indices, x_indices]
    
    return patch_ids


def compute_patch_statistics(df_pw, patch_ids, pathway_id):
    """
    Compute statistics for cloud properties in each patch.
    
    Returns:
        patch_stats: DataFrame with statistics per patch
    """
    df_pw_copy = df_pw.copy()
    df_pw_copy["patch_id"] = patch_ids
    
    # Properties to analyze
    properties = ["cma", "cth50", "cot50", "cth_very_high", "cot_thick", "precipitation99", 
                  "max_hail_intensity", "n_precip", "n_hail"]
    
    patch_stats_list = []
    
    # Exclude background (patch_id == 0) if desired, or include it
    for patch_id in np.unique(patch_ids):
        if patch_id == 0:
            continue  # Skip background
        
        patch_data = df_pw_copy[df_pw_copy["patch_id"] == patch_id]
        
        if len(patch_data) == 0:
            continue
        
        stats = {
            "pathway_id": pathway_id,
            "patch_id": int(patch_id),
            "n_samples": len(patch_data),
        }
        
        for prop in properties:
            if prop in patch_data.columns:
                valid_data = patch_data[prop].dropna()
                if len(valid_data) > 0:
                    stats[f"{prop}_mean"] = valid_data.mean()
                    stats[f"{prop}_std"] = valid_data.std()
                    stats[f"{prop}_median"] = valid_data.median()
                    stats[f"{prop}_min"] = valid_data.min()
                    stats[f"{prop}_max"] = valid_data.max()
        
        patch_stats_list.append(stats)
    
    return pd.DataFrame(patch_stats_list)


def visualize_patches(df_pw, test_vectors, xx, yy, labeled_array, patch_ids, 
                      df_tsne_train, pathway_id, pathway_name):
    """Create visualization of patches with statistics overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Density map with patches
    ax = axes[0]
    # Color training points by class label (if available) or gray
    if "label" in df_tsne_train.columns:
        color_map = {1: "steelblue", 2: "darkorange", 4: "green"}
        colors = [color_map.get(int(l), "gray") for l in df_tsne_train["label"].values]
        ax.scatter(df_tsne_train["tsne_dim_1"], df_tsne_train["tsne_dim_2"],
                   c=colors, s=3, alpha=0.05, linewidth=0)
    else:
        ax.scatter(df_tsne_train["tsne_dim_1"], df_tsne_train["tsne_dim_2"],
                   c="gray", s=3, alpha=0.05, linewidth=0)
    
    # Color each patch differently
    cmap = mpl.cm.get_cmap('tab20')
    for patch_id in np.unique(labeled_array):
        if patch_id == 0:
            continue
        mask = labeled_array == patch_id
        color_idx = int(patch_id) % 20
        patch_color = cmap(color_idx / 20.0)
        ax.contourf(xx, yy, mask.astype(float), levels=[0.5, 1.5], colors=[patch_color], alpha=0.5)
        ax.contour(xx, yy, mask.astype(float), levels=[0.5], colors=[patch_color], linewidths=1.5, alpha=0.8)
    
    # Overlay data points colored by patch
    unique_patches = np.unique(patch_ids[patch_ids != 0])
    for patch_id in unique_patches:
        mask = patch_ids == patch_id
        color_idx = int(patch_id) % 20
        patch_color = cmap(color_idx / 20.0)
        ax.scatter(test_vectors[mask, 0], test_vectors[mask, 1], 
                  s=20, color=patch_color, edgecolor="black", linewidth=0.5, 
                  alpha=0.7, label=f"Patch {patch_id}")
    
    ax.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax.set_title(f"Pathway {pathway_id}: {pathway_name} - Density Patches", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    
    # Right: Cloud property heatmap per patch
    ax = axes[1]
    ax.axis("off")
    
    # Create text summary
    patch_stats = compute_patch_statistics(df_pw, patch_ids, pathway_id)
    text_content = f"Pathway {pathway_id}: {pathway_name}\n\nPatch Statistics:\n"
    text_content += "="*50 + "\n"
    
    for _, row in patch_stats.iterrows():
        patch_id = int(row["patch_id"])
        n_samples = int(row["n_samples"])
        text_content += f"\nPatch {patch_id} (n={n_samples} samples):\n"
        
        for col in patch_stats.columns:
            if col not in ["pathway_id", "patch_id", "n_samples"] and col.endswith("_mean"):
                prop_name = col.replace("_mean", "")
                mean_val = row[f"{prop_name}_mean"]
                if not np.isnan(mean_val):
                    text_content += f"  {prop_name}: {mean_val:.3f}"
                    if f"{prop_name}_std" in row:
                        std_val = row[f"{prop_name}_std"]
                        text_content += f" ± {std_val:.3f}\n"
                    else:
                        text_content += "\n"
    
    ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    df, df_tsne_train, df_tsne_centroids = load_data()
    df_crop_cth_pct, df_crop_cot_pct = load_crop_percentiles(BASE_DIR)
    
    # Get unique pathways sorted by frequency
    pathway_counts = df.groupby("pathway_id").size().sort_values(ascending=False)
    most_common_pathways = pathway_counts.head(20).index.tolist()
    
    all_patch_stats = []
    
    print(f"\nAnalyzing {len(most_common_pathways)} pathways...")
    
    for pathway_id in most_common_pathways:
        df_pw = df[df["pathway_id"] == pathway_id].copy()
        pathway_name = df_pw["pathway"].iloc[0]
        n_samples = len(df_pw)
        
        print(f"\n  Pathway {pathway_id}: {pathway_name} ({n_samples} samples)")
        
        # Get t-SNE coordinates for this pathway
        test_vectors = df_pw[["tsne_dim_1", "tsne_dim_2"]].values
        
        if len(test_vectors) < 4:
            print(f"    Skipping: Too few samples ({len(test_vectors)})")
            continue
        
        # Identify patches
        x_min, x_max = df_tsne_train["tsne_dim_1"].min(), df_tsne_train["tsne_dim_1"].max()
        y_min, y_max = df_tsne_train["tsne_dim_2"].min(), df_tsne_train["tsne_dim_2"].max()
        
        labeled_array, grids, kde, density_threshold, zz = identify_patches(
            test_vectors, x_min, x_max, y_min, y_max
        )
        
        if labeled_array is None:
            print(f"    Skipping: Could not compute density")
            continue
        
        xx, yy = grids
        n_patches = np.max(labeled_array)
        print(f"    Found {n_patches} patch(es)")
        
        # Assign points to patches
        patch_ids = assign_points_to_patches(test_vectors, xx, yy, labeled_array)
        
        # Compute statistics (including cth50/cot50 aligned with panel d logic).
        df_pw_stats = add_cth_cot50_columns(df_pw, df_crop_cth_pct, df_crop_cot_pct)
        patch_stats = compute_patch_statistics(df_pw_stats, patch_ids, pathway_id)
        all_patch_stats.append(patch_stats)
        
        print(f"    Computed statistics for {len(patch_stats)} patch(es)")
        
        # Create visualization
        fig = visualize_patches(df_pw_stats, test_vectors, xx, yy, labeled_array, patch_ids,
                               df_tsne_train, pathway_id, pathway_name)
        
        fig_path = os.path.join(PATCHES_OUT_DIR, f"pathway_{pathway_id:03d}_patches.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved visualization to {fig_path}")
    
    # Combine and save all statistics
    if all_patch_stats:
        all_stats_df = pd.concat(all_patch_stats, ignore_index=True)
        stats_path = os.path.join(PATCHES_OUT_DIR, "patch_statistics_summary.csv")
        all_stats_df.to_csv(stats_path, index=False)
        print(f"\nSaved patch statistics to {stats_path}")
        print(f"Total patches analyzed: {len(all_stats_df)}")
        print("\nSample statistics:")
        print(all_stats_df.head(10))
    else:
        print("\nNo patches found in any pathway")


if __name__ == "__main__":
    main()
