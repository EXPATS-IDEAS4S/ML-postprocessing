"""
Embedding Visualization Script

This script loads dimensionality-reduced embeddings and their corresponding labels,
then generates visualizations such as scatter plots, grids of image crops, and
frame-wise embeddings for video crops. Configurable parameters are defined at the top,
making the workflow reproducible and adaptable to different runs, datasets, and 
visualization styles.
"""

import os
import pandas as pd
import sys

sys.path.append('/home/Daniele/codes/VISSL_postprocessing/')
from scripts.pretrain.embedding_visualization.plot_embedding_utils import (
    plot_average_crop_shapes,
    plot_embedding_crops_table,
    plot_embedding_crops_table_transposed,
    plot_embedding_crops_new,
    plot_embedding_dots_iterative_test_msg_icon,
    scale_to_01_range,
    name_to_rgb,
    extract_hour,
    plot_embedding_dots,
    plot_embedding_filled,
    plot_embedding_crops,
    plot_embedding_dots_iterative_case_study,
    plot_average_crop_values,
    plot_embedding_crops_grid,
    plot_embedding_crops_binned_grid,
    create_WV_IR_diff_colormap,
    plot_classwise_grids,
    plot_embedding_crops_table_transposed_new
)
from scripts.pretrain.embedding_visualization.prepare_df_with_img_path import load_labels

from utils.plotting.class_colors import COLORS_PER_CLASS, CLOUD_CLASS_INFO

from scripts.pretrain.embedding_visualization.config_visualization import (
    RUN_NAME,
    OUTPUT_PATH,
    TRAIN_LIST_IMAGE_CROPS,
    TEST_LIST_IMAGE_CROPS,
    N_FRAMES,
    FILENAME_TSNE,
    FILENAME_LABELS,
    VARIABLE_TYPE,
    CMAP,
    VIDEO,
    RANDOM_SEEDS,
    PERPLEXITY,
    YEAR,
    FILENAME
)

expanded_csv = os.path.join(
        os.path.dirname(OUTPUT_PATH),
        f"merged_tsne_crop_list_with_img_path.csv"
    )

def plot_main_embeddings(df: pd.DataFrame, filename: str = "tsne_embedding.png"):
    """Generate main embedding visualizations."""
    #keep only row with a certain year

    #get only 2012 rowa
    if YEAR is not None:
        df = df[df['year'] == YEAR]

    plot_embedding_dots(df, COLORS_PER_CLASS, OUTPUT_PATH, filename, 'Component_1', 'Component_2', perplexity=PERPLEXITY, year=YEAR)
    #plot_embedding_crops_table(df, OUTPUT_PATH, filename, n=5, selection="closest")
    #print(df['path'].iloc[0])
    #plot_embedding_crops_new(df, OUTPUT_PATH, filename)
    # Example alternatives:
    # plot_embedding_filled(df, COLORS_PER_CLASS, OUTPUT_PATH, FILENAME, df)
    # plot_classwise_grids(df, OUTPUT_PATH, FILENAME, CMAP, n=100, selection="closest")


def plot_video_frames(df_expanded: pd.DataFrame, df_centroids: pd.DataFrame, ordered_items, items_dict):
    """Plot embeddings for each video frame if VIDEO mode is enabled."""
    
    for frame_idx in range(N_FRAMES):
        df_frame = df_expanded[df_expanded["frame_idx"] == frame_idx]
        if not df_frame.empty:
            print(f"Plotting frame {frame_idx} with {len(df_frame)} samples...")
            plot_embedding_crops_grid(   
                df_frame,
                df_centroids,
                OUTPUT_PATH,
                FILENAME,
                VARIABLE_TYPE,
                items_dict,
                comp_1='tsne_dim_1',
                comp_2='tsne_dim_2',
                grid_size=20,
                zoom=0.33)
            plot_embedding_crops_table_transposed_new(
            df_frame,
            OUTPUT_PATH,
            f"feature_space_{RUN_NAME}.png",
            ordered_items,
            n_closest=5,
            n_random=5,
            random_seed=0,
            )
            
            if RANDOM_SEEDS is not None:
                for random_seed in RANDOM_SEEDS:
                    print(f"Plotting table for frame {frame_idx} with random seed {random_seed}...")
                    plot_embedding_crops_table_transposed(df_frame, 
                                            OUTPUT_PATH, 
                                            f"{os.path.splitext(FILENAME)[0]}_frame{frame_idx}.png", 
                                            n=10, 
                                            selection="closest",
                                            random_seed=random_seed
                                            )
  


# =============================================================================
# MAIN
# =============================================================================
def main():
    items = sorted(CLOUD_CLASS_INFO.items(), key=lambda x: x[1]["order"])
    items_dict = dict(items)
    print(items)
    labels = sorted([label for label, _ in items])
    print(labels)
    print(f"TRAIN image files: {len(TRAIN_LIST_IMAGE_CROPS)}")
    print(f"TEST image files: {len(TEST_LIST_IMAGE_CROPS)}")

    df_feat_labels, df_centroids = load_labels(OUTPUT_PATH, FILENAME_TSNE, FILENAME_LABELS, labels)
    print(df_feat_labels)
    print(df_centroids)

    #in df_feat_labels only retain split == TRAIN
    df_feat_labels = df_feat_labels[df_feat_labels["split"].isin(["TRAIN"])]
   
    if not os.path.exists(expanded_csv):
        raise FileNotFoundError(
            f"Expanded CSV not found: {expanded_csv}. Run prepare_df_with_img_path.py first."
        )

    df_feat_train = pd.read_csv(expanded_csv, low_memory=False)
    print(df_feat_train)
    #retain onlu splot == TRAIN
    df_feat_train = df_feat_train[df_feat_train["split"].isin(["TRAIN"])]


    # Backfill columns that may be missing in an older expanded CSV.
    # merge_cols = []
    # for col in ["path", "distance", "split", "vector_type"]:
    #     if col in df_feat_labels.columns:
    #         merge_cols.append(col)

    # if "distance" not in df_feat_train.columns and "path" in df_feat_train.columns and "path" in merge_cols:
    #     lookup = df_feat_labels[merge_cols].drop_duplicates(subset=["path"], keep="first")
    #     df_feat_train = df_feat_train.merge(
    #         lookup,
    #         on="path",
    #         how="left",
    #         suffixes=("", "_src")
    #     )

    #     for col in ["distance", "split", "vector_type"]:
    #         src_col = f"{col}_src"
    #         if src_col in df_feat_train.columns and col not in df_feat_train.columns:
    #             df_feat_train[col] = df_feat_train[src_col]
    #         elif src_col in df_feat_train.columns and col in df_feat_train.columns:
    #             df_feat_train[col] = df_feat_train[col].where(df_feat_train[col].notna(), df_feat_train[src_col])
    #         if src_col in df_feat_train.columns:
    #             df_feat_train = df_feat_train.drop(columns=[src_col])

    if "img_path" in df_feat_train.columns:
        df_feat_train["sample_path"] = df_feat_train["path"]
        df_feat_train["path"] = df_feat_train["img_path"]


    df_feat_train = df_feat_train[df_feat_train["label"] != -100]
 
    #plot_main_embeddings(df_feat_train, df_centroids, filename=FILENAME)
    if VIDEO:
        plot_video_frames(df_feat_train, df_centroids, items, items_dict)


if __name__ == "__main__":
    main()

#47648