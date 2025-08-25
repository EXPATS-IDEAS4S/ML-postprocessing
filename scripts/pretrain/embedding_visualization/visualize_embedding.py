"""
Embedding Visualization Script

This script loads dimensionality-reduced embeddings and their corresponding labels,
then generates visualizations such as scatter plots, grids of image crops, and
frame-wise embeddings for video crops. Configurable parameters are defined at the top,
making the workflow reproducible and adaptable to different runs, datasets, and 
visualization styles.
"""

import os
from glob import glob
import pandas as pd
import numpy as np

from scripts.pretrain.dim_reduction.plot_embedding_utils import (
    plot_average_crop_shapes,
    plot_embedding_crops_table,
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
)

# =============================================================================
# CONFIGURATION
# =============================================================================
RUN_NAME = "dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1"
CROPS_NAME = "clips_ir108_100x100_8frames_2013-2020"
RANDOM_STATE = 3
SAMPLING_TYPE = "all"            # Options: "all", "subsample"
REDUCTION_METHOD = "tsne"        # Options: "tsne", "isomap"
EPOCH = 800
FILE_EXTENSION = "png"
SUBSTITUTE_PATH = True
VARIABLE_TYPE = "IR_108_cm"      # e.g. "WV_062-IR_108"
VIDEO = True
N_FRAMES = 8

# Visualization settings
VMIN, CENTER, VMAX = -60, 0, 5
CMAP = "gray"  # or create_WV_IR_diff_colormap(VMIN, CENTER, VMAX)
OUTPUT_PATH = f"/data1/fig/{RUN_NAME}/epoch_{EPOCH}/{SAMPLING_TYPE}/"
FILENAME = f"{REDUCTION_METHOD}_pca_cosine_perp-50_{RUN_NAME}_{RANDOM_STATE}_epoch_{EPOCH}.npy"

# Input data
IMAGE_CROPS_PATH = f"/data1/crops/{CROPS_NAME}/img/{VARIABLE_TYPE}/1/"
LIST_IMAGE_CROPS = sorted(glob(IMAGE_CROPS_PATH + "*." + FILE_EXTENSION))

# Class color mapping
COLORS_PER_CLASS = {
    '0': 'darkgray',
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green',
    '9': 'goldenrod',
    '10': 'magenta',
    '11': 'dodgerblue',
    '12': 'darkorange',
    '13': 'olive',
    '14': 'crimson',
}


# =============================================================================
# FUNCTIONS
# =============================================================================
def load_labels() -> pd.DataFrame:
    """Load precomputed labels and dimensionality-reduced features."""
    csv_path = f"{OUTPUT_PATH}merged_tsne_crop_list_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}.csv"
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^color")]  # drop pre-existing color cols
    return df


def prepare_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid labels and map cluster labels to colors."""
    df_valid = df[df["label"] != -100].copy()
    df_valid["color"] = df_valid["label"].map(lambda x: COLORS_PER_CLASS[str(int(x))])
    return df_valid


def plot_main_embeddings(df: pd.DataFrame):
    """Generate main embedding visualizations."""
    plot_embedding_dots(df, COLORS_PER_CLASS, OUTPUT_PATH, FILENAME)
    # Example alternatives:
    # plot_embedding_filled(df, COLORS_PER_CLASS, OUTPUT_PATH, FILENAME, df)
    # plot_classwise_grids(df, OUTPUT_PATH, FILENAME, CMAP, n=100, selection="closest")


def plot_video_frames(df_labels: pd.DataFrame):
    """Plot embeddings for each video frame if VIDEO mode is enabled."""
    expanded_csv = os.path.join(
        os.path.dirname(OUTPUT_PATH),
        f"merged_tsne_crop_list_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}_expanded.csv"
    )

    if os.path.exists(expanded_csv):
        df_expanded = pd.read_csv(expanded_csv)
        df_expanded = df_expanded[df_expanded["label"] != -100]

        for frame_idx in range(N_FRAMES):
            df_frame = df_expanded[df_expanded["frame_idx"] == frame_idx]
            if not df_frame.empty:
                plot_embedding_crops_grid(
                    df_frame,
                    OUTPUT_PATH,
                    filename=f"{os.path.splitext(FILENAME)[0]}_frame{frame_idx}.png",
                    variable_type=VARIABLE_TYPE,
                    cmap=CMAP,
                    grid_size=20,
                    zoom=0.33,
                )
    else:
        substitute_paths_and_plot(df_labels)


def substitute_paths_and_plot(df_labels: pd.DataFrame):
    """Substitute image paths per frame and plot grids if expanded dataset is missing."""
    if SUBSTITUTE_PATH and VIDEO:
        for frame_idx in range(N_FRAMES):
            frame_rows = []
            for _, row in df_labels.iterrows():
                video_stem = os.path.splitext(os.path.basename(row["path"]))[0]
                frame_str = f"t{frame_idx}_"
                matches = [p for p in LIST_IMAGE_CROPS if video_stem in p and frame_str in p]
                if not matches:
                    continue
                new_row = row.copy()
                new_row["path"] = matches[0]
                new_row["frame_idx"] = frame_idx
                frame_rows.append(new_row)

            df_frame = pd.DataFrame(frame_rows)
            df_frame = df_frame[df_frame["label"] != -100]

            if not df_frame.empty:
                plot_embedding_crops_grid(
                    df_frame,
                    OUTPUT_PATH,
                    filename=f"{os.path.splitext(FILENAME)[0]}_frame{frame_idx}.png",
                    variable_type=VARIABLE_TYPE,
                    cmap=CMAP,
                    grid_size=20,
                    zoom=0.33,
                )
    else:
        if SUBSTITUTE_PATH:
            df_labels["path"] = df_labels["crop_index"].apply(
                lambda x: LIST_IMAGE_CROPS[int(x)]
            )
            df_labels = df_labels[df_labels["label"] != -100]

        plot_embedding_crops_grid(
            df_labels,
            OUTPUT_PATH,
            FILENAME,
            variable_type=VARIABLE_TYPE,
            cmap=CMAP,
            grid_size=20,
            zoom=0.33,
        )


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"n samples: {len(LIST_IMAGE_CROPS)}")

    df_labels = load_labels()
    df_prepared = prepare_colors(df_labels)

    plot_main_embeddings(df_prepared)
    if VIDEO:
        plot_video_frames(df_labels)


if __name__ == "__main__":
    main()
