"""
Embedding Visualization Script

This script loads dimensionality-reduced embeddings and their corresponding labels,
then generates visualizations such as scatter plots, grids of image crops, and
frame-wise embeddings for video crops. Configurable parameters are defined at the top,
making the workflow reproducible and adaptable to different runs, datasets, and 
visualization styles.
"""

import os
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
import sys
import pdb
from tqdm import tqdm

sys.path.append('/home/claudia/codes/ML_postprocessing/')
sys.path.append("/home/claudia/codes/ML_postprocessing")
from utils.configs import load_config

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
)



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
def load_labels(csv_path: str) -> pd.DataFrame:
    """Load precomputed labels and dimensionality-reduced features."""
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^color")]  # drop pre-existing color cols
    #print how many rows per label are there
    print("Label distribution:")
    print(df["label"].value_counts())
    #exit()
    return df


def prepare_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out invalid labels and map cluster labels to colors."""
    df_valid = df[df["label"] != -100].copy()
    df_valid["color"] = df_valid["label"].map(lambda x: COLORS_PER_CLASS[str(int(x))])
    return df_valid


def plot_main_embeddings(df: pd.DataFrame):
    """Generate main embedding visualizations."""
    plot_embedding_dots(df, COLORS_PER_CLASS, OUTPUT_PATH, FILENAME, 'Component_1', 'Component_2')
    #plot_embedding_crops_table(df, OUTPUT_PATH, FILENAME, n=5, selection="closest")
    #print(df['path'].iloc[0])
    #plot_embedding_crops_new(df, OUTPUT_PATH, FILENAME)
    # Example alternatives:
    # plot_embedding_filled(df, COLORS_PER_CLASS, OUTPUT_PATH, FILENAME, df)
    # plot_classwise_grids(df, OUTPUT_PATH, FILENAME, CMAP, n=100, selection="closest")


def plot_video_frames(df_labels: pd.DataFrame):
    """Plot embeddings for each video frame if VIDEO mode is enabled."""
    expanded_csv = os.path.join(
        os.path.dirname(OUTPUT_PATH),
        f"merged_tsne_crop_stats_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}.csv"
    )

    if os.path.exists(expanded_csv):
        df_expanded = pd.read_csv(expanded_csv)
        df_expanded = df_expanded[df_expanded["label"] != -100]
        print(df_expanded)
        
        for frame_idx in range(N_FRAMES):
            df_frame = df_expanded[df_expanded["frame_idx"] == frame_idx]
            if not df_frame.empty:
                # plot_embedding_crops_grid(
                #     df_frame,
                #     OUTPUT_PATH,
                #     filename=f"{os.path.splitext(FILENAME)[0]}_frame{frame_idx}.png",
                #     variable_type=VARIABLE_TYPE,
                #     cmap=CMAP,
                #     grid_size=20,
                #     zoom=0.33,
                # )
                for random_seed in RANDOM_SEEDS:
                    plot_embedding_crops_table_transposed(df_frame, 
                                            OUTPUT_PATH, 
                                            f"{os.path.splitext(FILENAME)[0]}_frame{frame_idx}.png", 
                                            n=10, 
                                            selection="random",
                                            random_seed=random_seed
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

            #save expanded dataframe
            expanded_csv = os.path.join(
                os.path.dirname(OUTPUT_PATH),
                f"merged_tsne_crop_list_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}_expanded.csv"
            )
            df_frame.to_csv(expanded_csv, index=False)

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
                plot_embedding_crops_table(df_frame, 
                                           OUTPUT_PATH, 
                                           f"{os.path.splitext(FILENAME)[0]}_frame{frame_idx}.png", 
                                           n=10, 
                                           selection="closest")
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

def create_expanded_csv(config):
    """"
    Create an expanded CSV by merging the original tsne features CSV with the list of image crops.
    This is necessary for video frame-wise plotting when the original CSV does not contain explicit frame indices.
    input: config dictionary with necessary parameters
    output: expanded CSV file saved to disk and returned as DataFrame

    """

    # list all images in the crops directory
    images_dir = config["visualization"]["image_crops_path"]
    LIST_IMAGE_CROPS = sorted(glob(os.path.join(images_dir, "*.png")))
    print(f"Found {len(LIST_IMAGE_CROPS)} image crops in {images_dir}")


    # read feature csv file merged_tsne_crop_stats_grl_2026_all_3_epoch_800.csv
    csv_path = OUTPUT_PATH
    ds_features_tsne = pd.read_csv(os.path.join(csv_path, f"merged_tsne_crop_stats_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}.csv"))

    ds_features_tsne["crop_key"] = ds_features_tsne["path"].map(
        lambda path: os.path.splitext(os.path.basename(path))[0]
    )

    image_keys = []
    for image_path in tqdm(LIST_IMAGE_CROPS, desc="Indexing image paths"):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_keys.append(image_name.rsplit("_t", 1)[0])

    image_index_df = pd.DataFrame(
        {
            "crop_key": image_keys,
            "image_path": LIST_IMAGE_CROPS,
        }
    )

    df_expanded = ds_features_tsne.merge(image_index_df, on="crop_key", how="left")
    df_expanded = df_expanded.drop(columns=["crop_key"])

    n_unmatched = df_expanded["image_path"].isna().sum()
    if n_unmatched:
        print(f"Warning: {n_unmatched} expanded rows do not have a matching image path")

    df_expanded = df_expanded.dropna(subset=["image_path"])
    expanded_csv = os.path.join(
        OUTPUT_PATH,
        f"merged_tsne_crop_stats_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}_expanded.csv",
    )
    df_expanded.to_csv(expanded_csv, index=False)
    print(f"Saved expanded DataFrame with {len(df_expanded)} rows to {expanded_csv}")
    return df_expanded

# =============================================================================
# MAIN
# =============================================================================
def main():
    global RUN_NAME, SAMPLING_TYPE, RANDOM_STATE, EPOCH, OUTPUT_PATH
    global VIDEO, VARIABLE_TYPE, CMAP, FILENAME, LIST_IMAGE_CROPS
    # read config parameters
    config = load_config(config_path)
    # run name
    RUN_NAME = config["experiment"]["run_names"][0]
    # sampling type    
    SAMPLING_TYPE = config["data"]["sampling_type"]
    # random state    
    RANDOM_STATE = config["experiment"]["random_state"]
    # epoch    
    EPOCH = config["experiment"]["epoch"]
    # output path    
    OUTPUT_PATH = os.path.join(config["visualization"]["output_path"], "")
    # video mode
    VIDEO = config["visualization"]["video"]
    # variable type for grid coloring
    VARIABLE_TYPE = config["visualization"]["variable_type"]
    # colormap for grid coloring
    CMAP = config["visualization"]["cmap"]
    # filename for main embedding plot
    FILENAME = f"embedding_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}.png"


    # check if expanded CSV exists, if not create it by merging tsne features with image paths
    expanded_csv = os.path.join(
        os.path.dirname(OUTPUT_PATH),
        f"merged_tsne_crop_stats_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}_expanded.csv",
    )
    if not os.path.exists(expanded_csv):
        print(f"Expanded CSV not found at {expanded_csv}. Attempting to create it by merging tsne features with image paths.")
        expanded_csv = create_expanded_csv(config)
    else:
        print(f"Expanded CSV already exists at {expanded_csv}. Skipping creation.")

    # load labels and prepare colors for plotting
    df_labels = load_labels(expanded_csv)

    # prepare colors for plotting
    df_prepared = prepare_colors(df_labels)

    #plot_main_embeddings(df_prepared)
    if VIDEO:
        plot_video_frames(df_labels)

if __name__ == "__main__":
    config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"

    main()
