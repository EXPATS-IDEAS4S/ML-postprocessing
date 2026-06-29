"""
Embedding Visualization Script For Video Crops

This script reads a dimensionality-reduction CSV for a video-crop experiment,
associates each embedding row with the corresponding rendered crop image, derives the
frame index for each image, and writes frame-wise visualization panels for the
selection mode requested in the visualization config. When the frame PNGs have been
written, the script also assembles them into an animation.

Configuration used by this script:
- /home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml

Configured inputs in the current setup:
- Expanded embedding CSV used at runtime when it already exists:
    /sat_data/output/grl_2026/csv/merged_tsne_crop_stats_grl_2026_all_3_epoch_800_expanded.csv
- Base embedding CSV used to create the expanded CSV when needed:
    /sat_data/output/grl_2026/csv/merged_tsne_crop_stats_grl_2026_all_3_epoch_800.csv
- Crop image directory used by the fallback reconstruction path and by expanded CSV
    creation:
    /sat_data/crops/grl_2026/img/

How the script works:
1. Load the visualization section from
     /home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml.
2. Reuse the expanded CSV if it already exists. This is the fast path.
3. If the expanded CSV does not exist, merge the base embedding CSV with image files
     found in /sat_data/crops/grl_2026/img/ and save a new expanded CSV.
4. Replace plotting paths with image_path when available and derive frame_idx from the
     image filename pattern `_t<frame>_`.
5. Normalize distance per label subset so the displayed value is intended to span
     from 0 for the crops treated as closest to the centroid to 1 for the crops treated
     as farthest.
6. Group rows by frame once, pre-sort each label subset once, then select the rows to
     plot according to the configured selection mode (`closest`, `farthest`, or `random`).
7. Save one grid PNG and one transposed crop-table PNG for each plotted frame.
8. Assemble the crop-table PNGs into a GIF and, when OpenCV is available, an MP4.
9. Assemble the grid PNGs into a GIF and, when OpenCV is available, an MP4.

Expanded CSV written by the creation path:
- /sat_data/output/grl_2026/csv/merged_tsne_crop_stats_grl_2026_all_3_epoch_800_expanded.csv

Frame-wise PNG outputs:
- Grid view for every frame:
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_frame<frame_idx>_IR_108_grid.png
- Non-random selection modes:
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_frame<frame_idx>_10_<selection>_crops_table_transposed.png
- Random selection mode:
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_frame<frame_idx>_10_random_crops_table_transposed_rs-<seed>.png

Animation outputs written after all frame PNGs are created:
- Grid animation:
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_IR_108_grid.gif
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_IR_108_grid.mp4
- Non-random selection modes:
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_10_<selection>_crops_table_transposed.gif
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_10_<selection>_crops_table_transposed.mp4
- Random selection mode:
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_10_random_crops_table_transposed_rs-<seed>.gif
    /sat_data/output/grl_2026/csv/tsne_embedding_grl_2026_epoch_800_10_random_crops_table_transposed_rs-<seed>.mp4

example call of the script:
conda run -n vissl python scripts/pretrain/embedding_visualization/visualize_embedding_video.py --mode=closest 
"""

import os
import re
from pathlib import Path
from glob import glob
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import sys
import pdb
from tqdm import tqdm
from PIL import Image

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

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "process_run_GRL.yaml"


def get_visualization_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load visualization settings from YAML."""
    config = load_config(str(config_path))
    visualization_config = config.get("visualization", {})

    if not visualization_config:
        raise ValueError(f"Missing 'visualization' section in config: {config_path}")

    return visualization_config


VIS_CONFIG = get_visualization_config()

# =============================================================================
# CONFIGURATION
# =============================================================================
RUN_NAME = VIS_CONFIG["run_name"]
CROPS_NAME = VIS_CONFIG["crops_name"]
RANDOM_STATE = VIS_CONFIG["random_state"]
SAMPLING_TYPE = VIS_CONFIG["sampling_type"]
REDUCTION_METHOD = VIS_CONFIG["reduction_method"]
EPOCH = VIS_CONFIG["epochs"][0]
FILE_EXTENSION = VIS_CONFIG["file_extension"]
SUBSTITUTE_PATH = VIS_CONFIG["substitute_path"]
VARIABLE_TYPE = VIS_CONFIG["variable_type"]
VIDEO = VIS_CONFIG["video"]
N_FRAMES = VIS_CONFIG["n_frames"]
RANDOM_SEEDS = VIS_CONFIG["random_seed"]
SELECTION_MODE = VIS_CONFIG.get("selection", "random")

# Visualization settings
CMAP = VIS_CONFIG["cmap"]
OUTPUT_PATH = os.path.join(VIS_CONFIG["output_path"], "")

# Input data
IMAGE_CROPS_PATH = VIS_CONFIG["image_crops_path"]
LIST_IMAGE_CROPS = None
IMAGE_CROP_LOOKUP = None
FILENAME = f"{REDUCTION_METHOD}_embedding_{RUN_NAME}_epoch_{EPOCH}.png"



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


def get_image_crops() -> List[str]:
    """Load image crop paths lazily to avoid an expensive directory scan at import time."""
    global LIST_IMAGE_CROPS

    if LIST_IMAGE_CROPS is None:
        print(f"Scanning image crops in {IMAGE_CROPS_PATH} ...")
        LIST_IMAGE_CROPS = sorted(glob(os.path.join(IMAGE_CROPS_PATH, f"*.{FILE_EXTENSION}")))
        print(f"Found {len(LIST_IMAGE_CROPS)} image crops")

    return LIST_IMAGE_CROPS


def get_image_crop_lookup() -> Dict[Tuple[str, str], str]:
    """Index image crops by base crop key and frame token for fast lookup."""
    global IMAGE_CROP_LOOKUP

    if IMAGE_CROP_LOOKUP is None:
        image_crop_lookup = {}
        for image_path in get_image_crops():
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            crop_key, separator, frame_suffix = image_name.rpartition("_t")
            if not separator:
                continue

            frame_token = frame_suffix.split("_", 1)[0]
            image_crop_lookup[(crop_key, frame_token)] = image_path

        IMAGE_CROP_LOOKUP = image_crop_lookup
        print(f"Indexed {len(IMAGE_CROP_LOOKUP)} crop/frame image paths")

    return IMAGE_CROP_LOOKUP


def with_plot_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer resolved image paths when they are already available in the expanded CSV."""
    if "image_path" not in df.columns:
        return df

    df_plot = df.copy()
    if "path" in df_plot.columns:
        df_plot["path"] = df_plot["image_path"].fillna(df_plot["path"])
    else:
        df_plot["path"] = df_plot["image_path"]

    return df_plot


def with_frame_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Populate frame_idx from image_path when the expanded CSV did not persist it."""
    if "frame_idx" in df.columns or "image_path" not in df.columns:
        return df

    df_with_frames = df.copy()
    # Expanded CSVs created before this optimization do not store frame_idx, but the
    # frame number is still encoded in image_path as `_t<frame>_`.
    frame_idx = df_with_frames["image_path"].str.extract(r"_t(\d+)_", expand=False)
    if frame_idx.notna().any():
        df_with_frames["frame_idx"] = pd.to_numeric(frame_idx, errors="coerce")
        print("Derived frame_idx from image_path in expanded CSV")

    return df_with_frames


def build_sorted_label_subsets(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Group a frame once by label and sort each label subset by distance."""
    sorted_label_subsets = {}
    # The plotting helper only needs per-label rows ordered by distance. Doing this
    # once per frame is cheaper than repeating the same groupby/sort for every seed.
    for label, label_df in df.groupby("label", sort=True):
        sorted_subset = label_df.sort_values(by="distance", ascending=True).reset_index(drop=True)
        distance_min = sorted_subset["distance"].min()
        distance_max = sorted_subset["distance"].max()

        if pd.isna(distance_min) or pd.isna(distance_max) or distance_max == distance_min:
            sorted_subset["distance_normalized"] = 0.0
        else:
            # In this dataset higher raw values correspond to crops that are treated as
            # closer to the centroid by the existing selection logic. Invert the min-max
            # scaling so displayed distances read 0 for closest and 1 for farthest.
            sorted_subset["distance_normalized"] = (
                (distance_max - sorted_subset["distance"]) / (distance_max - distance_min)
            )

        sorted_label_subsets[int(label)] = sorted_subset

    return sorted_label_subsets


def select_label_subsets(
    sorted_label_subsets: Dict[int, pd.DataFrame],
    n: int,
    selection: str,
    random_seed: int,
) -> Dict[int, pd.DataFrame]:
    """Select rows from pre-sorted label subsets without re-filtering the full frame DataFrame."""
    selected_label_subsets = {}

    for label, label_subset in sorted_label_subsets.items():
        # Reuse the cached distance ordering and only apply the seed-dependent selection.
        if selection == "closest":
            selected_subset = label_subset.tail(n)
        elif selection == "farthest":
            selected_subset = label_subset.head(n)
        elif selection == "random":
            selected_subset = label_subset.sample(n=min(n, len(label_subset)), random_state=random_seed)
        else:
            raise ValueError("Invalid selection method. Choose 'closest', 'farthest', or 'random'.")

        selected_label_subsets[label] = selected_subset.reset_index(drop=True)

    return selected_label_subsets


def create_animation_from_frames(frame_paths: List[str]) -> None:
    """Create a transparent GIF and, when OpenCV is available, an MP4 from ordered frame PNGs."""
    if not frame_paths:
        return

    first_frame = os.path.basename(frame_paths[0])
    animation_stem = re.sub(r"_frame\d+", "", os.path.splitext(first_frame)[0])
    output_gif = os.path.join(OUTPUT_PATH, f"{animation_stem}.gif")

    gif_frames = [Image.open(frame_path).convert("RGBA") for frame_path in frame_paths]
    gif_palette_frames = [frame.convert("P", palette=Image.ADAPTIVE) for frame in gif_frames]
    gif_palette_frames[0].save(
        output_gif,
        save_all=True,
        append_images=gif_palette_frames[1:],
        duration=1000,
        loop=0,
        transparency=0,
        disposal=2,
    )
    print(f"Saved animation GIF to {output_gif}")

    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping MP4 creation")
        return

    mp4_frames = [np.array(frame.convert("RGB")) for frame in gif_frames]
    height, width = mp4_frames[0].shape[:2]
    output_mp4 = os.path.join(OUTPUT_PATH, f"{animation_stem}.mp4")
    video_writer = cv2.VideoWriter(
        output_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        1,
        (width, height),
    )

    for frame in mp4_frames:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved animation MP4 to {output_mp4}")


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
    print(f"Preparing frame plots from {len(df_labels)} loaded rows")
    df_expanded = with_frame_idx(with_plot_paths(df_labels))
    generated_frame_paths: Dict[str, List[str]] = {}
    generated_grid_paths: List[str] = []

    if "frame_idx" not in df_expanded.columns:
        print("Loaded labels do not contain frame indices. Reconstructing frame paths from image crops.")
        substitute_paths_and_plot(df_labels)
        return

    df_expanded = df_expanded[df_expanded["label"] != -100].copy()
    df_expanded["frame_idx"] = pd.to_numeric(df_expanded["frame_idx"], errors="coerce")
    df_expanded = df_expanded.dropna(subset=["frame_idx"])
    df_expanded["frame_idx"] = df_expanded["frame_idx"].astype(int)
    print(f"Expanded rows after filtering: {len(df_expanded)}")

    # Materialize each frame bucket once so later loops do not rescan the full table.
    frame_groups = {
        int(frame_idx): frame_df.reset_index(drop=True)
        for frame_idx, frame_df in df_expanded.groupby("frame_idx", sort=True)
    }
    print(f"Grouped expanded rows into {len(frame_groups)} frame buckets")

    for frame_idx in range(N_FRAMES):
        df_frame = frame_groups.get(frame_idx)
        frame_size = 0 if df_frame is None else len(df_frame)
        print(f"Frame {frame_idx}: {frame_size} rows")
        if df_frame is None or df_frame.empty:
            continue

        # plot grid plot for all crops in the frame as a background reference
        grid_output_file = plot_embedding_crops_grid(
            df_frame.copy(),
            OUTPUT_PATH,
            filename=f"{os.path.splitext(FILENAME)[0]}_frame{frame_idx}.png",
            variable_type=VARIABLE_TYPE,
            cmap=CMAP,
            grid_size=20,
            zoom=0.33,
        )
        generated_grid_paths.append(grid_output_file)

        sorted_label_subsets = build_sorted_label_subsets(df_frame)
        print(f"Frame {frame_idx}: prepared {len(sorted_label_subsets)} sorted label subsets")

        selection_random_seeds = RANDOM_SEEDS if SELECTION_MODE == "random" else [None]

        for random_seed in selection_random_seeds:
            if random_seed is None:
                print(f"Saving frame {frame_idx} table with selection={SELECTION_MODE}")
            else:
                print(f"Saving frame {frame_idx} table for random_seed={random_seed}")
            selected_label_subsets = select_label_subsets(
                sorted_label_subsets,
                n=10,
                selection=SELECTION_MODE,
                random_seed=random_seed,
            )
            output_file = plot_embedding_crops_table_transposed(
                df_frame,
                OUTPUT_PATH,
                f"{os.path.splitext(FILENAME)[0]}_frame{frame_idx}.png",
                n=10,
                selection=SELECTION_MODE,
                random_seed=random_seed,
                label_subsets=selected_label_subsets,
            )

            animation_key = SELECTION_MODE if random_seed is None else f"{SELECTION_MODE}_rs-{random_seed}"
            generated_frame_paths.setdefault(animation_key, []).append(output_file)

    for animation_key, frame_paths in generated_frame_paths.items():
        print(f"Creating animation for {animation_key} from {len(frame_paths)} frames")
        create_animation_from_frames(frame_paths)

    if generated_grid_paths:
        print(f"Creating grid animation from {len(generated_grid_paths)} frames")
        create_animation_from_frames(generated_grid_paths)


def substitute_paths_and_plot(df_labels: pd.DataFrame):
    """Substitute image paths per frame and plot grids if expanded dataset is missing."""
    if SUBSTITUTE_PATH and VIDEO:
        image_crop_lookup = get_image_crop_lookup()
        for frame_idx in range(N_FRAMES):
            print(f"Reconstructing frame {frame_idx} paths")
            frame_rows = []
            for _, row in df_labels.iterrows():
                video_stem = os.path.splitext(os.path.basename(row["path"]))[0]
                match = image_crop_lookup.get((video_stem, str(frame_idx)))
                if not match:
                    continue
                new_row = row.copy()
                new_row["path"] = match
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
            image_crops = get_image_crops()
            df_labels["path"] = df_labels["crop_index"].apply(
                lambda x: image_crops[int(x)]
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

def create_expanded_csv():
    """"
    Create an expanded CSV by merging the original tsne features CSV with the list of image crops.
    This is necessary for video frame-wise plotting when the original CSV does not contain explicit frame indices.
    input: config dictionary with necessary parameters
    output: expanded CSV file saved to disk and returned as DataFrame

    """

    image_crops = get_image_crops()


    # read feature csv file merged_tsne_crop_stats_grl_2026_all_3_epoch_800.csv
    csv_path = OUTPUT_PATH
    print(f"Loading feature CSV from {csv_path}")
    ds_features_tsne = pd.read_csv(os.path.join(csv_path, f"merged_tsne_crop_stats_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}.csv"))
    print(f"Loaded {len(ds_features_tsne)} feature rows")

    ds_features_tsne["crop_key"] = ds_features_tsne["path"].map(
        lambda path: os.path.splitext(os.path.basename(path))[0]
    )

    image_rows = []
    for image_path in tqdm(image_crops, desc="Indexing image paths"):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        crop_key, separator, frame_suffix = image_name.rpartition("_t")
        if not separator:
            continue

        frame_token = frame_suffix.split("_", 1)[0]
        image_rows.append(
            {
                "crop_key": crop_key,
                "image_path": image_path,
                "frame_idx": frame_token,
            }
        )

    image_index_df = pd.DataFrame(image_rows)

    df_expanded = ds_features_tsne.merge(image_index_df, on="crop_key", how="left")
    df_expanded = df_expanded.drop(columns=["crop_key"])
    df_expanded["frame_idx"] = pd.to_numeric(
        df_expanded["image_path"].str.extract(r"_t(\d+)_", expand=False),
        errors="coerce",
    )

    n_unmatched = df_expanded["image_path"].isna().sum()
    if n_unmatched:
        print(f"Warning: {n_unmatched} expanded rows do not have a matching image path")

    df_expanded = df_expanded.dropna(subset=["image_path", "frame_idx"])
    df_expanded["frame_idx"] = df_expanded["frame_idx"].astype(int)
    df_expanded["path"] = df_expanded["image_path"]
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
    print(f"Loaded visualization config from {CONFIG_PATH}")
    
    # check if expanded CSV exists, if not create it by merging tsne features with image paths
    expanded_csv = os.path.join(
        os.path.dirname(OUTPUT_PATH),
        f"merged_tsne_crop_stats_{RUN_NAME}_{SAMPLING_TYPE}_{RANDOM_STATE}_epoch_{EPOCH}_expanded.csv",
    )
    if not os.path.exists(expanded_csv):
        print(f"Expanded CSV not found at {expanded_csv}. Attempting to create it by merging tsne features with image paths.")
        expanded_csv = create_expanded_csv()
    else:
        print(f"Expanded CSV already exists at {expanded_csv}. Skipping creation.")

    # load labels and prepare colors for plotting
    print(f"Loading labels from {expanded_csv}")
    df_labels = load_labels(expanded_csv)

    # prepare colors for plotting
    df_prepared = prepare_colors(df_labels)
    print(f"Prepared {len(df_prepared)} labeled rows for plotting")

    #plot_main_embeddings(df_prepared)
    if VIDEO:
        print("Starting frame-wise plotting")
        plot_video_frames(df_labels)

if __name__ == "__main__":
    main()
