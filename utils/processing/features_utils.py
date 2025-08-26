import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from glob import glob
import sys 

sys.path.append("/home/Daniele/codes/VISSL_postprocessing")
from utils.plotting.feature_space_plot_functions import scale_to_01_range


def parse_crop_datetime(filename: str) -> datetime:
    """Extract datetime object from crop filename (format: YYYYMMDD-HH:MM_...)."""
    try:
        timestamp_str = os.path.basename(filename).split('_')[0]  # e.g., '20130401-00:00'
        return datetime.strptime(timestamp_str, "%Y%m%d-%H:%M")
    except Exception as e:
        print(f"Could not parse datetime from {filename}: {e}")
        return None


def is_daytime(filename: str) -> bool:
    """Return True if crop timestamp is between 06–16 UTC."""
    dt = parse_crop_datetime(filename)
    return dt and (6 <= dt.hour <= 16)


def is_valid_imerg_minute(filename: str) -> bool:
    """Return True if crop timestamp minute is 00 or 30 (IMERG-compatible)."""
    dt = parse_crop_datetime(filename)
    return dt and (dt.minute in [0, 30])


def load_dataframes(base_path: str, run_name: str, crops_name: str, file_extension: str, epoch: int) -> pd.DataFrame:
    """Load crop paths, cluster assignments, and distances into a DataFrame."""
    labels_path = f'{base_path}/runs/{run_name}/checkpoints/assignments.pt'
    distances_path = f'{base_path}/runs/{run_name}/checkpoints/distances.pt'
    image_crops_path = f'{base_path}/crops/{crops_name}/{file_extension}/1/'

    print(f"Loading image crops from {image_crops_path} ...")
    list_image_crops = sorted(glob(os.path.join(image_crops_path, f'*.{file_extension}')))
    n_samples = len(list_image_crops)
    print('Initial n samples:', n_samples)

    # Load cluster data
    assignments = torch.load(labels_path, map_location='cpu')[0].numpy()
    distances = torch.load(distances_path, map_location='cpu')[0].numpy()
    data_index = np.arange(n_samples)

    return pd.DataFrame({
        'index': data_index,
        'path': list_image_crops,
        'assignment': assignments,
        'distance': distances
    })


def load_tsne_coordinates(output_path: str, filename: str) -> pd.DataFrame:
    """Load T-SNE coordinates from .npy file and return as DataFrame."""
    tsne = np.load(os.path.join(output_path, filename))
    tx = scale_to_01_range(tsne[:, 0])
    ty = scale_to_01_range(tsne[:, 1])
    return pd.DataFrame({"Component_1": tx, "Component_2": ty, "crop_index": np.arange(len(tsne))})