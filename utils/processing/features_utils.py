"""

Utility functions for handling crop files, timestamps, clustering results,
and dimensionality reduction outputs.

This module provides:
- Datetime parsing and validation from crop filenames.
- Filters to check daytime crops and IMERG-compatible timestamps.
- Loading cluster assignments, distances, and crop paths into DataFrames.
- Loading and scaling T-SNE coordinates for feature-space visualization.
- Counting the number of available crop files.

Intended to support postprocessing and analysis of VISSL training runs.
"""


import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from glob import glob
import sys 
import xarray as xr

sys.path.append("/home/Daniele/codes/VISSL_postprocessing")
from utils.plotting.feature_space_plot_functions import scale_to_01_range


def parse_crop_datetime_filename(filename: str) -> datetime:
    """Extract datetime object from crop filename (format: YYYYMMDD-HH:MM_...)."""
    try:
        timestamp_str = os.path.basename(filename).split('_')[0]  # e.g., '20130401-00:00'
        return datetime.strptime(timestamp_str, "%Y%m%d-%H:%M")
    except Exception as e:
        print(f"Could not parse datetime from {filename}: {e}")
        return None

def parse_crop_datetime_nc(filename: str) -> datetime:
    """Extract datetime object from crop filename (format: YYYYMMDD-HH:MM_...)."""
    try:
        #open netCDF file and read timestamp
        with xr.open_dataset(filename) as ds:
            timestamp = ds['time'].values[-1] #if multipletime steps, the last one is taken
            #print(timestamp)
            # Safe conversion
            py_datetime = pd.to_datetime(timestamp).to_pydatetime()
            return py_datetime
    except Exception as e:
        print(f"Could not parse datetime from {filename}: {e}")
        return None


def is_daytime(filename: str, file_extension: str) -> bool:
    """Return True if crop timestamp is between 06–16 UTC."""
    if file_extension == 'nc':
        dt = parse_crop_datetime_nc(filename)
    else:
        dt = parse_crop_datetime_filename(filename)
    return dt and (6 <= dt.hour <= 16)


def is_valid_imerg_minute(filename: str, file_extension: str) -> bool:
    """Return True if crop timestamp minute is 00 or 30 (IMERG-compatible)."""
    if file_extension == 'nc':
        dt = parse_crop_datetime_nc(filename)
        print(dt)
    else:
        dt = parse_crop_datetime_filename(filename)
    return dt and (dt.minute in [0, 30])



def get_num_crop(image_crops_path, extension='tif'):
    #image_crops_path = f'{basepath}/{crop_name}/{extension}/1/'
    list_image_crops = sorted(glob(image_crops_path + '*.' + extension))
    n_samples = len(list_image_crops)

    return n_samples

def load_dataframes(base_path: str, base_data_path: str, run_name: str, crops_name: str, file_extension: str, epoch: int) -> pd.DataFrame:
    """Load crop paths, cluster assignments, and distances into a DataFrame."""
    labels_path = f'{base_path}/{run_name}/checkpoints/assignments.pt'
    distances_path = f'{base_path}/{run_name}/checkpoints/distances.pt'
    image_crops_path = f'{base_data_path}/{crops_name}/{file_extension}/1/'

    print(f"Loading image crops from {image_crops_path} ...")
    n_samples = get_num_crop(image_crops_path, extension=file_extension)
    print('Initial n samples:', n_samples)

    # Load cluster data
    list_image_crops = sorted(glob(image_crops_path + '*.' + file_extension))
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




def apply_filters(df: pd.DataFrame, filter_daytime: bool, filter_imerg_minutes: bool, file_extension: str) -> pd.DataFrame:
    """Apply daytime and IMERG filters to the dataset."""
    if filter_daytime:
        df = df[df['path'].apply(is_daytime, file_extension=file_extension)]
        print(f"After daytime filter: {len(df)} samples")

    if filter_imerg_minutes:
        df = df[df['path'].apply(is_valid_imerg_minute, file_extension=file_extension)]
        print(f"After IMERG minute filter: {len(df)} samples")

    return df.reset_index(drop=True)