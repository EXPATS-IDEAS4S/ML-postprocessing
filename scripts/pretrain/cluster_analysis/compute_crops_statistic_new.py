"""
Compute crop-level and cluster-level statistics for CMSAF and IMERG datasets.

This script extracts statistics from S3 buckets for geospatial crops
(either in GeoTIFF or NetCDF format). It supports both serial and
parallel execution, depending on the `use_parallel` flag.

Main steps:
-----------
1. Load the crop list and labels from a CSV file.
2. For each crop:
   - Parse geographic bounding box (lat/lon) and timestamp from filename.
   - Retrieve corresponding CMSAF or IMERG data files from S3.
   - Subset the data in space and time.
   - Apply masking (using CMA variable where applicable).
   - Compute either:
        * Percentiles (e.g., 50th, 99th) for continuous variables, OR
        * Category counts/proportions for categorical variables.
3. Save per-crop statistics to a CSV.
4. Aggregate per-cluster statistics (mean, std) and save to another CSV.

Inputs:
-------
- Crop list CSV (paths + labels), generated in a prior preprocessing step.
- Crop files (GeoTIFFs or NetCDFs).
- CMSAF and IMERG datasets stored in S3 buckets.

Outputs:
--------
- CSV with crop-level statistics:
  `<output_path>/crops_stats_<run_name>_<sampling_type>_<n_subsample>.csv`
- CSV with cluster-level statistics:
  `<output_path>/clusters_stats_<run_name>_<sampling_type>_<n_subsample>.csv`

Configuration:
--------------
- `use_parallel` : bool
      Whether to use parallel processing via joblib or run serially.
- `data_format` : {'nc','tif'}
      Format of crop files.
- `vars`, `stats`, `categ_vars` : list
      Variables to process and statistics to compute.
- Bucket names, credentials, run_name, paths, etc.

Notes:
------
- Designed for large-scale analysis of satellite datasets stored on S3.
- Parallel processing significantly speeds up computations but increases memory usage.

How to run in batch mode:
------

NOTE: remember to activate the conda environment with the required dependencies before running the script.
source activate vissl

- Adjust configuration parameters in the `config` dictionary.
- Run the script using:
    conda run -n vissl python scripts/pretrain/cluster_analysis/compute_crops_statistic_new.py`
in batch mode use this:
cd /home/claudia/codes/ML_postprocessing
nohup conda run -n vissl python scripts/pretrain/cluster_analysis/compute_crops_statistic_new.py > logs/compute_crops_statistic_new_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID #  241881 launched at 15:00 on thu 7 may 2026

to check when reopening
------
ps -fp 241881
pgrep -P 241881
tail -f /home/claudia/codes/ML_postprocessing/logs/processing_crops_stats_per_frame.log

to check elapsed times
------
ps -o etime= -p 241881

to check memory usage: RSS (resident set size, actual RAM used), VSZ (virtual memory size)
------
ps -o pid,etime,rss,vsz,pmem,pcpu,cmd -p 241881

"""
import argparse
import os, sys, io
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

#sys.path.append(os.path.abspath("/home/Daniele/codes/VISSL_postprocessing"))
sys.path.append(os.path.abspath("/home/claudia/codes/ML_postprocessing"))

from utils.processing.stats_utils import compute_percentile
from utils.configs import load_config
from utils.buckets.credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
from utils.processing.coords_utils import extract_coordinates
from utils.processing.datetime_utils import extract_datetime
from utils.processing.stats_utils import compute_categorical_values, filter_cma_values
from utils.buckets.get_data_from_buckets import read_file, Initialize_s3_client
from utils.processing.features_utils import get_num_crop

import logging


_S3_CLIENT = None
_MISSING_BUCKET_OBJECTS = set()
_SUCCESS_BUCKET_OBJECTS = OrderedDict()
_MAX_SUCCESS_BUCKET_OBJECTS = 8


def get_s3_client():
    """"
    Initialize and cache the S3 client for efficient reuse across multiple bucket reads.
    Returns
    -------
    boto3.client
        An initialized S3 client.
    """
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = Initialize_s3_client(
            S3_ENDPOINT_URL,
            S3_ACCESS_KEY,
            S3_SECRET_ACCESS_KEY,
        )
    return _S3_CLIENT


def read_bucket_object(bucket_name, bucket_filename, logger, cache_misses=True, cache_success=True):
    """
    Read an object from an S3 bucket with optional caching for both successful reads and missing objects.
    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    bucket_filename : str
        Filename/key of the object in the S3 bucket.
    logger : logging.Logger
        Logger for logging messages.
    cache_misses : bool, optional
        If True, cache missing objects to avoid repeated failed attempts (default: True).
    cache_success : bool, optional
        If True, cache successfully read objects to speed up future accesses (default: True).
    Returns
    -------
    bytes or None
        The content of the bucket object as bytes if found, or None if not found.
    bool
        True if the object was a cache hit for a missing object, False otherwise.
    """

    cache_key = (bucket_name, bucket_filename)
    if cache_misses and cache_key in _MISSING_BUCKET_OBJECTS:
        logger.debug("Skipping cached missing object: %s/%s", bucket_name, bucket_filename)
        return None, True

    if cache_success and cache_key in _SUCCESS_BUCKET_OBJECTS:
        _SUCCESS_BUCKET_OBJECTS.move_to_end(cache_key)
        logger.debug("Using cached object: %s/%s", bucket_name, bucket_filename)
        return _SUCCESS_BUCKET_OBJECTS[cache_key], False

    my_obj = read_file(get_s3_client(), bucket_filename, bucket_name)
    if my_obj is None and cache_misses:
        _MISSING_BUCKET_OBJECTS.add(cache_key)
    elif my_obj is not None and cache_success:
        _SUCCESS_BUCKET_OBJECTS[cache_key] = my_obj
        _SUCCESS_BUCKET_OBJECTS.move_to_end(cache_key)
        if len(_SUCCESS_BUCKET_OBJECTS) > _MAX_SUCCESS_BUCKET_OBJECTS:
            _SUCCESS_BUCKET_OBJECTS.popitem(last=False)
    return my_obj, False


def extract_crop_metadata_from_nc(filename, dir_path, engine="netcdf4"):
    """Read crop coords and timestamps from one NetCDF open.
    Parameters
    ----------
    filename : str
        NetCDF file name.
    dir_path : str
        Directory where the file is located.
    engine : str, optional
        Engine used by xarray to open the dataset (default: 'netcdf4').
    Returns
    -------
    dict
        Dictionary containing lat_min, lat_max, lon_min, lon_max.   
    list of np.datetime64
        List of all timestamps found in the file.
    """
    filepath = os.path.join(dir_path, filename)

    with xr.open_dataset(filepath, engine=engine) as ds:
        coords = {
            "lat_min": round(ds["lat"].min().item(), 3),
            "lat_max": round(ds["lat"].max().item(), 3),
            "lon_min": round(ds["lon"].min().item(), 3),
            "lon_max": round(ds["lon"].max().item(), 3),
        }
        times = [np.datetime64(t) for t in ds["time"].values]

    return coords, times


def get_crop_metadata(row, config, image_crops_path):
    """
    Extract crop metadata (coords and timestamps) from either NetCDF or filename.
    Parameters
    ----------
    row : pd.Series
        Row from the DataFrame containing crop information.
    config : dict
        Configuration dictionary.
    image_crops_path : str
        Path to the directory containing image crops.

    Returns
    -------
    dict
        Dictionary containing crop metadata.
    """
    
    crop_filename = os.path.basename(row["path"])
    data_format = config["data"]["file_extension"]
    nc_engine = config["data"]["nc_engine"]

    if data_format == "nc":
        coords, times = extract_crop_metadata_from_nc(
            crop_filename,
            image_crops_path,
            nc_engine,
        )
    else:
        coords = extract_coordinates(crop_filename)
        datetime_info = extract_datetime(crop_filename)
        year, month, day, hour, minute = (
            datetime_info[k] for k in ["year", "month", "day", "hour", "minute"]
        )
        times = [
            np.datetime64(
                f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"
            )
        ]

    return {
        "crop_id": row.name,
        "crop_filename": crop_filename,
        "label": row.get("label", np.nan),
        "coords": coords,
        "times": list(times),
        "lat_min": coords["lat_min"],
        "lat_max": coords["lat_max"],
        "lon_min": coords["lon_min"],
        "lon_max": coords["lon_max"],
    }


def build_bucket_filename(day_key, var, var_meta):
    """"
    Build the S3 bucket filename based on the variable and day key.
    The naming convention depends on the variable type (e.g., RR vs. others).
    Parameters
    ----------
    day_key : str
        Day in the format 'YYYY-MM-DD'.
    var : str
        Variable name (e.g., 'RR', 'euclid_msg_grid').
    var_meta : dict
        Variable metadata from the config, containing filename prefixes/suffixes.
    Returns
    -------
    str
        The constructed bucket filename.
    """
    # crops are grouped into shared day blocks 
    y, m, d = map(int, day_key.split("-"))

    if var == "RR":
        return (
            f"{var_meta['bucket_filename_prefix']}{y:04d}{m:02d}{d:02d}"
            f"{var_meta['bucket_filename_suffix']}"
        )

    if var == "euclid_msg_grid":
        return (
            f"{y:04d}/{m:02d}/{var_meta['bucket_filename_prefix']}{y:04d}{m:02d}{d:02d}"
            f"{var_meta['bucket_filename_suffix']}"
        )

    return (
        f"{var_meta['bucket_filename_prefix']}{y:04d}-{m:02d}-{d:02d}"
        f"{var_meta['bucket_filename_suffix']}"
    )


def load_daily_bucket_objects(day_key, var, var_meta, logger):
    """"
    Load the daily bucket objects for the given variable and day key, with error handling and logging.
    Parameters
    ----------
    day_key : str
        Day in the format 'YYYY-MM-DD'.
    var : str
        Variable name (e.g., 'RR', 'euclid_msg_grid').
    var_meta : dict
        Variable metadata from the config, containing bucket names and filename patterns.
    logger : logging.Logger
        Logger for logging messages.
    Returns
    -------
    bytes or None
        The content of the main bucket object as bytes if found, or None if not found.
    bytes or None
        The content of the CMA bucket object as bytes if found, or None if not found.
    """
    y, m, d = map(int, day_key.split("-"))
    bucket_filename = build_bucket_filename(day_key, var, var_meta)

    my_obj, was_cached_missing = read_bucket_object(
        var_meta["bucket_name"],
        bucket_filename,
        logger,
    )
    if my_obj is None:
        if not was_cached_missing:
            logger.warning(f"File not found: {bucket_filename}")
        return None, None

    if var == "RR":
        import gzip

        with gzip.GzipFile(fileobj=io.BytesIO(my_obj)) as f:
            my_obj = f.read()

    cma_filename = f"MCP_{y:04d}-{m:02d}-{d:02d}_regrid.nc"
    my_obj_cma, cma_was_cached_missing = read_bucket_object(
        "expats-cmsaf-cloud",
        cma_filename,
        logger,
    )
    if my_obj_cma is None:
        if not cma_was_cached_missing:
            logger.warning(f"CMA file not found for day: {day_key}")
        return None, None

    return my_obj, my_obj_cma


def build_day_blocks(crop_requests, sel_vars, filter_imerg=False):
    """
    Group crop requests into blocks that share the same variable and day key.
    Parameters
    ----------
    crop_requests : list of dict
        List of crop metadata dictionaries, each containing 'times' and 'coords'.
    sel_vars : list of str
        List of variables to process (e.g., ['RR', 'euclid_msg_grid']).
    filter_imerg : bool, optional
        If True, restrict precipitation timestamps to IMERG-compatible minutes.
    Returns
    -------
    list of dict
        List of blocks, each containing 'var', 'day_key', and 'crop_requests'.
    """

    day_blocks = defaultdict(list)
    multi_day_blocks = []

    for crop_request in crop_requests:
        for var in sel_vars:
            # Get all days covered by this crop
            times = crop_request["times"]
            day_keys = sorted({str(np.datetime64(t, "D")) for t in times})
            if len(day_keys) > 1:
                # Multi-day crop: create a special block
                multi_day_blocks.append({
                    "var": var,
                    "day_keys": day_keys,
                    "crop_request": crop_request,
                })
            else:
                # Single day: normal block
                for day_key, times_for_day in get_time_windows(
                    times,
                    var=var,
                    filter_imerg=filter_imerg,
                ).items():
                    day_blocks[(var, day_key)].append(
                        {
                            "crop_id": crop_request["crop_id"],
                            "crop_filename": crop_request["crop_filename"],
                            "coords": crop_request["coords"],
                            "times_for_day": times_for_day,
                            "lat_min": crop_request["lat_min"],
                            "lat_max": crop_request["lat_max"],
                            "lon_min": crop_request["lon_min"],
                            "lon_max": crop_request["lon_max"],
                        }
                    )

    blocks = []
    for var, day_key in sorted(day_blocks.keys(), key=lambda item: (item[1], item[0])):
        blocks.append(
            {
                "var": var,
                "day_key": day_key,
                "crop_requests": day_blocks[(var, day_key)],
                "multi_day": False,
            }
        )

    # Add special multi-day blocks
    for entry in multi_day_blocks:
        blocks.append({
            "var": entry["var"],
            "day_keys": entry["day_keys"],
            "crop_requests": [
                {
                    "crop_id": entry["crop_request"]["crop_id"],
                    "crop_filename": entry["crop_request"]["crop_filename"],
                    "coords": entry["crop_request"]["coords"],
                    "times_for_day": entry["crop_request"]["times"],
                    "lat_min": entry["crop_request"]["lat_min"],
                    "lat_max": entry["crop_request"]["lat_max"],
                    "lon_min": entry["crop_request"]["lon_min"],
                    "lon_max": entry["crop_request"]["lon_max"],
                }
            ],
            "multi_day": True,
        })

    return blocks


def normalize_time_value(time_value):
    """
    Normalize a time value to a consistent format (numpy datetime64 in nanoseconds).    
    Parameters
    ----------
    time_value : any
        The input time value, which can be in various formats (e.g., string, datetime, numpy datetime64).
    Returns
    -------
    np.datetime64
        The normalized time value as a numpy datetime64 in nanoseconds.
    """
    return np.datetime64(time_value, "ns")


def get_coord_indices(coord_values, coord_min, coord_max):
    """
    Get the indices of coordinate values that fall within the specified min and max bounds.
    Parameters
    ----------
    coord_values : array-like
        Array of coordinate values (e.g., latitudes or longitudes).
    coord_min : float
        Minimum coordinate value for the bounding box.
    coord_max : float
        Maximum coordinate value for the bounding box.
    Returns
    -------
    np.ndarray
        Array of indices where the coordinate values are within the specified bounds.
    """

    coord_values = np.asarray(coord_values)
    return np.where((coord_values >= coord_min) & (coord_values <= coord_max))[0]


def build_block_context(ds_day, ds_day_cma, crop_requests):
    """"
    Build the context for processing a block of crop requests, including subsetting the data and creating time-to-index mapping.
    Parameters
    ----------
    ds_day : xarray.Dataset
        The dataset containing the main variable data for the day.
    ds_day_cma : xarray.Dataset
        The dataset containing the CMA variable data for the day.
    crop_requests : list of dict
        List of crop request dictionaries, each containing 'times' and spatial bounds.
    Returns
    -------
    dict
        Dictionary containing the subsetted data, CMA data, and time-to-index mapping for the block.
    """

    block_times = sorted(
        {
            normalize_time_value(time_value)
            for crop_request in crop_requests
            for time_value in crop_request["times_for_day"]
        }
    )

    lat_min = min(crop_request["lat_min"] for crop_request in crop_requests)
    lat_max = max(crop_request["lat_max"] for crop_request in crop_requests)
    lon_min = min(crop_request["lon_min"] for crop_request in crop_requests)
    lon_max = max(crop_request["lon_max"] for crop_request in crop_requests)


    # Detect variable type
    is_lightning = "euclid_msg_grid" in ds_day.data_vars
    is_precip = "precipitation" in ds_day.data_vars
    ds_day_proc = ds_day

    if is_lightning:
        # Lightning: resample to 15-min bins by summing, label left
        ds_day_proc = ds_day.resample(time="15min", label="left", closed="left").sum()
        proc_time_values = np.asarray([normalize_time_value(t) for t in ds_day_proc["time"].values])
        block_times_set = set(block_times)
        available_times = [time_value for time_value, norm_time in zip(ds_day_proc["time"].values, proc_time_values) if norm_time in block_times_set]
        ds_day_block = ds_day_proc.sel(time=available_times)
    elif is_precip:
        # Precipitation: forward-fill 30-min values to 15-min crop timestamps
        # (Assume ds_day is 30-min, crop_times are 15-min)
        # Align ds_day to block_times using reindex with method='ffill'
        precip_var = "precipitation"
        # Convert block_times to numpy datetime64[s] for matching
        block_times_arr = np.array(block_times, dtype="datetime64[s]")
        ds_day_proc = ds_day[precip_var]
        # Reindex to 15-min crop times, forward-fill
        ds_day_proc_15 = ds_day_proc.reindex(time=block_times_arr, method="ffill")
        # Wrap back into a Dataset for compatibility
        ds_day_block = ds_day_proc_15.to_dataset(name=precip_var)
    else:
        day_time_values = np.asarray([normalize_time_value(t) for t in ds_day["time"].values])
        block_times_set = set(block_times)
        available_times = [time_value for time_value, norm_time in zip(ds_day["time"].values, day_time_values) if norm_time in block_times_set]
        ds_day_block = ds_day.sel(time=available_times)

    # CMA block always subset as before
    cma_time_values = np.asarray([normalize_time_value(t) for t in ds_day_cma["time"].values])
    block_times_set = set(block_times)
    available_times_cma = [time_value for time_value, norm_time in zip(ds_day_cma["time"].values, cma_time_values) if norm_time in block_times_set]
    ds_day_cma_block = ds_day_cma.sel(time=available_times_cma)

    ds_day_block = ds_day_block.where(
        (ds_day_block["lat"] >= lat_min) & (ds_day_block["lat"] <= lat_max),
        drop=True,
    )
    ds_day_block = ds_day_block.where(
        (ds_day_block["lon"] >= lon_min) & (ds_day_block["lon"] <= lon_max),
        drop=True,
    )

    ds_day_cma_block = ds_day_cma_block.where(
        (ds_day_cma_block["lat"] >= lat_min) & (ds_day_cma_block["lat"] <= lat_max),
        drop=True,
    )
    ds_day_cma_block = ds_day_cma_block.where(
        (ds_day_cma_block["lon"] >= lon_min) & (ds_day_cma_block["lon"] <= lon_max),
        drop=True,
    )

    block_time_values = np.asarray([normalize_time_value(t) for t in ds_day_block["time"].values])
    return {
        "data": ds_day_block,
        "cma": ds_day_cma_block,
        "time_to_index": {
            normalize_time_value(time_value): index
            for index, time_value in enumerate(block_time_values)
        },
    }


def extract_block_values(block_context, var, crop_request, mode, logger, day_key):
    """"
    Extract values from the block context for a specific crop request, applying CMA filtering and handling missing data.
    Parameters
    ----------
    block_context : dict
        Dictionary containing the subsetted data, CMA data, and time-to-index mapping for the block.
    var : str
        Variable name (e.g., 'RR', 'euclid_msg_grid').
    crop_request : dict
        Dictionary containing crop metadata, including 'times_for_day' and spatial bounds.
    mode : str
        Mode of value extraction ('aggregated' or 'per_frame').
    logger : logging.Logger
        Logger for logging messages.
    day_key : str
        Day key for logging purposes (e.g., 'YYYY-MM-DD').
    Returns
    -------
    np.ndarray or list of dict
        Extracted values for the crop request, either aggregated into a single array or as a list of per-frame dictionaries.
    """
    try:
        available_times = [
            time_value
            for time_value in crop_request["times_for_day"]
            if normalize_time_value(time_value) in block_context["time_to_index"]
        ]

        if len(available_times) == 0:
            return np.array([np.nan]) if mode == "aggregated" else []

        ds_subset = block_context["data"].sel(
            lat=slice(crop_request["lat_min"], crop_request["lat_max"]),
            lon=slice(crop_request["lon_min"], crop_request["lon_max"]),
            time=available_times,
        )
        ds_subset_cma = block_context["cma"].sel(
            lat=slice(crop_request["lat_min"], crop_request["lat_max"]),
            lon=slice(crop_request["lon_min"], crop_request["lon_max"]),
            time=available_times,
        )

        if mode == "aggregated":
            values = ds_subset.values.reshape(-1)
            values_cma = ds_subset_cma.values.reshape(-1)
            return filter_cma_values(values, values_cma, var)

        per_frame_list = []
        for time_value in ds_subset.time.values:
            frame_values = ds_subset.sel(time=time_value).values.reshape(-1)
            frame_values_cma = ds_subset_cma.sel(time=time_value).values.reshape(-1)
            frame_values = filter_cma_values(frame_values, frame_values_cma, var)
            per_frame_list.append(
                {
                    "time": np.datetime64(time_value).astype("datetime64[s]").item(),
                    "values": frame_values,
                }
            )
        return per_frame_list
    except Exception:
        logger.error(
            "Error processing day %s for variable '%s' (crop %s)",
            day_key,
            var,
            crop_request["crop_filename"],
            exc_info=True,
        )
        return np.array([np.nan]) if mode == "aggregated" else []


def process_day_block(block, config, var_config, logger):
    """"
    Process a block of crop requests that share the same variable and day key.
    Parameters
    ----------
    block : dict
        Dictionary containing 'var', 'day_key', and 'crop_requests'.
    config : dict
        Configuration dictionary.
    var_config : dict
        Variable metadata from the config.
    logger : logging.Logger
        Logger for logging messages.
    Returns
    -------
    list of tuples
        List of tuples containing crop_id, variable, and extracted values.
    """
    
    var = block["var"]
    var_meta = var_config["variables"][var]
    mode = config["statistics"]["spatial"].get("mode", "aggregated")

    if block.get("multi_day", False):
        # Multi-day crop: load and merge all required days
        logger.debug(f"Processing multi-day block for {var} on days {block['day_keys']}")
        ds_list = []
        ds_cma_list = []
        for day_key in block["day_keys"]:
            my_obj, my_obj_cma = load_daily_bucket_objects(day_key, var, var_meta, logger)
            if my_obj is None or my_obj_cma is None:
                continue
            with xr.open_dataset(io.BytesIO(my_obj)) as ds_bucket, xr.open_dataset(io.BytesIO(my_obj_cma)) as ds_cma_bucket:
                ds = ds_bucket[var]
                ds_cma = ds_cma_bucket["cma"]
                if isinstance(ds.indexes["time"], xr.CFTimeIndex):
                    ds["time"] = ds["time"].astype("datetime64[ns]")
                    ds_cma["time"] = ds_cma["time"].astype("datetime64[ns]")
                ds_list.append(ds)
                ds_cma_list.append(ds_cma)
        if not ds_list or not ds_cma_list:
            empty_values = np.array([np.nan]) if mode == "aggregated" else []
            return [
                (crop_request["crop_id"], var, empty_values)
                for crop_request in block["crop_requests"]
            ]
        # Concatenate along time
        ds_day = xr.concat(ds_list, dim="time")
        ds_day_cma = xr.concat(ds_cma_list, dim="time")
        block_context = build_block_context(ds_day, ds_day_cma, block["crop_requests"])
        return [
            (
                crop_request["crop_id"],
                var,
                extract_block_values(block_context, var, crop_request, mode, logger, ",".join(block["day_keys"])),
            )
            for crop_request in block["crop_requests"]
        ]
    else:
        day_key = block["day_key"]
        logger.debug("Processing shared block for %s on %s", var, day_key)
        my_obj, my_obj_cma = load_daily_bucket_objects(day_key, var, var_meta, logger)
        if my_obj is None or my_obj_cma is None:
            empty_values = np.array([np.nan]) if mode == "aggregated" else []
            return [
                (crop_request["crop_id"], var, empty_values)
                for crop_request in block["crop_requests"]
            ]
        with xr.open_dataset(io.BytesIO(my_obj)) as ds_bucket, xr.open_dataset(io.BytesIO(my_obj_cma)) as ds_cma_bucket:
            ds_day = ds_bucket[var]
            ds_day_cma = ds_cma_bucket["cma"]
            if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
                ds_day["time"] = ds_day["time"].astype("datetime64[ns]")
                ds_day_cma["time"] = ds_day_cma["time"].astype("datetime64[ns]")
            block_context = build_block_context(ds_day, ds_day_cma, block["crop_requests"])
            return [
                (
                    crop_request["crop_id"],
                    var,
                    extract_block_values(block_context, var, crop_request, mode, logger, day_key),
                )
                for crop_request in block["crop_requests"]
            ]


def combine_block_results(block_results, crop_requests, config, var_config):
    """"
    Combine the results from processing all blocks into a flat list of statistics for each crop and variable.
    Parameters
    ----------
    block_results : list of lists of tuples
        List of results from each block, where each result is a list of tuples (crop_id, variable, values).
    crop_requests : list of dict
        List of crop metadata dictionaries, each containing 'crop_id', 'crop_filename', 'label', 'coords', and 'times'.
    config : dict
        Configuration dictionary.
    var_config : dict
        Variable metadata from the config.
    Returns
    -------
    list of dict
        List of dictionaries, each representing a row of computed statistics with metadata for a specific crop and variable.
    """
    values_by_crop_var = defaultdict(list)
    mode = config["statistics"]["spatial"].get("mode", "aggregated")
    sel_vars = config["statistics"]["spatial"]["sel_vars"]

    for block_result in block_results:
        for crop_id, var, values in block_result:
            if mode == "aggregated":
                values_by_crop_var[(crop_id, var)].append(values)
            else:
                values_by_crop_var[(crop_id, var)].extend(values)

    flat_results = []
    for crop_request in crop_requests:
        for var in sel_vars:
            # For precipitation, compute percentiles, sum[mm], and fraction
            if var == "precipitation":
                stats = config["statistics"]["spatial"]["percentiles"] + ["sum[mm]", "prec_fraction"]
            else:
                stats = (
                    ["None"]
                    if var_config["variables"][var]["categorical"]
                    else config["statistics"]["spatial"]["percentiles"]
                )
            values_append = values_by_crop_var.get((crop_request["crop_id"], var), [])
            if mode == "per_frame":
                values_append = sorted(values_append, key=lambda item: item["time"])

            flat_results.extend(
                compute_statistics(
                    values_append,
                    stats,
                    var,
                    mode,
                    crop_request["times"],
                    crop_request["crop_filename"],
                    crop_request["label"],
                    crop_request["coords"],
                )
            )

    return flat_results

def get_time_windows(times, var=None, filter_imerg=False):
    """
    Group a list of crop timestamps by day, with optional IMERG filtering.

    Parameters
    ----------
    times : list of np.datetime64
        List of timestamps (e.g., from crop file metadata).
    filter_imerg : bool, optional
        If True, apply IMERG-compatible filtering for precipitation by keeping
        only timestamps at minutes 00 and 30.
    var : str, optional
        Variable name (used for IMERG filtering).

    Returns
    -------
    dict
        Mapping {day (str YYYY-MM-DD) -> list of timestamps}.
    """
    # Ensure chronological order
    times = sorted(times)

    # Optional IMERG filtering for precipitation.
    if filter_imerg and var == "precipitation":
        times = [
            t for t in times
            if np.datetime64(t, "m").astype(object).minute in (0, 30)
        ]

    if len(times) == 0:
        return {}
    else:
        # Group by day (bucket files are daily)
        times_by_day = {}
        for t in times:
            day_key = str(np.datetime64(t, "D"))
            times_by_day.setdefault(day_key, []).append(t)

        return times_by_day

def compute_statistics(values_append, stats, var, mode, times, crop_filename, label, coords):
    """
    Compute stats either aggregated across frames or per-frame.
    Returns a list of flat dicts (rows).
    Parameters
    ----------
    values_append : list
        List of values to compute stats on (either one array for aggregated or list of dicts for per-frame).
    stats : list of str
        List of stats to compute (e.g., ['50', '99'] for percentiles or ['None'] for categorical).
    var : str
        Variable name (used for categorical processing).
    mode : str
        'aggregated' or 'per_frame'.
    times : list of np.datetime64
        List of timestamps for the crop (used for metadata).
    crop_filename : str
        Filename of the crop (used for metadata).
    label : any
        Label of the crop (used for metadata).
    coords : dict
        Dictionary with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max' (used for metadata).
    Returns
    -------
    list of dict
        List of dictionaries, each representing a row of computed statistics with metadata. 
    """
    rows = []

    lat_mid = (coords["lat_min"] + coords["lat_max"]) / 2
    lon_mid = (coords["lon_min"] + coords["lon_max"]) / 2

    if not values_append:
        if mode == "aggregated":
            row = {
                "crop": crop_filename,
                "label": label,
                "var": var,
                #"frame": None,
                "time": times[0],
                "lat_mid": lat_mid,
                "lon_mid": lon_mid,
            }
            for stat in stats:
                row[stat] = np.nan
            rows.append(row)
        elif mode == "per_frame":
            for frame_idx, t in enumerate(times):
                row = {
                    "crop": crop_filename,
                    "label": label,
                    "var": var,
                    #"frame": frame_idx,
                    "time": str(np.datetime64(t).astype("datetime64[s]").item()),
                    "lat_mid": lat_mid,
                    "lon_mid": lon_mid,
                }
                for stat in stats:
                    row[stat] = np.nan
                rows.append(row)
        return rows

    if mode == "aggregated":
        all_values = np.concatenate([np.atleast_1d(v) for v in values_append])
        row = {
            "crop": crop_filename,
            "label": label,
            "var": var,
            #"frame": None,
            "time": times[0],
            "lat_mid": lat_mid,
            "lon_mid": lon_mid,
        }
        for stat in stats:
            row[stat] = compute_single_stat(all_values, stat, var)
        rows.append(row)

    elif mode == "per_frame":
        for frame_dict in values_append:
            row = {
                "crop": crop_filename,
                "label": label,
                "var": var,
                #"frame": frame_dict["frame"],
                "time": str(frame_dict["time"]),
                "lat_mid": lat_mid,
                "lon_mid": lon_mid,
            }
            for stat in stats:
                row[stat] = compute_single_stat(frame_dict["values"], stat, var)
            rows.append(row)

    return rows


def compute_single_stat(values, stat, var):
    """
    Compute a single statistic (percentile or categorical) based on the stat type.
    Parameters
    ----------
    values : np.ndarray
        Array of values to compute the statistic on.
    stat : str
        Statistic to compute (e.g., '50', '99' for percentiles or 'None' for categorical). 
    var : str
        Variable name (used for categorical processing).
    Returns
    -------
    float or dict
        The computed statistic value (float for percentiles or dict for categorical).
    """
    if stat == "None":
        return compute_categorical_values(values, var)
    elif stat == "sum[mm]" and var == "precipitation":
        # True cumulated precipitation (mm): sum(mm/h * delta_t_hours)
        # By default, assume 30min (0.5h) time step for precipitation (IMERG)
        delta_t = 0.5  # hours
        return np.nansum(values) * delta_t
    elif stat == "prec_fraction" and var == "precipitation":
        total_pixels = len(values)
        prec_pixels = np.sum(values > 0)
        return prec_pixels / total_pixels if total_pixels > 0 else 0
    else:
        return compute_percentile(values, int(stat))


def extract_geotime_metadata(coords, times):
    """
    Extract basic geospatial and temporal metadata for a crop.

    Parameters
    ----------
    coords : dict
        Dictionary with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'.
    times : list of np.datetime64
        List of timestamps for the crop.

    Returns
    -------
    dict
        Dictionary with 'month', 'hour', 'lat_mid', 'lon_mid'.
    """
    if not times:
        month_end = np.nan
        hour_end = np.nan
    else:
        month_end = int(np.datetime64(times[-1], 'M').astype(object).month)
        hour_end = int(np.datetime64(times[-1], 'h').astype(object).hour)

    lat_mid = (coords['lat_min'] + coords['lat_max']) / 2
    lon_mid = (coords['lon_min'] + coords['lon_max']) / 2

    return {
        "month": month_end,
        "hour": hour_end,
        "lat_mid": lat_mid,
        "lon_mid": lon_mid
    }


# === MAIN ===
def main(
    config_path: str = "config.yaml",
    var_config_path: str = "variables_metadata.yaml",
    benchmark_rows: int = None,
    use_parallel_override: bool = None,
    n_cores_override: int = None,
):
        # ...existing code...
    
    # load configs
    config = load_config(config_path)
    var_config = load_config(var_config_path)

    # Configure logging
    level = config["logging"]["log_level"]
    logging.basicConfig(
        level=level,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),                       # Print to console
            logging.FileHandler(f"{config['logging']['logs_path']}/processing_crops_stats_per_frame.log", mode="w")  # Save to file
        ]
    )
    logger = logging.getLogger(__name__)

    # read all the config parameters
    run_name = config["experiment"]["run_names"][0] # name of the run, used to read the right files and save the output
    #base_path = config["experiment"]["base_path"]
    epoch = config["experiment"]["epoch"]   # epoch number, used to read the right files and save the output
    crops_name = config["data"]["crops_name"] # name of the crops folder, used to read the right files and save the output
    data_base_path = config["data"]["data_base_path"] # base path where the data is stored, used to read the right files and save the output
    data_format = config["data"]["file_extension"] #§ file extension of the crop files, used to read the right files
    sampling_type = config["data"].get("sampling_type", "all")  # default to 'all'
    use_parallel = config["statistics"]["use_parallel"] # whether to use parallel processing with joblib
    n_cores = config["statistics"]["n_cores"] # number of cores to use for parallel processing (if enabled)

    # settings for benchmark mode (process only a subset of rows for faster execution during testing)
    if use_parallel_override is not None:
        use_parallel = use_parallel_override
    if n_cores_override is not None:
        n_cores = n_cores_override


    output_root = config["experiment"]["path_out"] # root path where the output will be saved
    filter_daytime = config["data"]["filter_daytime"] # whether to filter crops to keep only daytime samples (based on timestamp)
    filter_imerg_minutes = config["data"]["filter_imerg"] # whether to filter crops based on IMERG minutes
    n_frames = config["data"]["n_frames"] # number of frames to process (if per-frame mode is enabled)

    # read paths from process_run_GRL.yaml file
    features_config = config["features_preparation"]

    # crops path
    crops_path = features_config['crops_path']  # f"{data_base_path}/{crops_name}/{data_format}/"
    # define images path
    image_crops_path = features_config['images_path']      #f"{data_base_path}/{crops_name}/{data_format}/1/"

    # define output path for csv files
    output_path = features_config['output_path']  # f"{output_root}/{run_name}/epoch_{epoch}/{sampling_type}/"

    # Load crop list
    n_samples = get_num_crop(crops_path, extension=data_format)
    logging.info(f"Total number of crop samples found: {n_samples}")

    #list_image_crops = sorted(glob(image_crops_path + "*." + data_format))
    #n_samples = len(list_image_crops)
    #print("n samples:", n_samples)

    # define number of subsamples as n_samples if processing all, otherwise read it from config
    n_subsample = n_samples if sampling_type == "all" else config["data"]["n_subsample"]
    logger.info(f"Number of subsamples: {n_subsample} over total samples {n_samples}")

    logging.info(f"processing crops from {crops_path} with data format {data_format}")
    logging.info(f"output path for csv files: {output_path}")
    logging.info(f"filter daytime: {filter_daytime}, filter imerg minutes: {filter_imerg_minutes}, n_frames: {n_frames}")
    logging.info(f"use_parallel: {use_parallel}, n_cores: {n_cores}")
    logging.info(f"n_frames: {n_frames}, sampling_type: {sampling_type}, n_subsample: {n_subsample}")
    logging.info(f"run_name: {run_name}, epoch: {epoch}")
    logging.info(f"variables to process: {config['statistics']['spatial']['sel_vars']}, stats to compute: {config['statistics']['spatial']['percentiles']}, mode: {config['statistics']['spatial'].get('mode', 'aggregated')}")
    logging.info("--------------------------------------------")

    # Construct filter tags for filename to recognize if processeing is done between 6-16 or at IMERG res of 30mins
    filter_tags = []
    if filter_daytime:
        filter_tags.append("daytime")
    if filter_imerg_minutes:
        filter_tags.append("imergmin")
    filter_suffix = "_" + "_".join(filter_tags) if filter_tags else ""

    # read csv file containing features and nc filenames with labels, generated in the previous step of the pipeline (create_csv_features.py)
    # construct csv output filename
    backbone = config["experiment"]["backbone"]
    crop_resolution = config["experiment"]["crop_resolution"]
    n_input_layers = config["experiment"]["n_input_layers"]
    csv_filename = f'{run_name}-features_backbone_{backbone}_cropres_{crop_resolution}_inputvars_{n_input_layers}_epochs_{epoch}.csv'
    csv_filename = os.path.join(output_path, csv_filename)    

    # read feature csv file
    df_labels = pd.read_csv(csv_filename)
    if benchmark_rows is not None:
        df_labels = df_labels.head(benchmark_rows).copy()
        logger.info(f"Benchmark mode enabled: processing first {len(df_labels)} rows.")
    print(df_labels)
    
    #print(f"Loaded {len(df_labels)} rows from crop list.")
    logger.info(f"Loaded {len(df_labels)} rows from crop list.")

    crop_requests = [
        get_crop_metadata(row, config, crops_path)
        for _, row in df_labels.iterrows()
    ]
    day_blocks = build_day_blocks(
        crop_requests,
        config["statistics"]["spatial"]["sel_vars"],
        filter_imerg=filter_imerg_minutes,
    )
    logger.info(
        "Prepared %s shared day blocks for %s crops.",
        len(day_blocks),
        len(crop_requests),
    )

    # Run processing by shared (variable, day) blocks
    block_results = []
    total_blocks = len(day_blocks)
    if use_parallel and day_blocks:
        num_cores = min(n_cores, os.cpu_count() - 1)
        logger.info(f"Processing {total_blocks} day blocks in parallel mode (no per-block progress print)")
        from joblib import Parallel, delayed
        block_results = Parallel(n_jobs=num_cores)(
            delayed(process_day_block)(block, config, var_config, logger)
            for block in day_blocks
        )
    else:
        # Serial version with progress print every 50 blocks
        for i, block in enumerate(day_blocks, 1):
            block_results.append(process_day_block(block, config, var_config, logger))
            if i % 50 == 0 or i == total_blocks:
                print(f"Processed {i}/{total_blocks} day blocks...")
                logger.info(f"Processed {i}/{total_blocks} day blocks...")


    # Collect stats and vars info for filename
    stats_str = "-".join(map(str, config["statistics"]["spatial"]["percentiles"]))
    vars_str = "-".join(config["statistics"]["spatial"]["sel_vars"])

    # Flags for optional processing
    time_flag = "timedim" if config["statistics"].get("time_dimension", False) else None
    geo_flag = "coords-datetime" if config["statistics"].get("include_geotime", False) else None


    # Prepare output CSV paths for each variable
    sel_vars = config["statistics"]["spatial"]["sel_vars"]
    var_to_csv = {}
    for var in sel_vars:
        parts = [
            "crops_stats",
            f"var-{var}",
            f"stats-{stats_str}",
            f"frames-{n_frames}",
            time_flag,
            geo_flag,
            run_name,
            sampling_type,
            str(n_subsample) + filter_suffix,
        ]
        filename = "_".join([p for p in parts if p]) + ".csv"
        output_csv_path = os.path.join(output_path, filename)
        var_to_csv[var] = output_csv_path
        # Remove file if exists to avoid appending to old data
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)

    # Write header for each variable file (using a sample result)
    for var in sel_vars:
        # Find a sample block result for this variable
        sample_block_results = None
        for block_result in block_results:
            flat_results = combine_block_results(
                [block_result],
                crop_requests,
                config,
                var_config,
            )
            if flat_results and flat_results[0]["var"] == var:
                sample_block_results = flat_results
                break
        if sample_block_results:
            pd.DataFrame(sample_block_results).head(0).to_csv(var_to_csv[var], index=False)

    # Now process and append each block's results to the correct variable file
    for i, block_result in enumerate(block_results, 1):
        flat_results = combine_block_results(
            [block_result],
            crop_requests,
            config,
            var_config,
        )
        if flat_results:
            # Group by variable and write to the corresponding file
            df_block = pd.DataFrame(flat_results)
            for var in sel_vars:
                df_var = df_block[df_block["var"] == var]
                if not df_var.empty:
                    df_var.to_csv(var_to_csv[var], mode='a', header=False, index=False)
        if i % 50 == 0 or i == len(block_results):
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                print(f"[INFO] Memory usage after {i} blocks: {mem_mb:.2f} MB")
                logger.info(f"Memory usage after {i} blocks: {mem_mb:.2f} MB")
            except ImportError:
                print("[WARNING] psutil not installed, cannot print memory usage.")
                logger.warning("psutil not installed, cannot print memory usage.")

    logger.info(f"Crop stats for each variable saved successfully in {output_path}.")

    # # Save overall stats
    # continuous_stats = df_labels.groupby("label").agg(["mean", "std"])
    # continuous_stats.columns = [
    #     "_".join(col).strip() for col in continuous_stats.columns.values
    # ]
    # continuous_stats.reset_index(inplace=True)
    # continuous_stats.to_csv(
    #     f"{output_path}clusters_stats_{run_name}_{sampling_type}_{n_subsample}{filter_suffix}CA.csv",
    #     index=False,
    # )
    # logger.info("Cluster-level stats saved successfully.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-rows", type=int, default=None)
    parser.add_argument("--use-parallel", action="store_true")
    parser.add_argument("--n-cores", type=int, default=None)
    args = parser.parse_args()

    config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_GRL.yaml"
    var_config_path = "/home/claudia/codes/ML_postprocessing/configs/variables_metadata.yaml"
    main(
        config_path,
        var_config_path,
        benchmark_rows=args.benchmark_rows,
        use_parallel_override=True if args.use_parallel else None,
        n_cores_override=args.n_cores,
    )
    # nohup  855373