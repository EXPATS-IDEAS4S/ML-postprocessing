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

#TODO:  
Check why the labels are not included in the output csv
CHeck frame index in the case of day change! Now it does restart from zero!!!!
"""
import os, sys, io
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


#sys.path.append(os.path.abspath("/home/Daniele/codes/VISSL_postprocessing"))
sys.path.append(os.path.abspath("//home/Daniele/codes/VISSL_postprocessing/"))

from utils.processing.stats_utils import compute_percentile
from utils.configs import load_config
from utils.buckets.credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
from utils.processing.coords_utils import extract_coordinates,  extract_coord_from_nc
from utils.processing.datetime_utils import extract_datetime,  extract_datetime_from_nc
from utils.processing.stats_utils import compute_categorical_values, filter_cma_values, apply_closing
from utils.buckets.get_data_from_buckets import read_file, Initialize_s3_client
from utils.processing.features_utils import get_num_crop

import logging

def get_time_windows(times, var=None):
    """
    Group a list of crop timestamps by day, with optional IMERG filtering.

    Parameters
    ----------
    times : list of np.datetime64
        List of timestamps (e.g., from crop file metadata).
    filter_imerg : bool, optional
        If True, apply IMERG-compatible filtering.
        - For precipitation: keep only minutes 00 and 30.
        - For other variables: add an extra +15 min timestamp.
    var : str, optional
        Variable name (used for IMERG filtering).

    Returns
    -------
    dict
        Mapping {day (str YYYY-MM-DD) -> list of timestamps}.
    """
    # Ensure chronological order
    times = sorted(times)

    # Optional IMERG filtering
    if var == "precipitation":
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


def process_row(row, config, var_config, image_crops_path, logger):
    """
    Process a single crop row: load data, mask with CMA, and compute statistics.
    Returns a list of flat dicts (rows), ready for CSV export.
    """
    crop_filename = os.path.basename(row["path"])
    crop_path = row["path"]
    print("Processing crop:", crop_filename)
    #check if vector type column exists
    if 'vector_type' in row:
        print("Vector type:", row["vector_type"])
        vector_type = row["vector_type"]
    else:
        vector_type = None
    data_format = config["data"]["file_extension"]
    nc_engine = config["data"]["nc_engine"]

    if data_format == "nc":
        coords = extract_coord_from_nc(crop_path, nc_engine)
        times = extract_datetime_from_nc(crop_path, nc_engine)
    else:
        coords = extract_coordinates(crop_filename)
        datetime_info = extract_datetime(crop_filename)
        year, month, day, hour, minute = (
            datetime_info[k] for k in ["year", "month", "day", "hour", "minute"]
        )
        times = np.datetime64(
            f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"
        )
    #print('crop times:', times)
    #print('crop coords:', coords)
    lat_min, lat_max = coords["lat_min"], coords["lat_max"]
    lon_min, lon_max = coords["lon_min"], coords["lon_max"]

    all_rows = []

    for var in config["statistics"]["spatial"]["sel_vars"]:
        var_rows = process_variable(
            var, times, lat_min, lat_max, lon_min, lon_max,
            config, var_config, logger, crop_filename, coords, vector_type
        )
        all_rows.extend(var_rows)
    
    return all_rows


def process_variable(var, times, lat_min, lat_max, lon_min, lon_max,
                     config, var_config, logger, crop_filename, coords, vector_type):
    """Process one variable across time windows."""
    var_meta = var_config['variables'][var]
    stats = ['None'] if var_meta['categorical'] else config["statistics"]["spatial"]["percentiles"]
    mode = config["statistics"]["spatial"].get("mode", "aggregated")

    values_append = []
    for day_key, times_for_day in get_time_windows(times, var=var).items():
        print('processing day:', day_key, 'var:', var)
        values = load_and_mask_data(day_key, var, var_meta, lat_min, lat_max, lon_min, lon_max, times_for_day, logger, mode)
        if mode == "aggregated":
            values_append.append(values)
        elif mode == "per_frame":
            values_append.extend(values)  # flatten all frames across days

    return compute_statistics(values_append, stats, var, mode, times, crop_filename, coords, vector_type)



def load_and_mask_data(day_key, var, var_meta, lat_min, lat_max, lon_min, lon_max, times, logger, mode):
    """
    Load daily dataset from S3, subset by coords/times, apply CMA mask.

    Returns
    -------
    - if mode == "aggregated": np.ndarray of values
    - if mode == "per_frame": list of dicts with keys {"time", "frame", "values"}
    """
    try:
        y, m, d = map(int, day_key.split("-"))
        if var == "euclid_msg_grid":
            bucket_filename =(f"{y:04d}/{m:02d}/" \
                f"{var_meta['bucket_filename_prefix']}{y:04d}{m:02d}{d:02d}" \
                f"{var_meta['bucket_filename_suffix']}"
            )
        else:
            bucket_filename = (
                f"{var_meta['bucket_filename_prefix']}{y:04d}-{m:02d}-{d:02d}"
                f"{var_meta['bucket_filename_suffix']}"
            )
        #print('bucket filename:', bucket_filename)
        s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

        # Load variable and CMA
        my_obj = read_file(s3, bucket_filename, var_meta['bucket_name'])
        if my_obj is None:
            logger.warning(f"File not found: {bucket_filename}")
            return np.array([np.nan]) if mode == "aggregated" else []

        cma_filename = f"MCP_{y:04d}-{m:02d}-{d:02d}_regrid.nc"
        my_obj_cma = read_file(s3, cma_filename, "expats-cmsaf-cloud")
        if my_obj_cma is None:
            logger.warning(f"CMA file not found for day: {day_key}")
            return np.array([np.nan]) if mode == "aggregated" else []
        #print(xr.open_dataset(io.BytesIO(my_obj)))
        # Open datasets
        ds_day = xr.open_dataset(io.BytesIO(my_obj))[var].sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )

        ds_day_cma = xr.open_dataset(io.BytesIO(my_obj_cma))["cma"].sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )

        # Normalize time index if needed
        if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
            ds_day["time"] = ds_day["time"].astype("datetime64[ns]")
            ds_day_cma["time"] = ds_day_cma["time"].astype("datetime64[ns]")

        if var=="euclid_msg_grid":
            #besdies the timestamp also take +5 min and +10 min
            ds_subset = ds_day.sel(time=times + np.timedelta64(0,'m')).combine_first(
                ds_day.sel(time=times + np.timedelta64(5,'m'))
            ).combine_first(
                ds_day.sel(time=times + np.timedelta64(10,'m'))
            )
        else:
            ds_subset = ds_day.sel(time=times)
        
        ds_subset_cma = ds_day_cma.sel(time=times)

        #apply closing
        ds_subset_cma = apply_closing(ds_subset_cma)
        if var == "cma":
            ds_subset = apply_closing(ds_subset)
        
        if mode == "aggregated":
            values = ds_subset.values.flatten()
            values_cma = ds_subset_cma.values.flatten()
            return filter_cma_values(values, values_cma, var)

        elif mode == "per_frame":
            per_frame_list = []
            for t_idx, t in enumerate(ds_subset.time.values):
                frame_values = ds_subset.sel(time=t).values.flatten()
                frame_values_cma = ds_subset_cma.sel(time=t).values.flatten()
                frame_values = filter_cma_values(frame_values, frame_values_cma, var)
                print('frame values:', frame_values)

                per_frame_list.append({
                    "time": np.datetime64(t).astype("datetime64[s]").item(),
                    #"frame": t_idx, TODO error here, when change of day, frame restart from zero!!
                    "values": frame_values,
                })
            return per_frame_list

    except Exception:
        logger.error(f"Error processing {day_key}, var {var}", exc_info=True)
        return np.array([np.nan]) if mode == "aggregated" else []



def compute_statistics(values_append, stats, var, mode, times, crop_filename, coords, vector_type):
    """
    Compute stats either aggregated across frames or per-frame.
    Returns a list of flat dicts (rows).
    """
    rows = []

    lat_mid = (coords["lat_min"] + coords["lat_max"]) / 2
    lon_mid = (coords["lon_min"] + coords["lon_max"]) / 2

    if not values_append:
        if mode == "aggregated":
            row = {
                "crop": crop_filename,
                "var": var,
                #"frame": None,
                "time": times[0],
                "lat_mid": lat_mid,
                "lon_mid": lon_mid,
                "vector_type": vector_type
            }
            for stat in stats:
                row[stat] = np.nan
            rows.append(row)
        elif mode == "per_frame":
            for frame_idx, t in enumerate(times):
                row = {
                    "crop": crop_filename,
                    "var": var,
                    #"frame": frame_idx,
                    "time": str(np.datetime64(t).astype("datetime64[s]").item()),
                    "lat_mid": lat_mid,
                    "lon_mid": lon_mid,
                    "vector_type": vector_type
                }
                for stat in stats:
                    row[stat] = np.nan
                rows.append(row)
        return rows

    if mode == "aggregated":
        all_values = np.concatenate([np.atleast_1d(v) for v in values_append])
        row = {
            "crop": crop_filename,
            "var": var,
            #"frame": None,
            "time": times[0],
            "lat_mid": lat_mid,
            "lon_mid": lon_mid,
            "vector_type": vector_type
        }
        for stat in stats:
            row[stat] = compute_single_stat(all_values, stat, var)
            #print(row[stat])
        rows.append(row)

    elif mode == "per_frame":
        for frame_dict in values_append:
            row = {
                "crop": crop_filename,
                "var": var,
                #"frame": frame_dict["frame"],
                "time": str(frame_dict["time"]),
                "lat_mid": lat_mid,
                "lon_mid": lon_mid,
                "vector_type": vector_type
            }
            for stat in stats:
                row[stat] = compute_single_stat(frame_dict["values"], stat, var)
                #print(row[stat])
            rows.append(row)

    return rows


def compute_single_stat(values, stat, var):
    if stat == "None":
        return compute_categorical_values(values, var)
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
def main(config_path: str = "config.yaml", var_config_path: str = "variables_metadata.yaml"):
    
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

    run_name = config["experiment"]["run_names"][0]
    #base_path = config["experiment"]["base_path"]
    epoch = config["experiment"]["epoch"]
    crops_name = config["data"]["crops_name"]
    data_base_path = config["data"]["data_base_path"]
    data_format = config["data"]["file_extension"]
    sampling_type = config["data"].get("sampling_type", "all")  # default to 'all'
    use_parallel = config["statistics"]["use_parallel"]
    n_cores = config["statistics"]["n_cores"]
    output_root = config["experiment"]["path_out"]
    filter_daytime = config["data"]["filter_daytime"]
    filter_imerg_minutes = config["data"]["filter_imerg"]
    n_frames = config["data"]["n_frames"]

    test_data_set = config["experiment"].get("test_set", False)

    # Build paths
    image_crops_path = f"{data_base_path}/{crops_name}/{data_format}/1/"
    
    # Load crop list
    n_samples = get_num_crop(image_crops_path, extension=data_format)
    #list_image_crops = sorted(glob(image_crops_path + "*." + data_format))
    #n_samples = len(list_image_crops)
    #print("n samples:", n_samples)

    n_subsample = n_samples if sampling_type == "all" else config["data"]["n_subsample"]
    #print(n_subsample)
    logger.info(f"Number of subsamples: {n_subsample} over total samples {n_samples}")

    # Construct filter tags for filename
    filter_tags = []
    if filter_daytime:
        filter_tags.append("daytime")
    if filter_imerg_minutes:
        filter_tags.append("imergmin")
    filter_suffix = "_" + "_".join(filter_tags) if filter_tags else ""

    if test_data_set:
        output_path = f"{output_root}/{run_name}/epoch_{epoch}/test_traj/"
        csv_filename = f"{output_path}features_train_test_{run_name}_2nd_labels.csv"
        df_labels = pd.read_csv(csv_filename)
        #print unique values of vector_type column
        #print(df_labels)
        #get only the rows with test set (remove vector_type column == TRAIN)
        df_labels = df_labels[df_labels["vector_type"] != "TRAIN"]
    else:
        output_path = f"{output_root}/{run_name}/epoch_{epoch}/{sampling_type}/"
        csv_filename = f"{output_path}crop_list_{run_name}_{sampling_type}_{n_subsample}{filter_suffix}.csv"
        df_labels = pd.read_csv(csv_filename)
    
    #print(f"Loaded {len(df_labels)} rows from crop list.")
    logger.info(f"Loaded {len(df_labels)} rows from crop list.")

    # Run processing
    if use_parallel:
        num_cores = min(n_cores, os.cpu_count() - 1) #check cpu count
        results = Parallel(n_jobs=num_cores)(
            delayed(process_row)(row, config, var_config, image_crops_path, logger)
            for _, row in df_labels.iterrows()
        )
    else:
        results = [process_row(row, config, var_config, image_crops_path, logger) for _, row in df_labels.iterrows()]

    # Flatten list of lists into one list of dicts
    flat_results = [r for crop_rows in results for r in crop_rows]

    # Build long-format DataFrame
    df_results = pd.DataFrame(flat_results)

    # Collect stats and vars info for filename
    stats_str = "-".join(map(str, config["statistics"]["spatial"]["percentiles"]))
    vars_str = "-".join(config["statistics"]["spatial"]["sel_vars"])

    # Flags for optional processing
    time_flag = "timedim" if config["statistics"].get("time_dimension", False) else None
    geo_flag = "coords-datetime" if config["statistics"].get("include_geotime", False) else None

    # Assemble filename parts (skip None/empty)
    parts = [
        "crops_stats",
        f"vars-{vars_str}",
        f"stats-{stats_str}",
        f"frames-{n_frames}",
        time_flag,
        geo_flag,
        run_name,
        sampling_type,
        str(n_subsample) + filter_suffix,
    ]
    filename = "_".join([p for p in parts if p]) + ".csv"

    # Save
    df_results.to_csv(os.path.join(output_path, filename), index=False)
    logger.info(f"Crop stats for {filename} saved successfully in {output_path}.")

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
    config_path = "/home/Daniele/codes/VISSL_postprocessing/configs/process_run_config.yaml"
    #config_path = "/home/claudia/codes/ML_postprocessing/configs/process_run_config.yaml"
    var_config_path = "/home/Daniele/codes/VISSL_postprocessing/configs/variables_metadata.yaml"
    #var_config_path = "/home/claudia/codes/ML_postprocessing/configs/variables_metadata.yaml"
    main(config_path, var_config_path)


#nohup stat test: 79853
#nohup stat train: 80305