"""

Helpers for working with S3 buckets and extracting variable data.
Handles bucket selection, file path generation, and variable extraction from NetCDFs in S3.
"""


import numpy as np
import io
import xarray as xr

from utils.buckets.get_data_from_buckets import read_file
from utils.processing.coords_utils import extract_coordinates
from utils.processing.datetime_utils import  extract_datetime


def get_bucket_and_filename(var, year, month, day, BUCKETS):
    """
    Determine the appropriate S3 bucket and file path for the given variable.
    """
    if var in ['IR_108', 'WV_062']:
        bucket = BUCKETS['crop']
        filename = (
            f"/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/"
            f"{year:04d}/{month:02d}/merged_MSG_CMSAF_{year:04d}-{month:02d}-{day:02d}.nc"
        )
    elif var == 'precipitation':
        bucket = BUCKETS['imerg']
        filename = f"IMERG_daily_{year:04d}-{month:02d}-{day:02d}.nc"
    else:
        bucket = BUCKETS['cmsaf']
        filename = f"MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc"
    
    return bucket, filename


def extract_variable_values(row, var, s3, **BUCKETS):
    """
    Extract variable values from an S3 bucket given a row of metadata and a variable name.

    Parameters:
    - row: DataFrame row containing 'path' with datetime and spatial metadata.
    - var: Name of the variable to extract.
    - s3: Initialized S3 client.
    - **BUCKETS: Dictionary with keys 'cmsaf', 'imerg', and 'crop' for bucket names.

    Returns:
    - 1D NumPy array of extracted values, or empty list if error occurs.
    """
    crop_filename = row['path'].split('/')[-1]
    coords = extract_coordinates(crop_filename)
    datetime_info = extract_datetime(crop_filename)

    lat_min, lat_max = coords['lat_min'], coords['lat_max']
    lon_min, lon_max = coords['lon_min'], coords['lon_max']
    year, month, day, hour, minute = (
        datetime_info['year'], datetime_info['month'], datetime_info['day'],
        datetime_info['hour'], datetime_info['minute']
    )
    datetime_obj = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00")
    #print(f"Processing timestamp: {datetime_obj}")

    # Skip known invalid precipitation times
    if var == 'precipitation' and minute in {15, 45}:
        return []

    # Determine S3 path
    bucket, filename = get_bucket_and_filename(var, year, month, day, BUCKETS)

    try:
        file_obj = read_file(s3, filename, bucket)
        ds = xr.open_dataset(io.BytesIO(file_obj))[var]

        # Convert time format if needed
        if isinstance(ds.indexes["time"], xr.CFTimeIndex):
            ds["time"] = ds["time"].astype("datetime64[ns]")

        # Subset spatially and temporally
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max), time=datetime_obj)

        return ds.values.flatten()

    except Exception as e:
        print(f"Error extracting '{var}' for {row['path']}: {e}")
        return []


