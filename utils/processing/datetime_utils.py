"""
datetime_utils.py

Datetime parsing helpers.
Provides functions to extract date and time information from NetCDF files and filenames.
"""


import pandas as pd
import os 
import xarray as xr



def get_time_from_ds(ds):
    """
    Extracts the year, month, day, and hour from the time variable in the dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the time variable.

    Returns:
    --------
    year : int
        The year extracted from the dataset's time.
    month : int
        The month extracted from the dataset's time.
    day : int
        The day extracted from the dataset's time.
    hour : int
        The hour extracted from the dataset's time.
    """
    time_crop = ds.time.values

    # Extract year, month, and hour from the crop time
    time_crop_dt = pd.to_datetime(time_crop)
    year = time_crop_dt.year
    month = time_crop_dt.month
    day = time_crop_dt.day
    hour = time_crop_dt.hour

    return year, month, day, hour





def extract_datetime_from_nc(filename, dir_path):
    """
    Extracts year, month, day, hour, and minute from a NetCDF filename.

    Args:
    filename (str): The name of the NetCDF file.
    dir_path (str): The directory path where the file is located.

    Returns:
    dict: A dictionary containing 'year', 'month', 'day', 'hour', and 'minute'.
    """
    filepath = os.path.join(dir_path, filename)
    
    try:
        ds = xr.open_dataset(filepath, engine='h5netcdf')
        time_var = ds['time']
        
        # Assuming time is in datetime64 format
        time_value = time_var.values[0]  # Get the first time value
        dt = np.datetime64(time_value)

        return {
            "year": dt.astype('datetime64[Y]').astype(int) + 1970,
            "month": dt.astype('datetime64[M]').astype(int) % 12 + 1,
            "day": dt.astype('datetime64[D]').astype(int) % 31 + 1,
            "hour": dt.astype('datetime64[h]').astype(int) % 24,
            "minute": dt.astype('datetime64[m]').astype(int) % 60
        }
    
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None
    

def extract_datetime(filename):
    """
    Extracts year, month, day, and hour from a given filename.

    Args:
    filename (str): The filename containing the date and time.

    Returns:
    dict: A dictionary containing 'year', 'month', 'day', 'hour', and 'minute'.
    """
    parts = filename.split("_")  # Split by underscore (_)

    if len(parts) < 2:
        raise ValueError(f"Filename format does not match expected pattern: {filename}")

    # Extract date and time
    date_part = parts[0].split('-')[0]  # YYYYMMDD
    time_part = parts[0].split('-')[1]  # HH:MM

    # Parse year, month, day
    year = int(date_part[:4])
    month = int(date_part[4:6])
    day = int(date_part[6:8])

    # Parse hour and minute
    time_subparts = time_part.split(":")
    hour = int(time_subparts[0])
    minute = int(time_subparts[1])

    return {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute
    }