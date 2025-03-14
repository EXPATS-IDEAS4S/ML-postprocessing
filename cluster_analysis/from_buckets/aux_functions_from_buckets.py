import re
import numpy as np
from glob import glob
import pandas as pd
import os

def extract_coordinates(filename):
    """
    Extracts latitude and longitude boundaries from a given filename.

    Args:
    filename (str): The filename containing coordinates and resolution.

    Returns:
    dict: A dictionary containing lat_min, lat_max, lon_min, lon_max.
    """
    parts = filename.split("_")  # Split by underscore (_)

    if len(parts) < 7:
        raise ValueError(f"Filename format does not match expected pattern: {filename}")

    # Extract values based on the known format
    UL_lat = float(parts[1])  # Upper-left latitude
    UL_lon = float(parts[2])  # Upper-left longitude
    pixel_size = float(parts[3])  # Pixel size in degrees
    x_pixels = int(parts[4].split('x')[0])  # Number of pixels in X direction
    y_pixels = int(parts[4].split('x')[1])  # Number of pixels in Y direction

    # Compute latitude and longitude boundaries
    lat_max = UL_lat
    lat_min = UL_lat - (y_pixels * pixel_size)
    lon_min = UL_lon
    lon_max = UL_lon + (x_pixels * pixel_size)

    return {
        "lat_min": round(lat_min, 4),
        "lat_max": round(lat_max, 4),
        "lon_min": round(lon_min, 4),
        "lon_max": round(lon_max, 4)
    }


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

def compute_categorical_values(values, var):
    """
    Compute categorical values for the given variable.

    Args:
    values (np.array): The values of the variable.
    var (str): The variable name.

    Returns:
    np.array: The computed categorical values.
    """
    if var == 'cma':
        # Compute the fraction of cloud pixels (value == 1) over total pixels
        total_pixels = len(values)
        cloud_pixels = np.sum(values == 1)
        fraction_cloudy = cloud_pixels / total_pixels if total_pixels > 0 else 0
        values = fraction_cloudy
    elif var == 'cph':
        # Compute the fraction of liquid clouds (value == 1) over cloudy pixels (value 1 or 2)
        cloudy_pixels = np.sum((values == 1) | (values == 2))  # pixels with value 1 or 2
        ice_pixels = np.sum(values == 2)  # ice clouds (value 2)
        
        if cloudy_pixels > 0:
            fraction_ice = ice_pixels / cloudy_pixels
        else:
            fraction_ice = 0  # If no cloudy pixels, set the fraction to 0
        values = fraction_ice
    else:
        raise ValueError('Wrong variable names!')

    return values


def get_num_crop(run_name, extenion='tif'):
    image_crops_path = f'/data1/crops/{run_name}/1/'
    list_image_crops = sorted(glob(image_crops_path + '*.' + extenion))
    n_samples = len(list_image_crops)

    return n_samples


def find_crops_with_coordinates(df, lat, lon):
    """
    Given a dataframe with a 'path' column, find crop files that contain the specified latitude and longitude.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'path' with crop file paths.
        lat (float): Latitude to search for.
        lon (float): Longitude to search for.

    Returns:
        list: A list of filenames (last part of the path) that contain the given coordinates.
    """
    matching_crops = []

    for path in df['path']:
        filename = os.path.basename(path)

        # Extract information from filename using regex
        parts = filename.split("_")
        
        crop_UL_lat = float(parts[1])
        crop_UL_lon = float(parts[2])
        resolution = float(parts[3])
        width = int(parts[4].split('x')[0])
        height = int(parts[4].split('x')[1])

        # Calculate crop boundaries using the upper left corner
        lat_min = crop_UL_lat - (resolution * height)
        lat_max = crop_UL_lat
        lon_min = crop_UL_lon
        lon_max = crop_UL_lon + (resolution * width)

        # Check if the given lat/lon falls within the boundaries
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            matching_crops.append(filename)

    return matching_crops



def get_variable_info(var_name):
    """
    Retrieves information for a specific variable.

    Parameters:
    -----------
    var_name : str
        The specific variable to retrieve information for (e.g., 'cot').

    Returns:
    --------
    var_info : dict
        Dictionary containing the variable's long name, unit, logscale, and direction.
        Returns None if the variable is not found.
    """

    variables = {
        'cot':  {'long_name': 'cloud optical thickness', 'unit': None,   'logscale': False, 'direction': 'incr'},
        'cth':  {'long_name': 'cloud top height',       'unit': 'm',     'logscale': False, 'direction': 'incr'},
        'cma':  {'long_name': 'cloud mask',            'unit': None,    'logscale': False, 'direction': 'incr'},
        'cph':  {'long_name': 'cloud phase',           'unit': None,    'logscale': False, 'direction': 'incr'},
        'precipitation': {'long_name': 'precipitation', 'unit': 'mm/h',  'logscale': False, 'direction': 'incr'},
        'hour': {'long_name': 'hour',                  'unit': 'UTC',   'logscale': False, 'direction': 'incr'},
        'month': {'long_name': 'month',                'unit': None,    'logscale': False, 'direction': 'incr'},
        'lat_mid': {'long_name': 'latitude middle point', 'unit': '°N', 'logscale': False, 'direction': 'incr'},
        'lon_mid': {'long_name': 'longitude middle point', 'unit': '°E', 'logscale': False, 'direction': 'incr'}
    }

    return variables.get(var_name, None)  # Returns None if var_name is not found
