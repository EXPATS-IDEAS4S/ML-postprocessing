import re
import numpy as np

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
