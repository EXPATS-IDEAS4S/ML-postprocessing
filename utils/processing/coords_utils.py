"""
coords_utils.py

Coordinate extraction and geospatial bounding box utilities.
Handles parsing of crop filenames and NetCDF metadata for latitude/longitude.
"""


import os
import xarray as xr

def find_latlon_boundaries_from_ds(ds_crops):
    """
    Extracts latitude and longitude boundaries from the crop dataset.

    Parameters:
    -----------
    ds_crops : xarray.Dataset
        The dataset containing latitude and longitude values.

    Returns:
    --------
    lat_min, lat_max, lon_min, lon_max : float
        The minimum and maximum latitude and longitude values defining the bounding box.
    """
    # Get the latitude and longitude values from the crop dataset
    lats_crops = ds_crops.lat.values
    lons_crops = ds_crops.lon.values
    
    # Define the bounding box for the crop dataset (min/max lat and lon)
    lat_min, lat_max = lats_crops.min(), lats_crops.max()
    lon_min, lon_max = lons_crops.min(), lons_crops.max()

    return lat_min, lat_max, lon_min, lon_max


def extract_coord_from_nc(filename, dir_path, engine='netcdf4'):
    """
    Extracts latitude and longitude boundaries from a NetCDF file.

    Args:
    filename (str): The name of the NetCDF file.
    dir_path (str): The directory path where the file is located.
    engine (str): The engine to use for reading the NetCDF file.

    Returns:
    dict: A dictionary containing lat_min, lat_max, lon_min, lon_max.
    """
    filepath = os.path.join(dir_path, filename)
    
    try:
        ds = xr.open_dataset(filepath, engine=engine)
        lat_min = ds['lat'].min().item()
        lat_max = ds['lat'].max().item()
        lon_min = ds['lon'].min().item()
        lon_max = ds['lon'].max().item()
        
        return {
            "lat_min": round(lat_min, 3),
            "lat_max": round(lat_max, 3),
            "lon_min": round(lon_min, 3),
            "lon_max": round(lon_max, 3)
        }
    
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None
                          

def extract_coordinates(filename):
    """
    Extracts latitude and longitude boundaries from a given filename.

    Args:
    filename (str): The filename containing coordinates and resolution.

    Returns:
    dict: A dictionary containing lat_min, lat_max, lon_min, lon_max.
    """
    print(filename)
    exit()
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


def find_crops_in_range(df, lat_min, lat_max, lon_min, lon_max):
    """
    Given a DataFrame with a 'path' column, find crop files that overlap with a specified latitude and longitude range.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'path' with crop file paths.
        lat_min (float): Minimum latitude of the search range.
        lat_max (float): Maximum latitude of the search range.
        lon_min (float): Minimum longitude of the search range.
        lon_max (float): Maximum longitude of the search range.

    Returns:
        list: A list of filenames (last part of the path) that intersect with the given coordinate range.
    """
    matching_crops = []

    for path in df['path']:
        filename = os.path.basename(path)

        # Extract information from filename
        parts = filename.split("_")
        crop_UL_lat = float(parts[1])  # Upper-left latitude
        crop_UL_lon = float(parts[2])  # Upper-left longitude
        resolution = float(parts[3])   # Resolution in degrees
        width = int(parts[4].split('x')[0])  # Width in pixels
        height = int(parts[4].split('x')[1]) # Height in pixels

        # Calculate crop boundaries using the upper-left corner
        crop_lat_min = crop_UL_lat - (resolution * height)
        crop_lat_max = crop_UL_lat
        crop_lon_min = crop_UL_lon
        crop_lon_max = crop_UL_lon + (resolution * width)
        #print(crop_lat_min, crop_lat_max, crop_lon_min, crop_lon_max)

        # Check if the crop intersects with the given range
        if not (crop_lat_max < lat_min or crop_lat_min > lat_max or crop_lon_max < lon_min or crop_lon_min > lon_max):
            matching_crops.append(filename)

    return matching_crops