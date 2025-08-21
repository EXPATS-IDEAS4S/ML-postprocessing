import re
import numpy as np
from glob import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import io

from get_data_from_buckets import read_file


def plot_cartopy_map(output_path, latmin, lonmin, latmax, lonmax, n_divs=5):
    """
    Plots a Cartopy map with country borders and a grid of vertical and horizontal lines.

    Parameters:
    -----------
    latmin : float
        Minimum latitude of the map.
    lonmin : float
        Minimum longitude of the map.
    latmax : float
        Maximum latitude of the map.
    lonmax : float
        Maximum longitude of the map.
    n_divs : int, optional (default=5)
        Number of vertical and horizontal grid lines.
    """

    # Define the figure and axis with a PlateCarree projection
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set map extent
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.BORDERS, linewidth=1)  # Country borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)  # Coastlines
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)  # Land color
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)  # Ocean color

    # Generate grid lines
    lon_ticks = np.linspace(lonmin, lonmax, n_divs+1)
    lat_ticks = np.linspace(latmin, latmax, n_divs+1)

    # Plot vertical grid lines (longitude)
    for lon in lon_ticks:
        ax.plot([lon, lon], [latmin, latmax], transform=ccrs.PlateCarree(), color='black', linestyle='--', alpha=0.7)

    # Plot horizontal grid lines (latitude)
    for lat in lat_ticks:
        ax.plot([lonmin, lonmax], [lat, lat], transform=ccrs.PlateCarree(), color='black', linestyle='--', alpha=0.7)

    # Add custom tick labels at the bottom (longitude)
    ax.set_xticks(lon_ticks)  
    ax.set_xticklabels([f"{lon:.1f}°E" if lon >= 0 else f"{-lon:.1f}°W" for lon in lon_ticks], fontsize=10)
    ax.xaxis.set_ticks_position('bottom')  # Only at the bottom

    # Add custom tick labels on the left (latitude)
    ax.set_yticks(lat_ticks)  
    ax.set_yticklabels([f"{lat:.1f}°N" if lat >= 0 else f"{-lat:.1f}°S" for lat in lat_ticks], fontsize=10)
    ax.yaxis.set_ticks_position('left')  # Only on the left

    # Hide top and right tick labels
    ax.tick_params(top=False, right=False)

    # Save the figure
    plt.savefig(f'{output_path}map_divided_{n_divs}.png', bbox_inches='tight', dpi=300)


def extract_coord_from_nc(filename, dir_path):
    """
    Extracts latitude and longitude boundaries from a NetCDF file.

    Args:
    filename (str): The name of the NetCDF file.
    dir_path (str): The directory path where the file is located.

    Returns:
    dict: A dictionary containing lat_min, lat_max, lon_min, lon_max.
    """
    filepath = os.path.join(dir_path, filename)
    
    try:
        ds = xr.open_dataset(filepath, engine='h5netcdf')
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
        'cot':  {'long_name': 'cloud optical thickness', 'unit': None,   'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},   
        'cth':  {'long_name': 'cloud top height',       'unit': 'm',     'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'cma':  {'long_name': 'cloud cover',            'unit': None,    'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'cph':  {'long_name': 'ice ratio',           'unit': None,    'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'precipitation': {'long_name': 'precipitation', 'unit': 'mm/h',  'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'hour': {'long_name': 'hour',                  'unit': 'UTC',   'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'month': {'long_name': 'month',                'unit': None,    'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'lat_mid': {'long_name': 'latitude middle point', 'unit': '°N', 'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'lon_mid': {'long_name': 'longitude middle point', 'unit': '°E', 'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100}
    }

    return variables.get(var_name, None)  # Returns None if var_name is not found





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



def filter_cma_values(values, cma_values, var_name, cma_filter=True):
    """
    Filter input values based on cloud mask (CMA) values.

    Parameters:
        values (array-like): Array of values to filter.
        cma_values (array-like): Corresponding cloud mask values (1 = cloudy).
        var_name (str): Variable name, e.g., 'precipitation'.
        cma_filter (bool): Whether to apply the cloud mask filter.

    Returns:
        np.ndarray or list: Filtered values or [0] if no valid values remain.
    """
    if not cma_filter:
        return values

    if var_name == 'precipitation':
        if len(values) == 0:
            return [np.nan]
        else:
            filtered = values[values > 0.1]
    else:
        if len(values) != len(cma_values):
            raise ValueError("Length mismatch: 'values' and 'cma_values' must be the same.")
        filtered = values[cma_values == 1]

    return filtered if len(filtered) > 0 else [0]