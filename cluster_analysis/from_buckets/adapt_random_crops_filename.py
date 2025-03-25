import os
import re
import shutil
import xarray as xr
import numpy as np

# Directories
DATA_DIR = "/data1/crops/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/1"
NC_DIR = "/data1/crops/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/cmsaf_2013-2014-2015-2016_expats/nc_clouds"

# Regular expression to match filenames in the TIFF directory
tif_pattern = re.compile(r"(\d{8})_(\d{2}):(\d{2})_EXPATS_(\d+)_([0-9]+K-[0-9]+K)_([a-zA-Z]+)_CMA\.tif")

# Regular expression to match filenames in the NetCDF directory
nc_pattern = re.compile(r"(\d{8})_(\d{2}):(\d{2})_EXPATS_(\d+)\.nc")

# Get all NetCDF files in the reference directory
nc_files = {
    f: os.path.join(NC_DIR, f)
    for f in sorted(os.listdir(NC_DIR))  # Sort filenames alphabetically
    if nc_pattern.match(f)
}

# Process each TIFF file in the directory
for file in sorted(os.listdir(DATA_DIR)):
    tif_match = tif_pattern.match(file)
    if not tif_match:
        print(f"Skipping: {file} (format not recognized)")
        continue

    # Extract parts from the TIFF filename
    date, hour, minute, last_number, bt_scale, color_scale = tif_match.groups()
    #print(date, hour, minute, last_number, bt_scale, color_scale)

    time_str = f"{hour}:{minute}"

    # Look for a matching NetCDF file
    matching_nc_file = None
    for nc_filename, nc_filepath in nc_files.items():
        #print(nc_filename)
        nc_match = nc_pattern.match(nc_filename)
        if nc_match:
            nc_date, nc_hour, nc_minute, nc_last_number = nc_match.groups()
            #print(nc_date, nc_hour, nc_minute, nc_last_number)
            if nc_date == date and nc_hour == hour and nc_minute == minute and nc_last_number == last_number:
                matching_nc_file = nc_filepath
                break

    if not matching_nc_file:
        print(f"No matching NetCDF file for: {file}")
        continue

    # Read lat/lon from the matched NetCDF file
    with xr.open_dataset(matching_nc_file, engine="h5netcdf") as ds:
        ul_lat = ds.lat.values.max() # Upper-left latitude (max latitude)
        ul_lon = ds.lon.values.min() # Upper-left longitude (min longitude)
        print(f"Upper-left coordinates: {ul_lat}, {ul_lon}")
   

    # Construct the new filename
    pixel_size = "0.04"
    x_pixels = "128"
    y_pixels = "128"
    cma_info = "CMA"
    ext = "tif"

    # Include only the first 2 decimants for lat and lon
    new_filename = f"{date}-{time_str}_{ul_lat:.2f}_{ul_lon:.2f}_{pixel_size}_{x_pixels}x{y_pixels}_{bt_scale}_{color_scale}_{cma_info}.{ext}"
    print(f"New filename: {new_filename}")
    
    # Rename the file
    old_path = os.path.join(DATA_DIR, file)
    new_path = os.path.join(DATA_DIR, new_filename)
    shutil.move(old_path, new_path)
    print(f"Renamed: {file} -> {new_filename}")

print("Filename conversion completed!")
