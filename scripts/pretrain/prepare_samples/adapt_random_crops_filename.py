"""
Crop File Renaming Script with NetCDF Geolocation

This script renames crop image files (TIFF, PNG, JPG, etc.) by including precise
geolocation information extracted from corresponding NetCDF files. The new filenames
contain:

- Date (YYYYMMDD)
- Time (HH:MM)
- Upper-left latitude and longitude (from NetCDF)
- Pixel size
- Image dimensions (XxY pixels)
- Brightness temperature scale (BT scale)
- Color scale
- CMA info

It automatically searches for matching NetCDF files and can handle multiple image
file extensions. Filenames that do not match the expected format are skipped.
"""

import os
import re
import shutil
import xarray as xr
import numpy as np

# =======================
# === CONFIGURATION =====
# =======================

CROP_DIR = "/data1/crops/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/1"
NC_DIR = "/data1/crops/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/cmsaf_2013-2014-2015-2016_expats/nc_clouds"
EXTENSIONS = ['tif', 'png', 'jpg']          # Image file extensions to process
PIXEL_SIZE = 0.04
X_PIXELS = 128
Y_PIXELS = 128
CMA_INFO = "CMA"

# Filename regex patterns
TIF_PATTERN = r"(\d{8})_(\d{2}):(\d{2})_EXPATS_(\d+)_([0-9]+K-[0-9]+K)_([a-zA-Z]+)_CMA\.{ext}"
NC_PATTERN = r"(\d{8})_(\d{2}):(\d{2})_EXPATS_(\d+)\.nc"


# =======================
# === HELPER FUNCTIONS ==
# =======================

def list_nc_files(nc_dir, nc_pattern):
    """Return a dictionary of NetCDF filenames and paths matching the regex pattern."""
    nc_regex = re.compile(nc_pattern)
    return {f: os.path.join(nc_dir, f) for f in sorted(os.listdir(nc_dir)) if nc_regex.match(f)}


def extract_ul_coordinates_from_nc(nc_file_path):
    """
    Extract upper-left latitude and longitude from a NetCDF dataset.

    Parameters
    ----------
    nc_file_path : str
        Full path to the NetCDF file.

    Returns
    -------
    tuple
        (upper_left_latitude, upper_left_longitude)
    """
    with xr.open_dataset(nc_file_path, engine="h5netcdf") as ds:
        ul_lat = ds.lat.values.max()
        ul_lon = ds.lon.values.min()
    return ul_lat, ul_lon


def construct_new_filename(date, time_str, ul_lat, ul_lon, pixel_size, x_pix, y_pix,
                           bt_scale, color_scale, cma_info, ext):
    """Construct the new filename using all relevant components."""
    return f"{date}-{time_str}_{ul_lat:.2f}_{ul_lon:.2f}_{pixel_size}_{x_pix}x{y_pix}_{bt_scale}_{color_scale}_{cma_info}.{ext}"


def rename_crop_files(crop_dir, nc_dir, extensions, tif_pattern, nc_pattern,
                      pixel_size, x_pixels, y_pixels, cma_info):
    """Main function to rename crop files based on NetCDF geolocation."""
    nc_files = list_nc_files(nc_dir, nc_pattern)
    nc_regex = re.compile(nc_pattern)

    for ext in extensions:
        file_regex = re.compile(tif_pattern.format(ext=ext))
        for crop_file in sorted(os.listdir(crop_dir)):
            if not crop_file.lower().endswith(ext):
                continue

            match = file_regex.match(crop_file)
            if not match:
                print(f"Skipping: {crop_file} (format not recognized)")
                continue

            date, hour, minute, last_number, bt_scale, color_scale = match.groups()
            time_str = f"{hour}:{minute}"

            # Find matching NetCDF
            matching_nc_file = None
            for nc_filename, nc_filepath in nc_files.items():
                nc_match = nc_regex.match(nc_filename)
                if nc_match:
                    nc_date, nc_hour, nc_minute, nc_last_number = nc_match.groups()
                    if nc_date == date and nc_hour == hour and nc_minute == minute and nc_last_number == last_number:
                        matching_nc_file = nc_filepath
                        break

            if not matching_nc_file:
                print(f"No matching NetCDF file for: {crop_file}")
                continue

            ul_lat, ul_lon = extract_ul_coordinates_from_nc(matching_nc_file)
            new_filename = construct_new_filename(
                date, time_str, ul_lat, ul_lon, pixel_size, x_pixels, y_pixels,
                bt_scale, color_scale, cma_info, ext
            )

            old_path = os.path.join(crop_dir, crop_file)
            new_path = os.path.join(crop_dir, new_filename)
            shutil.move(old_path, new_path)
            print(f"Renamed: {crop_file} -> {new_filename}")

    print("Filename conversion completed!")


# =======================
# === SCRIPT EXECUTION ==
# =======================

if __name__ == "__main__":
    rename_crop_files(
        CROP_DIR, NC_DIR, EXTENSIONS, TIF_PATTERN, NC_PATTERN,
        PIXEL_SIZE, X_PIXELS, Y_PIXELS, CMA_INFO
    )
