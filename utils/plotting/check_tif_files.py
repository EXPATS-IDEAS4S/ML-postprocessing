import rasterio
from rasterio.plot import show
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.io.img_tiles as cimgt
from glob import glob
import numpy as np


import rasterio

def inspect_tif_file(filename):
    with rasterio.open(filename) as src:
        # Print metadata
        print("Metadata:")
        for key, value in src.meta.items():
            print(f"{key}: {value}")

        # Print band count and details
        print("\nBand Count:", src.count)
        for i in range(1, src.count + 1):
            print(f"\nBand {i}:")
            print("Description:", src.descriptions[i-1])
            print("Data Type:", src.dtypes[i-1])

        # Print CRS (Coordinate Reference System)
        print("\nCRS (Coordinate Reference System):")
        print(src.crs)

        # Print bounds
        print("\nBounds:")
        print(src.bounds)

        # Print transform (affine transformation matrix)
        print("\nTransform (Affine Transformation Matrix):")
        print(src.transform)

        # Print other tags
        print("\nTags:")
        for key, value in src.tags().items():
            print(f"{key}: {value}")



def get_data(filename):
    with rasterio.open(filename) as src:
        # Check if the file is an RGB image
        if src.count == 3:
            # Read the RGB bands
            print('RGB')
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)
            data = np.dstack((red, green, blue))
        else:
            # Read the first band
            data = src.read(1)
        transform = src.transform
        bbox = src.bounds
        crs = src.crs
    return data, bbox, crs


# File path to the .tif file
tif_file_path = '/home/Daniele/data/all_128_germany/1/'
tif_filepattern = '*_germany_*.tif'

all_files = sorted(glob(tif_file_path+tif_filepattern))
#print(all_files)

for filename in all_files:
    #print(get_data(filename))
    inspect_tif_file(filename)
    exit()





