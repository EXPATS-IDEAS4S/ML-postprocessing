"""
Code to generate a a video of the ten closest crops to the centroid for each class grouped by lines.

Author: Claudia Acquistapace 
Date: 10 sept 2025

"""

import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import pdb
from array import array
from moviepy.editor import VideoFileClip
import imageio
import glob


# csv file with 2D+1 output
csv_file = '/data1/fig/dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1/epoch_800/closest/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000.csv'
output_dir = '/Users/claudia/Documents/data_ml_spacetime/figs'
path_img = '/Users/claudia/Documents/data_ml_spacetime/figs/'


# example of file name with date and t0 indicates the position of the frame in the video
# MSG_timeseries_2013-06-17_0615_crop2_t0_2013-06-17T06-15.png


def main():

    # list .png files in path_img    
    img_list = sorted(glob.glob(os.path.join(path_img, '*_closest_crops_table.png')))

    # create gif using imageio
    images = []
    for filename in img_list:
         images.append(imageio.imread(os.path.join(path_img, filename)))
    imageio.mimwrite(os.path.join(output_dir, 'closest_crops_table.gif'),
                     images)
    print("GIF created")

    


def read_csv_to_dataframe(csv_file):
    """Reads the CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_file)
    return df

if __name__ == "__main__":
    main()  