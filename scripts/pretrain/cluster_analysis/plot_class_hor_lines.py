"""
Code to generate a figure of the ten closest crops to the centroid for each class.
Author: Claudia Acquistapace and Daniele Corradini
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
import cv2 
import sys
import glob
import matplotlib.patches as mpatches
sys.path.append('/home/claudia/codes/ML_postprocessing')
from scripts.pretrain.embedding_visualization.plot_embedding_utils import plot_embedding_crops_table

from PIL import Image
# csv file with 2D+1 output
csv_file = '/data1/fig/dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1/epoch_800/closest/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000_debug.csv'
output_dir = '/home/claudia/figs/'
path_img = '/data1/crops/clips_ir108_100x100_8frames_2013-2020/img/IR_108_cm/1/'

# example of file name with date and t0 indicates the position of the frame in the video
# MSG_timeseries_2013-06-17_0615_crop2_t0_2013-06-17T06-15.png


def main():

    # read csv file
    df = read_csv_to_dataframe(csv_file)
    print("Column titles:", df.columns.tolist())

    # select one variable to be sure to select 8 frames
    df8 = df[df['var'] == 'cth']

    # loop on the frames to produce figures to be joined in a video
    frames = df8['frame'].unique()
    print(frames)

    # loop on the frames
    for frame in frames:

        # select by frame
        df8_frame = df8[df8['frame'] == frame]

        print('Frame:', frame)

        # select the 10 farthest crops to the centroid for each class, read their label and frame
        crops_plot = df8_frame.loc[df8_frame.groupby('label')['distance'].nlargest(10).index.get_level_values(1)]

        # print first 20 rows
        #print(crops_plot.head(20))

        # create a dataframe with image paths, labels, and distances.
        img_paths = []
        labels = []
        distances = []

        # loop on the rows of crops_plot
        for i, row in crops_plot.iterrows():
            crop = row['crop'].split(".")[0].split("_")[-1]  # get the crop number
            print("crop:", crop)
            label = row['label']
            distance = row['distance']
            time_stamp = row['time']
            print("time_stamp:", time_stamp)
            date, hour = time_stamp.split(' ')
            hh = hour[0:2]  # keep only HHMM
            mins = hour[3:5]
            print("date:", date, "hour:", hh, "mins:", mins)

            # construct the file name
            # search filename pattern in the folder
            file_pattern = f'*{crop}_t{frame}_{date}T{hh}-{mins}.png'
            print("file_pattern:", file_pattern)
            file_path = glob.glob(os.path.join(path_img, file_pattern))
            file_path = file_path[0] if file_path else None

            print("file_path:", file_path)

            # append strings to lists
            img_paths.append(file_path)
            labels.append(label)
            distances.append(distance)

        # create dataframe
        df_imgs = pd.DataFrame({'path': img_paths, 'label': labels, 'distance': distances})

        # sort df_imgs by label and distance largest to smallest (distance to centroid)
        df_imgs = df_imgs.sort_values(by=['label', 'distance'], ascending=[True, False])
        df_imgs = df_imgs.reset_index(drop=True)

        # print first 20 rows
        print(df_imgs.head(20))
        pdb.set_trace()

        # plot the crops in a table
        plot_embedding_crops_table(df_imgs, output_dir, f'10_closest_crops_table_{frame}.png', 10)
        print(f'Figure saved: {output_dir}/10_closest_crops_table_{frame}.png')
        

def read_csv_to_dataframe(csv_file):
    """Reads the CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_file)
    return df



if __name__ == "__main__":
    main()  