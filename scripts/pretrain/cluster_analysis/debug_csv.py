"""
Code to debug the csv file with crops statistics. Frame number is wrongly assigned. This code loops on the 
ncdf file names and the time stamps of the crop and assigns based on the time difference the frame number. 
It produces a new csv file with the corrected frame number.
author: Claudia Acquistapace Daniele Corradini
date:11 Sept 2025
"""

import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import pdb
from datetime import datetime
import re
import sys
sys.path.append('/home/claudia/codes/ML_postprocessing')


def main():

    # csv file with 2D+1 output
    csv_file = '/data1/fig/dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1/epoch_800/closest/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000.csv'
    output_csv_file = '/data1/fig/dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1/epoch_800/closest/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000_debug.csv'
    path_img = '/data1/crops/clips_ir108_100x100_8frames_2013-2020/img/IR_108_cm/1/'
    path_netcdf = '/data1/crops/clips_ir108_100x100_8frames_2013-2020/nc/IR_108_cm/1/'

    # read csv file
    df = read_csv_to_dataframe(csv_file)
    print("Column titles:", df.columns.tolist())

    # loop on df rows
    corrected_frames = []
    for index, row in df.iterrows():

        # get the ncdf file path for the crop
        nc_filename = row['path']

        # get the time stamp for the crop
        timestamp_crop = row['time']

        # exctract the date and the time from the nc_filename
        # example of file name /data1/crops/clips_ir108_100x100_8frames_2013-2020/nc/1/MSG_timeseries_2016-06-17_2330_crop2.nc        

        # split the string to extract starting time 
        base_nc_filename = os.path.basename(nc_filename)
        parts = base_nc_filename.split('_')
        date_part = parts[2]  # '2016-06-17'
        time_part = parts[3]  # '2330'
        starting_time_str = f"{date_part} {time_part[:2]}:{time_part[2:]}"
        starting_time = datetime.strptime(starting_time_str, '%Y-%m-%d %H:%M')  
        #print('Starting time:', starting_time)
        #print('Timestamp crop:', timestamp_crop)

        # convert timestamp_crop to datetime object
        timestamp_crop = pd.to_datetime(timestamp_crop)

        # calculate time difference in minutes
        time_diff = (timestamp_crop - starting_time).total_seconds() / 60.0
        #print('Time difference (minutes):', time_diff)

        # calculate frame number
        frame_number = int(time_diff / 15)  # assuming each frame is 15 minutes apart
        #print('Calculated frame number:', frame_number) 
        corrected_frames.append(frame_number)

    # add the corrected frame number to the dataframe for all rows and all variables
    df = df.copy()  # to avoid SettingWithCopyWarning
    df['frame'] = corrected_frames
    print(df[['path', 'time', 'frame']].head(20))

     # store a new csv file with the corrected frame number and the other columns from the original dataframe
    df.to_csv(output_csv_file, index=False)
    print(f"Corrected CSV file saved to {output_csv_file}")


def read_csv_to_dataframe(csv_file):
    """Reads the CSV file into a pandas DataFrame."""
    df = pd.read_csv(csv_file)
    return df

if __name__ == "__main__":
    main()  