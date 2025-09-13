"""
code to map geolocation of classes as dots over a map of the expats domain
Author: Claudia Acquistapace
Date: 13 sept 2025
"""


import sys
import os
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import pdb
from array import array 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
sys.path.append(os.path.abspath("/Users/claudia/Documents/ML-postprocessing"))
from utils.plotting.class_colors import colors_per_class1_names, class_groups
from scripts.pretrain.cluster_analysis.var_class_temporal_series import read_csv_to_dataframe

def main():

    # csv file with 2D+1 output
    csv_file = '/Users/claudia/Documents/data_ml_spacetime/crops_stats_vars-cth-cma-cot-cph_stats-50-99-25-75_frames-8_timedim_coords-datetime_dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1_closest_1000_debug.csv'
    output_dir = '/Users/claudia/Documents/data_ml_spacetime/figs/'

    # read csv file
    df = read_csv_to_dataframe(csv_file)
    print("Column titles:", df.columns.tolist())

    # select one variable to be sure to select 8 frames
    df8 = df[df['var'] == 'cth']

    # loop on df8 rowq to read lat, lon, label of first frame of each video
    lats = []
    lons = []
    labels = []
    # loop on df8 rows
    for i, row in df8.iterrows():
        lats.append(row['lat_mid'])
        lons.append(row['lon_mid'])
        labels.append(row['label'])
    lats = np.array(lats)
    lons = np.array(lons)
    labels = np.array(labels)

    r_tl = {'lon_min': 5, 'lon_max': 10.5, 'lat_min': 47, 'lat_max': 52}
    r_tr = {'lon_min': 10.5, 'lon_max': 16, 'lat_min': 47, 'lat_max': 52}
    r_bl = {'lon_min': 5, 'lon_max': 16, 'lat_min': 42, 'lat_max': 47}
    r_br = {'lon_min': 10.5, 'lon_max': 16, 'lat_min': 42, 'lat_max': 47}

    # find lat, lon, labels in each quadrant
    rects = [r_tl, r_tr, r_bl, r_br]
    quadrant_labels = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right']

    # plot histogram of occurrences of each class (x = class, y = number of occurrences)
    plt.figure(figsize=(10, 8))
    # create subplot for each quadrant
    for i, r in enumerate(rects):
        in_quadrant = np.where((lons >= r['lon_min']) & (lons < r['lon_max']) &
                               (lats >= r['lat_min']) & (lats < r['lat_max']))[0]
        labels_in_quadrant = labels[in_quadrant]
        unique, counts = np.unique(labels_in_quadrant, return_counts=True)
        class_counts = dict(zip(unique, counts))
        plt.subplot(2, 2, i+1)
        plt.title(f'{quadrant_labels[i]}', fontsize=20)
        # plot bars using colors from colors_per_class1_names
        colors = list(colors_per_class1_names.values())
        bar_colors = [colors[int(cls) % len(colors)] for cls in class_counts.keys()]
        # normalize counts to percentage
        total_counts = sum(class_counts.values())
        class_counts = {k: v / total_counts * 100 for k, v in class_counts.items()}
        print("Class counts (percentage):", class_counts)
        plt.bar(class_counts.keys(), class_counts.values(), color=bar_colors)
        plt.xlabel('Class', fontsize=18)
        plt.xlim(-0.5, 8.5)
        plt.ylabel('Class count percentage [%]', fontsize=18)
        # remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    # save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'class_occurrences_quadrants.png'), dpi=300, transparent=True)
    plt.close()

    # find number of points in each class

    # plot geolocation of quadrants with colors shadows for each quadrant
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([5, 16, 42, 52], crs=ccrs.PlateCarree())

    # split in 4 equal quadrants and add a rectangle to highlight each quadrant
    # add rectangle to highlight the top left quadrant
    rect_tl = mpatches.Rectangle((5, 47), 
                                 5.5, 
                                 5, 
                                 linewidth=3, 
                                 edgecolor='black',
                                 facecolor='blue', 
                                 alpha=0.5)
    # color the area of the rectangle in light blue

    ax.add_patch(rect_tl)
    # add rectangle to highlight the top right quadrant
    rect_tr = mpatches.Rectangle((10.5, 47), 
                                 5.5, 
                                 5, 
                                 linewidth=3, 
                                 edgecolor='black',
                                 facecolor='lightblue', 
                                 alpha=0.5)
    ax.add_patch(rect_tr)
    # add rectangle to highlight the bottom left quadrant
    rect_bl = mpatches.Rectangle((5, 42), 
                                 5.5, 
                                 5, 
                                 linewidth=3, 
                                 edgecolor='black', 
                                 facecolor='red', 
                                 alpha=0.5)
    
    ax.add_patch(rect_bl)
    # add rectangle to highlight the bottom right quadrant
    rect_br = mpatches.Rectangle((10.5, 42),
                                  5.5, 
                                  5, 
                                  linewidth=3, 
                                  edgecolor='black', 
                                  facecolor='purple', 
                                  alpha=0.5)
    ax.add_patch(rect_br)   

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.gridlines(draw_labels=True)  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'geolocation_classes.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()


