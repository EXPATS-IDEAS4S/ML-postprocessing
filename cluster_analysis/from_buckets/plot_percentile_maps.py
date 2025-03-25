import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import cmcrameri.cm as cmc
import os
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd

from aux_functions_from_buckets import get_variable_info

def plot_dataset_maps(ds, ds_oro, output_path, label, cmap, vmin_vmax_df):
    """
    Plots each variable in the dataset using Cartopy for borders and coastlines,
    setting extent based on lat/lon edges stored in metadata.
    Grid lines and label ticks align with the dataset's lat/lon centers, shown only on the bottom and left.
    Orography is plotted in the background in grayscale.

    Args:
        ds (xarray.Dataset): The dataset containing lat/lon and variables.
        ds_oro (xarray.Dataset): The dataset containing orography data.
        output_path (str): Path to save the plots.
        label (int): Label identifier for the dataset.
        cmap (str): Colormap for plotting.
    """

    # Extract lat/lon edges from metadata
    lat_edges = np.array(eval(ds.attrs["lat_edges"]))  # Convert string to array
    lon_edges = np.array(eval(ds.attrs["lon_edges"]))

    # Get lat/lon values (centers of pixels) 
    lat_centers = ds.lat.values
    lon_centers = ds.lon.values

    # Get orography data
    oro_lat = ds_oro.lat.values
    oro_lon = ds_oro.lon.values
    orography = ds_oro["DEM"].values  # Assuming DEM is the orography variable

    for var in ds.data_vars:
        print(f"Processing variable: {var}")

        # Get vmin and vmax for the variable
        vmin = vmin_vmax_df.loc[vmin_vmax_df['Variable'] == var, 'Min'].values[0]
        vmax = vmin_vmax_df.loc[vmin_vmax_df['Variable'] == var, 'Max'].values[0]
        print(f"vmin: {vmin}, vmax: {vmax}")
        
        # Handle variable name parsing
        if "-" in var:
            parts = var.split("-")
            var_name = parts[0]
            percentile = parts[1] if len(parts) > 1 else None
        else:
            var_name = var
            percentile = None

        var_info = get_variable_info(var_name)

        fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon_edges.min(), lon_edges.max(), lat_edges.min(), lat_edges.max()], crs=ccrs.PlateCarree())

        # Plot data variable
        pcm = ax.pcolormesh(lon_centers, lat_centers, ds[var], cmap=cmap, shading="auto", alpha=1, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

        # Plot orography as a transparent grayscale layer
        ax.pcolormesh(oro_lon, oro_lat, orography, cmap="Greys", shading="auto", alpha=0.3, transform=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.BORDERS, linewidth=1, color="black")
        ax.add_feature(cfeature.COASTLINE, linewidth=1, color="black")
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)

        # Add colorbar
        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.7)
        if var_info['unit'] is not None:
            cbar.set_label(f"{var_name} ({var_info['unit']})", fontsize=10)
        else:
            cbar.set_label(f"{var_name}", fontsize=10)

        # Labels and title
        title_text = f"{var_info['long_name']} map for class {label}"
        if percentile:
            title_text = f"{var_info['long_name']} {percentile}th percentile map for class {label}"
        plt.title(title_text, fontsize=12, fontweight="bold")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Add lat/lon grid lines aligning with data centers
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.7)
        gl.xlocator = plt.FixedLocator(lon_centers)  # Exact lon centers
        gl.ylocator = plt.FixedLocator(lat_centers)  # Exact lat centers

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Show labels only on bottom and left
        gl.right_labels = False
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = True

        # Increase font size of labels
        gl.xlabel_style = {'fontsize': 10}
        gl.ylabel_style = {'fontsize': 10}

        # Save the plot
        output_file = f"{output_path}percentile_{var}_maps_res_{label}.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved plot: {output_file}")

# Example usage
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
sampling_type = 'closest'
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
cmap = cmc.nuuk #RdYlBu_r'
n_div = 8

#path to orography
path_dem = '/data1/other_data/DEM_EXPATS_0.01x0.01.nc'
# Open DEM dataset 
ds_dem = xr.open_dataset(path_dem, decode_times=False, engine="h5netcdf")
print(ds_dem)

labels = np.arange(0, 9)  

# Open file with vmin and vmax
vmin_vmax_file = f"{output_path}variable_min_max.csv" 
vmin_vmax = pd.read_csv(vmin_vmax_file)
# Print the DataFrame
print(vmin_vmax)

for label in labels:

    # Open dataset
    ds = xr.open_dataset(f"{output_path}percentile_maps_res_{n_div}x{n_div}_label_{label}.nc", decode_times=False, engine="h5netcdf")
    print(ds)

    # Plot and save maps
    path_figures = f"{output_path}{label}/"
    os.makedirs(path_figures, exist_ok=True)
    plot_dataset_maps(ds, ds_dem, path_figures, label, cmap, vmin_vmax)
