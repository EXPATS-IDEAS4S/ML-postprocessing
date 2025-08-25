import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from glob import glob

run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3' #all visualization were made with random state 3
sampling_type = 'all' 
reduction_method = 'tsne' # Options: 'tsne', 'isomap',

output_path = f'/data1/fig/{run_name}/{sampling_type}/'

n_subsample = 1000

# Load DEM (Orography) Data
path_dem = '/data1/other_data/DEM_EXPATS_0.01x0.01.nc'
ds_dem = xr.open_dataset(path_dem, decode_times=False, engine="h5netcdf")
print(ds_dem)

# Extract elevation and coordinates
elevation = ds_dem['DEM'].values  # Adjust variable name if needed
lats = ds_dem['lat'].values
lons = ds_dem['lon'].values

# Load the DataFrame with labeled points
output_path = f'/data1/fig/{run_name}/{sampling_type}/'
df_labels = pd.read_csv(f'{output_path}merged_tsne_variables_{run_name}_{sampling_type}_{random_state}.csv')

# Create output directory for separate maps
output_dir = os.path.join(output_path, "crop_location_maps")
os.makedirs(output_dir, exist_ok=True)

# Get unique labels
unique_labels = df_labels['label'].unique()

# Loop through each label and generate a separate map
for label in unique_labels:
    df_subset = df_labels[df_labels['label'] == label]

    if n_subsample:
        #Select the closest n_subsample row basd on 'distance''
        #Take the largest values of distannce since a cosine is used
        df_subset = df_subset.nlargest(n_subsample, 'distance')

    # Initialize figure and axis with Cartopy projection
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set map extent based on DEM data
    ax.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', linewidth=0.8)
    ax.add_feature(cfeature.RIVERS, alpha=0.5, edgecolor='black')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='white')  # No colored sea

    # Plot orography in grayscale
    contour = ax.contourf(lons, lats, elevation, levels=100, cmap='Greys', alpha=0.6, transform=ccrs.PlateCarree())

    # Plot label-specific points
    ax.scatter(df_subset['lon_mid'], df_subset['lat_mid'], 
               color=df_subset['color'].values[0],  # Assuming one color per label
               label=f'Label {label}', 
               #edgecolor='black', 
               s=10, alpha=0.8, transform=ccrs.PlateCarree())

    # Add legend
    #plt.legend(title=f"Label {label}", loc='upper right')

    # Add color bar for elevation
    #cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
    #cbar.set_label('Elevation (m)')

    # Set title
    plt.title(f'Midpoints of Label {label}')

    # Save figure
    output_file = os.path.join(output_dir, f'map_label_{label}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

    print(f"Saved: {output_file}")