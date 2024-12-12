import rasterio
import numpy as np
import xarray as xr
import os
from glob import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_snow_cover(netcdf_path):
    # Open the NetCDF file
    ds = xr.open_dataset(netcdf_path)
    
    # Extract snow cover data
    snow_cover = ds['snow_cover']
    
    # Extract lat/lon values
    lats = snow_cover.lat.values
    lons = snow_cover.lon.values
    
    # Plot the data using Cartopy
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add features like coastlines, borders, etc.
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Plot the snow cover data
    snow_cover_plot = ax.pcolormesh(lons, lats, snow_cover, cmap='Blues', transform=ccrs.PlateCarree())

    # Add a colorbar for the snow cover data
    plt.colorbar(snow_cover_plot, ax=ax, label='Snow Cover')

    # Set a title and display the plot
    plt.title('Snow Cover')
    plt.show()


def plot_histogram(values, bins=100, title="Histogram", xlabel="Values", ylabel="Frequency"):
    plt.hist(values, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def tiff_to_netcdf(tiff_path, netcdf_path, check_fig=False):
    # Open the TIFF file
    with rasterio.open(tiff_path) as dataset:
        print(dataset.meta)
        print(dataset.tags())  # This will show metadata including possible scale/offset
        
        # Read the data from the TIFF file
        data = dataset.read(1)  # Read the first band (assuming snow cover data is in the first band)
        print(dataset)
        print(data)
        #plot_histogram(data)
    
        # Get the geographic transform and coordinate reference system
        transform = dataset.transform
        crs = dataset.crs
        print(transform)
        print(crs)
        
        if crs is None or not crs.is_geographic:
            print(f"File {tiff_path} does not contain geographic (lat/lon) information.")
            return
        
        # Get the bounds and resolution
        print(dataset.bounds)
        left, bottom, right, top = dataset.bounds
        res_x = transform[0]
        res_y = transform[4]
        print(res_x,res_y)
        
        # Create lat and lon arrays based on the geotransform
        lon = np.linspace(left + res_x / 2, right - res_x / 2, data.shape[1])
        lat = np.linspace(top + res_y / 2, bottom - res_y / 2, data.shape[0])

        # Check that data shape matches lat/lon lengths
        if data.shape != (len(lat), len(lon)):
            raise ValueError(f"Data shape {data.shape} does not match lat/lon dimensions ({len(lat)}, {len(lon)}).")
        
        print(lon.size)
        print(lat)
        print(data.size)

        # Make sure lat is in descending order
        if lat[0] < lat[-1]:
            lat = lat[::-1]
            data = np.flipud(data)
        
        # Create an xarray DataArray for snow cover data with lat/lon as coordinates
        snow_cover = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
            name="snow_cover"
        )
        
        # Set attributes for the DataArray
        snow_cover.attrs['units'] = '%'  # Assuming it's a binary snow cover (1 for snow, 0 for no snow)
        #snow_cover.attrs['description'] = 'Snow cover data'
        
        # Create a dataset and assign the DataArray
        dataset = xr.Dataset({'snow_cover': snow_cover})
        
        # Set global attributes
        dataset.attrs['crs'] = str(crs)
        dataset.attrs['title'] = 'Cryoland Snow cover data converted from GeoTIFF to NetCDF'

        print(dataset)
        
        # Save as NetCDF file
        print(tiff_path)
        filename = tiff_path.split('/')[-1].split('_')[0:4]
        print(filename)
        filename_nc = '_'.join(filename)+'_EXPATS.nc'
        dataset.to_netcdf(netcdf_path+filename_nc)
        print(f"Successfully converted {tiff_path} to {netcdf_path}.")
        if check_fig:
            plot_snow_cover(netcdf_path+filename_nc)
            #exit()

# Example usage
year = 2013
tiff_path = f'/home/daniele/Scaricati/cryoland/tif/{str(year)}/'
netcdf_path = f'/home/daniele/Scaricati/cryoland/netcdf/{str(year)}/'

# Create the directory if it doesn't exist
os.makedirs(netcdf_path, exist_ok=True)

list_tif = sorted(glob(tiff_path+'*tif'))
#print(list_tif)

for tif_file in list_tif:
    tiff_to_netcdf(tif_file, netcdf_path, True)
