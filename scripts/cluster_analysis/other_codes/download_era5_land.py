#!/usr/bin/env python
import xarray as xr
import cdsapi

#check download
path_file = '/home/daniele/Documenti/Data/ERA5-Land_t2m_snowc_u10_v10_sp_tp/2013/2013-04_expats.nc'
ds = xr.open_dataset(path_file)
print(ds.t2m)

"""

years = ["2013","2014"]
months = ["04","05","06","07","08","09"]
path_save = "/home/daniele/Scaricati/"

client = cdsapi.Client()

#for year in years:
#    for month in months:

dataset = "reanalysis-era5-land"
request = {
    "variable": [
        "2m_temperature",
        "snow_cover",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_pressure",
        "total_precipitation"
    ],
    "year": "2015",
    "month": "04",
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [51.5, 5, 42, 16]
}

#f'{path_save}ERA5-Land_{year}-{month}.nc'
client.retrieve(dataset, request).download(f"{path_save}ERA5-Land_2015-05.nc")

"""

