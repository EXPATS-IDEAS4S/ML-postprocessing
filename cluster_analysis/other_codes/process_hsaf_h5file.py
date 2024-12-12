import h5py
import xarray as xr

# Path to your .h5 file
file_path = '/home/daniele/Documenti/Data/hsaf/h10_2013/h10_20130401_day_merged.H5'

nc_file = '/home/daniele/Documenti/Data/hsaf/h10_2013/output_file.nc'
ds = xr.open_dataset(nc_file)
print(ds)
exit()


# Open the .h5 file in read mode
with h5py.File(file_path, 'r') as h5_file:
    # List all groups/datasets in the file
    print("Keys in the HDF5 file:")
    print(list(h5_file.keys()))  # This will show the top-level keys (groups or datasets)
    #['LAT', 'LON', 'SC', 'SC_Q_Flags', 'colormap'
    dataset_list = list(h5_file.keys())
    exit()
    # Access a specific group or dataset 
    for dataset_name in dataset_list:
        if dataset_name in h5_file:
            dataset = h5_file[dataset_name]
            print(f"\nDataset '{dataset_name}' details:")
            print(f"Shape: {dataset.shape}")
            print(f"Data type: {dataset.dtype}")

            # Read the data from the dataset (converting it to a NumPy array)
            data = dataset[:]
            print(f"Data: {data}")
        else:
            print(f"'{dataset_name}' not found in the HDF5 file.")
