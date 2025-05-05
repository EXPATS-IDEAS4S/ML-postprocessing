
import numpy as np
import pandas as pd
from glob import glob
import io
import xarray as xr
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cmcrameri.cm as cmc
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree

from get_data_from_buckets import read_file, Initialize_s3_client
from credentials_buckets import S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
from aux_functions_from_buckets import extract_coordinates, extract_datetime


# Bucket names
BUCKET_CMSAF_NAME = 'expats-cmsaf-cloud'
BUCKET_IMERG_NAME = 'expats-imerg-prec'
BUCKET_MSG_NAME = 'expats-msg-training'

def select_ds_from_dataframe(row, var):
    crop_filename = row['path'].split('/')[-1]
    coords = extract_coordinates(crop_filename)
    lat_min, lat_max, lon_min, lon_max = coords['lat_min'], coords['lat_max'], coords['lon_min'], coords['lon_max']

    datetime_info = extract_datetime(crop_filename)
    year, month, day, hour, minute = datetime_info['year'], datetime_info['month'], datetime_info['day'], datetime_info['hour'], datetime_info['minute']
    datetime_obj = np.datetime64(f'{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00')
    print('processing timestamp:', datetime_obj)
        
    if var == 'precipitation' and (minute == 15 or minute == 45):
        return None

    if var in ['IR_108', 'WV_062']:
        bucket_name = BUCKET_MSG_NAME
        bucket_filename = f'/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/{year:04d}/{month:02d}/merged_MSG_CMSAF_{year:04d}-{month:02d}-{day:02d}.nc'	
    elif var == 'precipitation':
        bucket_name = BUCKET_IMERG_NAME
        bucket_filename = f'IMERG_daily_{year:04d}-{month:02d}-{day:02d}.nc'
    else:
        bucket_name = BUCKET_CMSAF_NAME 
        bucket_filename = f'MCP_{year:04d}-{month:02d}-{day:02d}_regrid.nc'
    try:
        my_obj = read_file(s3, bucket_filename, bucket_name)
        ds_day = xr.open_dataset(io.BytesIO(my_obj))[var]

        if isinstance(ds_day.indexes["time"], xr.CFTimeIndex):
            ds_day["time"] = ds_day["time"].astype("datetime64[ns]")

        ds_day = ds_day.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        ds_day = ds_day.sel(time=datetime_obj)

        # Check if the dataset contain all nan
        if np.isnan(ds_day.values).all():
            print(f"All values are NaN for {var} at {datetime_obj}")
            return None
        else:
            return ds_day
        
    except Exception as e:
        print(f"Error processing {var} for {row['path']}: {e}")
        return None

def plot_nc_crops_scatter(df, output_path, filename, var, cmap, norm):
    """
    Plots NetCDF data on a scatter plot where images were originally plotted.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for idx, row in df.iterrows():
        ds = select_ds_from_dataframe(row, var)

        # If ds is None, attempt to find a valid dataset in subsequent rows
        search_idx = idx
        while ds is None and search_idx < len(df) - 1:
            search_idx += 1
            ds = select_ds_from_dataframe(df.iloc[search_idx], var)

        # If no valid ds is found after searching, skip this iteration
        if ds is None or not isinstance(ds, xr.DataArray) or ds.name != var:
            continue  

        img = ds.values.squeeze()  # Convert to 2D if needed
        img = np.flipud(img)  # Flip the image vertically
        imagebox = OffsetImage(img, zoom=0.3, cmap=cmap)

        if norm is not None and isinstance(norm, plt.Normalize):
            vmin, vmax = norm.vmin, norm.vmax  # Extract numerical values
        else:
            vmin, vmax = 0, 1  # Default values

        img = np.clip(img, vmin, vmax)  # Now `vmax` is a float, avoiding the TypeError

        ab = AnnotationBbox(imagebox, (row['Component_1'], row['Component_2']), frameon=False)
        ax.add_artist(ab)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    fig.savefig(output_path + filename.split('.')[0] + '_nc_scatter_' + var + '.png', bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot: {output_path + filename.split('.')[0]}_nc_scatter_{var}.png")


def plot_nc_crops_grid(df, output_path, filename, var, cmap, norm, grid_size=10):
    """
    Plots NetCDF crops on a regular grid using the closest data point to each grid cell.
    Avoids overlapping by ensuring one image per grid cell.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize component coordinates to [0, 1]
    x = df['Component_1'].values
    y = df['Component_2'].values
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Build KDTree for fast nearest neighbor search
    tree = cKDTree(np.c_[x_norm, y_norm])

    # Build a grid of evenly spaced points
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    used_indices = set()

    for point in grid_points:
        dist, idx = tree.query(point)

        if idx in used_indices:
            continue  # already used this crop

        used_indices.add(idx)
        row = df.iloc[idx]
        ds = select_ds_from_dataframe(row, var)

        if ds is None or not isinstance(ds, xr.DataArray) or ds.name != var:
            continue

        img = ds.values.squeeze()
        img = np.flipud(img)

        if norm is not None and isinstance(norm, plt.Normalize):
            vmin, vmax = norm.vmin, norm.vmax
        else:
            vmin, vmax = 0, 1

        img = np.clip(img, vmin, vmax)
        imagebox = OffsetImage(img, zoom=0.3, cmap=cmap)

        ab = AnnotationBbox(imagebox, point, frameon=False)
        ax.add_artist(ab)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.savefig(output_path + filename.split('.')[0] + '_nc_grid_' + var + '.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved grid plot: {output_path + filename.split('.')[0]}_nc_grid_{var}.png")


def plot_nc_crops_table(df, output_path, filename, var, cmap, norm, n=5, selection="closest"):
    """
    Plots NetCDF crops in a table format based on clustering labels.
    """
    labels = df['label'].unique()
    num_labels = len(labels)
    
    fig, axes = plt.subplots(num_labels, n, figsize=(n * 2, num_labels * 2))
    fig.suptitle(f"Crops Sorted by {selection.capitalize()} Distance", fontsize=14, fontweight="bold")
    
    if num_labels == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, label in enumerate(labels):
        subset = df[df['label'] == label].sort_values(by='distance')
        
        if selection == "closest":
            subset = subset.head(n)
        elif selection == "farthest":
            subset = subset.tail(n)
        elif selection == "random":
            subset = subset.sample(n=min(n, len(subset)), random_state=42)

        for j, (_, row) in enumerate(subset.iterrows()):
            ds = select_ds_from_dataframe(row, var)

            # If selection is random, try finding another row if ds is None
            if selection == "random" and ds is None:
                search_idx = j
                while ds is None and search_idx < len(subset) - 1:
                    search_idx += 1
                    ds = select_ds_from_dataframe(subset.iloc[search_idx], var)

            if ds is None or not isinstance(ds, xr.DataArray) or ds.name != var:
                continue  # Skip if still no valid dataset

            img = ds.values.squeeze()
            ax = axes[i, j] if num_labels > 1 else axes[j]
            img = np.flipud(img)  # Flip the image vertically
            ax.imshow(img, cmap=cmap)

            if norm is not None and isinstance(norm, plt.Normalize):
                vmin, vmax = norm.vmin, norm.vmax  # Extract numerical values
            else:
                vmin, vmax = 0, 1  # Default values

            img = np.clip(img, vmin, vmax)  # Now `vmax` is a float, avoiding the TypeError

            ax.axis('off')

            if j == 0:
                ax.set_ylabel(f"Label {label}", fontsize=12, fontweight='bold', rotation=0, labelpad=30, va='center')
    
    for j in range(n):
        axes[0, j].set_title(f"{j+1}", fontsize=12, fontweight="bold")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = f"{output_path}/{filename.split('.')[0]}_{n}_{selection}_nc_table_{var}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved table plot: {output_file}")



run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3' #all visualization were made with random state 3
sampling_type = 'all' 
reduction_method = 'tsne' # Options: 'tsne', 'isomap',
vars = ['IR_108','WV_062', 'cot', 'cth','precipitation','cma', 'cph']
#cmaps = [cmc.grayC, 'magma', 'cividis_r', cmc.batlowK, 'binary']
#vmaxs = [None, 20, None, 10, None   ]
sample_to_plot = 200

colormap_dict = {
    "IR_108": plt.get_cmap("Greys"),
    "WV_062": plt.get_cmap("Greys"),
    "cot": plt.get_cmap("magma"),
    "cth": plt.get_cmap("cividis"),
    "precipitation": cmc.batlowK,
    "cma": plt.get_cmap("binary_r"),
    "cph": ListedColormap(["black", "lightblue", "darkorange"])  # Custom colormap
}

norm_dict = {
    "IR_108": None,
    "cot": plt.Normalize(vmin=0, vmax=20),  # Compress values above 20
    "precipitation": plt.Normalize(vmin=0, vmax=10),  # Compress values above 10
    "WV_062": None,
    "cth": None,
    "cma": None,
    "cph": None
}

output_path = f'/data1/fig/{run_name}/{sampling_type}/'
filename = f'{reduction_method}_pca_cosine_{run_name}_{random_state}.npy'

# List of the image crops
image_crops_path = f'/data1/crops/{run_name}/1/'
list_image_crops = sorted(glob(image_crops_path+'*.tif'))
n_samples = len( list_image_crops)
print('n samples: ', n_samples)

# Read data
if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:
    n_subsample = 1000

# Open csv file with already labels and dim red features
df_labels = pd.read_csv(f'{output_path}merged_tsne_variables_{run_name}_{sampling_type}_{random_state}.csv')
#list all columns name
print(df_labels.columns)

# Remove column called color
df_labels = df_labels.loc[:, ~df_labels.columns.str.contains('^color')]

# # Define the color names for each class from dataframe
colors_per_class1_names = {
    '0': 'darkgray', 
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green'
}


# Filter out invalid labels (-100)
df_subset1 = df_labels[df_labels['label'] != -100]

# Map labels to colors
df_subset1['color'] = df_subset1['label'].map(lambda x: colors_per_class1_names[str(int(x))])

# Sample 20,000 points for plotting
df_subset2 = df_subset1.sample(n=sample_to_plot)

#INITIALIZE S3 CLIENT
s3 = Initialize_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY)

output_dir = f'{output_path}crop_embeddings/'
os.makedirs(output_dir, exist_ok=True)


for var in vars:
    print(var)
    #plot_nc_crops_scatter(df_subset2, output_dir, filename, var, colormap_dict[var], norm_dict[var] )
    #plot_nc_crops_table(df_subset1, output_dir, filename, var, colormap_dict[var], norm_dict[var], n=10, selection="random")
    #plot_nc_crops_table(df_subset1, output_dir, filename, var, colormap_dict[var], norm_dict[var], n=10, selection="closest")
    #plot_nc_crops_grid(df_subset2, output_dir, filename, var, colormap_dict[var], norm_dict[var], grid_size=10)
        