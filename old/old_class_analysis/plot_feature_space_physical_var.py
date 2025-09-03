import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from glob import glob
import torch
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
import cmcrameri.cm as cmc

from utils.plotting.feature_space_plot_functions import name_to_rgb, scale_to_01_range
from utils.plotting.feature_space_plot_functions import colors_per_class1_names
from utils.processing.aux_functions import get_variable_info

reduction_method = 'tsne' #'tsne
variable = 'cma'
data_type = 'categorical' #'continuous' 
scale = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3' #all visualization were made with random state 3
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'
stat = '50' #None  # '50'
cmap = cmc.imola #RdYlBu_r'

# Paths to CMSAF cloud properties crops
cloud_properties_path = '/data1/crops/cmsaf_2013-2014-2015-2016_expats/nc_clouds/'
cloud_properties_crop_list = sorted(glob(cloud_properties_path + '*.nc'))
n_samples = len(cloud_properties_crop_list)
n_subsample = n_samples #1000  # Number of samples per cluster

#Pick uynits
info_var = get_variable_info(data_type, variable)
print(info_var)

# Path to fig folder for outputs
output_path = f'/home/Daniele/fig/{scale}/{sampling_type}/'


tsne_path = f'/home/Daniele/fig/{scale}/'
filename = f'{reduction_method}_pca_cosine_{scale}_{random_state}.npy' 
#filename = f'{reduction_method}_cosine_{scale}_{n_subsample}.npy' 
tsne = np.load(tsne_path+filename)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)


# elif reduction_method=='isomap':
#     tsne_path = f'/home/Daniele/fig/cma_analysis/{scale}/'
#     filename = f'isomap_{scale}_2d_cosine_500ep_Nonesamples.csv'
#     df_tsne = pd.read_csv(tsne_path+filename)
#     #df_tsne = df_tsne.drop('Selected_Index')
#     print(df_tsne)
#     #exit()
# else:
#     print('reduction method not supported!')

image_path = f'/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA/1/' #cot_2013_128_germany, ir108_2013_128_germany
crop_path_list = sorted(glob(image_path+'*.tif'))
#print(len(crop_path_list))

filename1 = 'rank0_chunk0_train_heads_inds.npy' 
#filename2 = 'rank0_chunk0_train_heads_targets.npy'
#filename3 = 'rank0_chunk0_train_heads_features.npy'

path_feature = f'/data1/runs/{scale}/features/'

checkpoints_path = f'/data1/runs/{scale}/checkpoints/'  

assignments = torch.load(checkpoints_path+'assignments_800ep.pt',map_location='cpu')
sample_list = np.load(checkpoints_path+'samples_k7_800ep.npy')

data1 = np.load(path_feature+filename1)
print(data1)
if n_subsample < n_samples:
    indeces_filename = filename.split('.')[0] + '_indeces.npy' 
    indeces = np.load(tsne_path+indeces_filename)
    crop_path_list = [crop_path_list[i] for i in indeces]
    data1 = indeces

# Create a DataFrame for T-SNE coordinates
df_tsne = pd.DataFrame({'Component_1': tx, 'Component_2': ty, 'Selected_Index': data1})

print(assignments) #TODO it has 3 rows, DC used the first (why?)

# Path to cluster distances (from centroids)
#distances_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{scale}/checkpoints/distance_800ep.pt'

#Load the path and labels of the nc crops
df_labels = pd.read_csv(f'{output_path}crop_list_{scale}_{n_samples}_{sampling_type}.csv')
print(df_labels)


# Load the saved CSV into a new DataFrame
merged_df = pd.read_csv(f'{output_path}physical_feature_vectors_{scale}_{sampling_type}_{n_samples}_{stat}.csv')
print(merged_df)

if n_subsample < n_samples:
    #get the rows that correspond to the selected samples from the ordering of the rows
    merged_df = merged_df.iloc[data1]
    df_labels = df_labels.iloc[data1]


#Selected the variable that needs to be plot

df_variable = merged_df[variable]
df_cot = merged_df['cot']
print(df_variable)

#take only the useful columns from the dataframs and merge them
df_variable = pd.concat([df_variable,df_labels], axis=1)
df_variable_cot = pd.concat([df_cot,df_labels], axis=1)
print(df_variable)


# Create a DataFrame for labels
df_crop_path = pd.DataFrame({'y': '', 'index': data1})
df_crop_path.set_index('index', inplace=True)

for index in range(len(data1)):
    df_crop_path.loc[index] = assignments[0, :].cpu()[index]

df_crop_path['location'] = crop_path_list

print(df_crop_path) # dataframe with the path of the crops


# Reset the index of df2 to turn it into a column
df_tsne = df_tsne.reset_index(drop=True) # dadtaframe with the tsne componentss

# Merge tsne components with the crop location
df_tsne = pd.concat([df_tsne,df_crop_path], axis=1)

print(df_tsne)

#Extract the tsne component that matches the variables

# Extract the last part of the 'path' and 'location' columns (after the last '/')
df_variable['path_last_part'] = df_variable['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
df_variable_cot['path_last_part'] = df_variable_cot['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
df_tsne['path_last_part'] = df_tsne['location'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[0:4]))

print(df_variable)
print(df_tsne)

# Filter df_tsne to match the last part of df_variables' paths
filtered_df_tsne = df_tsne[df_tsne['path_last_part'].isin(df_variable['path_last_part'])]
filtered_df_tsne_cot = df_tsne[df_tsne['path_last_part'].isin(df_variable_cot['path_last_part'])]

print(filtered_df_tsne)

# Sort both DataFrames to ensure they align correctly before concatenation
df_variables_sorted = df_variable[df_variable['path_last_part'].isin(filtered_df_tsne['path_last_part'])].sort_values(by='path_last_part').reset_index(drop=True)
df_variables_sorted_cot = df_variable_cot[df_variable_cot['path_last_part'].isin(filtered_df_tsne_cot['path_last_part'])].sort_values(by='path_last_part').reset_index(drop=True)
filtered_df_tsne_sorted = filtered_df_tsne.sort_values(by='path_last_part').reset_index(drop=True)
filtered_df_tsne_sorted_cot = filtered_df_tsne_cot.sort_values(by='path_last_part').reset_index(drop=True)

print(df_variables_sorted)
print(filtered_df_tsne_sorted)

# Drop the helper columns used for matching
df_variables_sorted = df_variables_sorted.drop(columns=['path_last_part', 'path'])
df_variables_sorted_cot = df_variables_sorted_cot.drop(columns=['path_last_part', 'path'])
filtered_df_tsne_sorted = filtered_df_tsne_sorted.drop(columns=['path_last_part','y','location'])
filtered_df_tsne_sorted_cot = filtered_df_tsne_sorted_cot.drop(columns=['path_last_part','y','location'])

# Concatenate along axis 1
merged_tsne_variable_df = pd.concat([df_variables_sorted, filtered_df_tsne_sorted], axis=1)
merged_tsne_variable_df_cot = pd.concat([df_variables_sorted_cot, filtered_df_tsne_sorted_cot], axis=1)

# Display the merged DataFrame
print(merged_tsne_variable_df)


# Filter out invalid labels (-100)
#df_subset1 = df_subset[df_subset['y'] != -100]

# Map labels to colors
merged_tsne_variable_df['color'] = merged_tsne_variable_df['label'].map(lambda x: colors_per_class1_names[str(int(x))])
merged_tsne_variable_df_cot['color'] = merged_tsne_variable_df_cot['label'].map(lambda x: colors_per_class1_names[str(int(x))])
print(merged_tsne_variable_df)

# Sample 20,000 points for plotting
#df_subset2 = df_subset1.sample(n=20000)

#Convert color to RGB
colors_per_class1_rgb = {k: name_to_rgb(v) for k, v in colors_per_class1_names.items()}
print(colors_per_class1_rgb)


# Number of classes and subplot grid setup
n_classes = merged_tsne_variable_df['label'].nunique()
rows, cols = 3, 3  # For 9 classes, we want a 3x3 grid

# Set up a 3x3 grid of subplots
fig, axs = plt.subplots(rows, cols, figsize=(12, 10), sharex=True, sharey=True)

# Flatten the axis array for easy iteration (because axs is a 2D array)
axs = axs.flatten()

# Grid resolution for interpolation
grid_res = 100

# Get the min and max values of the variable to ensure consistent color mapping across all subplots
vmin, vmax = merged_tsne_variable_df[variable].min(), merged_tsne_variable_df[variable].max()

# Step 1: Loop through unique classes and plot in each subplot
for i, class_label in enumerate(merged_tsne_variable_df['label'].unique()):
    ax = axs[i]  # Select the current subplot
    
    # Filter the data for the current class
    class_data = merged_tsne_variable_df[merged_tsne_variable_df['label'] == class_label]
    
    # Drop rows where 'Component_1', 'Component_2', or the variable of interest has NaN values
    class_data = class_data.dropna(subset=['Component_1', 'Component_2', variable])
    
    # Ensure there's data to plot after removing NaNs
    if class_data.empty:
        print(f"Skipping class {class_label} due to missing data.")
        continue

    # Calculate the KDE for the points in Component_1 and Component_2
    kde = gaussian_kde(class_data[['Component_1', 'Component_2']].T)
    kde_values = kde(class_data[['Component_1', 'Component_2']].T)

    # Determine the 25th percentile of the KDE values
    kde_threshold = np.percentile(kde_values, 25)

    # Filter out points below the 25th percentile
    class_data = class_data[kde_values >= kde_threshold]

    # Ensure there's data to plot after filtering
    if class_data.empty:
        print(f"Skipping class {class_label} due to insufficient points after KDE filtering.")
        continue
        
    # Create a grid on Component_1 and Component_2 space
    grid_x, grid_y = np.mgrid[
        class_data['Component_1'].min():class_data['Component_1'].max():grid_res*1j, 
        class_data['Component_2'].min():class_data['Component_2'].max():grid_res*1j
    ]
    
    # Check if the ranges of grid_x or grid_y are valid (non-empty)
    if grid_x.size == 0 or grid_y.size == 0:
        print(f"Skipping class {class_label} due to invalid grid.")
        continue

    # Interpolate variable values onto the grid
    grid_z = griddata(
        (class_data['Component_1'], class_data['Component_2']),  # x, y coordinates
        class_data[variable],  # variable values
        (grid_x, grid_y),  # grid points where we want to interpolate
        method='linear',  # Interpolation method
        fill_value=np.nan  # Handle missing values by filling NaNs
    )
    
    # Ensure grid_z is not entirely NaN
    if np.isnan(grid_z).all():
        print(f"Skipping class {class_label} due to all NaN values after interpolation.")
        continue
      
    if variable == 'cot':
    
        contour = ax.contourf(
            grid_x, grid_y, grid_z, alpha=0.6, cmap=cmap,
            levels=np.linspace(vmin, 20, 100), extend='max'
        )
    else:    
        # Step 3: Plot contourf for the variable (same color map for all subplots)
        contour = ax.contourf(grid_x, grid_y, grid_z, cmap=cmap, alpha=0.5, levels=np.linspace(vmin, vmax, 100))
    
    
    # Step 2: Plot the KDE contours for each class
    sns.kdeplot(
        x=class_data['Component_1'],
        y=class_data['Component_2'],
        ax=ax,
        levels=[0.95],  #A vector argument must have increasing values in [0, 1]. Levels correspond to iso-proportions of the density: e.g., 20% of the probability mass will lie below the contour drawn for 0.2. 
        linewidths=1.,
        color= 'magenta', #colors_per_class1_names[str(int(class_label))],  # Use color corresponding to the class
        alpha=1.,
        label=f'Class {int(class_label)}'
    )

    
    # Set title for each subplot
    #ax.set_title(f'Class {int(class_label)}', fontsize=12)

    # Remove individual axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

# Step 4: Add shared x and y labels for the entire figure
fig.text(0.5, 0.05, 'Component 1', ha='center', fontsize=15)
fig.text(0.05, 0.5, 'Component 2', va='center', rotation='vertical', fontsize=15)

cbar = fig.colorbar(contour, ax=axs, orientation='vertical', fraction=0.03, pad=0.2)
unit = info_var['unit']
if unit:
    cbar.set_label(f'{unit}', fontsize=15)
else:
    cbar.set_label(f'', fontsize=15)

# Step 7: Format colorbar ticks to 2 decimal places
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# Step 6: Increase tick label size and thickness for the colorbar and axes
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=10, width=2)  # Adjust axis tick size and thickness
cbar.ax.tick_params(labelsize=12, width=2)  # Adjust colorbar label size and thickness

# Step 8: Add an overall title with larger bold font
if variable == 'cma':
    fig.suptitle(f'{reduction_method} 2d embedding distribution by class: cloud fraction', fontsize=16, fontweight='bold')
elif variable == 'cph':
    fig.suptitle(f'{reduction_method} 2d embedding distribution by class: liquid cloud fraction', fontsize=16, fontweight='bold')
else:
    fig.suptitle(f'{reduction_method} 2d Embedding Distribution by Class: {variable} - {stat}th quantile', fontsize=16, fontweight='bold')

# Step 9: Adjust spacing between subplots and make them more compact
plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.95, right=0.85)  # Decrease wspace and hspace for compact layout

# Step 7: Save the figure
filenamesave = output_path + filename.split('.')[0] + '_' + variable + '_contour_filled_cont_subplots.png'
fig.savefig(filenamesave, bbox_inches='tight')
print(f'Figure saved in: {filenamesave}')


#####################

# Plot classes separately

# Create separate plots for each class
for class_label in merged_tsne_variable_df['label'].unique():
    # Filter the data for the current class
    class_data = merged_tsne_variable_df[merged_tsne_variable_df['label'] == class_label]
    class_data_cot = merged_tsne_variable_df_cot[merged_tsne_variable_df_cot['label'] == class_label]
    
    # Drop rows where 'Component_1', 'Component_2', or the variable of interest has NaN values
    class_data = class_data.dropna(subset=['Component_1', 'Component_2', variable])
    class_data_cot = class_data_cot.dropna(subset=['Component_1', 'Component_2', 'cot'])

    #Extract the indices from both DataFrames
    indices_class_data = class_data.index
    indices_class_data_cot = class_data_cot.index

    # Find the common indices
    common_indices = indices_class_data.intersection(indices_class_data_cot)

    # Select the rows from both DataFrames that correspond to the common indices
    class_data = class_data.loc[common_indices]
    
    # Skip if there's no data after filtering
    if class_data.empty:
        print(f"Skipping class {class_label} due to missing data.")
        continue
    
    # KDE-based filtering (optional: adjust percentile as needed)
    kde = gaussian_kde(class_data[['Component_1', 'Component_2']].T)
    kde_values = kde(class_data[['Component_1', 'Component_2']].T)
    kde_threshold = np.percentile(kde_values, 25)
    class_data = class_data[kde_values >= kde_threshold]
    
    if class_data.empty:
        print(f"Skipping class {class_label} due to insufficient points after KDE filtering.")
        continue
    
    # Grid interpolation for the current class
    grid_x, grid_y = np.mgrid[
        class_data['Component_1'].min():class_data['Component_1'].max():grid_res * 1j, 
        class_data['Component_2'].min():class_data['Component_2'].max():grid_res * 1j
    ]
    
    grid_z = griddata(
        (class_data['Component_1'], class_data['Component_2']),
        class_data[variable],
        (grid_x, grid_y),
        method='linear',
        fill_value=np.nan
    )
    
    if np.isnan(grid_z).all():
        print(f"Skipping class {class_label} due to all NaN values after interpolation.")
        continue
    
    # Create a new figure for each class
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if variable == 'cot':
    
        contour = ax.contourf(
            grid_x, grid_y, grid_z, alpha=0.6, cmap=cmap,
            levels=np.linspace(vmin, 20, 100), extend='max'
        )
    else:    
        # Step 3: Plot contourf for the variable (same color map for all subplots)
        contour = ax.contourf(grid_x, grid_y, grid_z, cmap=cmap, alpha=0.6, levels=np.linspace(vmin, vmax, 100))

    
    # # Overlay actual points
    # scatter = ax.scatter(6    #     class_data['Component_1'], 
    #     class_data['Component_2'], 
    #     c=class_data[variable], 
    #     cmap=cmap, 
    #     edgecolor='k', 
    #     s=50, 
    #     vmin=vmin, vmax=vmax
    # )

    # # Step 2: Plot the KDE contours for each class
    # sns.kdeplot(
    #     x=class_data['Component_1'],
    #     y=class_data['Component_2'],
    #     ax=ax,
    #     levels=[0.95],  #A vector argument must have increasing values in [0, 1]. Levels correspond to iso-proportions of the density: e.g., 20% of the probability mass will lie below the contour drawn for 0.2. 
    #     linewidths=1.,
    #     color= 'magenta', #colors_per_class1_names[str(int(class_label))],  # Use color corresponding to the class
    #     alpha=1.,
    #     label=f'Class {int(class_label)}'
    # )

    
    # Add title and labels
    ax.set_title(f'Class {int(class_label)}', fontsize=14)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    
    # Add colorbar with shared range
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label(f'{variable} ({info_var["unit"] if "unit" in info_var else ""})', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Save the individual plot
    output_file = f"{output_path}{filename.split('.')[0]}_class_{int(class_label)}_{variable}_plot.png"
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    print(f"Saved plot for class {class_label}: {output_file}")