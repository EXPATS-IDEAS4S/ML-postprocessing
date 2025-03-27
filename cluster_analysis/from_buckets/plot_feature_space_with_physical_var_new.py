import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
import cmcrameri.cm as cmc
import os
import sys

from aux_functions_from_buckets import get_variable_info
sys.path.append(os.path.abspath("/home/Daniele/codes/visualization/cluster_analysis"))

reduction_method = 'tsne' #'tsne
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3' #all visualization were made with random state 3
sampling_type = 'all'  # Options: 'random', 'closest', 'farthest', 'all'
cmap = cmc.imola #RdYlBu_r'

# Path to fig folder for outputs
output_path = f'/data1/fig/{run_name}/{sampling_type}/'

#Open df with tsne and variables merged together
df_merged = pd.read_csv(f'{output_path}merged_tsne_variables_{run_name}_{sampling_type}_{random_state}.csv')
print(df_merged)

n_subsample = None #put None if no subsampling is needed

# Number of classes and subplot grid setup
n_classes = df_merged['label'].nunique()
#print(n_classes)

varlable_list = ['cma-None', 'cph-None']#['cot-50', 'cth-50', 'cot-99', 'cth-99', 'cma-None', 'cph-None', 'precipitation-50', 'precipitation-99']
for variable in varlable_list:
    print(variable)
    # Extract the statistic from the variable name
    stat = variable.split('-')[1]
    var_name = variable.split('-')[0]
    # Get variable information
    info_var = get_variable_info(variable.split('-')[0])
    print(info_var)

    rows, cols = 3, 3  # For 9 classes, we want a 3x3 grid

    # Set up a 3x3 grid of subplots
    fig, axs = plt.subplots(rows, cols, figsize=(10, 8), sharex=True, sharey=True)

    # Flatten the axis array for easy iteration (because axs is a 2D array)
    axs = axs.flatten()

    # Grid resolution for interpolation
    grid_res = 100

    # Get the min and max values of the variable to ensure consistent color mapping across all subplots
    vmin, vmax = df_merged[variable].min(), df_merged[variable].max()
    print(vmin, vmax)


    # Create a new directory for the variable 
    if n_subsample:
        output_dir = f'{output_path}/physical_embeddings/{variable}_{n_subsample}/'
    else:
        output_dir = f'{output_path}/{variable}/'
    os.makedirs(output_dir, exist_ok=True)


    # # Step 1: Loop through unique classes and plot in each subplot
    # for i, class_label in enumerate(df_merged['label'].unique()):
    #     ax = axs[i]  # Select the current subplot
        
    #     # Filter the data for the current class
    #     class_data = df_merged[df_merged['label'] == class_label]
        
    #     # Drop rows where 'Component_1', 'Component_2', or the variable of interest has NaN values
    #     class_data = class_data.dropna(subset=['Component_1', 'Component_2', variable])
        
    #     # Ensure there's data to plot after removing NaNs
    #     if class_data.empty:
    #         print(f"Skipping class {class_label} due to missing data.")
    #         continue

    #     # Calculate the KDE for the points in Component_1 and Component_2
    #     kde = gaussian_kde(class_data[['Component_1', 'Component_2']].T)
    #     kde_values = kde(class_data[['Component_1', 'Component_2']].T)

    #     # Determine the 25th percentile of the KDE values
    #     kde_threshold = np.percentile(kde_values, 25)

    #     # Filter out points below the 25th percentile
    #     class_data = class_data[kde_values >= kde_threshold]

    #     # Ensure there's data to plot after filtering
    #     if class_data.empty:
    #         print(f"Skipping class {class_label} due to insufficient points after KDE filtering.")
    #         continue
            
    #     # Create a grid on Component_1 and Component_2 space
    #     grid_x, grid_y = np.mgrid[
    #         class_data['Component_1'].min():class_data['Component_1'].max():grid_res*1j, 
    #         class_data['Component_2'].min():class_data['Component_2'].max():grid_res*1j
    #     ]
        
    #     # Check if the ranges of grid_x or grid_y are valid (non-empty)
    #     if grid_x.size == 0 or grid_y.size == 0:
    #         print(f"Skipping class {class_label} due to invalid grid.")
    #         continue

    #     # Interpolate variable values onto the grid
    #     grid_z = griddata(
    #         (class_data['Component_1'], class_data['Component_2']),  # x, y coordinates
    #         class_data[variable],  # variable values
    #         (grid_x, grid_y),  # grid points where we want to interpolate
    #         method='linear',  # Interpolation method
    #         fill_value=np.nan  # Handle missing values by filling NaNs
    #     )
        
    #     # Ensure grid_z is not entirely NaN
    #     if np.isnan(grid_z).all():
    #         print(f"Skipping class {class_label} due to all NaN values after interpolation.")
    #         continue
        
    #     if variable == 'cot':
        
    #         contour = ax.contourf(
    #             grid_x, grid_y, grid_z, alpha=0.6, cmap=cmap,
    #             levels=np.linspace(vmin, 20, 100), extend='max'
    #         )
    #     else:    
    #         # Step 3: Plot contourf for the variable (same color map for all subplots)
    #         contour = ax.contourf(grid_x, grid_y, grid_z, cmap=cmap, alpha=0.5, levels=np.linspace(vmin, vmax, 100))
        
        
    #     # Step 2: Plot the KDE contours for each class
    #     sns.kdeplot(
    #         x=class_data['Component_1'],
    #         y=class_data['Component_2'],
    #         ax=ax,
    #         levels=[0.95],  #A vector argument must have increasing values in [0, 1]. Levels correspond to iso-proportions of the density: e.g., 20% of the probability mass will lie below the contour drawn for 0.2. 
    #         linewidths=1.,
    #         color= 'magenta', 
    #         alpha=1.,
    #         label=f'Class {int(class_label)}'
    #     )

        
    #     # Set title for each subplot
    #     #ax.set_title(f'Class {int(class_label)}', fontsize=12)

    #     # Remove individual axis labels
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')

    # # Step 4: Add shared x and y labels for the entire figure
    # fig.text(0.5, 0.05, 'Component 1', ha='center', fontsize=15)
    # fig.text(0.05, 0.5, 'Component 2', va='center', rotation='vertical', fontsize=15)

    # cbar = fig.colorbar(contour, ax=axs, orientation='vertical', fraction=0.03, pad=0.2)
    # unit = info_var['unit']
    # if unit:
    #     cbar.set_label(f'{unit}', fontsize=15)
    # else:
    #     cbar.set_label(f'', fontsize=15)

    # # Step 7: Format colorbar ticks to 2 decimal places
    # cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    # # Step 6: Increase tick label size and thickness for the colorbar and axes
    # for ax in axs:
    #     ax.tick_params(axis='both', which='major', labelsize=10, width=2)  # Adjust axis tick size and thickness
    # cbar.ax.tick_params(labelsize=12, width=2)  # Adjust colorbar label size and thickness

    # # Step 8: Add an overall title with larger bold font
    # if variable == 'cma':
    #     fig.suptitle(f'{reduction_method} 2d embedding distribution by class: cloud fraction', fontsize=16, fontweight='bold')
    # elif variable == 'cph':
    #     fig.suptitle(f'{reduction_method} 2d embedding distribution by class: ice cloud fraction', fontsize=16, fontweight='bold')
    # else:
    #     fig.suptitle(f'{reduction_method} 2d Embedding Distribution by Class: {variable} - {stat}th quantile', fontsize=16, fontweight='bold')

    # # Step 9: Adjust spacing between subplots and make them more compact
    # plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.95, right=0.85)  # Decrease wspace and hspace for compact layout

    # # Step 7: Save the figure
    # filenamesave = output_path + run_name + '_' + variable + '_contour_filled_cont_subplots.png'
    # fig.savefig(filenamesave, bbox_inches='tight')
    # print(f'Figure saved in: {filenamesave}')

    #####################

    # Plot classes separately

    # Create separate plots for each class
    for class_label in df_merged['label'].unique():
        # Filter the data for the current class
        class_data = df_merged[df_merged['label'] == class_label]
        #class_data_cot = df_merged[df_merged['label'] == class_label]

        if n_subsample:
            # Select the closest n_subsample rows based on 'distance'
            # Take the largest values of distance since a cosine is used
            class_data = class_data.nlargest(n_subsample, 'distance')
           
        
        # Drop rows where 'Component_1', 'Component_2', or the variable of interest has NaN values
        class_data = class_data.dropna(subset=['Component_1', 'Component_2', variable])
        #class_data_cot = class_data_cot.dropna(subset=['Component_1', 'Component_2', 'cot'])

        #Extract the indices from both DataFrames
        #indices_class_data = class_data.index
        #indices_class_data_cot = class_data_cot.index

        # Find the common indices
        #common_indices = indices_class_data.intersection(indices_class_data_cot)

        # Select the rows from both DataFrames that correspond to the common indices
        #class_data = class_data.loc[common_indices]
        
        # Skip if there's no data after filtering
        if class_data.empty:
            print(f"Skipping class {class_label} due to missing data.")
            continue
        
        # KDE-based filtering (optional: adjust percentile as needed)
        #kde = gaussian_kde(class_data[['Component_1', 'Component_2']].T)
        #kde_values = kde(class_data[['Component_1', 'Component_2']].T)
        #kde_threshold = np.percentile(kde_values, 25)
        #class_data = class_data[kde_values >= kde_threshold]
        
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
        
        if var_name == 'cot':
        
            contour = ax.contourf(
                grid_x, grid_y, grid_z, alpha=0.6, cmap=cmap,
                levels=np.linspace(vmin, 20, 100), extend='max'
            )
        elif var_name == 'precipitation':
             contour = ax.contourf(
                grid_x, grid_y, grid_z, alpha=0.6, cmap=cmap,
                levels=np.linspace(vmin, 10, 100), extend='max'
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
        if info_var['unit'] == 'None':
            cbar.set_label(f'{info_var["long_name"]}', fontsize=12)
        else:
            cbar.set_label(f'{info_var["long_name"]} ({info_var["unit"]})', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Save the individual plot
        output_file = f"{output_dir}{run_name}_class_{int(class_label)}_{variable}_plot.png"
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved plot for class {class_label}: {output_file}")