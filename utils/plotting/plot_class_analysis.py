"""

Plotting utilities for visualizing variable distributions and class-based data.
Supports boxplots, countplots, and ridge (joy) plots with customization options
for labels, scales, and saving figures.
Includes functions for visualizing regions with Cartopy maps.
"""


import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

 
      
def plot_single_vars(df, n_subsample, var, long_name, unit, direction, scale, output_path, run_name, sampling_type, stat, legend=False, hue=None, boxplot=True):
    """
    Plots a boxplot for a single variable, with optional grouping by hue, and saves the figure.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to plot.
    n_subsample : int
        The number of samples being plotted.
    var : str
        The column name for the variable to be plotted.
    long_name : str
        The descriptive name of the variable for labeling the plot.
    unit : str
        The unit of the variable, included in the y-axis label (optional).
    direction : str
        If 'decr', the y-axis is inverted.
    scale : bool
        If True, the y-axis is set to a logarithmic scale.
    output_path : str
        The path where the plot will be saved.
    run_name : str
        A name or identifier for the current run, used in the file name.
    sampling_type : str
        The type of sampling used, included in the file name.
    stat : str
        A string representing additional statistics or identifiers for the plot.
    legend : bool or str, optional (default=False)
        If True or a string is provided, a legend is added to the plot. The legend title will be set to the value of this parameter.
    hue : str, optional
        The column name to group data by when plotting multiple groups (optional for hue-based boxplots).

    Returns:
    --------
    None
        The function saves the boxplot as a PNG file and closes the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    #sns.boxplot(data=df, x='label', y=var, ax=ax, showfliers=False)
    if boxplot:
        sns.boxplot(data=df, x='label', y=var, hue=hue, ax=ax, showfliers=False, palette='Set2')
        if unit:
            plt.ylabel(f'{var} ({unit})',fontsize=14)
        else:
            plt.ylabel(f'{var}',fontsize=14)
    else:
        sns.countplot(x='label', hue=var, data=df, ax=ax,
                  palette='Set2')#, hue_order=hue_order)
        ax.set_ylabel('Counts',fontsize=14)

    plt.title(f'{long_name} ({var}) - {n_subsample} samples',fontsize=14, fontweight='bold')
    plt.xlabel('Cloud Class Label',fontsize=14)

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    if scale:
        plt.yscale('log')
        # Exclude zero values
        non_zero_values = df[df[var] > 0][var]
        # Find the minimum of the non-zero values
        min_non_zero = non_zero_values.min()
        print(min_non_zero)
        #set y lim
        plt.ylim(bottom=min_non_zero)

    # Reverse y axis if direction is 'decr'
    if direction == 'decr':
        ax.invert_yaxis()

    if legend:
        # Move the legend outside on the right and center it vertically
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., title=legend, fontsize=12, title_fontsize=14)
    
    # Save the figure
    if boxplot:
        fig.savefig(f'{output_path}{var}_boxplot_{run_name}_{n_subsample}_{sampling_type}_{stat}.png', bbox_inches='tight')
    else:
        fig.savefig(f'{output_path}{var}_countplot_{run_name}_{n_subsample}_{sampling_type}_{stat}.png', bbox_inches='tight')
    print(f'Figure saved: {output_path}{var}_{n_subsample}.png')

    #Close Fig
    plt.close()



def plot_joyplot(df, class_label, variable_name, long_name, unit, n_subsample, output_path, run_name, sampling_type, legend=False):
    """
    Plots a joy plot (ridge plot) showing the distribution of a variable for the specified number of closest crops in a class,
    ordered by distance to the centroid, and saves the figure.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to plot. Expected columns: ['crop_name', 'distance', 'label', variable_name].
    class_label : str or int
        The class name or ID to filter data by.
    variable_name : str
        The column name for the variable to be plotted.
    long_name : str
        The descriptive name of the variable for labeling the plot.
    unit : str
        The unit of the variable, included in the x-axis label (optional).
    n_subsample : int
        The number of closest crops to plot.
    output_path : str
        The path where the plot will be saved.
    run_name : str
        A name or identifier for the current run, used in the file name.
    sampling_type : str
        The type of sampling used, included in the file name.
    legend : bool, optional (default=False)
        If True, a legend is added to the plot.
    """
    # Filter data for the specified class label
    class_df = df[df['label'] == class_label]

    # Sort by distance to order crops from closest to farthest
    class_df = class_df.sort_values(by='distance', ascending=False) #cosine distance, higher values (~1) means closer to centroid

    # Get list of names
    crop_paths = class_df['path'].to_list()  # Convert to list if needed
    crop_names = [os.path.basename(path).split('.')[0] for path in crop_paths]
    class_df['crop_name'] = crop_names  # Add to DataFrame

    # Get the first n unique crop names
    unique_crop_names = class_df['crop_name'].unique()[:n_subsample]
    # Filter the dataframe to keep rows with the first n unique crop names
    class_df = class_df[class_df['crop_name'].isin(unique_crop_names)]

    # Keep only the relevant columns
    class_df = class_df[['crop_name', 'distance', variable_name]]

    # Plot using seaborn's ridge plot (FacetGrid with kdeplot)
    g = sns.FacetGrid(class_df, row='crop_name', hue='distance', aspect=20, height=0.1, palette='viridis')
    g.map(sns.kdeplot, variable_name, bw_adjust=0.5, fill=True, alpha=0.6)
    g.map(sns.kdeplot, variable_name, clip_on=False, color="w", lw=1, bw_adjust=.5)

    # Remove axes details that don't play well with overlap
    g.set_titles('')  
    g.set(yticks=[], xticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Set x-axis labels only for the first plot
    #g.set_xlabels('')  # Remove x-axis labels from all plots
    #g.axes.flat[0].set_xlabel(f"{variable_name} ({unit})" if unit else long_name)  # Set x-axis label for the first plot

    # passing color=None to refline() uses the hue mapping
    #g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)
     
    #Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    #g.map(label, variable_name)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
    
    # Add a legend if specified
    if legend:
        g.add_legend(title="Crop")

    # Set a main title for the plot
    g.figure.subplots_adjust(top=0.95)
    g.figure.suptitle(f"{long_name} crop distribution ordered by distance (Class: {class_label})")

    # Save the plot
    output_file = f"{output_path}joyplot_{variable_name}_{sampling_type}_{n_subsample}_{class_label}_{run_name}.png"
    g.savefig(output_file, bbox_inches='tight')





def plot_cartopy_map(output_path, latmin, lonmin, latmax, lonmax, n_divs=5):
    """
    Plots a Cartopy map with country borders and a grid of vertical and horizontal lines.

    Parameters:
    -----------
    latmin : float
        Minimum latitude of the map.
    lonmin : float
        Minimum longitude of the map.
    latmax : float
        Maximum latitude of the map.
    lonmax : float
        Maximum longitude of the map.
    n_divs : int, optional (default=5)
        Number of vertical and horizontal grid lines.
    """

    # Define the figure and axis with a PlateCarree projection
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set map extent
    ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.BORDERS, linewidth=1)  # Country borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)  # Coastlines
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)  # Land color
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)  # Ocean color

    # Generate grid lines
    lon_ticks = np.linspace(lonmin, lonmax, n_divs+1)
    lat_ticks = np.linspace(latmin, latmax, n_divs+1)

    # Plot vertical grid lines (longitude)
    for lon in lon_ticks:
        ax.plot([lon, lon], [latmin, latmax], transform=ccrs.PlateCarree(), color='black', linestyle='--', alpha=0.7)

    # Plot horizontal grid lines (latitude)
    for lat in lat_ticks:
        ax.plot([lonmin, lonmax], [lat, lat], transform=ccrs.PlateCarree(), color='black', linestyle='--', alpha=0.7)

    # Add custom tick labels at the bottom (longitude)
    ax.set_xticks(lon_ticks)  
    ax.set_xticklabels([f"{lon:.1f}°E" if lon >= 0 else f"{-lon:.1f}°W" for lon in lon_ticks], fontsize=10)
    ax.xaxis.set_ticks_position('bottom')  # Only at the bottom

    # Add custom tick labels on the left (latitude)
    ax.set_yticks(lat_ticks)  
    ax.set_yticklabels([f"{lat:.1f}°N" if lat >= 0 else f"{-lat:.1f}°S" for lat in lat_ticks], fontsize=10)
    ax.yaxis.set_ticks_position('left')  # Only on the left

    # Hide top and right tick labels
    ax.tick_params(top=False, right=False)

    # Save the figure
    plt.savefig(f'{output_path}map_divided_{n_divs}.png', bbox_inches='tight', dpi=300)