import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def pick_variable(data_type):
    """
    Selects variable information based on the provided data type.

    Parameters:
    -----------
    data_type : str
        The type of data to retrieve variables for. Can be 'continuous', 'topography', or 'era5-land'.

    Returns:
    --------
    vars : list of str
        List of variable names for the selected data type.
    vars_long_name : list of str
        Long descriptive names for the variables.
    vars_units : list of str
        Units corresponding to each variable (may include None for dimensionless variables).
    vars_logscale : list of bool
        Indicates whether each variable should be plotted on a logarithmic scale.
    vars_dir : list of str
        Direction of each variable ('incr' for increasing, 'decr' for decreasing).

    Raises:
    -------
    SystemExit
        If an invalid data type is provided.
    """
    if data_type=='continuous':
        vars = ['cwp', 'cot','ctt', 'ctp', 'cth', 'cre']
        vars_long_name = ['cloud water path', 'cloud optical thickness', 'cloud top temperature', 'cloud top pressure', 'cloud top height', 'cloud particle effective radius']
        vars_units = ['kg/m^2', None , 'K', 'hPa', 'm', 'm']
        vars_logscale = [False, False, False, False, False, False]
        vars_dir = ['incr','incr', 'decr','decr','incr', 'incr']
    elif data_type == 'topography':
        vars = ['DEM', 'landseamask']
        vars_long_name = ['digital elevation model', 'land-sea mask']
        vars_units = ['m', None ]
        vars_logscale = [False, False]
        vars_dir = ['incr','incr']
    elif data_type == 'era5-land':
        vars = ['t2m', 'snowc','u10','v10','sp','tp']
        vars_long_name = ['2-m temperature', 'snow cover','10-m u wind speed', '10-m v wind speed', 'surface pressure','total precipitation']
        vars_units = ['K', '%','m/s','m/s','Pa','m' ]
        vars_logscale = [False, False,False,False,False,False]
        vars_dir = ['incr','incr','incr','incr','incr','incr']
    elif data_type =='space-time':
        vars = ['hour', 'month','lat_mid','lon_mid']
        vars_long_name = ['hour', 'month','latitude middle point','longitude middle point']
        vars_units = ['UTC', None,'°N','°E']
        vars_logscale = [False, False,False,False]
        vars_dir = ['incr','incr','incr','incr']  
    elif data_type =='categorical':
        vars = ['cma', 'cph']
        vars_long_name = ['cloud mask', 'cloud phase']
        vars_units = [None, None]
        vars_logscale = [False, False]
        vars_dir = ['incr','incr']  
    else:
        print('wrong data type name!')
        exit()

    return vars, vars_long_name, vars_units, vars_logscale, vars_dir



def get_variable_info(data_type, var_name):
    """
    Retrieves information for a specific variable based on the provided data type.

    Parameters:
    -----------
    data_type : str
        The type of data to retrieve variables for.
    var_name : str
        The specific variable to retrieve information for (e.g., 'ctp').

    Returns:
    --------
    var_info : dict
        Dictionary containing the variable's long name, unit, logscale, and direction.
    """
    # Call the pick_variable function to retrieve lists for the given data_type
    vars, vars_long_name, vars_units, vars_logscale, vars_dir = pick_variable(data_type)

    # Check if the requested variable exists in the vars list
    if var_name in vars:
        index = vars.index(var_name)  # Get the index of the variable
        
        # Create a dictionary with all the relevant information
        var_info = {
            'variable': var_name,
            'long_name': vars_long_name[index],
            'unit': vars_units[index],
            'logscale': vars_logscale[index],
            'direction': vars_dir[index]
        }
        return var_info
    else:
        return f"Variable '{var_name}' not found in the selected data type '{data_type}'."


def select_ds(var, datasets):
    """
    Selects the dataset that contains the specified variable.

    Parameters:
    -----------
    var : str
        The variable name to search for in the datasets.
    datasets : list of xarray.Dataset
        A list of datasets to search for the variable.

    Returns:
    --------
    xarray.Dataset or None
        The dataset containing the variable, or None if the variable is not found.
    """
    for ds in datasets:
        if var in ds.variables:
            return ds
    return None  # Return None if the variable is not found in any dataset


def get_time_from_ds(ds):
    """
    Extracts the year, month, day, and hour from the time variable in the dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the time variable.

    Returns:
    --------
    year : int
        The year extracted from the dataset's time.
    month : int
        The month extracted from the dataset's time.
    day : int
        The day extracted from the dataset's time.
    hour : int
        The hour extracted from the dataset's time.
    """
    time_crop = ds.time.values

    # Extract year, month, and hour from the crop time
    time_crop_dt = pd.to_datetime(time_crop)
    year = time_crop_dt.year
    month = time_crop_dt.month
    day = time_crop_dt.day
    hour = time_crop_dt.hour

    return year, month, day, hour


def find_latlon_boundaries_from_ds(ds_crops):
    """
    Extracts latitude and longitude boundaries from the crop dataset.

    Parameters:
    -----------
    ds_crops : xarray.Dataset
        The dataset containing latitude and longitude values.

    Returns:
    --------
    lat_min, lat_max, lon_min, lon_max : float
        The minimum and maximum latitude and longitude values defining the bounding box.
    """
    # Get the latitude and longitude values from the crop dataset
    lats_crops = ds_crops.lat.values
    lons_crops = ds_crops.lon.values
    
    # Define the bounding box for the crop dataset (min/max lat and lon)
    lat_min, lat_max = lats_crops.min(), lats_crops.max()
    lon_min, lon_max = lons_crops.min(), lons_crops.max()

    return lat_min, lat_max, lon_min, lon_max



def compute_percentile(values,stat):
    """
    Computes a percentile or range of percentiles (e.g., interquartile range) from a given array.

    Parameters:
    -----------
    values : array-like
        The input data from which to compute the percentiles.
    stat : int, list, or str
        The percentile(s) to compute. If a string in the form 'lower-upper' (e.g., '25-75'), 
        it computes the range between the specified percentiles (e.g., IQR).

    Returns:
    --------
    values : float or array-like
        The computed percentile(s) or range of percentiles.
    """
    if isinstance(stat, str) and '-' in stat:
        # Split the string into two percentile values
        lower, upper = map(int, stat.split('-'))
        
        # Compute the percentiles specified in the string (e.g., '25-75')
        q_upper, q_lower = np.nanpercentile(values, [upper, lower])
        
        # Compute the range (IQR or custom range)
        values = q_upper - q_lower
    else:
        # Compute the percentile if stat is a number or list of percentiles
        values = np.nanpercentile(values, stat)
    return values
    

def concatenate_values(values, var, data_dict):   
    """
    Appends or extends values to the specified key in a dictionary.

    Parameters:
    -----------
    values : array-like or single value
        The data to be added to the dictionary. If array-like, it extends the list;
        if a single value, it appends to the list.
    var : str
        The key in the dictionary where the values will be added.
    data_dict : dict
        The dictionary containing lists for different variables.

    Returns:
    --------
    data_dict : dict
        The updated dictionary with concatenated values for the specified key.
    """ 
    # Ensure values is a list or array before extending
    if isinstance(values, np.ndarray) or isinstance(values, list):
        data_dict[var].extend(values)
    else:
        data_dict[var].append(values)  # For single values

    return data_dict



def extend_labels(values, labels, row):
    """
    Extends the labels list based on the number of valid entries in the dataset.

    Parameters:
    -----------
    values : array-like or single value
        The data used to determine how many labels to extend.
    labels : list
        The list of labels to be extended.
    row : dict-like
        A row of data, containing the label to be added.

    Returns:
    --------
    labels : list
        The updated list of labels, extended according to the number of entries in `values`.
    """
    # Extend labels based on the number of valid entries for this dataset
    if isinstance(values, np.ndarray):
        labels.extend([row['label']] * len(values))  # Use len(values) for multiple valid entries
    else: # In case of a single value 
        labels.extend([row['label']])  # If only one valid entry

    return labels

 
      
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



