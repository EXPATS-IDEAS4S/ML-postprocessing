"""
stats_utils.py

Statistical helper functions for computing percentiles or percentile ranges
(e.g., interquartile range) from arrays of values.

Utilities for computing and filtering categorical variables such as cloud mask (CMA)
and cloud phase (CPH).


"""


import numpy as np

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




def compute_categorical_values(values, var):
    """
    Compute categorical values for the given variable.

    Args:
    values (np.array): The values of the variable.
    var (str): The variable name.

    Returns:
    np.array: The computed categorical values.
    """
    if var == 'cma':
        # Compute the fraction of cloud pixels (value == 1) over total pixels
        total_pixels = len(values)
        cloud_pixels = np.sum(values == 1)
        fraction_cloudy = cloud_pixels / total_pixels if total_pixels > 0 else 0
        values = fraction_cloudy
    elif var == 'cph':
        # Compute the fraction of liquid clouds (value == 1) over cloudy pixels (value 1 or 2)
        cloudy_pixels = np.sum((values == 1) | (values == 2))  # pixels with value 1 or 2
        ice_pixels = np.sum(values == 2)  # ice clouds (value 2)
        
        if cloudy_pixels > 0:
            fraction_ice = ice_pixels / cloudy_pixels
        else:
            fraction_ice = 0  # If no cloudy pixels, set the fraction to 0
        values = fraction_ice
    else:
        raise ValueError('Wrong variable names!')

    if var == 'lightning':
        # calculate total number of lightning in the frame by summing them up
        values = np.nansum(values) # total number of lightning in the frame

    if var == 'radar_prec':
        # calculate total number of radar_prec in the frame by summing  it up 
        values = np.nansum(values) # total mm/h in the frame

    return values



def filter_cma_values(values, cma_values, var_name, cma_filter=True):
    """
    Filter input values based on cloud mask (CMA) values.

    Parameters:
        values (array-like): Array of values to filter.
        cma_values (array-like): Corresponding cloud mask values (1 = cloudy).
        var_name (str): Variable name, e.g., 'precipitation'.
        cma_filter (bool): Whether to apply the cloud mask filter.

    Returns:
        np.ndarray or list: Filtered values or [0] if no valid values remain.
    """
    if not cma_filter:
        return values

    #if the variable to be filtered is the cloud mask itself, return all the variables, since clear sky pixels are still needed to compute cloud cover
    if var_name == 'cma':
        return values

    if var_name == 'precipitation':
        if len(values) == 0:
            return [np.nan]
        else:
            filtered = values[values > 0.1]
    else:
        if len(values) != len(cma_values):
            raise ValueError("Length mismatch: 'values' and 'cma_values' must be the same.")
        filtered = values[cma_values == 1]

    return filtered if len(filtered) > 0 else [0]