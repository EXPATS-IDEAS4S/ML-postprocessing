"""
dict_utils.py

Utility functions for managing dictionary-based data structures and labels.
Includes concatenating values into dictionaries and extending label lists
based on dataset entries.
"""


import numpy as np

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



def extend_labels(values, labels, row, column):
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
        labels.extend([row[column]] * len(values))  # Use len(values) for multiple valid entries
    else: # In case of a single value 
        labels.extend([row[column]])  # If only one valid entry

    return labels