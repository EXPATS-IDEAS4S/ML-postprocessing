"""
dataset_utils.py

Helper functions for extracting information from xarray datasets.
Includes dataset selection, time extraction, and geographic boundary retrieval.
"""


import pandas as pd

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

