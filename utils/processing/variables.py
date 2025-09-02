"""
variables.py

Utility functions for managing variable metadata used in datasets.
Provides mappings between variable names, their long names, units,
scale information, and directional properties.
"""




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
    elif data_type=='cmsaf':
        vars = ['cot','cth','cma','cph']
        vars_long_name = ['cloud optical thickness', 'cloud top height', 'cloud mask', 'cloud phase']
        vars_units = [None , 'm', None, None]
        vars_logscale = [False, False, False, False]
        vars_dir = ['incr','incr', 'incr', 'incr']
    elif data_type=='imerg':
        vars = ['precipitation']
        vars_long_name = ['precipitation']
        vars_units = ['mm/h']
        vars_logscale = [False]
        vars_dir = ['incr']
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
    






def get_variable_info(var_name):
    """
    Retrieves information for a specific variable.

    Parameters:
    -----------
    var_name : str
        The specific variable to retrieve information for (e.g., 'cot').

    Returns:
    --------
    var_info : dict
        Dictionary containing the variable's long name, unit, logscale, and direction.
        Returns None if the variable is not found.
    """

    variables = {
        'cot':  {'long_name': 'cloud optical thickness', 'unit': None,   'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},   
        'cth':  {'long_name': 'cloud top height',       'unit': 'm',     'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'cma':  {'long_name': 'cloud cover',            'unit': None,    'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'cph':  {'long_name': 'ice ratio',           'unit': None,    'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'precipitation': {'long_name': 'precipitation', 'unit': 'mm/h',  'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'hour': {'long_name': 'hour',                  'unit': 'UTC',   'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'month': {'long_name': 'month',                'unit': None,    'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'lat_mid': {'long_name': 'latitude middle point', 'unit': '°N', 'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100},
        'lon_mid': {'long_name': 'longitude middle point', 'unit': '°E', 'logscale': False, 'direction': 'incr', 'vmin': 0, 'vmax': 100}
    }

    return variables.get(var_name, None)  # Returns None if var_name is not found


