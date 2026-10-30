"""
old color combination for EUMETSAT talk 2025
colors_per_class1_names = {
    '0': 'darkgray',
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green',
    '9': 'goldenrod',
    '10': 'magenta',
    '11': 'dodgerblue',
    '12': 'darkorange',
    '13': 'olive',
    '14': 'crimson'
}

class_groups = {
        'Convection': [2, 3, 4],
        'Overcast': [5, 6, 7],
        'Broken Clouds': [0, 1, 8],
    }
"""

colors_per_class1_names = {
    '0': 'green',
    '1': 'orangered',
    '2': 'darkslategrey',
    '3': 'orange',
    '4': 'paleturquoise',
    '5': 'navajowhite',
    '6': 'orchid',
    '7': 'royalblue',
    '8': 'crimson',
    '9': 'goldenrod',
}

colors_per_class_codes_grl = {

    # daytime: warm colors
    '5': "#A43A00", # dark vermillion
    '6': "#E69F00", # orange
    '7': "#F0E442", # yellow

    # nighttime: blue family only
    '0': "#332288", # navy blue
    '8': "#808080", # medium grey
    '1': "#56B4E9", # sky blue
    '4': "#A7D5F2", # light sky blue derived from the blue family

    # anytime: green / teal family
    '2': "#009E73", # bluish green
    '3': "#117733", # dark green
    '9': "#66CCAA", # light teal
}

class_groups_diurnal_cycle = {
    'day': [5, 6, 7 ],
    'night': [0,1,4,8],
    'anytime': [2,3,9]
}

class_groups = {
        'Convection': [5, 6, 7, 8, ],
        'Overcast': [4, 7],
        'Broken Clouds': [0, 2],
    }



# define classes of interest for extreme events derived from analysis of the plots 
extreme_event_classes = {
    "all": [0, 1, 5, 6, 7, 8],
    "growing": [ 5, 6, 7], 
    "dissipating": [0, 1, 8],
}

