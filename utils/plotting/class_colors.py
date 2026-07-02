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
    '4': 'deepskyblue',
    '5': 'navajowhite',
    '6': 'orchid',
    '7': 'royalblue',
    '8': 'crimson',
    '9': 'goldenrod',
}



class_groups = {
        'Convection': [1, 3, 5, 8, 6],
        'Overcast': [4, 7],
        'Broken Clouds': [0, 2],
    }



# define classes of interest for extreme events derived from analysis of the plots 
extreme_event_classes = {
    "all": [0, 1, 5, 6, 7, 8],
    "growing": [ 5, 6, 7], 
    "dissipating": [0, 1, 8],
}

