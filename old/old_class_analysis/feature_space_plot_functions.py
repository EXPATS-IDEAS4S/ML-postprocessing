import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib import colors as mcolors 
import seaborn as sns

# Define the color names for each class
colors_per_class1_names = {
    '0': 'darkgray', 
    '1': 'black',
    '2': 'peru',
    '3': 'darkorange',
    '4': 'olivedrab',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green'
}


def name_to_rgb(color_name):
    """
    Convert a color name to its RGB representation scaled to 0-255.

    Parameters:
    color_name (str): The name of the color (e.g., 'red', 'blue').

    Returns:
    numpy.ndarray: An array with RGB values scaled to 0-255.
    """
    return np.array(mcolors.to_rgb(color_name)) * 255



def scale_to_01_range(x):
    """
    Scale the input array to the range [0, 1].

    Parameters:
    x (numpy.ndarray): Input array to be scaled.

    Returns:
    numpy.ndarray: Scaled array with values in the range [0, 1].
    """
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range




def compute_plot_coordinates(image, x, y, image_centers_area_size, offset, comp_bound):
    """
    Compute the top-left and bottom-right coordinates for positioning an image
    on a plot, scaled and centered based on the provided bounds and offsets.

    Parameters:
    image (numpy.ndarray): Image array to be plotted.
    x (float): X-coordinate for the image center.
    y (float): Y-coordinate for the image center.
    image_centers_area_size (tuple): Size of the plotting area (width, height).
    offset (tuple): Offset to apply to the coordinates (x_offset, y_offset).
    comp_bound (tuple): Bounding box for scaling x and y (x_min, x_max, y_min, y_max).

    Returns:
    tuple: Coordinates of the top-left and bottom-right corners (tl_x, tl_y, br_x, br_y).
    """
    image_height, image_width, _ = image.shape

    # Normalize x and y to be within the range [0, 1]
    x_min, x_max, y_min, y_max = comp_bound
    x = (x - x_min) / (x_max - x_min)
    y = (y - y_min) / (y_max - y_min)

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size[0] * x) + offset[0]

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size[1] * (1 - y)) + offset[1]

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def scale_image(image, max_image_size):
    """
    Rescale an image proportionally to fit within the specified maximum size.

    Parameters:
    image (numpy.ndarray): Image to be resized.
    max_image_size (int): Maximum allowed size for the image's width or height.

    Returns:
    numpy.ndarray: Rescaled image with preserved aspect ratio.
    """
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label, color_class_rgb):
    """
    Draws a colored rectangle around the image based on the class label.

    Parameters:
    image (numpy.ndarray): Image on which the rectangle will be drawn.
    label (str or int): Class label used to select the rectangle color.
    color_class_rgb (dict): Dictionary mapping class labels to RGB color values.

    Returns:
    numpy.ndarray: Image with a rectangle drawn around its border.
    """

    color = color_class_rgb[label]
    #print(color)
    image_height, image_width, _ = image.shape
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=4)

    return image