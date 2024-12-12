import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from matplotlib.colors import CSS4_COLORS
from matplotlib import colors as mcolors  # Correct import for colors
import re
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os


# Convert color names to RGB values
def name_to_rgb(color_name):
    return np.array(mcolors.to_rgb(color_name)) * 255


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset, comp_bound):
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
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label, colors_per_class1_rgb):
    color = colors_per_class1_rgb[label]
    #print(color)
    image_height, image_width, _ = image.shape
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=4)

    return image

def extract_hour(location):
    match = re.search(r'_(\d{2}):\d{2}_', location)
    if match:
        return int(match.group(1))  # Extract and convert the hour to an integer
    return None


def add_trajectory_case_study(df_subset, ax, fig, cmap, colorbar=False):
    
    #cmap = plt.cm.viridis  # You can choose any colormap you like
    norm = Normalize(vmin=0, vmax=23)  # Normalize the hour values between 0 and 23

    # Map the hour to colors
    df_subset['color_by_hour'] = df_subset['hour'].apply(lambda x: cmap(norm(x)) if x is not None else (0, 0, 0, 0))

    # Step 3: Create segments for the LineCollection
    points = np.array([df_subset['Component_1'], df_subset['Component_2']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Step 4: Filter segments for case study points
    case_study_points = df_subset[df_subset['case_study'] == True]
    case_study_segments = np.array([case_study_points['Component_1'], case_study_points['Component_2']]).T.reshape(-1, 1, 2)
    case_study_segments = np.concatenate([case_study_segments[:-1], case_study_segments[1:]], axis=1)

    # Colors for the case study segments based on the hour
    case_study_colors = case_study_points['color_by_hour'].iloc[:-1].tolist()

    # Add the colored line for the case study
    lc = LineCollection(case_study_segments, colors=case_study_colors, linewidths=10)
    ax.add_collection(lc)

    if colorbar:
        # Add a colorbar for the hour
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for the colorbar
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Hour of the Day', fontsize=16)
        cbar.ax.tick_params(labelsize=14)



def plot_embedding_dots(df_subset1, colors_per_class1_norm, output_path, filename, df_subset2=None):

    # Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    scatter = ax.scatter(df_subset1['Component_1'], df_subset1['Component_2'],
                        c=df_subset1['color'].tolist(), alpha=0.5, s=20)

    if df_subset2 is not None:
        add_trajectory_case_study(df_subset2, ax, fig, plt.cm.gray)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_per_class1_norm[label], markersize=10)
            for label in colors_per_class1_norm.keys()]

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Save figure
    fig.savefig(output_path + filename.split('.')[0] + '_dots.png', bbox_inches='tight')



def plot_embedding_dots_iterative_case_study(df_subset1, colors_per_class1_norm, output_path, filename, df_subset2, legend=False):
    """
    Plot embedding dots and iteratively add trajectory points with lines,
    considering only points with `case_study=True`.

    Parameters:
    - df_subset1: DataFrame for the full embedding space.
    - colors_per_class1_norm: Dictionary mapping class labels to colors.
    - output_path: Path to save output figures.
    - filename: Base filename for saving plots.
    - df_subset2: DataFrame for the trajectory to plot iteratively.
    """
    # Filter for `case_study=True`
    df_case_study = df_subset2[df_subset2['case_study'] == True].sort_index()
    #print(df_case_study)

    #extract the labels for the classes
    labels = df_subset2['y'].unique()  # Assuming 'label' column contains class labels

    # Plot base embedding
    for i in range(len(df_case_study)):
        # Create figure and scatter plot
        fig, ax = plt.subplots(figsize=(16, 12))
        scatter = ax.scatter(
            df_subset1['Component_1'], 
            df_subset1['Component_2'],
            c=df_subset1['color'].tolist(), 
            alpha=0.1, 
            s=5
        )
        
        # Get current subset of the trajectory
        current_trajectory = df_case_study.iloc[:i+1]
        
        # Extract trajectory components
        x_vals = current_trajectory['Component_1']
        y_vals = current_trajectory['Component_2']
        colors = current_trajectory['color']
        
        # Plot trajectory points
        ax.scatter(
            x_vals, 
            y_vals, 
            c=colors, 
            s=100, 
            edgecolor='k', 
            zorder=5, 
            label='Trajectory Points'
        )
        
        # Plot connecting lines
        if len(current_trajectory) > 1:
            ax.plot(
                x_vals, 
                y_vals, 
                color='black', 
                linewidth=2, 
                alpha=0.8, 
                label='Trajectory'
            )

        #add crop images on the upper left corner
        # Load the image (e.g., .tif)
        image_path = df_case_study.iloc[i]['location']
        image = Image.open(image_path)
        # Convert the image to a format usable by Matplotlib
        offset_image = OffsetImage(image, zoom=1.)  # Adjust `zoom` for size
        # Place the image at a specific position (e.g., upper-left corner)
        ab = AnnotationBbox(
            offset_image,  # The image
            (0.05, 0.95),  # Position in axis coordinates (0, 1 is top-left corner)
            frameon=False,  # No frame around the image
            xycoords='axes fraction'  # Coordinates relative to the axis (0-1)
        )
        ax.add_artist(ab)

        # for label in labels:
        #     class_color = colors_per_class1_norm.get(str(label), 'gray')
        #     compute_percentile_contour_levels(ax, df_subset2, label, class_color, 90, countour=True, contourf=False)

        
        # Add legend
        if legend:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_per_class1_norm[label], markersize=10)
                for label in colors_per_class1_norm.keys()
            ]
            ax.legend(handles, colors_per_class1_norm.keys(), title="Labels", fontsize=12)
        
        # Add timestamp of the last trajectory point to the title
        last_time = current_trajectory.iloc[-1]['location'].split('/')[-1].split('_')[4]
        print(last_time)
        ax.set_title(f"Marche Flood 15.09.22  - Hour: {last_time}", fontsize=16, fontweight='bold')
        
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        # # Set limits to shrink the scatter plot and contour lines
        padding = 0.03  # Add some padding to the edges of the plot
        x_min, x_max = np.min(df_subset2['Component_1']), np.max(df_subset2['Component_1'])
        y_min, y_max = np.min(df_subset2['Component_2']), np.max(df_subset2['Component_2'])
        ax.set_xlim([x_min - padding, x_max + padding])
        ax.set_ylim([y_min - padding, y_max + padding])

        # Add margins around the plot area
        #ax.margins(0.1)  # Adds 5% margin to the x and y axes
        
        # Save the figure
        step_filename = f"{output_path}{filename.split('.')[0]}_trajectory_step_{i+1}.png"
        fig.savefig(step_filename, bbox_inches='tight')
        plt.close(fig)

def plot_embedding_dots_iterative_test_msg_icon(
    df_subset1, colors_per_class1_norm, output_path, filename, df_subset2, legend=False
):
    """
    Plot embedding dots and iteratively add trajectory points with lines
    for both `case_study_msg` and `case_study_icon` in the same plot.

    Parameters:
    - df_subset1: DataFrame for the full embedding space.
    - colors_per_class1_norm: Dictionary mapping class labels to colors.
    - output_path: Path to save output figures.
    - filename: Base filename for saving plots.
    - df_subset2: DataFrame for the trajectories to plot iteratively.
    - legend: Whether to include a legend in the plot.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Filter case study points for MSG and ICON
    df_case_study_msg = df_subset2[df_subset2['case_study_msg'] == True].sort_index()
    df_case_study_icon = df_subset2[df_subset2['case_study_icon'] == True].sort_index()

    # Determine the maximum number of steps to iterate
    max_steps = max(len(df_case_study_msg), len(df_case_study_icon))

    # Plot base embedding
    for i in range(max_steps):
        fig, ax = plt.subplots(figsize=(12, 14))

        #extract time 
        time = df_case_study_msg.iloc[i]['location'].split('/')[-1].split('_')[4]
        print(time)

        # Scatter the full embedding space
        ax.scatter(
            df_subset1['Component_1'], 
            df_subset1['Component_2'],
            c=df_subset1['color'].tolist(), 
            alpha=0.1, 
            s=5
        )

        # Plot MSG trajectory up to step `i`
        if i < len(df_case_study_msg):
            current_msg = df_case_study_msg.iloc[:i+1]
            ax.scatter(
                current_msg['Component_1'], 
                current_msg['Component_2'], 
                c=current_msg['color'],  # Follow the same color coding as the background
                s=120, 
                edgecolor="red", 
                linewidth=2, 
                zorder=5, 
                label='MSG Trajectory Points' if i == 0 else ""
            )
            if len(current_msg) > 1:
                ax.plot(
                    current_msg['Component_1'], 
                    current_msg['Component_2'], 
                    color="red", 
                    linewidth=2, 
                    alpha=0.8, 
                    label='MSG Trajectory' if i == 0 else ""
                )

        # Plot ICON trajectory up to step `i`
        if i < len(df_case_study_icon):
            current_icon = df_case_study_icon.iloc[:i+1]
            ax.scatter(
                current_icon['Component_1'], 
                current_icon['Component_2'], 
                c=current_icon['color'],  # Follow the same color coding as the background
                s=120, 
                edgecolor="black", 
                linewidth=2, 
                zorder=5, 
                label='ICON Trajectory Points' if i == 0 else ""
            )
            if len(current_icon) > 1:
                ax.plot(
                    current_icon['Component_1'], 
                    current_icon['Component_2'], 
                    color="black", 
                    linewidth=2, 
                    alpha=0.8, 
                    label='ICON Trajectory' if i == 0 else ""
                )

        # Add MSG crop image in the upper-left corner

        if i < len(df_case_study_msg):
            # get image path and filename
            folder_msg_images = df_case_study_msg.iloc[i]['msg_image_path']
            msg_image_filename = df_case_study_msg.iloc[i]['location'].split('/')[-1]
            msg_image = Image.open(folder_msg_images + msg_image_filename)
            msg_offset_image = OffsetImage(msg_image, zoom=0.4)
            msg_ab = AnnotationBbox(
                msg_offset_image, 
                (0.2, 0.05),  # Upper-left corner
                frameon=False, 
                xycoords='axes fraction'
            )
            ax.add_artist(msg_ab)
            # Add 'MSG' label above the image
         

        # Add ICON crop image in the lower-left corner
        if i < len(df_case_study_icon):
            folder_icon_images = df_case_study_icon.iloc[i]['icon_image_path']
            icon_image_filename = df_case_study_icon.iloc[i]['location'].split('/')[-1]
            icon_image = Image.open(folder_icon_images + icon_image_filename)
            icon_offset_image = OffsetImage(icon_image, zoom=0.4)
            icon_ab = AnnotationBbox(
                icon_offset_image, 
                (0.8, 0.05),  # Lower-left corner
                frameon=False, 
                xycoords='axes fraction'
            )
            ax.add_artist(icon_ab)
            # Add 'ICON' label above the image

        # Add legend
        if legend:
            handles = [
                plt.Line2D([0], [0], color="red", label="MSG Trajectory"),
                plt.Line2D([0], [0], color="black", label="ICON Trajectory"),
            ]
            ax.legend(handles=handles, title="", fontsize=14, loc="upper right")
            


        # Set title and remove axes spines
        ax.set_title(f"Marche Flood 15.09.22 - {time}", fontsize=20, fontweight='bold')
        ax.title.set_position([0.5, 1.2]) 
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_visible(False)

        # Set limits
        # Set limits
        padding = 0.03
        extra_bottom_space = 0.3  # Proportion of extra space to add at the bottom
        x_min, x_max = np.min(df_subset2['Component_1']), np.max(df_subset2['Component_1'])
        y_min, y_max = np.min(df_subset2['Component_2']), np.max(df_subset2['Component_2'])

        ax.set_xlim([x_min - padding, x_max + padding])
        ax.set_ylim([y_min - padding - (y_max - y_min) * extra_bottom_space, y_max + padding])


        # Save the figure
        step_filename = f"{output_path}{filename.split('.')[0]}_trajectory_step_{i+1}.png"
        fig.savefig(step_filename, bbox_inches='tight')
        plt.close(fig)
        

    print("Plots for both MSG and ICON case studies saved.")





def plot_embedding_crops(indices, selected_images, df_conc, tsne_plot, output_path, filename, image_centers_area_size, offset, max_image_size, min_1, max_1, min_2, max_2, colors_per_class, df_subset2=None):
    #TODO how to plot crops with trajectory and line contours?
    #TODO maybe it is a problem of coordiantes, images and dots are not aligned!
    fig, ax = plt.subplots(figsize=(36,30))
    for i,index in enumerate(indices):
        image_path = selected_images[i] #df.iloc[index,2]
        row = df_conc.loc[index]  # 'index' here is the row number

        y = row['Component_1']
        x = row['Component_2']
        
        image = cv2.imread(image_path)
        image = scale_image(image, max_image_size)
      
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset, [min_2,max_2,min_1,max_1])

        # Ensure coordinates are within the bounds of tsne_plot
        if (tl_x >= 0 and tl_y >= 0 and br_x <= tsne_plot.shape[1] and br_y <= tsne_plot.shape[0]):
            tsne_plot[tl_y:br_y, tl_x:br_x, :] = image
        else:
            print(f"Skipping out-of-bounds slice: tl_x={tl_x}, br_x={br_x}, tl_y={tl_y}, br_y={br_y}")
    
    ax.imshow(tsne_plot[:, :, ::-1])#, cmap='viridis')
    ax.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    
    if df_subset2 is not None:
        #etract the labels for the classes
        unique_labels = df_subset2['y'].unique()  # Assuming 'label' column contains class labels
        #for eahc class, compute the contour levels
        for label in unique_labels:
            class_color = colors_per_class.get(str(label), 'gray')
            compute_percentile_contour_levels(ax, df_subset2, label, class_color, 75, countour=True, contourf=False)
        #add trajectory line
        add_trajectory_case_study(df_subset2, ax, fig, plt.cm.viridis, colorbar=True)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    fig.savefig(output_path+filename.split('.')[0]+'_crops.png',bbox_inches='tight')



def compute_percentile_contour_levels(ax, df_subset1, label, class_color,  percentile, countour=True, contourf=True):
    class_data = df_subset1[df_subset1['y'] == label]
    x, y = class_data['Component_1'], class_data['Component_2']
    #z = np.zeros_like(x)  # Placeholder for z-values, adjust if you have another dimension for contours

    # Skip if not enough points for contours
    if len(class_data) < 3:
        return None

    # Use Gaussian KDE to create a density map
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xi, yi = np.meshgrid(
        np.linspace(x.min(), x.max(), 100),
        np.linspace(y.min(), y.max(), 100)
    )
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

    # Calculate the 25th percentile level for contour
    percentile_value = np.percentile(zi, percentile)

    if contourf:
        # Plot the filled contour only inside the 25th percentile contour line
        ax.contourf(xi, yi, zi, levels=[percentile_value, 100], cmap=ListedColormap([class_color]), alpha=0.2)

    if countour:
        # Plot the 25th percentile contour line
        ax.contour(xi, yi, zi, levels=[percentile_value], colors=[class_color], linewidths=1.5, alpha=0.7)




def plot_embedding_filled(df_subset1, colors_per_class, output_path, filename, df_subset2=None):
    # Plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create filled contour and contour lines for each label
    unique_labels = df_subset1['y'].unique()  # Assuming 'label' column contains class labels
    for label in unique_labels:
        # Convert label to string to match dictionary keys
        class_color = colors_per_class.get(str(label), 'gray')
        
        compute_percentile_contour_levels(ax, df_subset1, label, class_color, 25, countour=True, contourf=True)
 
    # Add trajectory if df_subset2 is provided
    if df_subset2 is not None:
        add_trajectory_case_study(df_subset2, ax, fig, plt.cm.viridis, True)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_per_class[label], markersize=10)
               for label in colors_per_class.keys()]
    ax.legend(handles, colors_per_class.keys(), title="Labels", fontsize=12)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Save figure
    fig.savefig(output_path + filename.split('.')[0] + '_contours.png', bbox_inches='tight')


