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
from scipy.spatial import cKDTree
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.image as mpimg

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
    fig, ax = plt.subplots(figsize=(10, 10))
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
    #df_case_study_msg = df_subset2[df_subset2['case_study'] == True and df_subset2['vector_type']=='msg'].sort_index()
    df_case_study_msg = df_subset2[(df_subset2['case_study'] == True) & (df_subset2['vector_type'] == 'msg')].sort_index()
    df_case_study_icon = df_subset2[(df_subset2['case_study'] == True) & (df_subset2['vector_type'] == 'icon')].sort_index()
    #df_case_study_icon = df_subset2[df_subset2['case_study'] == True & df_subset2['vector_type']=='icon'].sort_index()

    # Determine the maximum number of steps to iterate
    max_steps = max(len(df_case_study_msg), len(df_case_study_icon))

    # Plot base embedding
    for i in range(max_steps):
        fig, ax = plt.subplots(figsize=(14, 12))

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
                c= "red", #current_msg['color'],  # Follow the same color coding as the background
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
                c= 'blue', #current_icon['color'],  # Follow the same color coding as the background
                s=120, 
                edgecolor="blue", 
                linewidth=2, 
                zorder=5, 
                label='ICON Trajectory Points' if i == 0 else ""
            )
            if len(current_icon) > 1:
                ax.plot(
                    current_icon['Component_1'], 
                    current_icon['Component_2'], 
                    color="blue", 
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
                plt.Line2D([0], [0], color="blue", label="ICON Trajectory"),
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
        extra_bottom_space = 0.4  # Proportion of extra space to add at the bottom
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
    fig, ax = plt.subplots(figsize=(20,20))
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


def create_WV_IR_diff_colormap(vmin, center, vmax, diverg_cmap=mpl.cm.seismic):
    if vmin is None:
        vmin = -1
    if vmax is None:
        vmax = 1
    # get number of colors above and below center point representing the respective range percentages
    n_pos = int(265*(vmax-center)/(vmax-vmin)) if vmax > center else 1
    n_neg = int(265*(center-vmin)/(vmax-vmin)) if vmin < center else 1
    # sample colors
    colors_pos = diverg_cmap(np.linspace(0.7, 1, n_pos))
    colors_neg = diverg_cmap(np.linspace(0, 0.5, n_neg))
    # combine them and build a new colormap
    colors = np.vstack((colors_neg, colors_pos))
    
    return mpl.colors.LinearSegmentedColormap.from_list('recentered_cmap', colors)


def plot_embedding_crops_grid(df, output_path, filename, variable_type, cmap, grid_size=10, zoom=0.3):
    """
    Plot image crops aligned on a regular grid using normalized Component_1 and Component_2,
    placing one image per grid bin based on minimal distance to bin center.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Normalize Component_1 and Component_2
    x = df['Component_1'].values
    y = df['Component_2'].values
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    df['x_norm'] = x_norm
    df['y_norm'] = y_norm

    # Assign grid bins
    ix = np.minimum((x_norm * grid_size).astype(int), grid_size - 1)
    iy = np.minimum((y_norm * grid_size).astype(int), grid_size - 1)
    centers = np.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size)

    #Select best image per bin (closest to center)
    placed = {}
    for i, j, xn, yn, idx in zip(ix, iy, x_norm, y_norm, df.index):
        bin_key = (i, j)
        cx, cy = centers[i], centers[j]
        dist = (xn - cx) ** 2 + (yn - cy) ** 2
        if bin_key not in placed or dist < placed[bin_key][0]:
            placed[bin_key] = (dist, idx)



    # Plot the selected images at bin centers
    for (i, j), (_, idx) in placed.items():
        row = df.loc[idx]
        cx, cy = centers[i], centers[j]
        # try:
        #     img = Image.open(row['path'])#.convert('L')
        #     imagebox = OffsetImage(img, zoom=zoom)#, cmap='gray')
        #     ab = AnnotationBbox(imagebox, (cx, cy), frameon=False)
        #     ax.add_artist(ab)
        # except Exception as e:
        #     print(f"Skipping image {row['path']}: {e}")
        #     continue

        try:
            # Load image as NumPy array (assumes grayscale .png saved with matplotlib)
            img_array = mpimg.imread(row['path'])
            img_rgba = img_array  # RGB or RGBA already

            imagebox = OffsetImage(img_rgba, zoom=zoom, cmap=cmap)
            ab = AnnotationBbox(imagebox, (cx, cy), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Skipping image {row['path']}: {e}")
            continue

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.invert_yaxis()
    ax.axis('off')

    plt.tight_layout()
    base_filename = os.path.splitext(filename)[0]
    save_path = os.path.join(output_path, base_filename + '_'+ variable_type + '_grid.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()





def plot_embedding_crops_binned_grid(df, output_path, filename, grid_size=10, zoom=0.3):
    """
    Place each crop into one of grid_size×grid_size bins in the (Component_1,Component_2) plane,
    then for each non-empty bin pick the point closest to the bin center and plot it there.
    This ensures a nicely filled grid with no overlapping crops.
    """
    fig, ax = plt.subplots(figsize=(10,10))

    # 1) normalize to [0,1]
    x = df['Component_1'].values
    y = df['Component_2'].values
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # 2) compute bin indices
    ix = np.minimum((x_norm * grid_size).astype(int), grid_size - 1)
    iy = np.minimum((y_norm * grid_size).astype(int), grid_size - 1)
    
    # 3) prepare bin-centers
    centers = np.linspace(0.5/grid_size, 1 - 0.5/grid_size, grid_size)
    
    placed = {}
    for i, j, xn, yn in zip(ix, iy, x_norm, y_norm):
        bin_key = (i,j)
        # compute distance to bin center
        cx, cy = centers[i], centers[j]
        d = (xn - cx)**2 + (yn - cy)**2
        # keep the closest point to the center
        if bin_key not in placed or d < placed[bin_key][0]:
            placed[bin_key] = (d, df.index[(ix==i)&(iy==j)][0])
    
    # 4) plot one per bin
    for (i,j), (_, idx) in placed.items():
        row = df.loc[idx]
        img = Image.open(row['path'])#.convert('L')
        print("Plotting image:", row['path'])
        exit()
        imbox = OffsetImage(img, zoom=zoom)# cmap='gray')
        # map bin coordinate to [0,1] for plotting
        px, py = centers[i], centers[j]
        ab = AnnotationBbox(imbox, (px,py), frameon=False)
        ax.add_artist(ab)

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout()

    base = os.path.splitext(filename)[0]
    save_path = os.path.join(output_path, base + '_binned_grid.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved:", save_path)


def plot_embedding_crops_table(df, output_path, filename, n=5, selection="closest"):
    """
    Plots crops in a table format where each row corresponds to a label, 
    and 'n' images are selected based on the specified selection method.

    Args:
        df (pd.DataFrame): DataFrame containing image paths, labels, and distances.
        output_path (str): Path to save the output plot.
        filename (str): Name of the output file.
        n (int): Number of crops to display per label.
        selection (str): Method of selection - "closest", "farthest", or "random".
    """
    labels = df['label'].unique()
    num_labels = len(labels)
    
    fig, axes = plt.subplots(num_labels, n, figsize=(n * 2, num_labels * 2))
    fig.suptitle(f"Crops Sorted by {selection.capitalize()} Distance from Centroid", fontsize=14, fontweight="bold")

    # Ensure axes is iterable even if there's only one row
    if num_labels == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, label in enumerate(labels):
        subset = df[df['label'] == label].sort_values(by='distance')

        # Select crops based on the specified method
        if selection == "closest":
            subset = subset.head(n)
        elif selection == "farthest":
            subset = subset.tail(n)
        elif selection == "random":
            subset = subset.sample(n=min(n, len(subset)), random_state=42)  # Ensure at least `n` samples
        else:
            raise ValueError("Invalid selection method. Choose 'closest', 'farthest', or 'random'.")

        for j, (_, row) in enumerate(subset.iterrows()):
            img_path = row['path']
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            ax = axes[i, j] if num_labels > 1 else axes[j]  # Select subplot
            ax.imshow(img, cmap='gray')
            ax.axis('off')

            # Row label (on the left)
            if j == 0:
                ax.set_ylabel(f"Label {label}", fontsize=12, fontweight='bold', rotation=0, labelpad=30, va='center')

    # Add column headers (1, 2, 3, ... n)
    for j in range(n):
        axes[0, j].set_title(f"{j+1}", fontsize=12, fontweight="bold")  # Add column numbers

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    output_file = f"{output_path}/{filename.split('.')[0]}_{n}_{selection}_crops_table.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved plot: {output_file}")


def plot_classwise_grids(df, output_path, filename,cmap, n=100, selection="closest"):
    """
    For each class (label) in the DataFrame, plot a 10x10 grid of image crops
    from that class only. Save each class grid as a separate image.

    Args:
        df (pd.DataFrame): DataFrame with 'path', 'label', and optionally 'distance' columns.
        output_path (str): Directory where the output images will be saved.
        selection (str): "closest", "farthest", or "random" selection strategy per label.
    """
    os.makedirs(output_path, exist_ok=True)
    labels = sorted(df['label'].unique())

    for label in labels:
        subset = df[df['label'] == label]

        # Apply selection strategy
        if selection == "closest" and 'distance' in subset:
            #sort values from highest to lowest distance
            subset = subset.sort_values(by='distance', ascending=False).head(n)
        elif selection == "farthest" and 'distance' in subset:
            subset = subset.sort_values(by='distance', ascending=True).head(n)
        elif selection == "random":
            subset = subset.sample(n=min(n, len(subset)), random_state=42)
        else:
            raise ValueError("Invalid selection method. Choose 'closest', 'farthest', or 'random'.")

        fig, axes = plt.subplots(10, 10, figsize=(12, 12))
        fig.suptitle(f"Label {label} – {selection.capitalize()} Samples", fontsize=16, fontweight="bold")

        for ax, (_, row) in zip(axes.flatten(), subset.iterrows()):
            try:
                #img_array = mpimg.imread(row['path'])
                #img_rgba = img_array  # RGB or RGBA already
                #imagebox = OffsetImage(img_rgba, zoom=zoom, cmap=cmap)
                img = Image.open(row['path']).convert('L')
                ax.imshow(img, cmap=cmap)
            except Exception as e:
                print(f"Error loading {row['path']}: {e}")
                ax.axis('off')
                continue
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        #filename = os.path.join(output_path, f"label_{label}_{selection}_grid.png")
        output_file = f"{output_path}/{filename.split('.')[0]}_label_{label}_{selection}_grid.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def plot_embedding_crops_new(df, output_path, filename):

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the scatter plot with component 1 as x and component 2 as y
    #ax.scatter(df['Component_1'], df['Component_2'], s=100, color='blue', alpha=0.6)

    # Iterate over each row of the dataframe and overlay the image
    for i, row in df.iterrows():
        img_path = row['path']
        # Load the image using PIL or matplotlib
        img = Image.open(img_path)  # PIL can be used to open image
        # Alternatively, you can use mpimg.imread() if you prefer
        # img = mpimg.imread(img_path)

        # Convert the image to greyscale
        img = img.convert('L')  # 'L' mode is for greyscale

        # Create an OffsetImage with the loaded image
        imagebox = OffsetImage(img, zoom=0.3, cmap='gray')  # You can adjust zoom to scale the image
        
        # Position the image at the corresponding (x, y) of component1 and component2
        ab = AnnotationBbox(imagebox, (row['Component_1'], row['Component_2']), frameon=False)
        
        # Add the image to the plot
        ax.add_artist(ab)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        fig.savefig(output_path+filename.split('.')[0]+'_crops_new.png',bbox_inches='tight')



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


def plot_average_crop_shapes(df, output_path, filename, n=10, selection="closest", alpha=0.05):
    """
    Overlays `n` crop images with high transparency to visualize the average shape for each label.

    Args:
        df (pd.DataFrame): DataFrame containing crop information including 'path', 'distance', and 'label'.
        output_path (str): Path to save the output images.
        filename (str): Base name for the output file.
        n (int): Number of images to use per label.
        selection (str): Selection method ('closest', 'farthest', 'random').
        alpha (float): Transparency level for each image.
    """

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
    unique_labels = df['label'].unique()  # Get all unique labels

    for label in unique_labels:
        # Filter dataset for the given label and sort by distance
        subset = df[df['label'] == label].sort_values(by='distance')

        if selection == "closest":
            subset = subset.head(n)
        elif selection == "farthest":
            subset = subset.tail(n)
        elif selection == "random":
            subset = subset.sample(n=min(n, len(subset)), random_state=42)

        image_paths = subset['path'].tolist()

        if not image_paths:
            print(f"No images found for label {label}. Skipping.")
            continue

        # Load images and convert to grayscale arrays
        images = [np.array(Image.open(path).convert('L'), dtype=np.float32) for path in image_paths]

        # Get dimensions (assume all images are the same size)
        height, width = images[0].shape

        # Compute the averaged image
        avg_image = np.zeros((height, width), dtype=np.float32)
        for img in images:
            avg_image += img / len(images)  # Normalized sum

        # Plot the averaged image
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(avg_image, cmap="gray", alpha=1)  # Show the final averaged shape
        ax.axis("off")

        # Save the output
        output_file = os.path.join(output_path, f"{filename.split('.')[0]}_label_{label}_{n}_{selection}_average.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved averaged shape plot for label {label}: {output_file}")



def plot_average_crop_values(df, output_path, filename, n=10, selection="closest"):
    """
    Computes and plots the average pixel values for `n` crop images per label.

    Args:
        df (pd.DataFrame): DataFrame containing crop information including 'path', 'distance', and 'label'.
        output_path (str): Path to save the output images.
        filename (str): Base name for the output file.
        n (int): Number of images to use per label.
        selection (str): Selection method ('closest', 'farthest', 'random').
    """

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
    unique_labels = df['label'].unique()  # Get all unique labels

    for label in unique_labels:
        # Filter dataset for the given label and sort by distance
        subset = df[df['label'] == label].sort_values(by='distance')

        if selection == "closest":
            subset = subset.head(n)
        elif selection == "farthest":
            subset = subset.tail(n)
        elif selection == "random":
            subset = subset.sample(n=min(n, len(subset)), random_state=42)

        image_paths = subset['path'].tolist()

        if not image_paths:
            print(f"No images found for label {label}. Skipping.")
            continue

        # Load images and convert to grayscale arrays
        images = [np.array(Image.open(path).convert('L'), dtype=np.float32) for path in image_paths]
        # Get dimensions (assume all images are the same size)
        height, width = images[0].shape

        # Compute the averaged pixel values
        avg_image = np.mean(images, axis=0)

        # Plot the averaged pixel values
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(avg_image, cmap="gray", interpolation="nearest")  # Show the final averaged shape
        ax.axis("off")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Average Pixel Intensity", fontsize=12)

        # Save the output
        output_file = os.path.join(output_path, f"{filename.split('.')[0]}_label_{label}_{n}_{selection}_average.png")
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved average pixel intensity plot for label {label}: {output_file}")