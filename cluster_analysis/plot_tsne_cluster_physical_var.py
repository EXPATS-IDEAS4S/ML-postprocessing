import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from glob import glob
import torch
from matplotlib.colors import CSS4_COLORS
from matplotlib import colors as mcolors  # Correct import for colors
import seaborn as sns

scale = '10th-90th_CMA'
random_state = '3' #all visualization were made with random state 3
#dcv_cot_128x128_k7_germany_60kcrops
#dcv2_ir_128x128_k7_germany_70kcrops
#output_path = f'/home/Daniele/fig/dcv_ir108_128x128_k9_30k_grey_{scale}/'

sampling_type = 'closest'

output_path = f'/home/Daniele/fig/cma_analysis/{scale}/{sampling_type}/'

tsne_path = f'/home/Daniele/fig/dcv_ir108_128x128_k9_30k_grey_{scale}/'

tsne_filename = f'tsne_pca_cosine_{scale}_{random_state}.npy' 
filename = tsne_filename
#filename = 'isomap_2d_cosine_800ep_20000samples.csv'
#tsne_filename = 'tsnegermany_pca_cosine_500annealing50_800ep.npy'
#tsne_filename = 'tsnegermany_pca_cosine_500multiscale50_800ep.npy'

image_path = f'/data1/crops/ir108_2013-2014_GS_{scale}/1/' #cot_2013_128_germany, ir108_2013_128_germany
crop_path_list = sorted(glob(image_path+'*.tif'))
#print(len(crop_path_list))

filename1 = 'rank0_chunk0_train_heads_inds.npy' 
filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

path_feature = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{scale}/features/'

checkpoints_path = f'/data1/runs/dcv2_ir108_128x128_k9_germany_30kcrops_grey_{scale}/checkpoints/'  

assignments = torch.load(checkpoints_path+'assignments_800ep.pt',map_location='cpu')
sample_list = np.load(checkpoints_path+'samples_k7_800ep.npy')

data1 = np.load(path_feature+filename1)
print(data1)

print(assignments) #TODO it has 3 rows, DC used the first (why?)

#load isomap and index
#df = pd.read_csv(tsne_path+filename)
#print(df)


colors_per_class1 = {
    '0' : [0, 0, 0], 
    '1' : [192, 192, 192],
    '2' : [3, 183, 255],
    '3' : [230, 202, 142],
    '4' : [188, 158, 33],
    '5' : [71, 48, 2],
    '6' : [151, 31, 52],
    '7' : [0, 133, 251],
    #'8' : [248,240,202 ],
    #'9' : [100, 100, 255],
}

# Convert RGB to 0-1 range
colors_per_class1_norm = {k: np.array(v) / 255.0 for k, v in colors_per_class1.items()}


# Define the color names for each class
colors_per_class1_names = {
    '0': 'darkgray', 
    '1': 'black',
    '2': 'peru',
    '3': 'beige',
    '4': 'olivedrab',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green'
}

# Convert color names to RGB values
def name_to_rgb(color_name):
    return np.array(mcolors.to_rgb(color_name)) * 255


colors_per_class1_rgb = {k: name_to_rgb(v) for k, v in colors_per_class1_names.items()}
print(colors_per_class1_rgb)


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


def draw_rectangle_by_class(image, label):
    color = colors_per_class1_rgb[label]
    #print(color)
    image_height, image_width, _ = image.shape
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=4)

    return image


tsne = np.load(tsne_path+tsne_filename)

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# Create a DataFrame for T-SNE coordinates
df = pd.DataFrame({'Component_1': tx, 'Component_2': ty})

# Create a DataFrame for labels
df_labels = pd.DataFrame({'y': '', 'index': data1})
df_labels.set_index('index', inplace=True)

for index in range(len(data1)):
    df_labels.loc[index] = assignments[0, :].cpu()[index]

df_labels['location'] = crop_path_list

print(df_labels)


# Reset the index of df2 to turn it into a column
df = df.reset_index()

print(df)

# Merge df1 with df2_reset on Selected_Index and index
df_subset = pd.merge(df, df_labels, on='index')# left_on='Selected_Index', right_on='index', how='left')

# Drop the 'index' column if it is no longer needed
df_subset = df_subset.drop(columns=['index'])

print(df_subset)


#df_subset['label'] = df_subset['y'].apply(lambda i: str(i))
#labels = df['label']

# Merge T-SNE coordinates and labels
#df_subset['y'] = df_labels['label'].values
#df_subset['location'] = crop_path_list



# Filter out invalid labels (-100)
df_subset1 = df_subset[df_subset['y'] != -100]

# Map labels to colors
df_subset1['color'] = df_subset1['y'].map(lambda x: colors_per_class1_names[str(int(x))])
print(df_subset1)

# Sample 20,000 points for plotting
df_subset2 = df_subset1.sample(n=20000)


# Set up the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Step 1: Loop through unique classes and plot KDE contours for each
for class_label in df_subset2['y'].unique():
    # Filter the data for the current class
    class_data = df_subset2[df_subset2['y'] == class_label]
    
    # Step 2: Plot the KDE (Kernel Density Estimate) contours
    sns.kdeplot(
        x=class_data['Component_1'],
        y=class_data['Component_2'],
        ax=ax,
        levels=1,  # Number of contour levels
        linewidths=2,
        color=colors_per_class1_names[str(int(class_label))],  # Use color corresponding to the class
        label=f'Class {int(class_label)}'
    )

# Step 3: Customize plot appearance
ax.set_title('Density Contour Plot by Class', fontsize=20)
ax.set_xlabel('Component 1', fontsize=15)
ax.set_ylabel('Component 2', fontsize=15)
ax.legend(title='Class Labels')

# Step 4: Save the figure
fig.savefig(output_path + filename.split('.')[0] + '_contour.png', bbox_inches='tight')

# Display the plot
plt.show()