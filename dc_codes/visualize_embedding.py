"""
This script visualizes high-dimensional feature embeddings reduced to 2D space using Isomap or t-SNE and
overlays image crops corresponding to each feature in the embedding plot. It employs random sampling, color-coded
labels, and image adjustments to improve visualization clarity.

Modules:
    - numpy: For numerical operations and data manipulation.
    - pandas: To create and manage data frames of feature embeddings and labels.
    - matplotlib: For plotting 2D scatter plots and annotated images.
    - OpenCV (cv2): For image manipulation and resizing.
    - torch: For handling PyTorch tensor data such as checkpoint assignments.
    - glob: For gathering file paths in directories.
    - CSS4_COLORS, colors: For mapping color names to RGB values.

Parameters:
    - scale: Identifier string for the dataset, used for setting file paths.
    - random_state: Integer seed for consistency across random samples.
    - reduction_method: Dimensionality reduction method ('tsne' or 'isomap').
    - n_random_samples: Number of samples for dimensionality reduction if random sampling is enabled.
    - image_path: Path to the directory containing image crops.
    - colors_per_class1 and colors_per_class1_names: Dictionaries defining color codes for each label class.

Workflow:
    1. **Data Preparation**: Loads feature indices, targets, and embeddings, along with checkpoint assignments.
    2. **Color Mapping**: Converts label colors to normalized RGB values for plotting.
    3. **Dimensionality Reduction**: Uses precomputed t-SNE or Isomap results for 2D scatter plots.
    4. **Scaling and Transformation**: Scales t-SNE or Isomap coordinates to fit within [0, 1] range.
    5. **Image Selection and Processing**:
        - Extracts paths for specific image crops based on class samples.
        - Resizes images to a specified size and draws color-coded borders based on label class.
    6. **Plot Creation**: Generates scatter plots and composite images with embedded image crops at their respective
       2D coordinates.

Functions:
    - `scale_to_01_range`: Scales input array to fit within [0, 1] range.
    - `compute_plot_coordinates`: Determines plot coordinates for image placement based on t-SNE values.
    - `scale_image`: Rescales images to a maximum size while maintaining aspect ratio.
    - `draw_rectangle_by_class`: Adds color-coded borders to images based on their class label.

Usage:
    - Set `scale`, `random_state`, `reduction_method`, `image_path`, and `output_path` to desired values.
    - To use other dimensionality reduction methods, adjust the `reduction_method` and filenames accordingly.

Notes:
    - The script checks for coordinates out of bounds when overlaying images and skips if necessary.
    - Adjust `n_random_samples` for quicker processing, especially on large datasets.
    - The saved figure shows the 2D embedding with randomly sampled image crops, color-coded by class label.
"""

import numpy as np
import pandas as pd
from glob import glob
import torch

from embedding_plotting_func import plot_embedding_dots_iterative_test_msg_icon, scale_to_01_range, name_to_rgb, extract_hour, plot_embedding_dots, plot_embedding_filled, plot_embedding_crops, plot_embedding_dots_iterative_case_study

scale = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon'
random_state = '3' #all visualization were made with random state 3

output_path = f'/home/Daniele/fig/{scale}/'

tsne_path = f'/home/Daniele/fig/{scale}/'

reduction_method = 'tsne' # Options: 'tsne', 'isomap',
n_random_samples = None #30000

case_study_msg = True
case_study_icon = True

if reduction_method == 'tsne':
    tsne_filename = f'{reduction_method}_pca_cosine_{scale}_{random_state}.npy'  
elif reduction_method == 'isomap':
    tsne_filename = f'{reduction_method}_cosine_{scale}_{n_random_samples}.npy'
    indeces_filename = f'{reduction_method}_cosine_{scale}_{n_random_samples}_indeces.npy' 
filename = tsne_filename

image_path = f'/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA_test_msg_icon/1/' #cot_2013_128_germany, ir108_2013_128_germany
crop_path_list = sorted(glob(image_path+'*.tif'))
#print(len(crop_path_list))

if case_study_msg:
    #get only the case study images to plot
    #image_path_msg = '/data1/other_data/Case_Studies/Marche_Flood_22/MSG/image_to_plot/'
    #case_study_msg_images = sorted(glob(image_path_msg+'*.tif'))
    #get the path to the crops used to test the model
    case_study_msg_crops = sorted(glob(image_path+'IR_108_128x128_20220915_*_200K-300K_greyscale_CMA.tif'))
    #print(len(case_study_msg_crops))
    #print(case_study_msg_images)

if case_study_icon:
    # get path to images to plot
    #image_path_icon = '/data1/other_data/Case_Studies/Marche_Flood_22/ICON/image_to_plot/'
    #case_study_icon_images = sorted(glob(image_path_icon+'*.tif'))
    #get the path to the crops used to test the model
    case_study_icon_crops = sorted(glob(image_path+'cropped_icon_*_200K-300K_greyscale.tif'))
    #print(case_study_icon_images)
    #print(len(case_study_icon_crops))

# Load the feature indices, targets, and embeddings
filename1 = 'rank0_chunk0_train_heads_inds.npy' 
filename2 = 'rank0_chunk0_train_heads_targets.npy'
filename3 = 'rank0_chunk0_train_heads_features.npy'

path_feature = f'/data1/runs/{scale}/features/'

checkpoints_path = f'/data1/runs/{scale}/checkpoints/'  

assignments = torch.load(checkpoints_path+'assignments_800ep.pt',map_location='cpu')
sample_list = np.load(checkpoints_path+'samples_k7_800ep.npy')

data1 = np.load(path_feature+filename1)
print(data1)
if n_random_samples:
    data1 = np.load(tsne_path+indeces_filename)

print(data1.shape, data1)
print(assignments) #TODO it has 3 rows, DC used the first (why?)


# Define the color names for each class
colors_per_class1_names = {
    '0': 'darkgray', 
    '1': 'darkslategrey',
    '2': 'peru',
    '3': 'orangered',
    '4': 'lightcoral',
    '5': 'deepskyblue',
    '6': 'purple',
    '7': 'lightblue',
    '8': 'green'
}

colors_per_class1_rgb = {k: name_to_rgb(v) for k, v in colors_per_class1_names.items()}
print(colors_per_class1_rgb)


tsne = np.load(tsne_path+tsne_filename)
print(tsne.shape, tsne)
#exit()

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]
print(np.sum(ty))


tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# Create a DataFrame for T-SNE coordinates
df = pd.DataFrame({'Component_1': tx, 'Component_2': ty})

# Create a DataFrame for labels
df_labels = pd.DataFrame({'y': '', 'index': data1})
df_labels.set_index('index', inplace=True)
print(df_labels)

labels = assignments[0, :].cpu().numpy()
selected_labels = labels[data1]

#for index in range(len(data1)):
#    df_labels.loc[index] = assignments[0, :].cpu()[index]

df_labels['y'] = selected_labels
print(df_labels)

print(type(data1))
print(data1)
# Using a list comprehension to get elements at indices in data1
selected_elements = [crop_path_list[i] for i in data1]

df_labels['location'] = selected_elements

df_labels['Component_1'] = tx
df_labels['Component_2'] = ty

if case_study_msg:
    # Add the 'case_study' column based on whether 'location' is in 'case_study_crops'
    df_labels['case_study_msg'] = df_labels['location'].isin(case_study_msg_crops)

if case_study_icon:
    df_labels['case_study_icon'] = df_labels['location'].isin(case_study_icon_crops)


# add column with path to images for case studied
# Add a new column based on condition
if case_study_msg:
    path_msg_images = '/data1/other_data/Case_Studies/Marche_Flood_22/MSG/image_to_plot/'
    # add the path to the images for the case study
    df_labels['msg_image_path'] = df_labels['case_study_msg'].map(lambda x: path_msg_images if x else '')

if case_study_icon:
    path_icon_images = '/data1/other_data/Case_Studies/Marche_Flood_22/ICON/image_to_plot/'
    # add the path to the images for the case study
    df_labels['icon_image_path'] = df_labels['case_study_icon'].map(lambda x: path_icon_images if x else '') 

    

print(df_labels)


#print(df_labels['case_study_msg'].value_counts())
#print(df_labels['case_study_icon'].value_counts())


# Drop the 'index' column if it is no longer needed
df_subset = df_labels#.drop(columns=['index'])



# Apply the extraction function to get the hours
df_subset['hour'] = df_subset['location'].apply(extract_hour)

print(df_subset)

# Filter out invalid labels (-100)
df_subset1 = df_subset[df_subset['y'] != -100]

# Map labels to colors
df_subset1['color'] = df_subset1['y'].map(lambda x: colors_per_class1_names[str(int(x))])
print(df_subset1)

# Sample 20,000 points for plotting
df_subset2 = df_subset1.sample(n=50000)

print(df_subset2)

# Plot embedding with dots
if case_study_msg and case_study_icon:
    #plot_embedding_dots(df_subset2, colors_per_class1_names, output_path, filename, df_subset)
    #plot_embedding_filled(df_subset2, colors_per_class1_names, output_path, filename, df_subset)
    #plot_embedding_dots_iterative_case_study(df_subset1, colors_per_class1_names, output_path+'trajectory_iter/', filename, df_subset1)
    plot_embedding_dots_iterative_test_msg_icon(
    df_subset1, colors_per_class1_names, output_path+'trajectory_iter/', filename, df_subset1, legend=True)
else:
    plot_embedding_dots(df_subset2, colors_per_class1_names, output_path, filename)
    plot_embedding_filled(df_subset2, colors_per_class1_names, output_path, filename)


"""

#########################################    
# Plot the crops in the embedding space #
#########################################


plot_size=[1000,1000]
max_image_size=80 #squared images

offset = [max_image_size // 2, max_image_size // 2]
image_centers_area_size = [plot_size[0] - max_image_size, plot_size[1] - max_image_size ] #2 * offset

tsne_plot = 255 * np.ones((plot_size[0], plot_size[1], 3), np.uint8)
print(tsne_plot.shape)


n1=50

a1=df_subset1.query("y == 0").sample(n=n1)
a2=df_subset1.query("y == 1").sample(n=n1)
a3=df_subset1.query("y == 2").sample(n=n1)
a4=df_subset1.query("y == 3").sample(n=n1)
a5=df_subset1.query("y == 4").sample(n=n1)
a6=df_subset1.query("y == 5").sample(n=n1)
a7=df_subset1.query("y == 6").sample(n=n1)
a8=df_subset1.query("y == 7").sample(n=n1)
a9=df_subset1.query("y == 8").sample(n=n1)
result = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
df_conc = pd.concat(result)


#df  = df_subset1.sample(n = n1)
print(df_conc)

# Get the list of indices
indices = df_conc.index.tolist()
print(indices)

#get list of component 1 and 2
comp_1_list = df_conc.Component_1.tolist()
comp_2_list = df_conc.Component_2.tolist()
#print(comp_1_list)
#find max and min of each component
max_1 = np.amax(comp_1_list)
min_1 = np.amin(comp_1_list)
max_2 = np.amax(comp_2_list)
min_2 = np.amin(comp_2_list)

#compute offsets


#get images path
image_path_list = sorted(glob(image_path+'*.tif'))

# Select the elements from image_path_list based on indices
selected_images = [image_path_list[i] for i in indices]

print(len(selected_images), len(df_conc))



if case_study:
    plot_embedding_crops(indices, selected_images, df_conc, tsne_plot, output_path, filename, image_centers_area_size, offset, max_image_size, min_1, max_1, min_2, max_2, colors_per_class1_names, df_subset)
else:
    plot_embedding_crops(indices, selected_images, df_conc, tsne_plot, output_path, filename, image_centers_area_size, offset, max_image_size, min_1, max_1, min_2, max_2, colors_per_class1_names)


    
"""