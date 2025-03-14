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
from matplotlib import pyplot as plt
import cv2
from glob import glob
import torch
from matplotlib.colors import CSS4_COLORS
from matplotlib import colors as mcolors  # Correct import for colors
import re
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

scale = 'dcv2_ir108_200x200_k9_expats_70k_200-300K_closed-CMA'
random_state = '3' #all visualization were made with random state 3
#dcv_cot_128x128_k7_germany_60kcrops
#dcv2_ir_128x128_k7_germany_70kcrops
output_path = f'/data1/fig/{scale}/'

tsne_path = f'/data1/fig/{scale}/'

reduction_method = 'tsne' # Options: 'tsne', 'isomap',
n_random_samples = None #30000

case_study = False

if reduction_method == 'tsne':
    tsne_filename = f'{reduction_method}_pca_cosine_{scale}_{random_state}.npy'  
elif reduction_method == 'isomap':
    tsne_filename = f'{reduction_method}_cosine_{scale}_{n_random_samples}.npy'
    indeces_filename = f'{reduction_method}_cosine_{scale}_{n_random_samples}_indeces.npy' 
filename = tsne_filename
#filename = 'isomap_2d_cosine_800ep_20000samples.csv'
#tsne_filename = 'tsnegermany_pca_cosine_500annealing50_800ep.npy'
#tsne_filename = 'tsnegermany_pca_cosine_500multiscale50_800ep.npy'

image_path = f'/data1/crops/ir108_2013-2014-2015-2016_GS_200-300_CMA-closed_200x200/1/' #cot_2013_128_germany, ir108_2013_128_germany
crop_path_list = sorted(glob(image_path+'*.tif'))
#print(len(crop_path_list))

if case_study:
    #get only the case study crops
    case_study_crops = sorted(glob(image_path+'*20220915*'))
    #print(len(case_study_crops))
    #print(case_study_crops) 


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
    '8' : [248,240,202 ],
    #'9' : [100, 100, 255],
}

# Convert RGB to 0-1 range
colors_per_class1_norm = {k: np.array(v) / 255.0 for k, v in colors_per_class1.items()}


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

def extract_hour(location):
    match = re.search(r'_(\d{2}):\d{2}_', location)
    if match:
        return int(match.group(1))  # Extract and convert the hour to an integer
    return None




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

if case_study:
    # Add the 'case_study' column based on whether 'location' is in 'case_study_crops'
    df_labels['case_study'] = df_labels['location'].isin(case_study_crops)


print(df_labels)


# Reset the index of df2 to turn it into a column
#df = df.reset_index()

#print(df)

# Merge df1 with df2_reset on Selected_Index and index
#df_subset = pd.merge(df, df_labels, on='index')# left_on='Selected_Index', right_on='index', how='left')

#print(df_subset)


# Drop the 'index' column if it is no longer needed
df_subset = df_labels#.drop(columns=['index'])



# Apply the extraction function to get the hours
df_subset['hour'] = df_subset['location'].apply(extract_hour)

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



# Plot
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(df_subset2['Component_1'], df_subset2['Component_2'],
                     c=df_subset2['color'].tolist(), alpha=0.5, s=20)

if case_study:
    # Filter case study points
    #case_study_points = df_subset[df_subset['case_study'] == True]
    #print(case_study_points['location'].tolist())    
    #exit()

    # Plot a continuous red line through the case study points
    #ax.plot(case_study_points['Component_1'], case_study_points['Component_2'],
    #    color='red', linewidth=2, linestyle='-', label='Case Study')

    # Step 2: Create a colormap for the hours
    cmap = plt.cm.viridis  # You can choose any colormap you like
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

    # Add a colorbar for the hour
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for the colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Hour of the Day', fontsize=16)
    cbar.ax.tick_params(labelsize=14)



# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_per_class1_norm[label], markersize=10)
           for label in colors_per_class1_norm.keys()]
#ax.legend(handles, colors_per_class1_norm.keys(), title="Labels")

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=20)

# Save figure
fig.savefig(output_path + filename.split('.')[0] + '_dots.png', bbox_inches='tight')




# df_subset1 = df_subset[df_subset.y != -100]

# print(len(df_subset), len(df_subset1))

# df_subset2 = df_subset1.sample(n = 20000)


# #print dots
# fig = plt.figure(figsize=(16,10))
# ax = fig.add_subplot(111)

# plt.scatter(df_subset1['tsne-dcv22d-one'],df_subset1['tsne-dcv22d-two'],alpha=0.5,s=20)
# ax.tick_params(axis='both', which='major', labelsize=20)

# fig.savefig(output_path+tsne_filename.split('.')[0]+'_dots.png',bbox_inches='tight')



#data1 = np.load(path_feature+filename2)
#print(data1.shape)


plot_size=[1000,1000]
max_image_size=80 #squared images

offset = [max_image_size // 2, max_image_size // 2]
image_centers_area_size = [plot_size[0] - max_image_size, plot_size[1] - max_image_size ] #2 * offset

tsne_plot = 255 * np.ones((plot_size[0], plot_size[1], 3), np.uint8)
print(tsne_plot.shape)

#df_subset2 = df_subset1.sample(n = 20000)

#number of crops to plot in the feature space
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

fig = plt.figure(figsize=(36,30))
for i,index in enumerate(indices):
    image_path = selected_images[i] #df.iloc[index,2]
    row = df_conc.loc[index]  # 'index' here is the row number
    label = row['y']
    #print(label)
    y = row['Component_1']
    x = row['Component_2']
    
    # label = df_conc.loc[df_conc['index'] == index, 'y'].item()
    # y=df_conc.loc[df_conc['index'] == index,'Component_1'].item()
    # x=df_conc.loc[df_conc['index'] == index,'Component_2'].item()
    image = cv2.imread(image_path)
    image = scale_image(image, max_image_size)
    #image = draw_rectangle_by_class(image, str(label))
    #print(image.shape)

    tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset, [min_2,max_2,min_1,max_1])
    #print( br_x,tl_x, br_y,tl_y )
    #print( br_x-tl_x, br_y-tl_y )

    # Ensure coordinates are within the bounds of tsne_plot
    if (tl_x >= 0 and tl_y >= 0 and br_x <= tsne_plot.shape[1] and br_y <= tsne_plot.shape[0]):
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image
    else:
        print(f"Skipping out-of-bounds slice: tl_x={tl_x}, br_x={br_x}, tl_y={tl_y}, br_y={br_y}")

plt.imshow(tsne_plot[:, :, ::-1])#, cmap='viridis')
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

fig.savefig(output_path+filename.split('.')[0]+'_crops.png',bbox_inches='tight')


#3rd plot


n1=10
k_class = 9

a1=df_subset1.query("y == 0").sample(n=n1)
a2=df_subset1.query("y == 1").sample(n=n1)
a3=df_subset1.query("y == 2").sample(n=n1)
a4=df_subset1.query("y == 3").sample(n=n1)
a5=df_subset1.query("y == 4").sample(n=n1)
a6=df_subset1.query("y == 5").sample(n=n1)
a7=df_subset1.query("y == 6").sample(n=n1)
a8=df_subset1.query("y == 7").sample(n=n1)
a9=df_subset1.query("y == 8").sample(n=n1)

a1_loc = a1['location'].tolist()
a2_loc = a2['location'].tolist()
a3_loc = a3['location'].tolist()
a4_loc = a4['location'].tolist()
a5_loc = a5['location'].tolist()
a6_loc = a6['location'].tolist()
a7_loc = a7['location'].tolist()
a8_loc = a8['location'].tolist()
a9_loc = a9['location'].tolist()

result = [a1_loc,a2_loc,a3_loc,a4_loc,a5_loc,a6_loc,a7_loc,a8_loc,a9_loc] #,a7_loc

ftick=18

fig = plt.figure(figsize=(10.5, 10.5),constrained_layout=True)
gs0 = fig.add_gridspec(1, 1)

gs00 = gs0[0].subgridspec(k_class, n1)


pattern = result

pattern = np.array(pattern)
pattern = np.reshape(pattern,[k_class,n1])


for a in range(k_class):
    for b in range(n1):
        ax = fig.add_subplot(gs00[a, b])
        file = str(pattern[a,b])
        file = file[0:len(file)]
        image = plt.imread(file)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        if a == 0 and b == 0:
            ax.set_ylabel('CR:1',fontsize=ftick)
            ax.set_title('rand 1',fontsize=ftick)
        if a == 1 and b == 0:
            ax.set_ylabel('CR:2',fontsize=ftick)
        if a == 2 and b == 0:
            ax.set_ylabel('CR:3',fontsize=ftick)
        if a == 3 and b == 0:
            ax.set_ylabel('CR:4',fontsize=ftick)
        if a == 4 and b == 0:
            ax.set_ylabel('CR:5',fontsize=ftick)
        if a == 5 and b == 0:
            ax.set_ylabel('CR:6',fontsize=ftick)
        if a == 6 and b == 0:
            ax.set_ylabel('CR:7',fontsize=ftick)
        if a == 0 and b == 1:
            ax.set_title('rand 2',fontsize=ftick)
        if a == 0 and b == 2:
            ax.set_title('rand 3',fontsize=ftick)
        if a == 0 and b == 3:
            ax.set_title('rand 4',fontsize=ftick)
        if a == 0 and b == 4:
            ax.set_title('rand 5',fontsize=ftick)
        if a == 0 and b == 5:
            ax.set_title('rand 6',fontsize=ftick)
        if a == 0 and b == 6:
            ax.set_title('rand 7',fontsize=ftick)
        if a == 0 and b == 7:
            ax.set_title('rand 8',fontsize=ftick)
        if a == 0 and b == 8:
            ax.set_title('rand 9',fontsize=ftick)
        if a == 0 and b == 9:
            ax.set_title('rand 10',fontsize=ftick)
        if a == 0 and b == 10:
            ax.set_title('rand 11',fontsize=ftick)
        if a == 0 and b ==11:
            ax.set_title('rand 12',fontsize=ftick)
        if a == 0 and b == 12:
            ax.set_title('rand 13',fontsize=ftick)
        if a == 0 and b == 13:
            ax.set_title('rand 14',fontsize=ftick)
        if a == 0 and b == 14:
            ax.set_title('rand 15',fontsize=ftick)
        if a == 0 and b == 15:
            ax.set_title('rand 16',fontsize=ftick)
        if a == 0 and b == 16:
            ax.set_title('rand 17',fontsize=ftick)
        if a == 0 and b == 17:
            ax.set_title('rand 18',fontsize=ftick)
        if a == 0 and b == 18:
            ax.set_title('rand 19',fontsize=ftick)
        if a == 0 and b == 19:
            ax.set_title('rand 20',fontsize=ftick)

fig.set_figheight(14)
fig.set_figwidth(25)

fig.savefig(output_path+filename.split('.')[0]+'_table.png',bbox_inches='tight')