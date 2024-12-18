import numpy as np
import pandas as pd
from glob import glob
import torch
import rasterio
from scipy.ndimage import binary_closing
import seaborn as sns
import matplotlib.pyplot as plt

run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'

output_path = f'/home/Daniele/fig/{run_name}/'

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

# load list of training crops
image_train_path = f'/data1/crops/ir108_2013-2014-2015-2016_200K-300K_CMA/1/'
crop_train_path_list = sorted(glob(image_train_path+'*.tif'))

print(len(crop_train_path_list))    

checkpoints_path = f'/data1/runs/{run_name}/checkpoints/'  
assignments = torch.load(checkpoints_path+'assignments_800ep.pt',map_location='cpu')
labels = assignments[0, :].cpu().numpy()

# create a dataframe with labels and crop path
df_labels = pd.DataFrame({'y': labels, 'path': crop_train_path_list})

print(df_labels)

# loop over the crop, open the tif image and convert in a numpy array
for index, row in df_labels.iterrows():
    print(row['path'])
    # open the tif image
    with rasterio.open(row['path']) as src:
        # read the image
        img = src.read(1)
        #print(img.shape)
        #print(img)
        # convert the image in a numpy array
        img_array = np.array(img)
        #print(img_array.shape)
        #print(img_array)
        # convert values above 0 to 1
        img_array[img_array > 0] = 1
        #print(img_array)
        
        # Apply binary closing with two different structures
        closed_mask_3x3 = binary_closing(img_array, structure=np.ones((3, 3)))
        closed_mask_5x5 = binary_closing(img_array, structure=np.ones((5, 5)))

        # Identify patches by taking the difference between the original and closed mask
        granular_mask_3x3 = (closed_mask_3x3 - img_array) == 1  # Where 1 represents patchy clear in cloudy areas
        granular_mask_5x5 = (closed_mask_5x5 - img_array) == 1

        num_holes_3x3 = np.sum(granular_mask_3x3)
        num_holes_5x5 = np.sum(granular_mask_5x5)
        #print(num_holes_3x3, num_holes_5x5)

        # add the number of holes to the dataframe
        #df_labels.at[index, 'num_holes_3x3'] = num_holes_3x3
        #df_labels.at[index, 'num_holes_5x5'] = num_holes_5x5

        #compute the percentage of holes over the points with value 1
        total_pixels = np.sum(img_array == 1)
        df_labels.at[index, 'perc_holes_3x3'] = num_holes_3x3/total_pixels
        df_labels.at[index, 'perc_holes_5x5'] = num_holes_5x5/total_pixels
        

print(df_labels)

# delete the rows with -100 values as y
df_labels = df_labels[df_labels['y'] != -100]

# save the dataframe to a csv file
df_labels.to_csv(output_path+'cma_holes_perc.csv', index=False)

# Plot boxplot with seaborn with distribution of num of holes (y axis) and label y (x axis) and hue the num of holes 3x3 and 5x5

# Melt the DataFrame for better compatibility with Seaborn
melted_df = df_labels.melt(id_vars=['y'], 
                    value_vars=['perc_holes_3x3', 'perc_holes_5x5'], 
                    var_name='Type', 
                    value_name='Value')

# Rename for readability
melted_df['Type'] = melted_df['Type'].replace({
    'perc_holes_3x3': '3x3',
    'perc_holes_5x5': '5x5'
})
print(melted_df)

# Plotting with Seaborn
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_df, x='y', y='Value', hue='Type', showfliers=False)
plt.title('Distribution of holes by Label and Mask Type', fontsize=16, fontweight='bold')
plt.xlabel('Class label', fontsize=14)
plt.ylabel('Holes Ratio', fontsize=14)
plt.legend(title='Mask Type')    

#save the plot
plt.savefig(output_path+'cma_holes_ratio.png', bbox_inches='tight') 
