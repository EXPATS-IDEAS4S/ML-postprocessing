
import numpy as np
import pandas as pd
from glob import glob
import torch
import os

from embedding_plotting_func import plot_average_crop_shapes, plot_embedding_crops_table, plot_embedding_crops_new, plot_embedding_dots_iterative_test_msg_icon, scale_to_01_range, name_to_rgb, extract_hour, plot_embedding_dots, plot_embedding_filled, plot_embedding_crops, plot_embedding_dots_iterative_case_study
from embedding_plotting_func import plot_average_crop_values, plot_embedding_crops_grid, plot_embedding_crops_binned_grid, create_WV_IR_diff_colormap, plot_classwise_grids

run_name = 'dcv2_ir108-cm_100x100_8frames_k9_70k_nc_r2dplus1'
crops_name = 'clips_ir108_100x100_8frames_2013-2020'  # Name of the crops
random_state = '3' #all visualization were made with random state 3
sampling_type = 'all' 
reduction_method = 'tsne' # Options: 'tsne', 'isomap',
epoch = 800  # Epoch number for the run
file_extension = 'png'  # Image file extension
substitute_path = True
variable_type = 'IR_108_cm' #'WV_062-IR_108'  # Variable type for the crops, e.g., 'ir108', 'wv062'
VIDEO = True
N_FRAMES = 8


vmin, center, vmax = -60, 0, 5
cmap = 'gray' #create_WV_IR_diff_colormap(vmin, center, vmax)  #'gray' #

output_path = f'/data1/fig/{run_name}/epoch_{epoch}/{sampling_type}/'
filename = f'{reduction_method}_pca_cosine_perp-50_{run_name}_{random_state}_epoch_{epoch}.npy'

# List of the image crops
image_crops_path = f'/data1/crops/{crops_name}/img/{variable_type}/1/'
#image_crops_path = f'/data1/crops/{crops_name}/img/1/'
list_image_crops = sorted(glob(image_crops_path+'*.'+ file_extension))

n_samples = len( list_image_crops)
print('n samples: ', n_samples)

# Read data
if sampling_type == 'all':
    n_subsample = n_samples  # Number of samples per cluster
else:
    n_subsample = 1000

# Open csv file with already labels and dim red features
df_labels = pd.read_csv(f'{output_path}merged_tsne_crop_list_{run_name}_{sampling_type}_{random_state}_epoch_{epoch}.csv')
#list all columns name
df_labels = df_labels.loc[:, ~df_labels.columns.str.contains('^color')]
print(df_labels.columns)
print(len(df_labels))
#print(df_labels)
#print(df_labels['crop_index'].tolist()[:5])

#print(df_labels['crop_index'].tolist()[:-5])
#print(df_labels['path'].tolist()[5:10])  # Print first 5 paths to check



# # Define the color names for each class from dataframe

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


# add column with path to images for case studied
# Add a new column based on condition

# path_msg_images = '/data1/other_data/Case_Studies/Marche_Flood_22/MSG/image_to_plot/'
# # add the path to the images for the case study
# #df_labels['msg_image_path'] = df_labels['case_study_msg'].map(lambda x: path_msg_images if x else '')
# df_labels['msg_image_path'] = df_labels.apply(
# lambda row: path_msg_images if row['vector_type'] == 'msg' and row['case_study'] else '',
# axis=1
# )


# path_icon_images = '/data1/other_data/Case_Studies/Marche_Flood_22/ICON/image_to_plot/'
# # add the path to the images for the case study
# #df_labels['icon_image_path'] = df_labels['case_study_icon'].map(lambda x: path_icon_images if x else '')
# df_labels['icon_image_path'] = df_labels.apply(
# lambda row: path_icon_images if row['vector_type'] == 'icon' and row['case_study'] else '',
# axis=1
# ) 

# Apply the extraction function to get the hours
#df_labels['hour'] = df_labels['location'].apply(extract_hour)

# Filter out invalid labels (-100)
df_subset1 = df_labels[df_labels['label'] != -100]

# Map labels to colors
df_subset1['color'] = df_subset1['label'].map(lambda x: colors_per_class1_names[str(int(x))])

# Sample 20,000 points for plotting
df_subset2 = df_subset1.sample(n=200)



plot_embedding_dots(df_subset1, colors_per_class1_names, output_path, filename)
#plot_embedding_filled(df_subset2, colors_per_class1_names, output_path, filename, df_subset)
#plot_embedding_dots_iterative_case_study(df_subset1, colors_per_class1_names, output_path+'trajectory_iter/', filename, df_subset1)
#plot_embedding_dots_iterative_test_msg_icon(df_subset1, colors_per_class1_names, output_path+'trajectory_iter/', filename, df_subset1, legend=True)
#plot_embedding_crops_new(df_subset2, output_path, filename)
#plot_embedding_crops_table(df_subset1, output_path, filename, n=10 ,selection='random')
#plot_classwise_grids(df_subset1, output_path, filename,cmap, n=100, selection="closest")
#plot_average_crop_shapes(df_subset1, output_path+'shade_maps/', filename, n=1000, selection="closest", alpha=0.001)
#plot_average_crop_values(df_subset1, output_path+'avarage_maps/', filename, n=1000, selection="closest")
#plot_embedding_crops_binned_grid(df_subset1, output_path, filename, grid_size=20, zoom=0.28)

#plot_embedding_crops_grid(df_subset1, output_path, filename, variable_type, cmap, grid_size=20, zoom=0.33)

# # Sample df labels and list of image crops to get same index
# # Step 1: sample crop indices from df_labels
# sampled_indices = df_labels['crop_index'].drop_duplicates().sample(n=1000, random_state=42)

# # Step 2: filter df_labels to only those crop indices
# df_labels = df_labels[df_labels['crop_index'].isin(sampled_indices)]

# # Step 3: also filter list_image_crops
# list_image_crops = [p for p in list_image_crops 
#                     if any(f"crop{idx}_" in p for idx in sampled_indices)]
#

#open the expanded dataset if alreasdy availabel
expanded_csv_path = os.path.join(
    os.path.dirname(output_path),
    f"merged_tsne_crop_list_{run_name}_{sampling_type}_{random_state}_epoch_{epoch}_expanded.csv"
)
if os.path.exists(expanded_csv_path):
    df_frame = pd.read_csv(expanded_csv_path)
    print(df_frame)

    dupes = (
    df_frame
    .groupby("crop_index")["path"]
    .nunique()
    )

    print((dupes == 1).sum(), "crops have identical paths across frames")
    print((dupes > 1).sum(), "crops have different paths across frames")

    # Filter out ignored labels
    df_frame = df_frame[df_frame['label'] != -100]

    #df_frame['path'] = df_frame['crop_index'].apply(lambda x: list_image_crops[int(x)])

    if not df_frame.empty:
        for frame_idx in range(N_FRAMES):
            print(f'plotting frame {frame_idx}')
            #filter df_frame
            df_frame_filtered = df_frame[df_frame['frame_idx'] == frame_idx]
            print(df_frame_filtered[['frame_idx', 'Component_1', 'Component_2', 'path']].head(20))

            #print the columns
            #print(df_frame_filtered['path'].tolist())
            # Plot immediately
            #print("Unique paths in this frame:", df_frame_filtered['path'].nunique())
            #print(df_frame_filtered[['frame_idx', 'crop_index', 'path']].head(20))

            plot_embedding_crops_grid(
                df_frame_filtered, 
                output_path, 
                filename=f"{os.path.splitext(filename)[0]}_frame{frame_idx}.png",
                variable_type=variable_type, 
                cmap=cmap, 
                grid_size=20, 
                zoom=0.33
            )
else:
    if substitute_path and VIDEO:
        for frame_idx in range(N_FRAMES):
            frame_rows = []

            for _, row in df_labels.iterrows():
                # video stem from .nc path
                video_stem = os.path.splitext(os.path.basename(row['path']))[0]

                frame_str = f"t{frame_idx}_"
                matches = [p for p in list_image_crops if video_stem in p and frame_str in p]

                if not matches:
                    # skip this video for this frame
                    continue  

                new_row = row.copy()
                new_row['path'] = matches[0]
                new_row['frame_idx'] = frame_idx
                frame_rows.append(new_row)

            # Build dataframe for this frame
            df_frame = pd.DataFrame(frame_rows)

            # Filter out ignored labels
            df_frame = df_frame[df_frame['label'] != -100]

            if not df_frame.empty:
                # Plot immediately
                plot_embedding_crops_grid(
                    df_frame, 
                    output_path, 
                    filename=f"{os.path.splitext(filename)[0]}_frame{frame_idx}.png",
                    variable_type=variable_type, 
                    cmap=cmap, 
                    grid_size=20, 
                    zoom=0.33
                )

    else:
        if substitute_path:
            # Single-frame crops assignment
            df_labels['path'] = df_labels['crop_index'].apply(
                lambda x: list_image_crops[int(x)]
            )
            df_labels = df_labels[df_labels['label'] != -100]

        # Plot normally
        plot_embedding_crops_grid(
            df_labels, 
            output_path, 
            filename, 
            variable_type, 
            cmap, 
            grid_size=20, 
            zoom=0.33
        )
