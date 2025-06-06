
import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import os
from scipy.ndimage import binary_closing
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Configuration
reduction_method = 'tsne'
run_name = 'dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA'
random_state = '3'
sampling_type = 'all'
output_path = f'/data1/fig/{run_name}/{sampling_type}/'

csv_path = f'{output_path}merged_{reduction_method}_variables_{run_name}_{sampling_type}_{random_state}.csv'

# Load the DataFrame
df = pd.read_csv(csv_path)
print(df['label'].unique())

def fill_closed_black_patches(img, structure=np.ones((3, 3))):
    """
    Apply morphological closing to black patches (pixels with value 0)
    and replace them with the mean of surrounding pixels.
    """
    img = img.copy()
    mask = img == 0
    closed_mask = binary_closing(mask, structure=structure)
    filled_mask = closed_mask & ~mask  # Pixels that were filled by closing

    # Replace each filled pixel with the mean of its valid (non-zero) neighbors
    for y, x in zip(*np.where(filled_mask)):
        neighborhood = img[max(0, y-1):y+2, max(0, x-1):x+2]
        valid_neighbors = neighborhood[neighborhood > 0]
        if valid_neighbors.size > 0:
            img[y, x] = np.mean(valid_neighbors)

    return img

def plot_sample_grid_per_class(
    df,
    n_cols=5,
    selection_mode='closest',
    output_dir=None,
    title="Sample Images per Class",
    close_black_patches=False
):


    # Determine classes
    unique_labels = sorted(df['label'].unique())

    # Set up plot grid
    n_rows = len(unique_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)  # Handle single row case
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)  # Handle single col case

    for row_idx, label in enumerate(unique_labels):
        class_df = df[df['label'] == label].copy()

        if selection_mode == 'closest':
            class_df = class_df.sort_values(by='distance', ascending=False)
        elif selection_mode == 'furthest':
            class_df = class_df.sort_values(by='distance', ascending=True)
        elif selection_mode == 'random':
            class_df = class_df.sample(frac=1, random_state=42)

        selected_paths = class_df['path'].head(n_cols).to_list()

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis('off')

            if col_idx >= len(selected_paths):
                continue  # Leave empty if fewer samples than n_cols

            crop_path = selected_paths[col_idx]
            if not os.path.exists(crop_path):
                continue

            try:
                img = imageio.imread(crop_path)

                # Optional closing operation to remove black patches
                if close_black_patches:
                    img = fill_closed_black_patches(img, structure=np.ones((3, 3)))
                
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)

                if row_idx == 0:
                    ax.set_title(f"#{col_idx + 1}", fontsize=16)

                if col_idx == 0:
                    ax.set_ylabel(f"Label {label}", fontsize=16)

            except Exception as e:
                print(f"Failed to load image: {crop_path} | Error: {e}")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"sample_grid_{selection_mode}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
        print(f"Saved: {output_file}")
    else:
        plt.show()

    plt.close()


def process_and_save_crops(df, output_base_dir, apply_closing=True, n_samples=None, selection_mode='all'):
    """
    Processes and saves image crops as PNGs per label/class.
    Images are ordered and selected based on distance and selection_mode.

    Parameters:
        df (pd.DataFrame): DataFrame with 'path', 'label', and 'distance'.
        output_base_dir (str): Directory where output PNGs will be saved.
        apply_closing (bool): Apply 3x3 morphological closing to images.
        n_samples (int or None): Number of images to save per class. If None, all are saved.
        selection_mode (str): One of {'closest', 'furthest', 'random', 'all'}.
    """
    assert selection_mode in ['closest', 'furthest', 'random', 'all'], "Invalid selection_mode"

    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) if apply_closing else None

    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label].copy()

        if selection_mode == 'closest':
            class_df = class_df.sort_values(by='distance', ascending=False)  # Closest = highest distance
        elif selection_mode == 'furthest':
            class_df = class_df.sort_values(by='distance', ascending=True)   # Furthest = lowest distance
        elif selection_mode == 'random':
            class_df = class_df.sample(frac=1, random_state=42)

        if n_samples is not None:
            class_df = class_df.head(n_samples).reset_index(drop=True)
        else:
            class_df = class_df.reset_index(drop=True)

        label_dir = os.path.join(output_base_dir, selection_mode, str(label))
        print(f"Creating directory: {label_dir}")
        os.makedirs(label_dir, exist_ok=True)

        print(f"Processing label {label} - mode: {selection_mode} - saving {len(class_df)} images")

        for idx, row in tqdm(class_df.iterrows(), total=len(class_df), desc=f"Label {label}"):
            try:
                img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: could not load image at {row['path']}")
                    continue

                if apply_closing:
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, struct_element)

                out_path = os.path.join(label_dir, f"{idx + 1}.png")
                cv2.imwrite(out_path, img)

            except Exception as e:
                print(f"Error processing {row['path']}: {e}")


#main execution
output_png_base = f'/data1/fig/{run_name}/{sampling_type}/crops_png'
os.makedirs(output_png_base, exist_ok=True)
# process_and_save_crops(df, 
#                        output_png_base, 
#                        apply_closing=True, 
#                        n_samples=50, 
#                        selection_mode='closest')

plot_sample_grid_per_class(
    df=df,
    n_cols=10,
    selection_mode='furthest',
    output_dir="/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/all/sample_grids",
    close_black_patches=True
)