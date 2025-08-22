import os
from PIL import Image
import numpy as np
import os
import re
import glob

def create_gif_from_images(image_list, output_gif_path, duration=500):
    """
    Creates a GIF from JPEG images stored in a specified folder.

    Parameters:
    - folder_path: str, the path to the folder containing the JPEG images.
    - output_gif_path: str, the path where the GIF should be saved.
    - duration: int, duration in milliseconds between frames in the GIF.
    """
    # Create a list to store the images
    images = []

    for image_path in image_list:
        # Open each image and append to the list
        #image_path = os.path.join(folder_path, image_file)
        img = Image.open(image_path)
        images.append(img)

    # Save the images as a GIF
    if images:
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {output_gif_path}")
    else:
        print("No images found in the specified folder.")

def extract_hour_number(filepath):
    # Extracts the number after 'hour_' in the filename
    filename = os.path.basename(filepath)
    match = re.search(r'hour_(\d+)', filename)
    return int(match.group(1)) if match else -1

labels = np.arange(0, 9)

for label in labels:
   
    folder_path = f'/data1/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/all/percentile_maps/{label}/'
    output_gif_path = f'{folder_path}evolving_weather_maps_label_{label}.gif'
    # Get a list of all JPEG images in the folder, sorted by filename
    image_files = sorted(
    glob.glob(os.path.join(folder_path, '*.png')),
    key=extract_hour_number)
    print(image_files)
  
    create_gif_from_images(image_files, output_gif_path, duration=500)
