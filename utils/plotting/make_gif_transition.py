
import os
from PIL import Image
import re

def create_gif_from_images(folder_path, output_gif_path, duration=500):
    """
    Creates a GIF from JPEG images stored in a specified folder.

    Parameters:
    - folder_path: str, the path to the folder containing the JPEG images.
    - output_gif_path: str, the path where the GIF should be saved.
    - duration: int, duration in milliseconds between frames in the GIF.
    """

    # Get a list of all PNG files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Extract step number from filename and sort numerically
    image_files.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))


    # Create a list to store the images
    images = []

    for image_file in image_files:
        # Open each image and append to the list
        image_path = os.path.join(folder_path, image_file)
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

# Example usage:
folder_path = '/home/Daniele/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon/trajectory_iter'
output_gif_path = '/home/Daniele/fig/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA_test_msg_icon/marche_flood_22_msg_icon_transition.gif' 

create_gif_from_images(folder_path, output_gif_path, duration=500)

