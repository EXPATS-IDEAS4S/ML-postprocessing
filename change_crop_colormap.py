import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define the paths
input_path = '/home/Daniele/data/ir108_2013_128_germany/1/'
output_path = '/home/Daniele/data/ir108_2013_128_germany/1/'
colormap = 'Spectral'  # Change this to any matplotlib colormap

# Ensure the output path exists
os.makedirs(output_path, exist_ok=True)

# List all TIFF files in the input directory
tiff_files = sorted([f for f in os.listdir(input_path) if f.endswith('.tif')])

def apply_colormap_to_image(image_path, colormap):
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Ensure the image has 3 bands (RGB)
    if image.mode != 'RGB':
        raise ValueError(f"Image {image_path} is not an RGB image.")
    
    # Convert the image to grayscale (luminance)
    grayscale_image = image.convert('L')
    
    # Convert grayscale image to numpy array
    grayscale_array = np.array(grayscale_image)
    
    # Normalize the grayscale image to range 0-1
    normalized_array = grayscale_array / 255.0
    
    # Apply the colormap
    colormap_func = cm.get_cmap(colormap)
    colored_array = colormap_func(normalized_array)
    
    # Convert the result to 8-bit RGB format
    colored_image = Image.fromarray((colored_array[:, :, :3] * 255).astype(np.uint8))
    
    return colored_image

# Process each TIFF file
for tiff_file in tiff_files:
    input_file_path = os.path.join(input_path, tiff_file)
    output_file_path = os.path.join(output_path, tiff_file)
    
    try:
        # Apply colormap to the image
        modified_image = apply_colormap_to_image(input_file_path, colormap)
        
        # Save the modified image
        modified_image.save(output_file_path)
        print(f'Saved modified image to: {output_file_path}')
    except ValueError as e:
        print(e)

print('Processing completed.')

