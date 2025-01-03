from torchvision import transforms
from PIL import Image
import numpy as np

# Load an image
image_path = "path/to/your/image.jpg"
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjust for your dataset normalization
])
input_image = transform(image).unsqueeze(0).to(device)

# Hook to capture activations
activations = {}

def hook_fn(module, input, output):
    activations[module] = output

# Register hook on a specific convolutional layer
layer_to_hook = conv_layers[0]
hook = layer_to_hook.register_forward_hook(hook_fn)

# Forward pass
model(input_image)

# Get the activation
activation = activations[layer_to_hook].squeeze().cpu().numpy()
print(f"Activation shape: {activation.shape}")  # (num_filters, h, w)

# Visualize the first 10 activations
fig, axes = plt.subplots(1, 10, figsize=(15, 15))
for i, ax in enumerate(axes):
    if i >= activation.shape[0]: break
    ax.imshow(activation[i, :, :], cmap="viridis")
    ax.axis("off")
plt.show()

# Remove hook after use
hook.remove()
