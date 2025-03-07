import torch
import matplotlib.pyplot as plt

model_path = "/data1/runs/dcv2_ir108_128x128_k9_expats_70k_200-300K_CMA/checkpoints/model_final_checkpoint_phase799.torch"
checkpoint = torch.load(model_path)
print(checkpoint.keys())
state_dict = checkpoint['classy_state_dict']  # This is where the model weights are likely stored.
print(state_dict.keys())
model_state_dict = state_dict['base_model']
print(model_state_dict.keys())
model_weights = model_state_dict['model']
print(model_weights.keys())
print('...')
print(model_weights['trunk'].keys())
meta = model_state_dict['meta']

exit()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YourModelClass()  # Replace with your model definition
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Visualize convolutional layer weights
conv_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
print(f"Number of Conv layers: {len(conv_layers)}")
exit()

# Plot filters of the first conv layer
first_conv = conv_layers[0]
weights = first_conv.weight.data.cpu().numpy()
print(f"Shape of filters: {weights.shape}")  # (out_channels, in_channels, h, w)

# Visualizing first 10 filters
fig, axes = plt.subplots(1, 10, figsize=(15, 15))
for i, ax in enumerate(axes):
    if i >= weights.shape[0]: break
    ax.imshow(weights[i, 0, :, :], cmap="viridis")
    ax.axis("off")
plt.show()
