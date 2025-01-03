from torch.optim import Adam

# Make the model layers trainable
for param in model.parameters():
    param.requires_grad = False

# Start with a random noise image
dream_image = torch.randn(1, 3, 128, 128, device=device, requires_grad=True)
optimizer = Adam([dream_image], lr=0.1)

# Target layer and feature map
target_layer = conv_layers[0]
feature_map_idx = 0

# Optimization loop
for step in range(30):
    optimizer.zero_grad()
    activations = target_layer(dream_image)
    loss = -activations[0, feature_map_idx].mean()  # Maximize activation
    loss.backward()
    optimizer.step()

# Visualize the dream image
dream_image_np = dream_image.squeeze().detach().cpu().numpy()
plt.imshow(np.transpose(dream_image_np, (1, 2, 0)))
plt.show()
