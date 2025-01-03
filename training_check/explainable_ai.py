from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Define the target layer (e.g., last conv layer of ResNet)
target_layer = conv_layers[-1]

# Initialize Grad-CAM
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

# Generate CAM for a specific class (e.g., class index 0)
grayscale_cam = cam(input_tensor=input_image, targets=None)
grayscale_cam = grayscale_cam[0, :]  # Get the first image's CAM

# Overlay CAM on the input image
input_image_np = np.transpose(input_image.squeeze().cpu().numpy(), (1, 2, 0))
input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())
cam_image = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)

# Show the Grad-CAM
plt.imshow(cam_image)
plt.axis("off")
plt.show()
