import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
from utils.draw_helper import draw_all_circles
from utils.io_helper import torch_read_image

# Function to convert image to grayscale


def convert_to_grayscale(image):
    transform = transforms.Grayscale()
    grayscale_img = transform(image)
    return grayscale_img

# Function to generate the Laplacian of Gaussian filter


def log_filter(size, sigma):
    n = size // 2
    y, x = np.ogrid[-n:n+1, -n:n+1]
    y_filter = (y ** 2 + x ** 2 - 2 * sigma ** 2) / (sigma ** 4)
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    log = y_filter * gaussian
    log /= np.sum(np.abs(log))  # Normalize the filter
    return torch.tensor(log, dtype=torch.float32)

# Step 3: Build Laplacian scale space


def laplacian_scale_space(image, init_sigma=1.0, num_scales=10, scale_factor=1.2):
    sigma = init_sigma
    laplacian_responses = []
    for _ in range(num_scales):
        # Create LoG filter at the current scale
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        log_kernel = log_filter(kernel_size, sigma)

        # Apply filter to the image
        filtered_image = F.conv2d(image.unsqueeze(0).unsqueeze(
            0), log_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        laplacian_responses.append((filtered_image.squeeze().numpy()) ** 2)

        # Update sigma
        sigma *= scale_factor

    return laplacian_responses

# Step 4: Non-maximum suppression


def non_max_suppression(scale_space, threshold):
    scale_space_np = np.stack(scale_space, axis=2)
    local_max = ndimage.maximum_filter(scale_space_np, size=(3, 3, 3))
    maxima = (scale_space_np == local_max) & (scale_space_np > threshold)

    keypoints = np.argwhere(maxima)
    return keypoints

# Step 5: Visualize the keypoints with the provided function
# def draw_all_circle(image, cx, cy, rad, filename):
#     img_copy = image.copy()
#     for (x, y, r) in zip(cx, cy, rad):
#         cv2.circle(img_copy, (x, y), r, (255, 0, 0), 2)

#     plt.imshow(img_copy, cmap='gray')
#     plt.title('Detected Blobs')
#     plt.axis('off')
#     plt.savefig(filename)
#     plt.show()

# Main function to detect blobs


def detect_blobs(image_path, num_scales=10, scale_factor=1.1, threshold=0.1):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_tensor = torch.tensor(image, dtype=torch.float32)
    # image_tensor = torch_read_image(image_path, gray=True)

    # Step 3: Build scale space
    scale_space = laplacian_scale_space(
        image_tensor, num_scales=num_scales, scale_factor=scale_factor)

    # Step 4: Perform non-maximum suppression
    keypoints = non_max_suppression(scale_space, threshold)

    # Extract keypoint coordinates and scales
    cx, cy, rad = keypoints[:, 1], keypoints[:, 0], keypoints[:, 2]

    # Step 5: Draw the circles
    draw_all_circles(image, cx, cy, rad, "output.png")


# Example usage:
detect_blobs("data/part2/sunflowers.jpg")
