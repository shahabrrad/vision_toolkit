import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.io as io

# Load and convert image to grayscale


def load_image(image_path):
    image = io.read_image(image_path).float() / 255.0
    if image.shape[0] == 3:  # Convert to grayscale if RGB
        transform = T.Grayscale(num_output_channels=1)
        image = transform(image)
    return image

# Define the Laplacian of Gaussian filter


def log_filter(sigma, kernel_size):
    """Generates a Laplacian of Gaussian filter."""
    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = torch.meshgrid([ax, ax], indexing="ij")
    square_distance = (xx**2 + yy**2).float()
    log_kernel = (1 / (2 * torch.pi * sigma ** 2)) * (1 - square_distance /
                                                      (2 * sigma ** 2)) * torch.exp(-square_distance / (2 * sigma ** 2))
    return log_kernel - log_kernel.mean()  # Normalizing to avoid bias

# Build Laplacian Scale Space


def build_laplacian_scale_space(image, n=10, initial_sigma=2.0, k=2**(0.35)):
    scale_space = []
    current_sigma = initial_sigma
    current_image = image

    for i in range(n):
        # Fixed kernel size based on initial sigma
        kernel_size = int(2 * round(2.5 * initial_sigma) + 1)
        log_kernel = log_filter(initial_sigma, kernel_size).unsqueeze(
            0).unsqueeze(0)  # Add batch and channel dims

        # Apply filter to the current image (downsampled version)
        response = F.conv2d(current_image.unsqueeze(
            0), log_kernel, padding=kernel_size//2)
        response_sq = response ** 2

        # Upsample response back to original size for maxima detection
        if current_image.shape[-2:] != image.shape[-2:]:
            response_sq = F.interpolate(
                response_sq, size=image.shape[-2:], mode='bilinear', align_corners=False)

        # Store the upsampled square response
        scale_space.append(response_sq.squeeze(0))

        # Downsample the image by factor 1/k
        current_image = F.interpolate(current_image.unsqueeze(
            0), scale_factor=1/k, mode='bilinear', align_corners=False).squeeze(0)
        current_sigma *= k  # Update scale
        print(f"Scale {i+1} done")

    return scale_space

# Perform non-maximum suppression in scale space
# def non_max_suppression(scale_space, threshold=0.01):
#     keypoints = []

#     for i, response in enumerate(scale_space):
#         response = response.squeeze(0)  # Remove batch dimension

#         # Iterate over each pixel in the response
#         for x in range(1, response.shape[0] - 1):
#             for y in range(1, response.shape[1] - 1):

#                 # Extract the 3x3 neighborhood
#                 local_patch = response[x-1:x+2, y-1:y+2]
#                 center_value = response[x, y]

#                 # Check if the center is the maximum in the neighborhood and above threshold
#                 if center_value == local_patch.max() and center_value > threshold:
#                     keypoints.append((x, y, i))  # Append x, y coordinates and scale index (i)

#     return keypoints


def non_max_suppression(scale_space, neighborhood_size=5, threshold=0.01):
    keypoints = []

    # Half size of the neighborhood
    half_size = neighborhood_size // 2

    for i, response in enumerate(scale_space):
        response = response.squeeze(0)  # Remove batch dimension

        # Iterate over each pixel in the response
        for x in range(half_size, response.shape[0] - half_size):
            for y in range(half_size, response.shape[1] - half_size):

                # Extract the neighborhood (larger than 3x3)
                local_patch = response[x-half_size:x +
                                       half_size+1, y-half_size:y+half_size+1]
                center_value = response[x, y]

                # Check if the center is the maximum in the neighborhood and above threshold
                if center_value == local_patch.max() and center_value > threshold:
                    # Append x, y coordinates and scale index (i)
                    keypoints.append((x, y, i))

    return keypoints


# Visualization of keypoints
def draw_all_circle(image, cx, cy, rad, filename):
    fig, ax = plt.subplots(1)
    ax.imshow(image[0], cmap='gray')

    for x, y, r in zip(cx, cy, rad):
        circle = plt.Circle((y, x), r, color='r', fill=False)
        ax.add_patch(circle)

    plt.savefig(filename)
    plt.show()


# Example usage
image_path = 'data/part2/butterfly.jpg'
image = load_image(image_path)

# Build scale space with n levels
n = 10
scale_space = build_laplacian_scale_space(image, n=n)

# Perform non-max suppression to find keypoints
keypoints = non_max_suppression(scale_space)

# Extract coordinates and radius for each keypoint
cx = [kp[0] for kp in keypoints]
cy = [kp[1] for kp in keypoints]
rad = [int(kp[2]) for kp in keypoints]

# Visualize the keypoints on the original image
draw_all_circle(image.numpy(), cx, cy, rad, 'keypoints.png')
