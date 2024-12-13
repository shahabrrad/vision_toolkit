import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from utils.io_helper import torch_read_image, torch_save_image
from utils.draw_helper import draw_all_circles


def create_gaussian_kernel(size, sigma, device='cpu'):
    """Create a 2D Gaussian kernel using PyTorch."""
    x = torch.linspace(-(size-1)/2, (size-1)/2, size, device=device)
    x, y = torch.meshgrid(x, x, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2)/(2*sigma**2))
    return kernel / kernel.sum()


def create_laplacian_of_gaussian(size, sigma, device='cpu'):
    """Create a Laplacian of Gaussian (LoG) kernel using PyTorch."""
    x = torch.linspace(-(size-1)/2, (size-1)/2, size, device=device)
    x, y = torch.meshgrid(x, x, indexing='ij')

    # LoG equation
    log = -(1/(np.pi * sigma**4)) * (1 - (x**2 + y**2)/(2*sigma**2)) * \
        torch.exp(-(x**2 + y**2)/(2*sigma**2))
    return log - log.mean()  # normalize to zero sum


def convolve2d(image, kernel):
    """Apply 2D convolution using PyTorch."""
    # Add batch and channel dimensions
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)

    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply convolution
    return F.conv2d(image, kernel, padding=kernel.shape[-1]//2)


def build_scale_space(image_tensor, n_scales, k, initial_sigma, device='cpu'):
    """Build the scale space pyramid using PyTorch operations."""
    scales = []
    responses = []
    current_sigma = initial_sigma

    # Ensure image is on the correct device
    current_image = image_tensor.to(device)

    for i in range(n_scales):
        # Create LoG kernel for current scale
        kernel_size = 2 * int(3 * current_sigma) + 1
        kernel = create_laplacian_of_gaussian(
            kernel_size, current_sigma, device)

        # Apply filter and normalize by scale
        response = current_sigma**2 * convolve2d(current_image, kernel)
        response = response.squeeze()  # Remove batch and channel dimensions

        # Store results
        scales.append(current_sigma)
        responses.append(response)

        # Downsample image for next iteration
        if i < n_scales - 1:
            new_size = (
                int(current_image.shape[-2]/k), int(current_image.shape[-1]/k))
            current_image = F.interpolate(
                current_image, size=new_size, mode='bilinear', align_corners=False)
            current_sigma *= k

    return scales, responses


def detect_blobs(image_path, n_scales=10, k=1.4, initial_sigma=1.0, threshold=0.1, device='gpu'):
    """Detect blobs in the image using Laplacian scale space."""
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'L':
        image = image.convert('L')

    # Convert to tensor and normalize
    image_tensor = TF.to_tensor(image).to(device)

    # Build scale space
    scales, responses = build_scale_space(
        image_tensor, n_scales, k, initial_sigma, device)

    # Initialize lists for keypoints
    keypoints_x = []
    keypoints_y = []
    keypoints_sigma = []

    original_size = (image_tensor.shape[-2], image_tensor.shape[-1])

    # Process each scale level
    for i in range(1, len(responses)-1):
        # Resize current level responses to original image size
        current = F.interpolate(responses[i].unsqueeze(0).unsqueeze(0),
                                size=original_size,
                                mode='bilinear',
                                align_corners=False).squeeze()
        prev = F.interpolate(responses[i-1].unsqueeze(0).unsqueeze(0),
                             size=original_size,
                             mode='bilinear',
                             align_corners=False).squeeze()
        next = F.interpolate(responses[i+1].unsqueeze(0).unsqueeze(0),
                             size=original_size,
                             mode='bilinear',
                             align_corners=False).squeeze()

        # Find local maxima in 3x3x3 neighborhood
        for y in range(1, current.shape[0]-1):
            for x in range(1, current.shape[1]-1):
                # Check if current point is local maximum in spatial domain
                window = current[y-1:y+2, x-1:x+2]
                if current[y, x] != torch.max(window):
                    continue

                # Check if local maximum in scale domain
                if current[y, x] < prev[y, x] or current[y, x] < next[y, x]:
                    continue

                # Threshold check
                if current[y, x] < threshold:
                    continue

                # Store keypoint
                keypoints_y.append(y)
                keypoints_x.append(x)
                keypoints_sigma.append(scales[i])

    return keypoints_x, keypoints_y, keypoints_sigma


def detect_and_draw_blobs(image_path, output_path, n_scales=10, k=1.4, initial_sigma=1.0, threshold=0.1, device='cpu'):
    """Detect blobs and visualize them using the provided draw_all_circle function."""
    # Load image for visualization
    image = Image.open(image_path)
    image_np = np.array(image)

    # Detect blobs
    cx, cy, sigmas = detect_blobs(
        image_path, n_scales, k, initial_sigma, threshold, device)

    # Convert sigmas to radii (approximately 1.414 * sigma)
    radii = [int(1.414 * s) for s in sigmas]

    # Draw circles
    draw_all_circles(image_np, cx, cy, radii, output_path)

    return cx, cy, sigmas


# Example usage:
"""
# Select device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Detect and visualize blobs
image_path = "your_image.jpg"
output_path = "output_image.jpg"
cx, cy, sigmas = detect_and_draw_blobs(
    image_path, 
    output_path,
    n_scales=10,
    k=1.4,
    initial_sigma=1.0,
    threshold=0.1,
    device=device
)
"""


detect_and_draw_blobs("data/part2/butterfly.jpg", "output")
