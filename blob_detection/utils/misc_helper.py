"""Implements Misc helper functions."""

import torch
import torch.nn.functional as F


def custom_shifts(input, shifts, dims=None, padding='circular'):
    """Shifts the input tensor by the specified shifts along the specified dimensions.
       Supports circular and zero padding.
    """
    ret = torch.roll(input, shifts, dims)
    if padding == 'zero':
        ret[:shifts[0], :shifts[1]] = 0
    return ret


def log_filter(sigma, kernel_size):
    """Generates a Laplacian of Gaussian filter."""
    # 1. Create a grid of coordinates for the kernel (from -size//2 to size//2)
    ax = torch.arange(-(kernel_size // 2), kernel_size //
                      2 + 1)  # Range of x and y
    # Create a 2D grid of coordinates
    xx, yy = torch.meshgrid([ax, ax], indexing="ij")

    # 2. Calculate the squared distance from the center for each point in the grid
    square_distance = (xx**2 + yy**2).float()

    # 3. Apply the Laplacian of Gaussian formula
    log_kernel = (1 / (2 * torch.pi * sigma ** 2)) * (
        1 - square_distance / (2 * sigma ** 2)
    ) * torch.exp(-square_distance / (2 * sigma ** 2))

    # 4. Normalize the kernel by subtracting the mean to ensure no bias
    return log_kernel - log_kernel.mean()  # Subtract mean to normalize


def build_laplacian_scale_space(image, n=5, initial_sigma=1.0, ksize=7, k=2**(0.35)):
    scale_space = []
    sigmas = []
    current_sigma = initial_sigma
    current_image = image

    for i in range(n):
        print(f"Processing scale {i+1}...")

        # Use a constant kernel size across all scales
        print(f"Using constant kernel size: {ksize}")
        # kernel_size = int(2 * round(3 * current_sigma) + 1)
        # ksize = kernel_size
        # Apply the Laplacian of Gaussian filter with a constant kernel size
        log_kernel = log_filter(current_sigma, ksize).unsqueeze(0).unsqueeze(0)

        # Apply filter
        response = F.conv2d(current_image.unsqueeze(0),
                            log_kernel, padding=ksize // 2)
        response_sq = response ** 2

        # Upsample response back to original size for maxima detection
        if current_image.shape[-2:] != image.shape[-2:]:
            response_sq = F.interpolate(
                response_sq, size=image.shape[-2:], mode='bilinear', align_corners=False)

        scale_space.append(response_sq.squeeze(0))
        # sigmas.append(current_sigma)
        # Store the sigma for the current scale
        sigmas.append(current_sigma * (k**i))

        # Downsample image for the next scale
        current_image = F.interpolate(current_image.unsqueeze(
            0), scale_factor=1/k, mode='bilinear', align_corners=False).squeeze(0)
        # current_sigma *= k  # Update sigma for the next scale

    return scale_space, sigmas


def non_max_suppression(scale_space, sigmas, threshold=0.01):
    keypoints = []
    print("len", len(scale_space))
    for i, response in enumerate(scale_space):
        print("response shape", response.shape)
        response = response.squeeze(0)  # Remove batch dimension

        # Iterate over each pixel in the response
        for x in range(1, response.shape[0] - 1):
            for y in range(1, response.shape[1] - 1):

                # Extract the 3x3 neighborhood
                local_patch = response[x-1:x+2, y-1:y+2]
                center_value = response[x, y]

                # Check if the center is the maximum in the neighborhood and above threshold
                if center_value == local_patch.max() and center_value > threshold:

                    sigma = sigmas[i]  # Use the actual sigma instead of index
                    # Append x, y coordinates and sigma (not the index)
                    keypoints.append((x, y, sigma))

    return keypoints
