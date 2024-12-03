import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def apply_alignment(source, translation):
    """Shifts the source image based on the transformation provided"""
    # Generate final aligned image
    source = source.unsqueeze(0)
    batch_size, channels, height, width = source.shape
    with torch.no_grad():
        theta = torch.eye(2, 3).unsqueeze(0)
        theta[0, :, 2] = translation
        grid = F.affine_grid(theta, source.size(), align_corners=False)
        aligned = F.grid_sample(source, grid, align_corners=False)
    aligned = aligned.squeeze(0)
    dx_pixels = -translation[0] * width / 2
    dy_pixels = -translation[1] * height / 2
    return aligned, (dx_pixels, dy_pixels)


def normalized_cross_correlation(x, y):
    x = x - torch.mean(x)
    y = y - torch.mean(y)
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)
    ncc = torch.sum(x * y) / (norm_x * norm_y + 1e-8)
    # Return 1 - NCC to make it similar to a loss (minimize NCC)
    return 1 - ncc


def align_images(source, target, max_iterations=100, learning_rate=0.01, loss_type='mse'):
    """
    Align source image to target image using only x and y translations

    Args:
        source: Source image tensor of shape (1, channels, height, width)
        target: Target image tensor of shape (1, channels, height, width)
        max_iterations: Maximum number of optimization iterations
        learning_rate: Learning rate for the optimizer

    Returns:
        Aligned source image and the final translation parameters
    """
    # Initialize translation parameters

    source = source.unsqueeze(0)
    target = target.unsqueeze(0)
    translation = torch.zeros(2, requires_grad=True)
    optimizer = torch.optim.Adam([translation], lr=learning_rate)

    # batch_size, channels, height, width = source.shape

    losss = []
    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Create identity matrix with translation
        theta = torch.eye(2, 3).unsqueeze(0)
        theta[0, :, 2] = translation  # Apply x,y translation

        # Create sampling grid
        grid = F.affine_grid(theta, source.size(), align_corners=False)

        # Warp the source image
        warped = F.grid_sample(source, grid, align_corners=False)

        # Calculate loss based on user selection
        if loss_type == 'mse':
            loss = F.mse_loss(warped, target)
        elif loss_type == 'ncc':
            loss = normalized_cross_correlation(warped, target)
        losss.append(loss.item())
        # Backpropagate and optimize
        loss.backward()
        optimizer.step()

        # uncomment this for iteration monitoring
        # if iteration % 10 == 0:
        #     print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
        #     print(f"Translation: dx={translation[0].item():.4f}, dy={translation[1].item():.4f}")

        # Early stopping if loss is small enough
        if loss.item() < 1e-6:
            break

    # uncomment this to see the plot of loss
    # plt.plot(losss)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Translation Alignment Loss')
    # plt.show()

    return loss, translation
