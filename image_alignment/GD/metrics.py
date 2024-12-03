"""Implements different image metrics."""

import torch
from skimage.metrics import structural_similarity


def ncc(img1, img2):
    """Takes two image and compute the negative normalized cross correlation.

       Lower the value, better the alignment.
    """
    ret = 0
    # Ensure images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")

    # Compute mean per image
    mean1 = torch.mean(img1, dim=[1, 2], keepdim=True)
    mean2 = torch.mean(img2, dim=[1, 2], keepdim=True)

    # Subtract mean from the images
    img1_centered = img1 - mean1
    img2_centered = img2 - mean2

    # Compute the numerator (dot product between centered images)
    numerator = torch.sum(img1_centered * img2_centered, dim=[1, 2])

    # Compute the denominator (product of image norms)
    denominator = torch.sqrt(
        torch.sum(img1_centered**2, dim=[1, 2]) * torch.sum(img2_centered**2, dim=[1, 2]))

    # Compute the NCC
    # Add a small constant for numerical stability
    ncc = numerator / (denominator + 1e-8)
    ret = - ncc.mean()
    # Return the negative NCC
    return ret


def mse(img1, img2):
    """Takes two image and compute the mean squared error.
       Lower the value, better the alignment.
    """
    ret = 0
    # Ensure images have the same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    # Compute the element-wise squared difference between the images
    squared_diff = (img1 - img2)**2
    # Compute the mean squared error
    mse = torch.mean(squared_diff)
    ret = mse
    return ret


def ssim(img1, img2):
    """Takes two image and compute the negative structural similarity.

    This function is given to you, nothing to do here.

    Please refer to the classic paper by Wang et al. of Image quality 
    assessment: from error visibility to structural similarity.
    """
    img1 = img1.numpy()
    img2 = img2.numpy()
    return -structural_similarity(img1, img2, data_range=img1.max() - img2.min())
