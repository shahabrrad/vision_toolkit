"""Implements image stitching."""

import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import kornia
import kornia.feature as KF
from helpers import plot_inlier_matches, compute_harris_response, get_harris_points
import matplotlib.pyplot as plt

from skimage.transform import ProjectiveTransform, warp
from skimage.util import img_as_float


class ImageStitcher(object):
    def __init__(self, img1, img2, keypoint_type='harris', descriptor_type='pixel'):
        """
        Inputs:
            img1: h x w tensor.
            img2: h x w tensor.
            keypoint_type: string in ['harris']
            descriptor_type: string in ['pixel', 'hynet']
        """
        self.img1 = img1
        self.img2 = img2
        self.keypoint_type = keypoint_type
        self.descriptor_type = descriptor_type
        #### Your Implementation Below ####

        # Extract keypoints
        self.keypoints1 = self._get_keypoints(self.img1)
        self.keypoints2 = self._get_keypoints(self.img2)

        # Extract descriptors at each keypoint
        self.desc1 = self._get_descriptors(self.img1, self.keypoints1)

        self.desc2 = self._get_descriptors(self.img2, self.keypoints2)

        # Compute putative matches and match the keypoints.
        matches = self._get_putative_matches(self.desc1, self.desc2)

        matched_keypoints = torch.zeros((matches.shape[1], 4))
        for i in range(matches.shape[1]):
            if matches[0, i] == -1 or matches[1, i] == -1:
                continue
            # matched_keypoints[i, :2] = torch.tensor(self.keypoints1[matches[0, i]])
            # matched_keypoints[i, 2:] = torch.tensor(self.keypoints2[matches[1, i]])
            matched_keypoints[i, 1], matched_keypoints[i,
                                                       0] = torch.tensor(self.keypoints1[matches[0, i]])
            matched_keypoints[i, 3], matched_keypoints[i,
                                                       2] = torch.tensor(self.keypoints2[matches[1, i]])

        # Perform RANSAC to find the best homography and inliers
        inliers, final_homography = self._ransac(matched_keypoints)
        # inliers = None
        print(inliers, final_homography)

        # Plot the inliers
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_inlier_matches(ax,
                            kornia.utils.tensor_to_image(img1),
                            kornia.utils.tensor_to_image(img2),
                            matched_keypoints[inliers])
        plt.savefig('inlier_matches_%s.png' % self.descriptor_type)

        # Refit with all inliers to get the final homography
        stitched = self.stitch(final_homography)

        plt.figure()
        plt.imshow(stitched)
        plt.gray()
        plt.savefig('stitched_%s.png' % self.descriptor_type)

    def _get_keypoints(self, img):
        """
        Extract keypoints from the image.

        Inputs:
            img: h x w tensor.
        Outputs:
            keypoints: N x 2 numpy array.
        """
        harrisim = compute_harris_response(img.squeeze())
        keypoints = get_harris_points(harrisim, min_distance=10, threshold=0.1)
        return keypoints

    def _get_descriptors(self, img, keypoints, patch_size=32):
        """
        Extract descriptors from the image at the given keypoints.

        Inputs:
            img: h x w tensor.
            keypoints: N x 2 tensor.
        Outputs:
            descriptors: N x D tensor.
        """

        # Reshape the image from (1, 1, x, y) to (1, x, y)
        if img.dim() == 4 and img.shape[1] == 1:
            img = img.squeeze(1)

        descriptors = []

        # Check the type of descriptor
        if self.descriptor_type == 'pixel':
            descriptors = extract_descriptors_pixel(
                img, keypoints, patch_size=patch_size)
            descriptors = torch.stack(descriptors)
        elif self.descriptor_type == 'hynet':
            descriptors = extract_hynet_descriptors(img, keypoints)

        return descriptors

    def _get_putative_matches(self, desc1, desc2, max_num_matches=100):
        """
        Compute putative matches between two sets of descriptors.

        Inputs:
            desc1: N x D tensor.
            desc2: M x D tensor.
            max_num_matches: Integer
        Outputs:
            matches: 2 x max_num_matches tensor.
        """
        distances = torch.cdist(desc1, desc2, p=2)
        matches = []
        top_k = max_num_matches
        # threshold not used (change from none to the value you want if used)
        threshold = None
        if threshold is not None:  # NOT USED
            # Approach 1: Select all pairs with distances below the threshold
            below_threshold = (distances < threshold).nonzero(as_tuple=False)
            for match in below_threshold:
                i, j = match[0].item(), match[1].item()
                matches.append((i, j))
        elif top_k is not None:
            # Approach 2: Select the top_k matches with the smallest distances
            # Flatten and sort to get top k
            flat_indices = torch.argsort(distances.view(-1))[:top_k]
            # Convert back to 2D indices
            indices = torch.unravel_index(flat_indices, distances.shape)
            # Stack to get shape (2, num_matches)
            matches = torch.stack(indices, dim=0)

        if matches.shape[1] < max_num_matches:
            padding = max_num_matches - matches.shape[1]
            # Use -1 for unmatched positions
            pad_tensor = torch.full((2, padding), -1, dtype=torch.long)
            matches = torch.cat((matches, pad_tensor), dim=1)
        return matches

    def _get_homography(self, matched_keypoints):
        """
        Compute the homography between two images.

        Inputs:
            matched_keypoints: N x 4 tensor.
        Outputs:
            homography: 3 x 3 tensor.
        """

        # Split matched points into two sets: points in image1 and corresponding points in image2
        matches1 = matched_keypoints[:, :2]
        matches2 = matched_keypoints[:, 2:]

        # Construct matrix A for the homogeneous least squares system Ax = 0
        A = []
        for (x1, y1), (x2, y2) in zip(matches1, matches2):
            A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
        A = torch.tensor(A, dtype=torch.float32)

        # Perform SVD on A and extract the homography from the last singular vector
        _, _, V = torch.svd(A)
        # Last column of V corresponds to the smallest singular value
        H = V[:, -1].reshape(3, 3)
        homography = H / H[-1, -1]  # Normalize so that H[2,2] = 1
        # homography = None
        return homography

    def _homography_inliers(self, H, matched_keypoints, inlier_threshold=20):
        """
        Compute the inliers for the given homography.

        Inputs:
            H: Homography 3 x 3 tensor.
            matched_keypoints: N x 4 tensor.
            inlier_threshold: upper bounds on what counts as inlier.
        Outputs:
            inliers: N tensor, indicates whether each matched keypoint is an inlier.
        """
        matches1 = matched_keypoints[:, :2]
        matches2 = matched_keypoints[:, 2:]

        # Transform matches1 points using H
        ones = torch.ones((matches1.shape[0], 1))
        # Convert to homogeneous coordinates
        pts1_hom = torch.cat([matches1, ones], dim=1)
        pts2_proj = (H @ pts1_hom.T).T  # Apply homography
        # Normalize to get cartesian coordinates
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2:3]

        # Calculate reprojection error
        errors = torch.norm(matches2 - pts2_proj, dim=1)
        inliers = errors < inlier_threshold  # Inliers are those within the threshold
        num_inliers = inliers.sum().item()

        if num_inliers > 0:
            average_residual = errors[inliers].mean().item()
        else:
            average_residual = float('inf')  # No inliers; set to infinity
        return inliers.sum().item(), inliers, average_residual

    def _ransac(self, matched_keypoints, num_iterations=30000, inlier_threshold=2):
        """
        Perform RANSAC to find the best homography.

        Inputs:
          matched_keypoints: N x 4 tensor.
          num_iterations: Number of iteration to run RANSAC.
          inlier_threshold: upper bounds on what counts as inlier.
        Outputs:
          best_inliers: N tensor, indicates whether each matched keypoint is an inlier.
          best_homography: 3 x 3 tensor
        """
        max_inliers = 0
        best_H = None
        best_inliers = None

        for _ in range(num_iterations):
            # Randomly sample four matches
            indices = torch.randperm(matched_keypoints.shape[0])[:4]
            sample_matches = matched_keypoints[indices]

            # Estimate homography using the four sampled points
            H = self._get_homography(sample_matches)

            # Count inliers based on reprojection error
            num_inliers, inliers, average_residual = self._homography_inliers(
                H, matched_keypoints, inlier_threshold)

            # Update if this homography has the most inliers
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H
                best_inliers = inliers
                best_residual = average_residual

        print("max_inliers", max_inliers)
        print("best_residual", best_residual)
        return best_inliers, best_H

    def stitch(self, final_homography):
        """
        Stitch the two images together.

        Inputs:
            final_homography: 3 x 3 tensor.
        Outputs:
            stitched: h x w tensor.
        """
        # Convert homography from torch tensor to numpy array for skimage compatibility
        H = final_homography  # .cpu().numpy()

        # Set up the transformation using the learned homography
        transform_h = ProjectiveTransform(H)
        canvas_size = (800, 1600)
        image2 = self.img2.squeeze()
        image1 = self.img1.squeeze()
        # Warp image2 onto the coordinate system of image1
        print("before warp", image1.shape, image2.shape)
        warped_image2 = warp(image2, transform_h, output_shape=canvas_size)

        print("warped shape", warped_image2.shape)
        # Place image1 on a blank canvas of the specified size
        canvas = np.zeros((canvas_size[0], canvas_size[1]))

        canvas[:image1.shape[0], :image1.shape[1]] = img_as_float(image1)

        print("canvas shape", canvas.shape)
        # Create a mask of where each image is placed on the canvas
        mask1 = np.zeros_like(canvas, dtype=bool)
        mask1[:image1.shape[0], :image1.shape[1]] = True
        print("mask1 shape", mask1.shape)
        mask2 = warped_image2 > 0  # Mask for warped_image2's non-zero regions

        # Composite images on the canvas
        overlap = mask1 & mask2
        # Averaging overlap areas
        canvas[overlap] = (canvas[overlap] + warped_image2[overlap]) / 2
        print("averaged")
        # return canvas
        # Non-overlapping areas of warped_image2
        canvas[mask2 & ~overlap] = warped_image2[mask2 & ~overlap]

        plt.figure()
        plt.imshow(canvas, cmap='gray')
        plt.show()

        return canvas


def extract_descriptors_pixel(image_tensor, keypoints, patch_size=32):
    """
    Extract descriptors from local neighborhoods around each keypoint, 
    padding for keypoints near edges to keep all points.

    Parameters:
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        keypoints (list of tuples): List of (x, y) tuples indicating keypoint positions.
        patch_size (int): Size of the local neighborhood around each keypoint.

    Returns:
        list of torch.Tensor: List of feature descriptors for each keypoint.
    """
    descriptors = []
    C, H, W = image_tensor.shape
    half_patch = patch_size // 2

    # Pad the image to handle edge keypoints
    # padded_image = image_tensor
    padded_image = F.pad(
        image_tensor, (half_patch, half_patch, half_patch, half_patch), mode='reflect')
    # Apply median filter to the padded image
    # padded_image = kornia.filters.median_blur(padded_image.unsqueeze(0), (3, 3)).squeeze(0)
    # padded_image = T.functional.gaussian_blur(padded_image.unsqueeze(0), kernel_size=(3, 3)).squeeze(0)

    for y, x in keypoints:
        # Adjust coordinates to match the padded image
        x_p, y_p = x + half_patch, y + half_patch
        # x_p, y_p = x, y

        # Extract local neighborhood
        patch = padded_image[:, y_p - half_patch:y_p +
                             half_patch, x_p - half_patch:x_p + half_patch]

        # Flatten the patch to form a descriptor
        descriptor = patch.reshape(-1)

        # Subtract the mean and normalize to unit length
        descriptor = descriptor - descriptor.mean()
        norm = torch.norm(descriptor)
        if norm != 0:
            descriptor /= norm

        descriptors.append(descriptor)
        if descriptor.shape[0] == 0:
            print("empty descriptor", x, y)
            print(patch.shape)

    return descriptors


def extract_hynet_descriptors(image_tensor, keypoints, patch_size=32):
    """
    Extracts descriptors for each keypoint in an image using Kornia's HyNet model.

    Parameters:
        image_tensor (torch.Tensor): The input image tensor of shape (1, H, W) for grayscale, or (3, H, W) for RGB.
        keypoints (list of tuples): List of (x, y) tuples indicating keypoint positions.
        patch_size (int): Size of the patch around each keypoint.

    Returns:
        torch.Tensor: Descriptors for each keypoint, shape (N, D) where D is the descriptor dimension (e.g., 128 for HyNet).
    """
    device = image_tensor.device  # Use the device of the input tensor
    # Load HyNet model on the same device
    hynet = KF.HyNet(pretrained=True).to(device).eval()

    half_patch = patch_size // 2
    descriptors = []

    # Convert image to grayscale if it's in RGB
    if image_tensor.shape[0] == 3:
        image_tensor = kornia.color.rgb_to_grayscale(image_tensor)

    # Pad the image to handle edge keypoints
    # padded_image = image_tensor
    padded_image = torch.nn.functional.pad(
        image_tensor, (half_patch, half_patch, half_patch, half_patch), mode='reflect')
    # padded_image = kornia.filters.median_blur(padded_image.unsqueeze(0), (3, 3)).squeeze(0)
    # Extract patches around each keypoint and prepare for batch processing
    patches = []
    # for x, y in keypoints:
    for y, x in keypoints:
        # Adjust keypoint coordinates for the padded image
        x_p, y_p = x + half_patch, y + half_patch
        patch = padded_image[:, y_p - half_patch:y_p +
                             half_patch, x_p - half_patch:x_p + half_patch]
        patches.append(patch)

    # Stack patches into a single tensor for batch processing with HyNet
    patches_tensor = torch.stack(patches).to(device)

    # Pass patches through HyNet to get descriptors
    with torch.no_grad():
        descriptors = hynet(patches_tensor)

    return descriptors
