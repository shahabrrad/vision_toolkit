"""Implements Shape from shading."""
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from helpers import LoadFaceImages, display_output, plot_surface_normals


class ShapeFromShading(object):
    def __init__(self, full_path, subject_name, integration_method):
        ambient_image, imarray, light_dirs = LoadFaceImages(full_path,
                                                            subject_name,
                                                            64)

        # Preprocess the data
        processed_imarray = self._preprocess(ambient_image, imarray)

        # Compute albedo and surface normals
        albedo_image, surface_normals = self._photometric_stereo(processed_imarray,
                                                                 light_dirs)

        # Save the output
        self.save_outputs(subject_name, albedo_image, surface_normals)

        plot_surface_normals(surface_normals)
        # Compute height map
        height_map = self._get_surface(surface_normals, integration_method)

        # Save output results
        display_output(albedo_image, height_map)

        plt.savefig('%s_height_map_%s.jpg' %
                    (subject_name, integration_method))

    def _preprocess(self, ambimage, imarray):
        """
        preprocess the data:
            1. subtract ambient_image from each image in imarray.
            2. make sure no pixel is less than zero.
            3. rescale values in imarray to be between 0 and 1.
        Inputs:
            ambimage: h x w
            imarray: Nimages x h x w
        Outputs:
            processed_imarray: Nimages x h x w
        """
        # for image in imarray:

        imarray = imarray - ambimage
        imarray[imarray < 0] = 0
        # print(torch.max(imarray))
        processed_imarray = imarray / torch.max(imarray)

        return processed_imarray

    def _photometric_stereo(self, imarray, light_dirs):
        """
        Inputs:
            imarray:  N x h x w
            light_dirs: N x 3
        Outputs:
            albedo_image: h x w
            surface_norms: h x w x 3
        """

        h, w = imarray[0].shape
        n = len(imarray)
        N = h * w  # Total number of pixels

        # Stack all images into a matrix of shape (n, N)
        I = np.array([img.flatten() for img in imarray])  # shape (n, N)

        # Convert light directions to a numpy array
        L = np.array(light_dirs)  # shape (n, 3)

        # Solve for G using least-squares: LG = I
        # G will have shape (3, N), where each column is a g vector for a pixel
        G, _, _, _ = np.linalg.lstsq(L, I, rcond=None)  # shape (3, N)

        # Calculate albedo as the norm of each column in G
        albedo = np.linalg.norm(G, axis=0).reshape(h, w)

        # Normalize G to get the surface normals
        normals = (G / albedo.flatten()).T.reshape(h, w, 3)

        return torch.tensor(albedo), torch.tensor(normals)

        # return albedo_image, surface_normals

    def _get_surface(self, surface_normals, integration_method):
        """
        Inputs:
            surface_normals:h x w x 3
            integration_method: string in ['average', 'column', 'row', 'random']
        Outputs:
            height_map: h x w
        """
        if integration_method == 'average':
            height_map = self.integrate_average(surface_normals)
        elif integration_method == 'column':
            height_map = self.integrate_col(surface_normals)
        elif integration_method == 'row':
            height_map = self.integrate_row(surface_normals)
        elif integration_method == 'random':
            height_map = self.integrate_random(surface_normals, 20)
        # height_map = None
        return height_map

    def save_outputs(self, subject_name, albedo_image, surface_normals):
        im = Image.fromarray((albedo_image*255).numpy().astype(np.uint8))
        im.save("%s_albedo.jpg" % subject_name)
        im = Image.fromarray(
            (surface_normals[:, :, 0]*128+128).numpy().astype(np.uint8))
        im.save("%s_normals_x.jpg" % subject_name)
        im = Image.fromarray(
            (surface_normals[:, :, 1]*128+128).numpy().astype(np.uint8))
        im.save("%s_normals_y.jpg" % subject_name)
        im = Image.fromarray(
            (surface_normals[:, :, 2]*128+128).numpy().astype(np.uint8))
        im.save("%s_normals_z.jpg" % subject_name)

    def integrate_row(self, normals):
        h, w, _ = normals.shape

        # Compute partial derivatives from surface normals
        dzdx = normals[:, :, 0] / normals[:, :, 2]  # ∂z/∂x
        dzdy = normals[:, :, 1] / normals[:, :, 2]  # ∂z/∂y
        # First path: Integrate rows first, then columns
        height_map_rows_then_columns = torch.cumsum(dzdx, dim=1)
        height_map_rows_then_columns2 = torch.cumsum(dzdy, dim=0)

        return height_map_rows_then_columns[0] + height_map_rows_then_columns2

    def integrate_col(self, normals):
        h, w, _ = normals.shape

        # Compute partial derivatives from surface normals
        dzdx = normals[:, :, 0] / normals[:, :, 2]  # ∂z/∂x
        dzdy = normals[:, :, 1] / normals[:, :, 2]  # ∂z/∂y

        # Second path: Integrate columns first, then rows
        height_map_columns_then_rows = torch.cumsum(dzdy, dim=0)
        height_map_columns_then_rows2 = torch.cumsum(dzdx, dim=1)
        ret_Val = height_map_columns_then_rows2 + \
            height_map_columns_then_rows[:, 0].unsqueeze(1)

        return ret_Val

    def integrate_average(self, normals):
        # Average the two height maps
        height_map = 0.5 * (self.integrate_row(normals) +
                            self.integrate_col(normals))

        return height_map

    def integrate_random(self, normals, num_random_paths):
        h, w, _ = normals.shape

        # Compute partial derivatives from surface normals
        dzdx = normals[:, :, 0] / normals[:, :, 2]  # ∂z/∂x
        dzdy = normals[:, :, 1] / normals[:, :, 2]  # ∂z/∂y

        height_map_random_paths = torch.zeros((h, w))
        for _ in range(num_random_paths):
            # Generate a random path for integration
            random_path = torch.zeros((h, w))
            for i in range(h):
                for j in range(w):
                    if i > 0 and j > 0:
                        # Randomly pick to accumulate from top or left
                        if torch.rand(1).item() < 0.5:
                            random_path[i, j] = random_path[i -
                                                            1, j] + dzdy[i, j]
                        else:
                            random_path[i, j] = random_path[i,
                                                            j-1] + dzdx[i, j]
                    elif i > 0:
                        random_path[i, j] = random_path[i-1, j] + dzdy[i, j]
                    elif j > 0:
                        random_path[i, j] = random_path[i, j-1] + dzdx[i, j]

            height_map_random_paths += random_path

        # Average the random paths
        height_map_random_paths /= num_random_paths
        print(height_map_random_paths)
        return height_map_random_paths
