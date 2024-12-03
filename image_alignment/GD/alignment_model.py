"""Implements the alignment algorithm."""

import torch
import torchvision
from metrics import ncc, mse, ssim
from helpers import custom_shifts, plot_white_border_threshold, plot_cutoff_lines
from PIL import Image
from metrics import ncc, mse, ssim
import matplotlib.pyplot as plt
import numpy as np
from imagealigner import align_images, apply_alignment


class AlignmentModel:
    def __init__(self, image_name, metric='mse', padding='circular'):
        # Image name
        self.image_name = image_name
        # Metric to use for alignment
        self.metric = metric
        # Padding mode for custom_shifts
        self.padding = padding

    def save(self, output_name):
        torchvision.utils.save_image(self.rgb, output_name)

    def align(self):
        """Aligns the image using the metric specified in the constructor.
           Experiment with the ordering of the alignment.

           Finally, outputs the rgb image in self.rgb.
        """
        self.img = self._load_image()

        # uncomment this if you want to see how white borders are removed
        # plot_white_border_threshold(self.img)

        b, g, r = self._crop_and_divide_image()

        # alignment method: chose one image as the base, then align the other two to it

        translation1, score1 = self._align_pairs(g, b)
        translation2, score2 = self._align_pairs(g, r)
        base_g_metric = score1 + score2

        translation3, score3 = self._align_pairs(b, g)
        translation4, score4 = self._align_pairs(b, r)
        base_b_metric = score3 + score4

        translation5, score5 = self._align_pairs(r, b)
        translation6, score6 = self._align_pairs(r, g)
        base_r_metric = score5 + score6

        # find the best alignment order (with lowest score), then shift the other two channels accordingly
        best_score = min(base_g_metric, base_b_metric, base_r_metric)
        print(self.image_name + " " + self.metric +
              " Best score found: ", str(best_score.item()))
        if best_score == base_g_metric:
            print(self.image_name + " " + self.metric +
                  " Green channel used as the base for alignment of image")

            b, (dx1, dy1) = apply_alignment(b, translation1)
            r, (dx2, dy2) = apply_alignment(r, translation2)
            print(self.image_name + " " + self.metric +
                  " Blue channel translation: ", dx1.item(), dy1.item())
            print(self.image_name + " " + self.metric +
                  " Red channel translation: ", dx2.item(), dy2.item())
        elif best_score == base_b_metric:
            print(self.image_name + " " + self.metric +
                  " Blue channel used as the base for alignment of image")
            g, (dx1, dy1) = apply_alignment(g, translation3)
            r, (dx2, dy2) = apply_alignment(r, translation4)
            print(self.image_name + " " + self.metric +
                  " Green channel translation: ", dx1.item(), dy1.item())
            print(self.image_name + " " + self.metric +
                  " Red channel translation: ", dx2.item(), dy2.item())
        else:
            print(self.image_name + " " + self.metric +
                  " Red channel used as the base for alignment of image")
            b, (dx1, dy1) = apply_alignment(b, translation5)
            g, (dx2, dy2) = apply_alignment(g, translation6)
            print(self.image_name + " " + self.metric +
                  " Blue channel translation: ", dx1.item(), dy1.item())
            print(self.image_name + " " + self.metric +
                  " Green channel translation: ", dx2.item(), dy2.item())

        stacked_image = torch.cat([r, g, b], dim=0)

        # uncomment this if you want to see the final image
        # torchvision.transforms.ToPILImage()(stacked_image).show()

        self.rgb = stacked_image.unsqueeze(0).permute(0, 1, 2, 3)

    def save(self, output_name):
        torchvision.utils.save_image(self.rgb, output_name)

    def _load_image(self):
        """Load the image from the image_name path,
           typecast it to float, and normalize it.

           Returns: torch.Tensor of shape (H, W)
        """
        ret = None
        img = torchvision.io.read_image(self.image_name)
        img = img.float()
        img = img / 255.0
        ret = img
        # plot_average_values(ret)
        return ret

    def _crop_and_divide_image(self):
        """Crop the image boundary and divide the image into three parts, padded to the same size.

           Feel free to be creative about this.
           You can eyeball the boundary values, or write code to find approximate cut-offs.
           Hint: Plot out the average values per row / column and visualize it!

           Returns: B, G, R torch.Tensor of shape (roughly H//3, W)
        """
        b_channel = None
        g_channel = None
        r_channel = None

        image = self._remove_white_borders(self.img)  # [0]
        cut_offs = self._find_cut_offs(image)

        # uncomment this if you want to see the cut-off lines on the plot
        # plot_cutoff_lines(image, cut_offs)

        image_segments = self._split_at_borders(image, cut_offs, 300)

        max_height = max([segment.shape[1] for segment in image_segments])
        max_width = max([segment.shape[2] for segment in image_segments])

        padded_segments = []
        # select the padding type
        mode = 'circular'
        if self.padding == 'zero':
            mode = 'constant'

        for segment in image_segments:
            pad_height = max_height - segment.shape[1]
            pad_width = max_width - segment.shape[2]
            padded_segment = torch.nn.functional.pad(
                segment, (0, pad_width, 0, pad_height), mode=mode)
            padded_segments.append(padded_segment)

        b_channel, g_channel, r_channel = padded_segments

        # Save the individual channels as image files
        # torchvision.utils.save_image(b_channel, 'b_channel.png')
        # torchvision.utils.save_image(g_channel, 'g_channel.png')
        # torchvision.utils.save_image(r_channel, 'r_channel.png')

        return b_channel, g_channel, r_channel

    def _align_pairs(self, img1, img2, delta=15):
        """
        Aligns two images using the metric specified in the constructor.
        Returns: Tuple of (u, v) shifts that minimizes the metric.
        img2 is the source and img1 is the target
        we want ot align source on target
        """
        loss, translation = align_images(
            img2, img1, max_iterations=100, learning_rate=0.01, loss_type=self.metric)

        return translation, loss

    def _remove_white_borders(self, img):
        """
        Remove white borders from the image.
        """
        ret = None
        # Remove rows with average close to 1
        row_means = torch.mean(img, dim=(0, 2))
        image = img[:, row_means < 0.95, :]
        # Remove columns with average close to 1
        col_means = torch.mean(image, dim=(0, 1))
        image = image[:, :, col_means < 0.9]
        ret = image
        return ret

    def _find_cut_offs(self, image):
        """
        Find the cut-offs rows in the image which satisfy the criteria for image borders
        Used to separate the channels
        """
        cut_offs = []
        image = image[0]
        row_means = torch.mean(image, dim=1)
        for i in range(len(row_means)):
            # the condition for cutting image... it is based on experimentation
            # explained in the report
            if ((row_means[i] < 0.2) and
                    (abs(row_means[i-1] - row_means[i]) > 0.1 or abs(row_means[i+1] - row_means[i]) > 0.1)) or \
                    (abs(row_means[i-1] - row_means[i]) > 0.4):
                cut_offs.append(i)

        return cut_offs

    def _split_at_borders(self, tensor, borders, min_segment_height):
        """
        Split the image tensor on the rows specified by the borders list.
        each segment should have a minimum height of min_segment_height
        """
        segments = []
        start = 0
        for border in borders:
            if border - start >= min_segment_height:  # if this border does not create a segment with min height, skip it
                segments.append(tensor[:, start:border, :])
            start = border
        # add the last segment if it has enough height
        if tensor.shape[1] - start >= min_segment_height:
            segments.append(tensor[:, start:, :])

        return segments
