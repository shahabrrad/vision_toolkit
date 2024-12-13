# Blob Detection Utility

This project is designed to detect and visualize blobs in grayscale images using a multi-scale Laplacian of Gaussian (LoG) approach. The program reads an input image, builds a scale-space using Laplacian filters, detects blob-like structures, and finally visualizes them by drawing circles around detected blobs on the image.

A number of examples of the output of the program with different parameters can be seen in the `outputs` folder.

## File Descriptions

1. **`draw_helper.py`**
   - This file provides functionality for visualizing detected blobs on images. The main function, `draw_all_circles`, takes a grayscale image and a set of blob centers and radii and draws corresponding circles on the image using Matplotlib.

2. **`io_helper.py`**
   - This file contains utility functions for reading and saving images in PyTorch tensors. The `torch_read_image` function reads an image (optionally converting it to grayscale), and `torch_save_image` saves an image to disk.

3. **`misc_helper.py`**
   - Contains miscellaneous helper functions, including `log_filter` which is the laplace of gaussian filter. The `build_laplacian_scale_space` iterates through the scales and builds the laplacian space. The `non_max_suppression` function checks that the keypoints that are detected are maximum in a neighbouring window.

4. **`main_p2.py`**
   - The main script that handles blob detection. It reads the input image, applies the Laplacian of Gaussian (LoG) technique at multiple scales to detect blobs, and finally draws and saves the output image with circles around detected blobs.

## Running the Program
```sh
    python main_p2.py -i <file name> -s <sigma> -k <kernel size> -n <number of scales>
    ```

