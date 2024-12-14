# Image Stitching

This project includes Python scripts designed to stitch images together using feature detection and matching techniques. The primary files are `main.py`, `helpers.py`, and `image_stitcher.py`.

## Files

- **main.py**: The main entry point for running the image stitching process. This script loads images, calls the helper functions, and displays the final stitched image.
- **helpers.py**: Contains utility functions to support the image stitching process, such as image transformation and feature extraction.
- **image_stitcher.py**: Implements the core image stitching logic, including image alignment, feature matching, and blending.

## How to run

Other than the script below for changing the descriptor type, we can finetune the parameters by changing them in the code.

```bash
python main.py -d pixel
python main.py -d hynet

