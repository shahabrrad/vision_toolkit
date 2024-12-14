# vision_toolkit
A number of simple computer vision projects

## Projects description:

### Image alignment:
Two different methods for aligning three channels of an image inorder to make an RGB image. This project takes digitized Prokudin-Gorskii glass plate images and automatically produce a color image with as few visual artifacts as possible. The alignment methods are either through gradient descent or through exhaustive search for finding the alignment with the smallest error.

### Blob detection
This project is designed to detect and visualize blobs in grayscale images using a multi-scale Laplacian of Gaussian (LoG) approach. The program reads an input image, builds a scale-space using Laplacian filters, detects blob-like structures, and finally visualizes them by drawing circles around detected blobs on the image.

### Stitching
This project stitches two images together based on keypoint matching between two images. This can be used to create panoramic images from multiple images.
