# Shape From Shading Project

This project implements a basic shape-from-shading algorithm, a technique in computer vision to reconstruct 3D surfaces from a set of images with varying illumination. This project uses the Yale face database. The files should be placed in a folder called data. One example have been placed in the folder.

## Files

- **main.py**: 
  - Entry point for running the shape-from-shading algorithms.
  - Manages input data, initializes the necessary functions, and displays results.
  
- **helpers.py**: 
  - Contains utility functions that are used across different modules.
  - Provides support for data preprocessing, visualization, and mathematical operations.
  
- **shape_from_shading.py**: 
  - Implements the core shape-from-shading algorithm.
  - Contains methods for processing image data and reconstructing 3D surfaces from shading information.

#### Usage

Run the file using the following command:

-s or --subject_name: The name of the subject to use (e.g., yaleB01).
-i or --integration_method: The integration method for generating the height map. Supports random, average, column, and row.

```bash
python main.py -s <subject_name> -i <integration_method>


