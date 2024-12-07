# Image alignment using exhaustivesearch from grayscale channels to RGB color

## Project Overview
This project involves aligning three channels of an image inorder to make an RGB image.

We take the digitized Prokudin-Gorskii glass plate images and automatically produce a color image with as few visual artifacts as possible. To do this, we extract the three color channel images, place them on top of each other, and align them so that they form a single RGB color image automatically

## File Structure
- `main.py`: The main script to run the project.
- `alignment_model.py`: The class and functions used for spliting, processing, aligning, and generating RGB image.
- `metrics.py`: The three metric functions used for alignment are here.
- `helpers.py`: Utility functions used throughout the project.
- `requirements.txt`: List of dependencies required to run the project.
- `data/`: Directory containing input and output data for the project.
- `plots/`: Directory where the plots used in the project are stored.
- `alignments_zero.txt`: Contains the alignment results with zero-padding.
- `alignments_circular.txt`: Contains the alignment results with circular-padding.


## Functions
### main.py
- `main()`: The entry point of the program. It orchestrates the execution of different parts of the project.
- `run_all()`: The function that can be used to run the alignment model on all images with all metrics.

### helpers.py
- `custom_shifts()`: Shifts the input tensor by the specified shifts along the specified dimensions. Supports circular and zero padding. The shifts and dimensions in the argument are either lists or tuples.
- `plot_average_values()`: Plots the average values of rows and columns of all input images on the same graph.
- `plot_white_border_threshold()`: Plots the thresholds used to cut out the borders of images
- `plot_cutoff_lines()`: Plots the lines that were used to sparate the image into 3 channels.

### metrics.py
- `ncc()`: Calculates the negative normalized cross correlation.
- `mse()`: Calculates the mean square error.
- `ssim()`: Calculates the negative structural similarity.

### alignment_model.py
- `_load_image()`: Loads the specified image based on its name.
- `align()`: The main action happens here. Different orders of alignment are calculated and the best one is selected, then the three channels are combined to create the RGB image.
- `_crop_and_divide_image()`: The preprocessing step. It removes the white borders and cuts the image into separate channels.
- `_align_pairs()`: Calculates the alignment score between two images from -15 to +15 displacements along both axis and returns the best result.
- `_remove_white_borders()`: Removes the white borders from around the image
- `_find_cut_offs()`: Finds the rows within the image that shoud be used to cut the image to separet the three channels
- `_split_at_borders()`: Based on the list of cut-off rows provided, it separates the three channels from the image


## How to Run the Code
1. **Install Dependencies**: Ensure you have Python installed. Then, install the required dependencies using:
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the Main Script**: Execute the main script to run the project:
    ```sh
    python main.py -i all
    ```

3. **View Results**: Check the `data/` directory for the results.

4. **Arguments**: You can select to run the program for specific images with the `-i` or `--image_name` argument.<br />
You can also select to run the progrma with specific metrics with the `-m` or `--metric` argument which can be either `ssim`, `mse`, or `ncc`.<br />
For example, teh script below runs the alignment for image 3 with ncc as a metric:
    ```sh
    python main.py -i data/1.jpg -m ncc
    ```

5. **Plots** In order to reproduce the plots that were used in this project, you can uncomment the specified pieces of code within the files, and run the program as usual.

