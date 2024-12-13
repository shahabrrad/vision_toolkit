import numpy as np
import cv2
from matplotlib import pyplot as plt
from test import detect_and_draw_blobs


def visualize_scale_space_coverage(min_sigma, max_sigma, k, n_scales):
    """Visualize how the scales cover the space of possible blob sizes"""
    sigmas = [min_sigma * (k**i) for i in range(n_scales)]

    # Calculate the effective radius for each sigma (≈ 1.414 * sigma)
    radii = [s * 1.414 for s in sigmas]

    # Create visualization
    plt.figure(figsize=(10, 4))
    for r in radii:
        plt.axvline(x=r, color='b', alpha=0.3)
    plt.xlabel('Blob Radius (pixels)')
    plt.title('Scale Space Coverage')
    plt.xlim(0, max(radii) + 5)
    plt.grid(True)
    plt.show()


def parameter_selection_guide(image_path):
    """Guide for parameter selection based on image properties"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read image")

    # Image properties
    height, width = image.shape
    min_dimension = min(height, width)

    # Recommended parameters based on image size
    recommended_params = {
        # Base scale relative to image size
        'initial_sigma': max(1.0, min_dimension / 500),
        'k': 1.4,  # Standard scale factor (√2 ≈ 1.414)
        # Cover reasonable range
        'n_scales': int(np.log(min_dimension/4) / np.log(1.4)),
        'threshold': 0.1  # Starting threshold, needs adjustment
    }

    # Calculate expected blob size range
    min_detectable_radius = recommended_params['initial_sigma'] * 1.414
    max_detectable_radius = min_detectable_radius * \
        (recommended_params['k'] ** (recommended_params['n_scales']-1))

    print(f"""
Parameter Selection Guide for image {image_path}
----------------------------------------------
Image size: {width}x{height}

Recommended Parameters:
1. initial_sigma = {recommended_params['initial_sigma']:.2f}
   - Minimum detectable blob radius: {min_detectable_radius:.1f} pixels
   - Adjust based on smallest features of interest

2. k = {recommended_params['k']}
   - Standard value based on scale-space theory
   - Increase for faster computation but less precise detection
   - Decrease for more precise detection but slower computation
   - Recommended range: 1.2 to 1.6

3. n_scales = {recommended_params['n_scales']}
   - Will detect blobs from {min_detectable_radius:.1f} to {max_detectable_radius:.1f} pixels
   - Increase if larger blobs are missing
   - Decrease if computation is too slow

4. threshold = {recommended_params['threshold']}
   - Start with this value and adjust based on results:
     * Increase if too many false detections
     * Decrease if missing obvious blobs
   - Typical range: 0.01 to 0.3

Steps for Parameter Tuning:
1. Start with these recommended values
2. Adjust initial_sigma based on smallest blob size of interest
3. Tune threshold to balance detection sensitivity
4. Modify n_scales if needed for larger/smaller blob range
5. Adjust k only if needing to fine-tune scale space coverage
""")

    return recommended_params


def test_parameters(image_path, params, output_path):
    """Test detection with given parameters and show results"""
    # Using the detect_and_draw_blobs function defined earlier
    try:
        cx, cy, sigmas = detect_and_draw_blobs(
            image_path,
            output_path,
            n_scales=params['n_scales'],
            k=params['k'],
            initial_sigma=params['initial_sigma'],
            threshold=params['threshold']
        )
        print(f"Detected {len(cx)} blobs")

        # Visualize scale space coverage
        visualize_scale_space_coverage(
            params['initial_sigma'],
            params['initial_sigma'] * (params['k'] ** (params['n_scales']-1)),
            params['k'],
            params['n_scales']
        )

        return cx, cy, sigmas
    except Exception as e:
        print(f"Error during detection: {e}")
        return None, None, None


# Example usage:
"""
# Load image and get recommended parameters
image_path = "your_image.jpg"
output_path = "result.jpg"
recommended_params = parameter_selection_guide(image_path)

# Test with recommended parameters
cx, cy, sigmas = test_parameters(image_path, recommended_params, output_path)

# Adjust parameters based on results
adjusted_params = recommended_params.copy()
adjusted_params['threshold'] = 0.15  # Example adjustment
cx, cy, sigmas = test_parameters(image_path, adjusted_params, "adjusted_result.jpg")
"""

image_path = "data/part2/butterfly.jpg"
output_path = "result.jpg"
recommended_params = parameter_selection_guide(image_path)
cx, cy, sigmas = test_parameters(image_path, recommended_params, output_path)
