import argparse
import os
import torch

from utils.io_helper import torch_read_image, torch_save_image
from utils.draw_helper import draw_all_circles
from utils.misc_helper import build_laplacian_scale_space, non_max_suppression


def main(args) -> None:
    os.makedirs('outputs', exist_ok=True)
    if args.input_name == 'all':
        run_all(args)
        return
    blob_detection(
        args.input_name, 'outputs/blob.jpg',
        ksize=args.ksize, sigma=args.sigma, n=args.n)


def run_all(args) -> None:
    """Run the blob detection on all images."""
    for image_name in [
        'butterfly', 'einstein', 'fishes', 'sunflowers'
    ]:
        input_name = 'data/part2/%s.jpg' % image_name
        output_name = 'outputs/%s-blob.jpg' % image_name
        blob_detection(
            input_name, output_name,
            ksize=args.ksize, sigma=args.sigma, n=args.n)


def blob_detection(
    input_name: str,
    output_name: str,
    ksize: int,
    sigma: float,
    n: int
) -> None:
    # Step 1: Read RGB image as Grayscale
    image = torch_read_image(input_name, gray=True)
    image_tensor = torch.tensor(image, dtype=torch.float32)

    scale_space, sigmas = build_laplacian_scale_space(
        image_tensor, n=n, initial_sigma=sigma, ksize=ksize)
    keypoints = non_max_suppression(scale_space, sigmas, threshold=0.01)

    cx = [kp[1] for kp in keypoints]
    cy = [kp[0] for kp in keypoints]
    rad = [int(kp[2]) for kp in keypoints]

    # Visualize the keypoints on the original image
    draw_all_circles(image_tensor, cx, cy, rad, output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CS59300CVD Assignment 2 Part 2')
    parser.add_argument('-i', '--input_name', required=True,
                        type=str, help='Input image path')
    parser.add_argument('-s', '--sigma', type=float)
    parser.add_argument('-k', '--ksize', type=int)
    parser.add_argument('-n', type=int)
    args = parser.parse_args()
    assert (args.ksize % 2 == 1)
    main(args)
