"""Implements helper functions."""

import torch
import matplotlib.pyplot as plt
import torchvision


def custom_shifts(input, shifts, dims=None, padding='circular'):
    """Shifts the input tensor by the specified shifts along
       the specified dimensions. Supports circular and zero padding.

       Input: Tensor, shifts (list of integers), dims (list of integers),
       It will shift dims[i] by shifts[i] for all i.
       Returns: Shifted Tensor along the specified dimension
         padded following the padding scheme.
    """
    ret = input

    if len(shifts) != len(dims):
        raise ValueError(
            "The number of shifts must match the number of dimensions.")

    for shift, dim in zip(shifts, dims):
        if shift == 0:
            continue  # No shift needed

        # Perform circular shift using torch.cat
        if padding == 'circular':
            if shift > 0:
                ret = torch.cat((ret.narrow(dim, ret.size(dim) - shift, shift),
                                 ret.narrow(dim, 0, ret.size(dim) - shift)), dim=dim)
            else:
                shift = -shift  # Convert to positive equivalent for the left/up shift
                ret = torch.cat((ret.narrow(dim, shift, ret.size(dim) - shift),
                                 ret.narrow(dim, 0, shift)), dim=dim)
        elif padding == 'zero':
            # Create a zero tensor for padding
            zeros_shape = list(ret.size())
            zeros_shape[dim] = shift if shift > 0 else -shift
            zeros_tensor = torch.zeros(zeros_shape, dtype=ret.dtype)

            if shift > 0:
                # Zero padding shift to the right/down (positive shift)
                ret = torch.cat((zeros_tensor, ret.narrow(
                    dim, 0, ret.size(dim) - shift)), dim=dim)
            else:
                # Zero padding shift to the left/up (negative shift)
                shift = -shift
                ret = torch.cat(
                    (ret.narrow(dim, shift, ret.size(dim) - shift), zeros_tensor), dim=dim)
        else:
            raise ValueError(
                "Unsupported padding type. Use 'circular' or 'zero'.")

    return ret


def plot_average_values():
    """Calculate average values per row and per column and plots it for all images"""
    row_avgs = []
    col_avgs = []
    for i in range(1, 7):
        img = torchvision.io.read_image('data/%d.jpg' % i)
        # torchvision.transforms.ToPILImage()(img).show()
        img = img.float()
        img = img / 255.0
        row_avg = torch.mean(img, dim=1)
        row_avgs.append(row_avg)
        col_avg = torch.mean(img, dim=2)
        col_avgs.append(col_avg)

    # Plot average values per row
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.plot(row_avgs[i][0])
    plt.title('Average Values per Row')
    plt.xlabel('Row')
    plt.ylabel('Average Value')
    plt.show()

    # Plot average values per column
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.plot(col_avgs[i][0])
    plt.title('Average Values per Column')
    plt.xlabel('Column')
    plt.ylabel('Average Value')
    plt.show()

    #  based on the plots, we can devide pictures based on where the average value goes below 0.2 on both rows and columns


def plot_white_border_threshold(img):
    """plot the average values per row and per column and add a threshold line for white border detection"""
    row_means = torch.mean(img, dim=(0, 2))
    col_means = torch.mean(img, dim=(0, 1))
    plt.plot(row_means)
    plt.axhline(y=0.95, color='r', linestyle='--',
                label='White border Threshold')
    plt.title('Average Values per Row')
    plt.xlabel('Row')
    plt.ylabel('Average Value')
    plt.legend()
    plt.show()

    # change the name of file if you want to save
    # plt.savefig('wb_rows_6.png')

    plt.plot(col_means)
    plt.axhline(y=0.9, color='r', linestyle='--',
                label='White border Threshold')
    plt.title('Average Values per Column')
    plt.xlabel('Column')
    plt.ylabel('Average Value')
    plt.legend()
    # plt.show()
    # change the name of file if you want to save
    # plt.savefig('wb_columns_6.png')


def plot_cutoff_lines(img, cut_offs):
    """Plot the average values per row and add vertical lines for the cut-offs to separate images"""
    image = img[0]
    row_means = torch.mean(image, dim=1)
    plt.figure(figsize=(10, 5))
    plt.plot(row_means)
    for i in range(len(cut_offs)):
        plt.axvline(x=cut_offs[i], color='r', linestyle='--', alpha=0.5)
    plt.title('Cut-ff Rows to separate images')
    plt.xlabel('Row')
    plt.ylabel('Average Value')
    # change the name of the file based on the imgage that you are ploting
    # plt.savefig('cut_offs_6.png')
    plt.show()
