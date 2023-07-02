#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions
    oh = h - kh + 1
    ow = w - kw + 1

    # Create an empty array for the convolved images
    images_conv = np.zeros((m, oh, ow))

    # Iterate over each image
    for i in range(m):
        # Iterate over each pixel position in the image
        for j in range(oh):
            for k in range(ow):
                # Extract the patch from the image
                patch = images[i, j:j+kh, k:k+kw]
                # Perform element-wise multiplication of patch & kernel
                convolved_patch = np.multiply(patch, kernel)
                # Calculate the sum of the convolved patch
                convolved_value = np.sum(convolved_patch)
                # Assign the convolved value to the output image
                images_conv[i, j, k] = convolved_value

    return images_conv
