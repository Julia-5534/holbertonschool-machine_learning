#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the output dimensions
    oh = h - kh + 1
    ow = w - kw + 1

    # Create an empty array to store the convolved images
    convolved_images = np.empty((m, oh, ow))

    # Perform the convolution
    for i in range(oh):
        for j in range(ow):
            convolved_images[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return convolved_images
