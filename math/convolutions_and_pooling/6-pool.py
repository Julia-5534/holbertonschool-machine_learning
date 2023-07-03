#!/usr/bin/env python3
"""Task 6"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate the output shape based on kernel size and stride
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    # Initialize the pooled images array
    pooled_images = np.zeros((m, out_h, out_w, c))

    # Perform pooling
    for i in range(out_h):
        for j in range(out_w):
            # Calculate the pooling region for the current position
            start_h = i * sh
            end_h = start_h + kh
            start_w = j * sw
            end_w = start_w + kw

            # Extract the region from the images
            region = images[:, start_h:end_h, start_w:end_w, :]

            # Perform pooling based on the specified mode
            if mode == 'max':
                pooled_value = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                pooled_value = np.mean(region, axis=(1, 2))

            # Assign pooled value to corresponding position in output array
            pooled_images[:, i, j, :] = pooled_value

    return pooled_images
