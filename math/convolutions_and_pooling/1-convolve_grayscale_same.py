#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the padding sizes
    ph = kh // 2
    pw = kw // 2

    # Pad the images with zeros
    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Calculate the output dimensions
    oh = h
    ow = w

    # Create an empty array to store the convolved images
    convolved_images = np.empty((m, oh, ow))

    # Perform the convolution
    for i in range(oh):
        for j in range(ow):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel)

    return convolved_images
