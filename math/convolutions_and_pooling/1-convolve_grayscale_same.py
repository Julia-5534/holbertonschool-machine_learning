#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the padding sizes
    ph, pw = kh // 2, kw // 2

    # Pad the images with zeros
    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Calculate the output dimensions
    oh = h
    ow = w

    # Create an empty array to store the convolved images
    convolved_images = np.zeros((m, oh, ow))

    image = np.arange(m)

    # Perform the convolution
    for i in range(oh):
        for j in range(ow):
            patch = padded_images[image, i:i+kh, j:j+kw]
            convolved_images[image, i, j] = (np.sum(
                patch * kernel, axis=(1, 2)))

    return convolved_images
