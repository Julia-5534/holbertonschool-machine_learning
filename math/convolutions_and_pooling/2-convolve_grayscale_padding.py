#!/usr/bin/env python3
"""Task 2"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs convolution on grayscale images with custom padding:"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    oh = h - kh + 2 * ph + 1
    ow = w - kw + 2 * pw + 1
    padded_images = np.pad(images,
                           [(0, 0), (ph, ph), (pw, pw)],
                           mode='constant')

    convolved_images = np.zeros(shape=(m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            convolved_images[:, i, j] = np.sum(
                (kernel * padded_images[:, i:i + kh, j:j + kw]),
                axis=(1, 2)
            )

    return convolved_images
