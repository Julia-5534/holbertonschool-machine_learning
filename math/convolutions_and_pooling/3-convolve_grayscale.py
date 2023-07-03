#!/usr/bin/env python3
"""Task 3"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = int((((h - 1) * sh + kh - h) / 2) + 1)
        pad_w = int((((w - 1) * sw + kw - w) / 2) + 1)
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding

    padded_images = np.pad(images,
                           ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')

    output_h = int((h + (2 * pad_h) - kh) / sh) + 1
    output_w = int((w + (2 * pad_w) - kw) / sw) + 1
    output = np.zeros((m, output_h, output_w))
    image = np.arange(m)

    for i in range(output_h):
        for j in range(output_w):
            img_region = padded_images[image, i*sh:i*sh+kh, j*sw:j*sw+kw]
            output[image, i, j] = np.sum(img_region * kernel, axis=(1, 2))

    return output
