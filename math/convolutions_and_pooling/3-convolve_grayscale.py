#!/usr/bin/env python3
"""Task 3"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = int(np.ceil((h - 1) / sh))
        pad_w = int(np.ceil((w - 1) / sw))
        padding = (pad_h, pad_w)
    elif padding == 'valid':
        padding = (0, 0)
    else:
        padding = padding

    pad_h, pad_w = padding
    padded_images = np.pad(images,
                           ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')

    output_h = int((h + (2 * pad_h) - kh) / sh) + 1
    output_w = int((w + (2 * pad_w) - kw) / sw) + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(0, output_h):
        for j in range(0, output_w):
            img_region = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            conv_result = np.sum(img_region * kernel, axis=(1, 2))
            output[:, i, j] = conv_result

    return output
