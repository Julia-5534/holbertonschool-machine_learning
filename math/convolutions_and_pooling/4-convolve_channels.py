#!/usr/bin/env python3
"""Task 4"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh - h + kh) / 2)
        pw = int(((w - 1) * sw - w + kw) / 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')
    conv_h = int((h - kh + 2 * ph) / sh) + 1
    conv_w = int((w - kw + 2 * pw) / sw) + 1

    convolved_images = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            i_start = i * sh
            i_end = i_start + kh
            j_start = j * sw
            j_end = j_start + kw
            image_patches = padded_images[:, i_start:i_end, j_start:j_end, :]
            convolved_images[:, i, j] = np.sum(np.multiply(
                image_patches, kernel), axis=(1, 2, 3))

    return convolved_images
