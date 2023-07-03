#!/usr/bin/env python3
"""Task 4"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == "valid":
        ph, pw = 0, 0
    elif padding == "same":
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    else:
        ph, pw = padding

    padded_images = np.pad(images,
                           [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                           mode='constant')
    conv_h = int((h - kh + 2 * ph) / sh + 1)
    conv_w = int((w - kw + 2 * pw) / sw + 1)

    convolved_images = np.zeros(shape=(m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            convolved_images[:, i, j] = np.sum(
                (kernel * padded_images[:, i *
                 sh:i * sh + kh, j * sw:j * sw + kw]),
                axis=(1, 2, 3)
            )

    return convolved_images
