#!/usr/bin/env python3
"""Task 5"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil((h * sh - sh + kh - h) / 2))
        pw = int(np.ceil((w * sw - sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(images,
                           [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                           mode='constant')
    out_h = int((h - kh + 2 * ph) / sh) + 1
    out_w = int((w - kw + 2 * pw) / sw) + 1

    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    padded_images[
                        :, i*sh:i*sh+kh, j*sw:j*sw+kw, :] * kernels[
                            :, :, :, k], axis=(1, 2, 3))

    return output
