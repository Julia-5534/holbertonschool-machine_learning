#!/usr/bin/env python3
"""Task 3"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = int(np.ceil((h - kh + 1) / sh))
        pad_w = int(np.ceil((w - kw + 1) / sw))
        pad_top = int(np.floor((pad_h - 1) / 2))
        pad_bottom = pad_h - 1 - pad_top
        pad_left = int(np.floor((pad_w - 1) / 2))
        pad_right = pad_w - 1 - pad_left
    elif padding == 'valid':
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
    else:
        pad_top, pad_bottom, pad_left, pad_right = padding

    padded_images = np.pad(images, ((0, 0),
                                    (pad_top, pad_bottom),
                                    (pad_left, pad_right)),
                           mode='constant')

    output_h = int((h - kh + pad_top + pad_bottom) / sh) + 1
    output_w = int((w - kw + pad_left + pad_right) / sw) + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            img_region = padded_images[
                :, i * sh:i * sh + kh, j * sw:j * sw + kw]
            conv_result = np.sum(img_region * kernel, axis=(1, 2))
            output[:, i, j] = conv_result

    return output
