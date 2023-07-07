#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a
    convolutional layer of a neural network"""
    # Retrieve dimensions from the input
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Calculate output dimensions
    if padding == "same":
        pad_h = int(np.ceil((h_prev * sh - sh + kh - h_prev) / 2))
        pad_w = int(np.ceil((w_prev * sw - sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    # Apply padding to input if necessary
    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                        mode='constant')

    # Calculate output dimensions
    h_out = int((h_prev - kh + 2 * pad_h) / sh) + 1
    w_out = int((w_prev - kw + 2 * pad_w) / sw) + 1

    # Initialize output
    Z = np.zeros((m, h_out, w_out, c_new))

    # Perform the convolution operation
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw
                A_slice_prev = A_prev_pad[
                    :, vert_start:vert_end,
                    horiz_start:horiz_end, :]
                Z[:, i, j, k] = np.sum(
                    A_slice_prev * W[:, :, :, k],
                    axis=(1, 2, 3))

    # Add bias term
    Z = Z + b

    # Apply activation function
    A = activation(Z)

    return A
