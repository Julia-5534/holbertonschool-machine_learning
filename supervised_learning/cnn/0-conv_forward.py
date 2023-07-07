#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a
    convolutional layer of a neural network"""
    # Retrieve dimensions from input
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Compute output dimensions
    if padding == "same":
        pad_h = int(np.ceil((h_prev * sh - sh + kh - h_prev) / 2))
        pad_w = int(np.ceil((w_prev * sw - sw + kw - w_prev) / 2))
    else:
        pad_h = 0
        pad_w = 0

    # Pad input with zeros if necessary
    A_prev_pad = np.pad(A_prev,
                        ((0, 0), (pad_h, pad_h),
                         (pad_w, pad_w), (0, 0)),
                        mode='constant')

    # Compute output dimensions after padding
    h_out = int((h_prev + 2 * pad_h - kh) / sh + 1)
    w_out = int((w_prev + 2 * pad_w - kw) / sw + 1)

    # Initialize output tensor
    Z = np.zeros((m, h_out, w_out, c_new))

    # Perform the convolution
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_new):
                    # Compute the corners of the current slice
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Extract the slice from A_prev_pad
                    a_slice_prev = A_prev_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end, :]

                    # Convolve the slice with the corresponding kernel
                    Z[i, h, w, c] = np.sum(
                        a_slice_prev * W[:, :, :, c]) + \
                        float(b[:, :, :, c])

    # Apply activation function
    A = activation(Z)

    return A
