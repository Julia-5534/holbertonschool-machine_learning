#!/usr/bin/env python3
"""Task 3"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over
    a pooling layer of a neural network"""
    # Retrieve dimensions from the input
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    h_out = dA.shape[1]
    w_out = dA.shape[2]

    # Initialize gradient with respect to previous layer
    dA_prev = np.zeros_like(A_prev)

    # Perform pooling backward pass
    for i in range(m):
        a_prev = A_prev[i]
        da = dA[i]
        for h in range(h_out):
            for w in range(w_out):
                for ch in range(c):
                    # Find the corners of the current slice
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        # Perform the backward pass for max pooling
                        a_slice = a_prev[
                            vert_start:vert_end,
                            horiz_start:horiz_end, ch]
                        mask = (a_slice == np.max(a_slice))
                        da_prev_slice = mask * da[h, w, ch]
                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end, ch] += da_prev_slice
                    elif mode == 'avg':
                        # Perform the backward pass for average pooling
                        da_prev_slice = da[h, w, ch] / (kh * kw)
                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            ch
                        ] += np.ones((kh, kw)) * da_prev_slice

    return dA_prev
