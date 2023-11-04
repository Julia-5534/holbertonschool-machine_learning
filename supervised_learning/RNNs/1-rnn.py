#!/usr/bin/env python3
"""Task 1: Forward Prop for a simple RNN"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    Args:
        rnn_cell (RNNCell): An instance of the RNNCell to be
        used for forward propagation.
        X (np.ndarray): The input data with shape (t, m, i).
        h_0 (np.ndarray): The initial hidden state with shape (m, h).

    Returns:
        H (np.ndarray): All hidden states with shape (t, m, h).
        Y (np.ndarray): All outputs with shape (t, m, o).
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    h_t = h_0

    for step in range(t):
        x_t = X[step]
        h_t, y_t = rnn_cell.forward(h_t, x_t)
        H[step] = h_t
        Y[step] = y_t

    return H, Y
