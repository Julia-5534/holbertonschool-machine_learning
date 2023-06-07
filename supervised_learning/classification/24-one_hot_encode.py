#!/usr/bin/env python3
"""Task 24"""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None

    m = len(Y)
    oh_mat = np.zeros((classes, m))

    for i in range(m):
        label = Y[i]
        if label < 0 or label >= classes:
            return None
        oh_mat[label, i] = 1

    return oh_mat
