#!/usr/bin/env python3
"""Task 2"""

import numpy as np


def variance(X, C):
    """Placeholder"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    n, d = X.shape
    k = C.shape[0]
    if d != C.shape[1]:
        return None
    distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
    min_distances = np.min(distances, axis=0)
    var = (min_distances**2).sum()
    return var
