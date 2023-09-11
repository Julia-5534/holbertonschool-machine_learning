#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a dataset"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            cov[i, j] = np.sum((
                X[:, i] - mean[0, i]) * (X[:, j] - mean[0, j])) / (n - 1)

    return mean, cov
