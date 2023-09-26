#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means:
    X is a numpy.ndarray of shape (n, d) containing the
    dataset that will be used for K-means clustering
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None

    n, d = X.shape
    centroids = np.random.uniform(
        low=X.min(axis=0),
        high=X.max(axis=0),
        size=(k, d))

    return centroids
