#!/usr/bin/env python3
"""Task 1"""

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


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset:
    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    clss = None

    for _ in range(iterations):
        C_prev = np.copy(C)
        dist = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        clss = np.argmin(dist, axis=0)

        for j in range(k):
            if X[clss == j].size == 0:
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        if np.all(C_prev == C):
            break

    return C, clss
