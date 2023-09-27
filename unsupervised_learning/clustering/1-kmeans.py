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
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
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

    centroids = initialize(X, k)
    clss = None

    for _ in range(iterations):
        cen_prev = np.copy(centroids)
        dist = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=-1))
        clss = np.argmin(dist, axis=0)

        for j in range(k):
            if X[clss == j].size == 0:
                centroids[j] = initialize(X, 1)
            else:
                centroids[j] = np.mean(X[clss == j], axis=0)

        if np.all(cen_prev == centroids):
            break

    return centroids, clss
