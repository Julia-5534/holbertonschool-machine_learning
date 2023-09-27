#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def pca(X, var=0.95):
    """Perform Principal Component Analysis"""
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute explained variance ratio
    explained_variance_ratio = (S ** 2) / np.sum(S ** 2)

    # Compute cumulative sum of explained variance
    cum_explained_variance = np.cumsum(explained_variance_ratio)

    # Find number of components needed to maintain given variance
    nd = np.argmax(cum_explained_variance >= var) + 1

    # Return the weights matrix
    W = Vt[:nd, :].T

    return W
