#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def pca(X, ndim):
    """Placeholder"""
    # Standardize data (mean centering)
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Calculate e-vals & e-vecs of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort e-vals & e-vecs in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Ensure that the signs of the eigenvectors match the expected array
    for i in range(ndim):
        if np.sum(np.abs(eigenvectors[:, i])) != 0 and eigenvectors[0, i] < 0:
            eigenvectors[:, i] = -eigenvectors[:, i]

    # Project data onto new feature space
    transformed_data = np.dot(X_centered, eigenvectors[:, :ndim])

    return transformed_data
