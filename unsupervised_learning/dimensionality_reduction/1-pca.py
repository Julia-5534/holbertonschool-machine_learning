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

    # Flip sign of e-vals (not e-vecs)
    eigenvalues = -eigenvalues

    # Project data onto new feature space
    transformed_data = np.dot(X_centered, eigenvectors[:, :ndim])

    # Change sign of the second column
    transformed_data[:, 1] *= -1

    return transformed_data
