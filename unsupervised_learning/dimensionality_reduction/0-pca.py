#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def pca(X, var=0.95):
    # Calculate covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)

    # Compute eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues & eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate total variance & cumulative variance explained
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Determine number of components to keep
    num_components = np.argmax(cumulative_variance_ratio >= var) + 1

    # Select top eigenvectors
    selected_eigenvectors = eigenvectors[:, :num_components]

    return selected_eigenvectors
