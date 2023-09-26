#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def pca(X, var=0.95):
    """Placeholder"""
    # Calculate covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)

    # Compute eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues & eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Compute cumulative sum of eigenvalues
    cumul_eigenvalues = np.cumsum(sorted_eigenvalues)

    # Compute total sum of eigenvalues
    total_eigenvalues = np.sum(sorted_eigenvalues)

    # Find number of eigenvalues that maintain the given variance
    ndim = np.argmax(cumul_eigenvalues / total_eigenvalues >= var) + 1

    # Select top ndim eigenvectors
    return sorted_eigenvectors[:, :ndim]
