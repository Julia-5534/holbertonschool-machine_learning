#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def pca(X, ndim=3):
    """Placeholder"""
    # Calculate covariance matrix
    covariance_matrix = np.cov(X, rowvar=False)
    
    # Compute eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues & eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select top ndim eigenvectors
    selected_eigenvectors = eigenvectors[:, :ndim]
    
    return selected_eigenvectors
