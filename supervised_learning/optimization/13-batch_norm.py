#!/usr/bin/env python3
"""Task 13"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes unactivated output of
    neural network using batch normalization"""
    # Calculate the mean and variance of Z
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift the normalized Z
    Z_scaled = gamma * Z_norm + beta

    return Z_scaled
