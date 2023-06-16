#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def normalization_constants(X):
    """Calculates norm constants of a matrix"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
