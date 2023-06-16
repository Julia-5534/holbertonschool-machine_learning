#!/usr/bin/env python3
"""Task 2"""

import numpy as np

def shuffle_data(X, Y):
    """Shuffles data points in 2 matrices the same way"""
    assert X.shape[0] == Y.shape[0]
    
    # Generate a random permutation
    perm = np.random.permutation(X.shape[0])
    
    # Shuffle the matrices based on the permutation
    shuffled_X = X[perm]
    shuffled_Y = Y[perm]
    
    return shuffled_X, shuffled_Y
