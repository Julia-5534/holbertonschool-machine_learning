#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural
    network with L2 regularization"""
    regularization_term = 0

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        regularization_term += np.sum(np.square(W))

    l2_regularization_cost = cost + (lambtha / (2 * m)) * regularization_term
    return l2_regularization_cost
