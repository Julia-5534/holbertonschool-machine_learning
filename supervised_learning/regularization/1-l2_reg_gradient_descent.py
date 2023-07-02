#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights & biases of neural network using
    gradient descent with L2 regularization"""
    m = Y.shape[1]

    # Backpropagation
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev))

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
