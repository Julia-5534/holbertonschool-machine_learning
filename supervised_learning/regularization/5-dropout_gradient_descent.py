#!/usr/bin/env python3
"""Task 5"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights of neural network with
    Dropout regularization using gradient descent"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        dW = np.matmul(dZ, A.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(W.T, dZ)
        if i > 1:
            dA = dA * (1 - A * A)
            dA = dA * cache["D" + str(i - 1)]
            dA = dA / keep_prob
        dZ = dA
        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db
