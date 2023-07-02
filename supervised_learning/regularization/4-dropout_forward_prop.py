#!/usr/bin/env python3
"""Task 4"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {}
    A = X
    cache['A0'] = A

    for i in range(1, L+1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.dot(W, A) + b
        if i < L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D / keep_prob
            cache['D' + str(i)] = D
        else:
            expZ = np.exp(Z)
            A = expZ / np.sum(expZ, axis=0, keepdims=True)
        cache['Z' + str(i)] = Z
        cache['A' + str(i)] = A

    return cache
