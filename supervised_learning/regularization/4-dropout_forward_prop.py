#!/usr/bin/env python3
"""Task 4"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X

    for i in range(L):
        w = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        a = cache['A' + str(i)]
        z = np.matmul(w, a) + b
        if i == L - 1:
            t = np.exp(z)
            cache['A' + str(i + 1)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            a = np.tanh(z)
            d = np.random.rand(a.shape[0], a.shape[1])
            d = np.where(d < keep_prob, 1, 0)
            cache['D' + str(i + 1)] = d
            cache['A' + str(i + 1)] = a * d / keep_prob
    return cache
