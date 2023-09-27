#!/usr/bin/env python3
"""Task 6"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Placeholder"""
    if (
        type(X) is not np.ndarray or
        type(pi) is not np.ndarray or
        type(m) is not np.ndarray or
        type(S) is not np.ndarray
    ):
        return None, None

    if (
        len(X.shape) != 2 or
        len(pi.shape) != 1 or
        len(m.shape) != 2 or
        len(S.shape) != 3
    ):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    g = np.zeros((k, n))

    P = np.zeros((k, n))
    for i in range(k):
        P = pi[i] * pdf(X, m[i], S[i])
        g[i] = P

    g_sum = np.sum(g, axis=0)
    g /= g_sum

    l_unambig = np.sum(np.log(g_sum))

    return g, l_unambig
