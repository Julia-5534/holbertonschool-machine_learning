#!/usr/bin/env python3
"""Task 9"""


def summation_i_squared(n):
    """Calculates ""sum_{i=1}^{n} i^2"""
    if not isinstance(n, int) or n < 1:
        return None

    return sum([i**2 for i in range(1, n+1)])
