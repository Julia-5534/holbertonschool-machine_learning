#!/usr/bin/env python3
"""Task 10"""


def np_shape(matrix):
    """Calculates the shape of a numpy.ndarray"""
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    return (rows, cols)
