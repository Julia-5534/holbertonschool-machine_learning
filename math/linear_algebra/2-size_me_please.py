#!/usr/bin/env python3
"""Task 2"""


def matrix_shape(matrix):
    """Calculates shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
