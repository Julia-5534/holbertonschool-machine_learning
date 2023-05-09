#!/usr/bin/env python3
"""Task 8"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication."""
    if len(mat1[0]) != len(mat2):
        return None
    result = [[0 for j in range(len(mat2[0]))] for i in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            scalar_product = 0
            for k in range(len(mat2)):
                scalar_product += mat1[i][k] * mat2[k][j]
            result[i][j] = scalar_product
    return result
