#!/usr/bin/env python3
"""Task 12"""
import numpy as np


def np_elementwise(mat1, mat2):
    """Element-wise +, -, *, /,"""
    sum_ = np.add(mat1, mat2)
    diff = np.subtract(mat1, mat2)
    prod = np.multiply(mat1, mat2)
    quot = np.divide(mat1, mat2)
    return sum_, diff, prod, quot
