#!/usr/bin/env python3
"""Task 12"""


def np_elementwise(mat1, mat2):
    """Element-wise +, -, *, /,"""
    sum_result = mat1 + mat2
    diff_result = mat1 - mat2
    prod_result = mat1 * mat2
    div_result = mat1 / mat2
    return sum_result, diff_result, prod_result, div_result
