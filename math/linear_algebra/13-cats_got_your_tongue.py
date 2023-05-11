#!/usr/bin/env python3
"""Task 13"""


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices on a given axis"""
    funcs = {0: lambda a, b: a + b,
             1: lambda a, b: [x + y for x, y in zip(a, b)]}
    return funcs[axis]([list(row) for row in mat1],
                       [list(row) for row in mat2])
