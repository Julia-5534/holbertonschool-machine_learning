#!/usr/bin/env python3
"""Task 17"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not all(isinstance(coeff, int) for coeff in poly) \
            or not isinstance(C, int):
        return None

    integral = [coeff // (index + 1) if coeff % (index + 1) == 0 else coeff / (index + 1)
                for index, coeff in enumerate(poly)]
    integral.insert(0, C)

    return integral
