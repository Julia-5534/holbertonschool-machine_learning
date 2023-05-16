#!/usr/bin/env python3
"""Task 17"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or \
       not all(isinstance(coeff, (int, float)) for coeff in poly) or \
       not isinstance(C, int):
        return None

    integral = [C]
    for i, coeff in enumerate(poly, start=1):
        if not isinstance(coeff, int):
            coeff = float(coeff)
        term = coeff / i
        integral.append(term)

    return integral
