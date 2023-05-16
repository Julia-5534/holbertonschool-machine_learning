#!/usr/bin/env python3
"""Task 10"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    if len(derivative) == 0:
        return [0]

    return derivative
