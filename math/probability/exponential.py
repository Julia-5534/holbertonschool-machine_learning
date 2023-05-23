#!/usr/bin/env python3
"""Exponential"""


class Exponential:
    """Represents an Exponential Distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize the distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1.0 / (sum(data) / len(data))
