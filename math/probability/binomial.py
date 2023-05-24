#!/usr/bin/env python3
"""Binomial"""


class Binomial:
    """Binomial Distribution Class"""
    def __init__(self, data=None, n=1, p=0.5):
        """Initializes Binomial Distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.n, self.p = self.estimate_parameters(data)

    def estimate_parameters(self, data):
        p = sum(data) / len(data)
        n = round(p * len(data))
        p = n / len(data)
        return int(n), float(p)
