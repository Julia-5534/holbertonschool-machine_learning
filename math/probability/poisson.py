#!/usr/bin/env python3
"""Poisson Distribution"""


class Poisson:
    """Represents a Poisson Distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson Distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = self.calculate_lambtha(data)

    def calculate_lambtha(self, data):
        """Calculates lambtha of the distribution"""
        n = len(data)
        total = sum(data)
        return total / n

    def pmf(self, k):
        """Calculates value of PMF for given number of 'successes'"""
        k = int(k)
        if k < 0:
            return 0
        else:
            e_to_the_power = -self.lambtha
            product = 1
            for i in range(1, k+1):
                product *= self.lambtha / i
            return product * pow(2.71828, e_to_the_power)
