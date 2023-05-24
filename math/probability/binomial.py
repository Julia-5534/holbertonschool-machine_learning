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
            mean = sum(data) / len(data)
            vary = sum([(result - mean) ** 2 for result in data]) / len(data)
            self.p = 1 - (vary / mean)
            self.n = round((sum(data) / self.p) / len(data))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        k = int(k)  # Convert k to an integer
        if k < 0 or k > self.n:
            return 0  # If k is out of range, return 0
        coefficient = self._calculate_coefficient(k)
        probability = self._calculate_probability(k)
        return coefficient * probability

    def _calculate_coefficient(self, k):
        """Calculates the coefficient of the binomial distribution"""
        numerator = self._factorial(self.n)
        denominator = self._factorial(self.n - k) * self._factorial(k)
        coefficient = numerator / denominator
        return coefficient

    def _calculate_probability(self, k):
        """Calculates the probability of k successes"""
        probability = self.p ** k * (1 - self.p) ** (self.n - k)
        return probability

    @staticmethod
    def _factorial(n):
        """Calculates the factorial of a number"""
        factorial = 1
        for i in range(1, n + 1):
            factorial *= i
        return factorial
