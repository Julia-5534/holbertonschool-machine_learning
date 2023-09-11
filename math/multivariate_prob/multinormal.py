#!/usr/bin/env python3
"""Tasks 2 & 3"""

import numpy as np


class MultiNormal:
    """Multivariate Normal Distribution"""
    def __init__(self, data):
        """Initializes the Class"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.zeros((d, d))

        for i in range(d):
            for j in range(d):
                self.cov[i, j] = np.sum(
                    (data[i] - self.mean[i]) * (
                        data[j] - self.mean[j])) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != self.mean.shape:
            raise ValueError(
                "x must have the shape {}".format(self.mean.shape))

        d = self.mean.shape[0]
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        x_minus_mean = x - self.mean
        exponent = -0.5 * x_minus_mean.T @ inv_cov @ x_minus_mean
        normalization = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_cov))

        return '{:.19f}'.format(normalization * np.exp(exponent)[0, 0])
