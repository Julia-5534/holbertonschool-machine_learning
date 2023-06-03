#!/usr/bin/env python3
"""Task 0"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Initialize the Neuron class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")

        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0

    def forward_prop(self, X):
        """Perform forward propagation"""
        z = np.matmul(self.W, X) + self.b
        self.A = 1 / (1 + np.exp(-z))
        return self.A
