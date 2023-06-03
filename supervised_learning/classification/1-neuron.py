#!/usr/bin/env python3
"""Task 1"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Initialize the Neuron class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A

    def forward_prop(self, X):
        """Perform forward propagation"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
