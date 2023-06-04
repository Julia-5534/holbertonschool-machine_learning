#!/usr/bin/env python3
"""Task 16"""

import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """Initialize the DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l_ind in range(1, self.__L + 1):
            if isinstance(layers[l_ind - 1], int) and layers[l_ind - 1] > 0:
                key_w = 'W' + str(l_ind)
                key_b = 'b' + str(l_ind)
                self.__weights[key_w] = (
                    np.random.randn(layers[l_ind - 1], nx) *
                    np.sqrt(2 / nx)
                    )

                self.__weights[key_b] = np.zeros((layers[l_ind - 1], 1))
            else:
                raise TypeError("layers must be a list of positive integers")

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @L.setter
    def L(self, value):
        if not isinstance(value, int):
            raise TypeError("L must be an integer")
        if value < 1:
            raise ValueError("L must be a positive integer")
        self.__L = value
