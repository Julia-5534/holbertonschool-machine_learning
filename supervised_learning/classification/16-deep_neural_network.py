#!/usr/bin/env python3
"""Task 16"""

import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """Initializes DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lix in range(1, self.__L + 1):
            if lix == 1:
                self.__weights['W' + str(lix)] = np.random.randn(layers[lix-1],
                                                                 nx) \
                                                     * np.sqrt(2/nx)
            else:
                self.__weights['W' + str(lix)] = \
                    np.random.randn(layers[lix-1], layers[lix-2]) \
                    * np.sqrt(2/layers[lix-2])
            self.__weights['b' + str(lix)] = np.zeros((layers[lix-1], 1))

    @property
    def L(self):
        return self.__L

    @L.setter
    def L(self, value):
        if not isinstance(value, int):
            raise TypeError("L must be an integer")
        if value < 1:
            raise ValueError("L must be a positive integer")
        self.__L = value

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
