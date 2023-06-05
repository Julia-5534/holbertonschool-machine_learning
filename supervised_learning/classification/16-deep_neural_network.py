#!/usr/bin/env python3
"""Task 16"""

import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
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

        for lix in range(self.__L):
            if not isinstance(layers[lix], int) or layers[lix] < 1:
                raise TypeError("layers must be a list of positive integers")

            if lix == 0:
                self.__weights['W' + str(lix + 1)] = np.random.randn(layers[lix], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(lix + 1)] = np.random.randn(layers[lix], layers[lix - 1]) \
                    * np.sqrt(2 / layers[lix - 1])

            self.__weights['b' + str(lix + 1)] = np.zeros((layers[lix], 1))

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
