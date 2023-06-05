#!/usr/bin/env python3
"""Task 17"""

import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lix, layer_size in enumerate(layers, 1):
            if type(layer_size) is not int:
                raise TypeError("layers must be a list of positive integers")
            w = np.random.randn(layer_size, nx) * np.sqrt(2/nx)
            self.__weights["W{}".format(lix)] = w
            self.__weights["b{}".format(lix)] = np.zeros((layer_size, 1))
            nx = layer_size

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
