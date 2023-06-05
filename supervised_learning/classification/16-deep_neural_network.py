#!/usr/bin/env python3
"""Task 16"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural network performing binary classification"""
    def __init__(self, nx, layers):
        """Initializes DeepNeuralNetwork"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for lix in range(len(layers)):
            if type(layers[lix]) is not int:
                raise TypeError("layers must be a list of positive integers")
            w = np.random.randn(layers[lix], nx)*np.sqrt(2/nx)
            self.weights["W{}".format(lix + 1)] = w
            self.weights["b{}".format(lix + 1)] = np.zeros((layers[lix], 1))
            nx = layers[lix]
