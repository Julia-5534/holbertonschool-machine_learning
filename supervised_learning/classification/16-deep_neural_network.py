#!/usr/bin/env python3
"""Task 16"""

import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """Initialize DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l_ind, nodes in enumerate(layers, 1):
            if isinstance(nodes, int) and nodes > 0:
                self.weights[f"W{l_ind}"] = (
                    np.random.randn(nodes, nx) *
                    np.sqrt(2 / nx)
                    )

                self.weights[f"b{l_ind}"] = np.zeros((nodes, 1))
                nx = nodes
            else:
                raise TypeError("layers must be a list of positive integers")
