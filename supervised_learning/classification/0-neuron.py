#!/usr/bin/env python3

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classifications"""
    def __init__(self, nx):
        """Initializes Neuron Class"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(nx, 1))
        self.b = 0
        self.A = 0
