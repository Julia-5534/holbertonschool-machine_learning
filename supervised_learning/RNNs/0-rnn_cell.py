#!/usr/bin/env python3
"""Task 0"""

import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        # Initialize the dimensions of the data, hidden state, and outputs
        self.i = i
        self.h = h
        self.o = o
        
        # Initialize the weights and biases
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
    
    def forward(self, h_prev, x_t):
        # Perform forward propagation for one time step
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True) # Apply softmax activation function
        
        return h_next, y
