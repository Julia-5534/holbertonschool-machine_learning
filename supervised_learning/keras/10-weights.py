#!/usr/bin/env python3
"""Task 10"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Save weights"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Load weights"""
    network.load_weights(filename)
