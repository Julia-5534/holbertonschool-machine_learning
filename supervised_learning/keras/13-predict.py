#!/usr/bin/env python3
"""Task 13"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes predictions using a neural network"""
    predictions = network.predict(data, verbose=verbose)
    return predictions
