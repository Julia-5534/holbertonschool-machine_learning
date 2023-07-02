#!/usr/bin/env python3
"""Task 12"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network"""
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return [loss, accuracy]
