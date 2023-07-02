#!/usr/bin/env python3
"""Task 11"""

import tensorflow.keras as K


def save_config(network, filename):
    """Save the configuration"""
    config = network.to_json()
    with open(filename, 'w') as file:
        file.write(config)


def load_config(filename):
    """Load the configuration"""
    with open(filename, 'r') as file:
        config = file.read()
    return K.models.model_from_json(config)
