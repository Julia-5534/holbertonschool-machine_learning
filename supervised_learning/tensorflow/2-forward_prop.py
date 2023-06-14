#!/usr/bin/env python3
"""Task 2"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Perform forward propagation through each layer"""
    for i in range(len(layer_sizes)):
        # Create a layer with the specified size and activation function
        outtie = create_layer(x, layer_sizes[i], activations[i])

    return outtie
