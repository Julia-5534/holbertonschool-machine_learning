#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Function to create a tf layer with L2 regularization
    Args:
        prev: tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation:  the activation function that should be used on the layer
        lambtha: L2 regularization parameter
    Returns: output of the new layer
    """
    layer_weights = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=layer_weights,
        name="layer",
        kernel_regularizer=tf.contrib.layers.l2_regularizer(lambtha)
    )

    return layer(prev)
