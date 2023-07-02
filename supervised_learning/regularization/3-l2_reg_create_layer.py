#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization"""
    reggie = tf.contrib.layers.l2_regularizer(lambtha)
    initi = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernal_initializer=initi,
        kernal_regularizer=reggie)
    return (layer(prev))
