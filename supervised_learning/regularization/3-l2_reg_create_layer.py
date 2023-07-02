#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization"""
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, activation=activation,
        kernel_regularizer=regularizer, kernal_initializer=init)
    return layer(prev)
