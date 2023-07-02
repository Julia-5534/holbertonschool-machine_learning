#!/usr/bin/env python3
"""Task 6"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using Dropout"""
    initializer = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(keep_prob)
    dense = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer)
    dropout_layer = dropout(dense(prev))
    return dropout_layer
