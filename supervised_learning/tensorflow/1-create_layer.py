#!/usr/bin/env python3
"""Task 1"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates layers"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=init)
    return layer
