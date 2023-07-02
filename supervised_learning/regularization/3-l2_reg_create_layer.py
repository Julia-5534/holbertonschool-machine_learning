#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    kernel_regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_regularizer=kernel_regularizer)
    output = layer(prev)
    return output
