#!/usr/bin/env python3
"""Task 1"""

import tensorflow as tf

def create_layer(prev, n, activation):
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, act=activation,
                            kernel_initializer=init, name='layer')
    output = layer(prev)
    return output
