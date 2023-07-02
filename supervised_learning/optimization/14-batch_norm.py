#!/usr/bin/env python3
"""Task 14"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates batch normalization layer for neural network in tensorflow"""
    # Create a Dense layer as the base layer
    dense_layer = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"
            )
        )
    # Perform the forward pass
    Z = dense_layer(prev)

    # Calculate the mean and variance of Z
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Create trainable parameters gamma and beta
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    # Normalize Z
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-8)

    # Apply activation function to the normalized Z
    A = activation(Z_norm)

    return A
