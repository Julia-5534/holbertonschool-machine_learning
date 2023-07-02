#!/usr/bin/env python3
"""Task 8"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates training op for neural network using
    RMSProp optimization algorithm.

    Args:
        loss: The loss of the network.
        alpha: The learning rate.
        beta2: The RMSProp weight.
        epsilon: A small number to avoid division by zero.

    Returns:
        The RMSProp optimization operation.
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
