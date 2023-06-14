#!/usr/bin/env python3
"""Task 5"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates training operation for network using gradient descent"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
