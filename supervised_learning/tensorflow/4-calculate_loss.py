#!/usr/bin/env python3
"""Task 4"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate the softmax cross-entropy loss"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
