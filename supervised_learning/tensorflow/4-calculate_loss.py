#!/usr/bin/env python3
"""Task 4"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate the softmax cross-entropy loss"""
    labels = y
    logits = y_pred
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels,
                                                                  logits))

    return loss
