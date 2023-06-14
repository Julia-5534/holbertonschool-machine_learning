#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate Accuracy"""
    # Count the number of correct predictions
    correct_predictions = tf.equal(tf.argmax(y, axis=1),
                                   tf.argmax(y_pred, axis=1))

    # Calculate accuracy as ratio of correct predictions to all predictions
    accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                      dtype=tf.float32))

    return accuracy
