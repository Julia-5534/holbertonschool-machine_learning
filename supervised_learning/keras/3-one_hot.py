#!/usr/bin/env python3
"""Task 3"""

import tensorflow.keras.utils as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""
    one_hot_matrix = K.to_categorical(labels, num_classes=classes)
    return one_hot_matrix
