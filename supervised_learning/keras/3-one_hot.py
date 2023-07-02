#!/usr/bin/env python3
"""Task 3"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""
    return K.utils.to_categorical(labels, classes)
