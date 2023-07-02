#!/usr/bin/env python3
"""Task 0"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a Confusion Matrix"""
    assert labels.shape == logits.shape

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.float32)

    for i in range(labels.shape[0]):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion[true_label, predicted_label] += 1.0

    return confusion
