#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix"""
    classes = confusion.shape[0]
    sensitivity_values = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        false_negatives = np.sum(confusion[i, :])
        sensitivity_values[i] = true_positives / false_negatives

    return sensitivity_values
