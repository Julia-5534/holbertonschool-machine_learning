#!/usr/bin/env python3
"""Task 2"""

import numpy as np

def precision(confusion):
    classes = confusion.shape[0]
    precision_values = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        false_positives = np.sum(confusion[:, i])
        precision_values[i] = true_positives / false_positives

    return precision_values
