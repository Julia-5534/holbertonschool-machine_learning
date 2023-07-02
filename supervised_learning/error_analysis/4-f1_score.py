#!/usr/bin/env python3
"""Task 4"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score of a Confusion Matrix"""
    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)

    sensitivities = sensitivity(confusion)
    precisions = precision(confusion)

    for i in range(classes):
        if sensitivities[i] + precisions[i] != 0:
            f1_scores[i] = 2 * (
                sensitivities[i] * precisions[i]) / (
                    sensitivities[i] + precisions[i])

    return f1_scores
