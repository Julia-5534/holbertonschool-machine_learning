#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def normalize(X, m, s):
    """Normalizes a matrix"""
    normalized_X = (X - m) / s
    return normalized_X
