#!/usr/bin/env python3
"""Task 8"""

import numpy as np


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates training op for neural network using
    RMSProp optimization algorithm.

    Args:
        loss: The loss of the network.
        alpha: The learning rate.
        beta2: The RMSProp weight.
        epsilon: A small number to avoid division by zero.

    Returns:
        The RMSProp optimization operation.
    """
    var_list = np.array(list(loss.keys()))
    s = {}
    for var in var_list:
        s[var] = np.zeros_like(loss[var])

    def train_op():
        nonlocal s
        for var in var_list:
            s[var] = beta2 * s[var] + (1 - beta2) * np.square(loss[var])
            loss[var] -= (alpha * loss[var]) / (np.sqrt(s[var]) + epsilon)

    return train_op
