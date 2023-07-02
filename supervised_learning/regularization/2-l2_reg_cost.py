#!/usr/bin/env python3
"""Task 2"""

import tensorflow as tf


def l2_reg_cost(cost):
    """Calculates the cost of neural
    network with L2 regularization"""
    return (cost + tf.losses.get_regularization_losses())
