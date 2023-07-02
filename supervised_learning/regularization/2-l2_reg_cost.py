#!/usr/bin/env python3
"""Task 2"""

import tensorflow as tf


def l2_reg_cost(cost):
    """Calculates the cost of neural
    network with L2 regularization"""
    l2_loss = tf.losses.get_regularization_loss()
    l2_cost = cost + l2_loss
    return l2_cost
