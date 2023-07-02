#!/usr/bin/env python3
"""Task 5"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates variable using grad des w/ momentum optimization algorithm"""
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new
    return var_new, v_new
