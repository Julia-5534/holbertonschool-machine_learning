#!/usr/bin/env python3
"""Task 11"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates learning rate using inverse time decay in numpy"""
    updated_alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return updated_alpha
