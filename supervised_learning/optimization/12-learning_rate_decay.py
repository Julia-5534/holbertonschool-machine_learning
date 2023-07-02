#!/usr/bin/env python3
"""Task 12"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    decayed_learning_rate = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate)
    return decayed_learning_rate
