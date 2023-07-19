#!/usr/bin/env python3
"""Task 3"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters

    # Shortcut path
    shortcut = Conv2D(
        F12, kernel_size=(1, 1),
        strides=(s, s), padding='valid',
        kernel_initializer='he_normal')(A_prev)
    shortcut = BatchNormalization(axis=3)(shortcut)

    # Main path
    x = Conv2D(
        F11, kernel_size=(1, 1),
        strides=(s, s), padding='valid',
        kernel_initializer='he_normal')(A_prev)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(
        F3, kernel_size=(3, 3),
        strides=(1, 1), padding='same',
        kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(
        F12, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    # Add the main and shortcut paths
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x
