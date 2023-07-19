#!/usr/bin/env python3
"""Task 3"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters

    X_shortcut = A_prev

    # Main path
    X = K.layers.Conv2D(
        F11, kernel_size=(1, 1),
        strides=(s, s), padding='valid',
        kernel_initializer='he_normal')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        F3, kernel_size=(3, 3),
        strides=(1, 1), padding='same',
        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        F12, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    X_shortcut = K.layers.Conv2D(
        F12, kernel_size=(1, 1),
        strides=(s, s), padding='valid',
        kernel_initializer='he_normal')(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add the main and shortcut paths
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
