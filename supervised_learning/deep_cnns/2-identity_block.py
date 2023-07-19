#!/usr/bin/env python3
"""Task 2"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Implementation of the identity block as defined in
    Deep Residual Learning for Image Recognition (2015)

    Arguments:
    A_prev -- output tensor of previous layer,
    of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- tuple or list containing F11, F3, F12, respectively:
                F11 -- number of filters in the first 1x1 convolution
                F3 -- number of filters in the 3x3 convolution
                F12 -- number of filters in the second 1x1 convolution

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Retrieve Filters
    F11, F3, F12 = filters

    # Save the input value
    X_shortcut = A_prev

    # First component of main path
    X = K.layers.Conv2D(
        F11, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        kernel_initializer='he_normal')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(
        F3, kernel_size=(3, 3),
        strides=(1, 1), padding='same',
        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(
        F12, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add shortcut value to main path & pass it through a RELU activation
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
