#!/usr/bin/env python3
"""Task 5"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in
    Densely Connected Convolutional Networks"""
    for i in range(layers):
        # Bottleneck layer (1x1 Convolution)
        X_bottleneck = K.layers.BatchNormalization()(X)
        X_bottleneck = K.layers.Activation('relu')(X_bottleneck)
        X_bottleneck = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer='he_normal')(X_bottleneck)

        # Convolution layer (3x3 Convolution)
        X_conv = K.layers.BatchNormalization()(X_bottleneck)
        X_conv = K.layers.Activation('relu')(X_conv)
        X_conv = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_normal')(X_conv)

        # Concatenate the output of each layer with the input X
        X = K.layers.concatenate([X, X_conv])
        nb_filters += growth_rate

    return X, nb_filters
