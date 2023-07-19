#!/usr/bin/env python3
"""Task 6"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in
    Densely Connected Convolutional Networks"""
    nb_filters = int(nb_filters * compression)

    # 1x1 Convolution (Bottleneck layer)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer='he_normal')(X)

    # Average Pooling
    X = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(X)

    return X, nb_filters
