#!/usr/bin/env python3
"""Task 7"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks"""
    # Input layer
    inputs = K.Input(shape=(224, 224, 3))
    nb_filters = 64

    # Initial Convolution (7x7, strides=2)
    X = K.layers.Conv2D(filters=nb_filters,
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer='he_normal')(inputs)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 6)

    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    X = K.layers.GlobalAveragePooling2D()(X)

    # Fully Connected Layer
    outputs = K.layers.Dense(units=1000, activation='softmax')(X)

    # Create the model
    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
