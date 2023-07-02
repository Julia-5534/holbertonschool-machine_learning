#!/usr/bin/env python3
"""Task 1"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds neural network with Keras library"""
    inputs = K.Input(shape=(nx,))
    x = inputs
    regularizer = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=regularizer)(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
