#!/usr/bin/env python3
"""Task 5"""

import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras
    """
    init = K.initializers.he_normal(seed=None)

    layer1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                             padding='same', activation='relu',
                             kernel_initializer=init)(X)
    layer2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer1)
    layer3 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                             padding='valid', activation='relu',
                             kernel_initializer=init)(layer2)
    layer4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer3)
    layer5 = K.layers.Flatten()(layer4)
    layer6 = K.layers.Dense(units=120, activation='relu',
                            kernel_initializer=init)(layer5)
    layer7 = K.layers.Dense(units=84, activation='relu',
                            kernel_initializer=init)(layer6)
    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=init)(layer7)

    model = K.Model(inputs=X, outputs=output)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
