#!/usr/bin/env python3
"""Task 5"""

import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture"""
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv1 = K.layers.Conv2D(
        filters=6, kernel_size=(5, 5),
        padding='same', activation='relu',
        kernel_initializer=K.initializers.he_normal())(X)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv2 = K.layers.Conv2D(
        filters=16, kernel_size=(5, 5),
        padding='valid', activation='relu',
        kernel_initializer=K.initializers.he_normal())(pool1)
    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten the previous output
    flatten = K.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes
    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=K.initializers.
                         he_normal())(flatten)

    # Fully connected layer with 84 nodes
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=K.initializers.he_normal())(fc1)

    # Fully connected softmax output layer with 10 nodes
    softmax_output = K.layers.Dense(units=10, activation='softmax',
                                    kernel_initializer=K.initializers.
                                    he_normal())(fc2)

    # Create the model
    model = K.models.Model(inputs=X, outputs=softmax_output)

    # Compile the model
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
