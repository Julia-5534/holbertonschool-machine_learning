#!/usr/bin/env python3
"""Task 1"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described in
    Going Deeper with Convolutions (2014)"""
    input_data = Input(shape=(224, 224, 3))

    # First Convolutional layer
    x = Conv2D(
        64, kernel_size=(7, 7),
        strides=(2, 2), padding='same',
        activation='relu')(input_data)
    x = MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2), padding='same')(x)

    # Second Convolutional layer
    x = Conv2D(
        64, kernel_size=(1, 1),
        strides=(1, 1), padding='same',
        activation='relu')(x)
    x = Conv2D(
        192, kernel_size=(3, 3),
        strides=(1, 1), padding='same',
        activation='relu')(x)
    x = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2),
        padding='same')(x)

    # First Inception Block
    x = inception_block(x, [64, 128, 32, 32])

    # Second Inception Block
    x = inception_block(x, [128, 192, 96, 64])

    # MaxPooling layer
    x = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2),
        padding='same')(x)

    # Third Inception Block
    x = inception_block(x, [192, 208, 48, 64])

    # Fourth Inception Block
    x = inception_block(x, [160, 224, 64, 64])

    # Fifth Inception Block
    x = inception_block(x, [128, 256, 64, 64])

    # MaxPooling layer
    x = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2),
        padding='same')(x)

    # Sixth Inception Block
    x = inception_block(x, [256, 320, 128, 128])

    # Seventh Inception Block
    x = inception_block(x, [384, 384, 128, 128])

    # Global Average Pooling
    x = K.layers.GlobalAveragePooling2D()(x)

    # Dropout and Fully Connected layer
    x = Dropout(0.4)(x)
    output = Dense(units=1000, activation='softmax')(x)

    model = K.models.Model(inputs=input_data, outputs=output)

    return model
