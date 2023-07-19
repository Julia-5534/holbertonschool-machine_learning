#!/usr/bin/env python3
"""Task 4"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture.

    Returns:
    model -- a Keras model representing ResNet-50
    """
    input_shape = (224, 224, 3)

    # Define the input tensor
    X_input = K.layers.Input(input_shape)

    # Stage 1: Convolutional layer followed by BatchNorm and ReLU
    X = K.layers.Conv2D(
        64, kernel_size=(7, 7),
        strides=(2, 2), padding='same',
        kernel_initializer='he_normal')(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2: Projection block followed by two Identity blocks
    X = projection_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, filters=[64, 64, 256])
    X = identity_block(X, filters=[64, 64, 256])

    # Stage 3: Projection block followed by three Identity blocks
    X = projection_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])

    # Stage 4: Projection block followed by five Identity blocks
    X = projection_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])

    # Stage 5: Projection block followed by two Identity blocks
    X = projection_block(X, filters=[512, 512, 2048])
    X = identity_block(X, filters=[512, 512, 2048])
    X = identity_block(X, filters=[512, 512, 2048])

    # Average pooling
    X = K.layers.GlobalAveragePooling2D()(X)

    # Fully connected layer
    X = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer='he_normal')(X)

    # Create the model
    model = K.models.Model(
        inputs=X_input,
        outputs=X, name='ResNet50')

    return model
