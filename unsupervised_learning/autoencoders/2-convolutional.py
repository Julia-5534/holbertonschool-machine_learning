#!/usr/bin/env python3
"""Task 2"""

import tensorflow as tf
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Create a convolutional autoencoder model for dimensionality reduction
    and feature learning.

    Parameters:
    - input_dims (tuple): A tuple of integers containing the dimensions
      of the model input.
    - filters (list): A list containing the number of filters for each
      convolutional layer in the encoder, respectively. The filters should
      be reversed for the decoder.
    - latent_dims (tuple): A tuple of integers containing the dimensions of
      the latent space representation.

    Returns:
    - encoder (keras.Model): The encoder model that maps input data to the
      latent space. Each convolution in the encoder uses a kernel size of
      (3, 3) with same padding and relu activation, followed by max pooling
      of size (2, 2).
    - decoder (keras.Model): The decoder model that maps data from the latent
      space to the output space, attempting to reconstruct the input data. Each
      convolution in the decoder, except for the last two, uses a filter size
      of (3, 3) with same padding and relu activation, followed by upsampling
      of size (2, 2). The second to last convolution uses valid padding. The
      last convolution has the same number of filters as the number of channels
      in input_dims with sigmoid activation and no upsampling.
    - auto (keras.Model): The full autoencoder model that combines the encoder
      and  decoder.

    This function defines & returns a convolutional autoencoder model which
    consists of an encoder and a decoder. The encoder takes the input data
    and maps it to a lower-dimensional latent space using convolutional layers,
    while the decoder maps data from the latent space back to the original
    input space using transposed convolutional layers, aiming to reconstruct
    the input data. The encoder and decoder models can be used independently,
    and the full autoencoder model can be used for training and inference.

    The autoencoder model is compiled using Adam optimization and binary
    cross-entropy loss. All layers in the encoder and decoder use the ReLU
    activation function,except for the last layer in the decoder, which uses
    the sigmoid activation function for output reconstruction.
    """
    # Input layer that takes data with 'input_dims' features
    input_layer = keras.layers.Input(shape=(input_dims,))

    # Encoder network
    x = input_layer
    for f in filters:
        # Create a Conv2D layer with 'f' filters and ReLU activation
        x = keras.layers.Conv2D(f, (3, 3),
                                activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')

    target_shape_after_flattening = keras.backend.int_shape(x)
    # Flatten the 2D feature maps into a 1D vector
    x = keras.layers.Flatten()(x)

    # Create a dense layer for the latent space
    encodeD = keras.layers.Dense(latent_dims, activation='relu')(x)

    # Create an encoder model that maps the input to the latent space
    encoder = keras.Model(input_layer, encodeD)

    # Decoder network
    decoder_input = keras.layers.Input(shape=latent_dims)
    # Reshape the 1D latent vector into 2D feature maps
    x = keras.layers.Reshape(target_shape_after_flattening)(decoder_input)
    for f in reversed(filters[:-1]):
        # Create a Conv2D layer with 'f' filters, ReLU activation,
        # and same padding
        x = keras.layers.Conv2D(f, (3, 3),
                                activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Second to last convolution with 'valid' padding
    x = keras.layers.Conv2D(filters[-1], (3, 3),
                            activation='relu', padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Output layer that reconstructs the input data
    decoded_output = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                         activation='sigmoid',
                                         padding='same')(x)

    # Create a decoder that maps the latent space to the output
    decoder = keras.Model(decoder_input, decoded_output)

    # Combine the encoder and decoder to create the full autoencoder model
    autoencoder_output = decoder(encoder(input_layer))
    auto = keras.Model(input_layer, autoencoder_output)

    # Compile autoencoder model with Adam optimization & Binary Crossentropy
    auto.compile(optimizer=tf.optimizers.Adam(),
                 loss=tf.losses.binary_crossentropy)

    # Return the encoder, decoder, and the sparse autoencoder models
    return encoder, decoder, auto
