#!/usr/bin/env python3
"""Task 1"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a Sparse Autoencoder"""
    # Input layer that takes data with 'input_dims' features
    input_layer = keras.layers.Input(shape=(input_dims,))

    # Encoder network
    x = input_layer
    for n in hidden_layers:
        # Create a Dense layer with 'n' neurons,
        # ReLU activation, and sparsity constraint
        x = keras.layers.Dense(n,
                               activation='relu',
                               activity_regularizer=keras.regularizers.l1(
                                   lambtha))(x)
    # Latent layer, the bottleneck of the autoencoder
    encodeD = keras.layers.Dense(latent_dims, activation='relu')(x)

    # Create an encoder model that maps the input to the latent space
    encoder = keras.Model(input_layer, encodeD)

    # Decoder network
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    x = decoder_input
    for n in reversed(hidden_layers):
        # Create a Dense layer with 'n' neurons and ReLU activation
        x = keras.layers.Dense(n, activation='relu')(x)

    # Output layer that reconstructs the input data
    decoded_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Create a decoder that maps the latent space to the output
    decoder = keras.Model(decoder_input, decoded_output)

    # Combine the encoder and decoder to create the full autoencoder model
    autoencoder_output = decoder(encoder(input_layer))
    auto = keras.Model(input_layer, autoencoder_output)

    # Compile autoencoder model with Adam optimization & Binary Crossentropy
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    # Return the encoder, decoder, and the full autoencoder models
    return encoder, decoder, auto
