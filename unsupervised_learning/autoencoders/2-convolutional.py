#!/usr/bin/env python3
"""Task 2"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Placeholder. Documentations and comments
    to be added later."""
    input_layer = keras.Input(shape=input_dims)
    x = input_layer
    for f in filters:
      x = keras.layers.Conv2D(f, (3, 3),
                              activation='relu',
                              padding='same')(x)
      x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
      
    encoder = keras.Model(input_layer, x)
      
    decoder_input = keras.layers.Input(shape=latent_dims)
    x = decoder_input

    # Decoder
    for f in reversed(filters):
      x = keras.layers.Conv2D(f, (3, 3),
                              activation='relu',
                              padding='same')(x)
      x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(input_dims[-1], (3, 3),
                            activation='sigmoid',
                            padding='same')(x)

    # Create Decoder
    decoder = keras.Model(decoder_input, x)

    #Create Autoencoder
    auto = keras.Model(input_layer, decoder(encoder(input_layer)))

    # Compile model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
