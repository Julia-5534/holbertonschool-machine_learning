#!/usr/bin/env python3
"""Task 3"""

# Import necessary libraries
import tensorflow as tf
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder model for dimensionality reduction
    and feature learning.

    Parameters:
    - input_dims (int): The number of features or dimensions in the model
      input.
    - hidden_layers (list): A list containing the number of nodes for each
      hidden layer in the encoder, respectively. The hidden layers should
      be reversed for the decoder.
    - latent_dims (int): The dimensions of the latent space representation.

    Returns:
    - encoder (keras.Model): The encoder model, which should output the
      latent representation, the mean, and the log variance, respectively.
    - decoder (keras.Model): The decoder model.
    - auto (keras.Model): The full autoencoder model.

    This function defines and returns a variational autoencoder model, which
    consists of an encoder and a decoder. The encoder takes the input data
    and maps it to a lower-dimensional latent space.
    Unlike a standard autoencoder, this encoder outputs two parameters (the
    mean and log variance) that define a Gaussian distribution in the latent
    space. A sample from this distribution is taken as the latent
    representation.

    The decoder maps data from the latent space back to the original input
    space, aiming to reconstruct the input data. The encoder and decoder
    models can be used independently, and the full autoencoder model can be
    used for training and inference.

    The autoencoder model is compiled using Adam optimization and binary
    cross-entropy loss. All layers in the encoder and decoder use the ReLU
    activation function, except for the mean and log variance layers in the
    encoder, which should use None, and the last layer in the decoder, which
    should use sigmoid.
    """
    # Input layer that takes data with 'input_dims' features
    inputs = keras.layers.Input(shape=(input_dims,))

    # Encoder network
    x = inputs
    for n in hidden_layers:
        x = keras.layers.Dense(n, activation='relu')(x)

    # Latent layer, the bottleneck of the autoencoder
    mean_log = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick
    def sampling(args):
        mean_log, log_var = args
        batch = tf.shape(mean_log)[0]
        dim = tf.shape(mean_log)[1]
        epsilon = tf.keras.backend.random_normal(
          shape=(batch, dim), mean=0.0, stddev=1.0)
        return mean_log + tf.exp(0.5 * log_var) * epsilon
    bridge = tf.keras.layers.Lambda(sampling)([mean_log, log_var])

    # Create an encoder model that maps the input to the latent space
    encoder = keras.Model(inputs, [bridge, mean_log, log_var], name="encoder")

    # Decoder network
    latent_inputs = keras.layers.Input(shape=(latent_dims,))
    x = latent_inputs
    for n in reversed(hidden_layers):
        x = keras.layers.Dense(n, activation='relu')(x)

    # Output layer that reconstructs the input data
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Create a decoder that maps the latent space to the output
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # Combine the encoder and decoder to create the full autoencoder model
    outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, outputs, name="vae")
    
    # Define VAE loss
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + log_var - tf.square(mean_log) - tf.exp(log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    # Compile autoencoder model with Adam optimization & VAE loss
    auto.add_loss(vae_loss)

    # Compile autoencoder model with Adam optimization
    auto.compile(optimizer='adam')

    # Return the encoder, decoder, and full autoencoder models
    return encoder, decoder, auto
