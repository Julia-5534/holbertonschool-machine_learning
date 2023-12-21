#!/usr/bin/env python3
"""Task 4"""

import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for a sequence.

    Parameters:
    seq (tf.Tensor): The sequence to mask.

    Returns:
    seq (tf.Tensor): A padding mask for the sequence.
    """
    # Cast the sequence to float32 and invert it
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # Add extra dimensions to the sequence
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask for a sequence.

    Parameters:
    size (int): The size of the sequence.

    Returns:
    mask (tf.Tensor): A look-ahead mask for the sequence.
    """
    # Create a matrix of ones and subtract its lower triangular part
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    return mask  # (seq_len, seq_len)


def create_masks(inputs, target):
    """
    Creates padding and look-ahead masks for input and target sequences.

    Parameters:
    inputs (tf.Tensor): The input sequence.
    target (tf.Tensor): The target sequence.

    Returns:
    encoder_mask (tf.Tensor): A padding mask for the input sequence.
    combined_mask (tf.Tensor): A combined mask for the target sequence.
    decoder_mask (tf.Tensor): A padding mask for the input sequence.
    """
    # Create a padding mask for the input sequence
    encoder_mask = create_padding_mask(inputs)

    # Create a padding mask for the input sequence
    decoder_mask = create_padding_mask(inputs)

    # Create a look-ahead mask for the target sequence
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    # Create a padding mask for the target sequence
    dec_target_padding_mask = create_padding_mask(target)

    # Combine the look-ahead and padding masks
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
