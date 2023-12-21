#!/usr/bin/env python3
"""Task 3"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """
    This class handles the loading, tokenization, encoding, filtering,
    and batching of the ted_hrlr_translate/pt_to_en dataset from
    TensorFlow Datasets.
    """

    def __init__(self, batch_size, max_len):
        """
        Constructor method. Loads the train and validation splits of the
        dataset, tokenizes them, encodes them, filters them by length,
        batches them, and prefetches them.

        Parameters:
        batch_size (int): The size of the batches.
        max_len (int): The maximum length of sequences.
        """
        # Set the batch size and maximum length
        self.batch_size = batch_size
        self.max_len = max_len

        # Load the training split of the dataset
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)

        # Load the validation split of the dataset
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        # Tokenize the training dataset
        self.toke_pt, self.toke_en = self.tokenize_dataset(self.data_train)

        # Map the encoding function to the training data
        self.data_train = self.data_train.map(self.tf_encode)

        # Map the encoding function to the validation data
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Filter the training data by length
        self.data_train = self.data_train.filter(self.filter_max_length)

        # Cache the training data
        self.data_train = self.data_train.cache()

        # Batch the training data
        self.data_train = self.data_train.padded_batch(self.batch_size)

        # Prefetch the training data
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        # Filter the validation data by length
        self.data_valid = self.data_valid.filter(self.filter_max_length)

        # Batch the validation data
        self.data_valid = self.data_valid.padded_batch(self.batch_size)

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset.

        Parameters:
        data (tf.data.Dataset): The dataset to tokenize.

        Returns:
        toke_pt (tfds.deprecated.text.SubwordTextEncoder):
        The tokenizer for Portuguese text.
        toke_en (tfds.deprecated.text.SubwordTextEncoder):
        The tokenizer for English text.
        """
        # Build a tokenizer for the Portuguese text
        toke_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)

        # Build a tokenizer for the English text
        toke_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)

        return toke_pt, toke_en

    def encode(self, pt, en):
        """
        Encodes a translation pair into tokens.

        Parameters:
        pt (tf.Tensor): The Portuguese text.
        en (tf.Tensor): The English text.

        Returns:
        pt_tokens (List[int]): The tokenized Portuguese text.
        en_tokens (List[int]): The tokenized English text.
        """
        # Tokenize the Portuguese text
        pt_tokens = [self.toke_pt.vocab_size] + self.toke_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size+1]

        # Tokenize the English text
        en_tokens = [self.toke_en.vocab_size] + self.toke_en.encode(
            en.numpy()) + [self.toke_en.vocab_size+1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Wraps the encode method in a TensorFlow py_function.

        Parameters:
        pt (tf.Tensor): The Portuguese text.
        en (tf.Tensor): The English text.

        Returns:
        pt (tf.Tensor): The tokenized Portuguese text.
        en (tf.Tensor): The tokenized English text.
        """
        # Wrap the encode method in a TensorFlow py_function
        pt, en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])

        # Set the shape of the tensors
        pt.set_shape([None])
        en.set_shape([None])

        return pt, en

    def filter_max_length(self, pt, en):
        """
        Filters sequences by length.

        Parameters:
        pt (tf.Tensor): The Portuguese text.
        en (tf.Tensor): The English text.

        Returns:
        (tf.Tensor): A boolean tensor indicating whether both
        sequences are shorter than max_len.
        """
        # Check if both sequences are shorter than max_len
        return tf.logical_and(tf.size(pt) <= self.max_len,
                              tf.size(en) <= self.max_len)
