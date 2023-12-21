#!/usr/bin/env python3
"""Task 0"""

import tensorflow_datasets as tfds


class Dataset:
    """
    This class handles the loading and tokenization of the
    ted_hrlr_translate/pt_to_en dataset from TensorFlow Datasets.
    """

    def __init__(self):
        """
        Constructor method. Loads the train and
        validation splits of the dataset and tokenizes them.
        """
        # Load the training split of the dataset
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)

        # Load the validation split of the dataset
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        # Tokenize the training dataset
        self.toke_pt, self.toke_en = self.tokenize_dataset(
            self.data_train)

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
