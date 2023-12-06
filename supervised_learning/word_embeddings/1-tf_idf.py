#!/usr/bin/env python3
"""Task 1"""

import numpy as np


def calc_tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix manually.
      - sentences: A list of sentences to analyze.
      - vocab: A list of vocabulary words to use for analysis.
        If None, all words within sentences are used.
      - embeddings: A numpy.ndarray of shape (s, f) containing
        the embeddings.
      - features: A list of the features used for embeddings.
      - Tokenize sentences into words.
      - Create a vocabulary if not provided.
      - Create a mapping from words to indices.
      - Initialize a matrix to store TF-IDF values.
      - Calculate TF-IDF values.
      - Get the feature names(words).
    """

    tokenized_sentences = [sentence.lower().split() for sentence in sentences]

    if vocab is None:
        vocab = set(
          word for sentence in tokenized_sentences for word in sentence)

    word_to_index = {word: index for index, word in enumerate(vocab)}
    tfidf_matrix = np.zeros((len(sentences), len(vocab)))

    for i, sentence in enumerate(tokenized_sentences):
        word_counts = {word: sentence.count(word) for word in set(sentence)}
        for word, count in word_counts.items():
            if word in vocab:
                tf = count / len(sentence)
                idf = np.log(len(sentences) / (1 + sum(
                    1 for s in tokenized_sentences if word in s))) + 1
                tfidf_matrix[i, word_to_index[word]] = tf * idf

    features = list(vocab)

    return tfidf_matrix, features


def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding matrix.
      - Call the calc_tf_idf function to calculate TF-IDF values.
    """

    tfidf_matrix, features = calc_tf_idf(sentences, vocab=vocab)

    return tfidf_matrix, features
