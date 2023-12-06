#!/usr/bin/env python3
"""Task 0"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag-of-words embedding matrix.
      - Create an instance of CountVectorizer
      - Fit & transform the sentences
      - Get the feature names (words)
      - Convert the sparse matrix to a dense matrix
    """

    vectorizer = CountVectorizer(vocabulary=vocab)
    wc_matrix = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = wc_matrix.toarray()

    return embeddings, features
