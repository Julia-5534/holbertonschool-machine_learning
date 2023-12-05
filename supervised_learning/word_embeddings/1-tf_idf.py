#!/usr/bin/env python3
"""Task 1"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Create a TF-IDF embedding matrix.
      - Create an instance of TfidfVectorizer
      - Fit and transform the sentences
      - Get the feature names (words)
      - Convert the sparse matrix to a dense matrix
    """

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names_out()
    embeddings = tfidf_matrix.toarray()

    return embeddings, features
