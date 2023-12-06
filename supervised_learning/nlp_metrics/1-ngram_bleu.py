#!/usr/bin/env python3
"""Task 1"""

from collections import Counter
import math
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the N-gram BLEU score for a given sentence and reference(s).
      - references (list): List of reference sentences (list of words).
      - sentence (list): Candidate sentence (list of words) to be evaluated.
      - n (int): Maximum n-gram order for N-gram BLEU score calculation.
      - bleu_score (float): N-gram BLEU score for the candidate sentence.
      - Initialize a list to store precision values for different
        n-gram orders.
      - Iterate over different n-gram orders (up to the specified maximum).
      - Initialize counters for n-grams in references and the
        candidate sentence.
      - Update counters with n-grams for references and the candidate sentence.
      - Calculate the common n-grams between the candidate and references.
      - Calculate precision for the current n-gram order.
      - Append precision to the list.
      - Calculate brevity penalty based on sentence length and
        minimum reference length.
      - Calculate N-gram BLEU score using precision values and
        average over different n-gram orders.
      - Return the calculated N-gram BLEU score.
    """

    precisions = []
    for i in range(1, n + 1):
        reference_ngrams = []
        sentence_ngrams = []

        for ref in references:
            ref_ngrams = [
              tuple(ref[j:j + i]) for j in range(len(ref) - i + 1)]
            reference_ngrams.extend(ref_ngrams)

        sentence_ngrams = [
          tuple(sentence[j:j + i]) for j in range(len(sentence) - i + 1)]

        overlap_count = sum(
          1 for ngram in sentence_ngrams if ngram in reference_ngrams)

        precision = overlap_count / max(len(sentence_ngrams), 1)
        precisions.append(precision)

    bleu = np.exp(np.sum(np.log(precisions)) / n)

    closest_ref_length = min(
      references, key=lambda ref: abs(len(ref) - len(sentence)))
    brevity_penalty = min(1, len(sentence) / len(closest_ref_length))

    bleu_score = brevity_penalty * bleu
    return bleu_score
