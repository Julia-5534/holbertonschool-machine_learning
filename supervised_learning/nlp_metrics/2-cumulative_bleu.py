#!/usr/bin/env python3
"""Task 2"""

from collections import Counter
import math


def cumulative_bleu(references, sentence, n):
    """
    Calculates the Cumulative BLEU score for a given
    sentence and reference(s).
      - references (list): List of reference sentences (list of words).
      - sentence (list): Candidate sentence (list of words) to be evaluated.
      - n (int): Maximum n-gram order for cumulative BLEU score calculation.
      - bleu_score (float): Cumulative BLEU score for the candidate sentence.
      - Initialize a list to store precision values for different
        n-gram orders.
      - Iterate over different n-gram orders (up to the specified maximum).
      - Initialize counters for n-grams in references and the
        candidate sentence.
      - Update counters with n-grams for references and the candidate sentence.
      - Calculate the common n-grams between the candidate and references.
      - Calculate precision for the current n-gram order.
      - Append precision to the list.
      - Calculate brevity penalty based on sentence length and minimum
        reference length.
      - Calculate cumulative BLEU score using precision values and
        specified n-gram order.
      - Return the calculated cumulative BLEU score.
    """
    precisions = []
    for i in range(1, n + 1):
        reference_ngrams = Counter()
        candidate_ngrams = Counter()

        for reference in references:
            reference_ngrams.update(
                zip(*[reference[j:] for j in range(i)]))

        candidate_ngrams.update(
            zip(*[sentence[j:] for j in range(i)]))

        common_ngrams = candidate_ngrams & reference_ngrams
        precision = sum(common_ngrams.values()) / max(
            1, sum(candidate_ngrams.values()))

        precisions.append(precision)

    brevity_penalty = min(1.0, len(sentence) / min(
        len(ref) for ref in references))
    bleu_score = brevity_penalty * math.exp(
        sum(math.log(p) for p in precisions) / n)

    return bleu_score
