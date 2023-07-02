#!/usr/bin/env python3
"""Task 4"""


def moving_average(data, beta):
    """Calculates weighted moving average of a dataset"""
    moving_averages = []
    bias_correction = 1 - beta

    weighted_average = 0
    for i, value in enumerate(data):
        weighted_average = (beta * weighted_average) + ((1 - beta) * value)
        bias_corrected_average = weighted_average / bias_correction
        moving_averages.append(bias_corrected_average)

    return moving_averages
