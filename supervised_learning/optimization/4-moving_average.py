#!/usr/bin/env python3
"""Task 4"""


def moving_average(data, beta):
    """Calulates weighted moving average of a dataset"""
    moving_averages = []
    avg = 0
    bias_correction = avg / (1 - beta ** (i + 1))

    for i in range(len(data)):
        avg = beta * avg + (1 - beta) * data[i]
        moving_averages.append(bias_correction)

    return moving_averages
