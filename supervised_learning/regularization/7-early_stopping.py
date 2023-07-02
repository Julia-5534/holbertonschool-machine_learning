#!/usr/bin/env python3
"""Task 7"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop Gradient Descent early"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    if count != patience:
        return False, count
    else:
        return True, count
