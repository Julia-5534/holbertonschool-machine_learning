#!/usr/bin/env python3
"""Task 4"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise."""
    # Check if the arrays have the same length
    if len(arr1) != len(arr2):
        return None

    # Create a new list to store the result
    result = []

    # Iterate over the arrays and add their elements
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    # Return the resulting list
    return result
