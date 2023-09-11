#!/usr/bin/env python3
"""Task 5"""

import numpy as np


def determinant(matrix):
    """Calculates the Determinant of a Matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for i in range(n):
        submatrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += (-1) ** i * matrix[0][i] * determinant(submatrix)
    return det


def minor(matrix):
    """Calculates the Minor of a Matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j+1:] for row in (
                matrix[:i]+matrix[i+1:])]
            det = determinant(sub_matrix)
            minor_row.append(det)
        minor_matrix.append(minor_row)
    return minor_matrix


def cofactor(matrix):
    """Calculates the Cofactor of a Matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    cofactor_matrix = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j+1:] for row in (
                matrix[:i]+matrix[i+1:])]
            det = determinant(sub_matrix)
            cofactor_row.append(((-1)**(i+j))*det)
        cofactor_matrix.append(cofactor_row)
    return cofactor_matrix


def adjugate(matrix):
    """Calculates the Adjugate matrix of a Matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Calculate the cofactor matrix
    cofactor_matrix = cofactor(matrix)

    # Transpose the cofactor matrix to get the adjugate matrix
    adjugate_matrix = [[
        cofactor_matrix[j][i] for j in range(n)] for i in range(n)]

    return adjugate_matrix


def inverse(matrix):
    """Calculates the Inverse of a Matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Calculate the determinant of the matrix
    det = determinant(matrix)

    # Check if the matrix is singular (determinant is zero)
    if det == 0:
        return None

    # Calculate the adjugate matrix
    adjugate_matrix = adjugate(matrix)

    # Calculate the inverse matrix by dividing each
    # element of the adjugate matrix by the determinant
    inverse_matrix = [[
        adjugate_matrix[i][j] / det for j in range(n)] for i in range(n)]

    return inverse_matrix


def definiteness(matrix):
    """Calculates the Definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    pos_eigenvalues = np.sum(eigenvalues > 0)
    zero_eigenvalues = np.sum(eigenvalues == 0)

    if pos_eigenvalues == matrix.shape[0]:
        return "Positive definite"
    elif pos_eigenvalues > 0 and zero_eigenvalues > 0:
        return "Positive semi-definite"
    elif zero_eigenvalues == matrix.shape[0]:
        return "Indefinite"
    elif pos_eigenvalues == 0:
        neg_eigenvalues = np.sum(eigenvalues < 0)
        if neg_eigenvalues == matrix.shape[0]:
            return "Negative definite"
        elif neg_eigenvalues > 0:
            return "Negative semi-definite"

    return None
