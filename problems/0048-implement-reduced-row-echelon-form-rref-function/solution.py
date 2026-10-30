import numpy as np

def rref(matrix):
    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape

    pivot_row = 0
    tol = 1e-10

    for k in range(n):
        if pivot_row >= m:
            break

        pivot_idx = pivot_row + np.argmax(
            np.abs(matrix[pivot_row:, k])
        )

        matrix[[pivot_row, pivot_idx]] = matrix[[pivot_idx, pivot_row]]

        pivot = matrix[pivot_row, k]

        if abs(pivot) < tol:
            continue

        matrix[pivot_row] /= pivot

        for j in range(m):
            if j != pivot_row:
                matrix[j] -= matrix[j, k] * matrix[pivot_row]

        pivot_row += 1

    matrix[np.abs(matrix) < tol] = 0

    return matrix