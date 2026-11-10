import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    if not vectors:return True
    # Your code here
    vectors = np.array(vectors, dtype=float)

    M = vectors.copy()
    m, n = vectors.shape

    tol = 1e-6

    for i in range(min(m, n)):
        if abs(M[i, i]) <= tol:
            non_zero_rows = np.where(abs(M[i:, i]) > tol)[0]
            if len(non_zero_rows) == 0:
                continue
            pivot_row = i + non_zero_rows[0]
            M[[i, pivot_row]] = M[[pivot_row, i]]

        pivot = M[i, i]
        if abs(pivot) <= tol:
            continue

        for j in range(i + 1, m):
            factor = M[j, i] / pivot
            M[j] -= factor * M[i]

    row_indices = np.where(np.any(np.abs(M) > tol, axis=1))[0]
    return len(row_indices) == m