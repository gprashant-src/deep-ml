import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    A = np.array(A, dtype=float)
    M = A.copy()
    m, n = A.shape

    for i in range(min(m, n)):
        if abs(M[i, i]) <= tol:
            non_zero_rows = np.where(abs(M[i:, i]) > tol)[0]
            if len(non_zero_rows) == 0:
                continue
            pivot_row = i + non_zero_rows[0]
            M[[i, pivot_row]] = M[[pivot_row, i]]
            A[[i, pivot_row]] = A[[pivot_row, i]]

        pivot = M[i, i]
        if abs(pivot) <= tol:
            continue

        for j in range(i + 1, m):
            factor = M[j, i] / pivot
            M[j] -= factor * M[i]

    row_indices = np.where(np.any(np.abs(M) > tol, axis=1))[0]
    return len(row_indices)